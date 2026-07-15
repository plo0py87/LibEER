"""
Same AFTL (Adapter-Finetuned Transfer Learning) protocol as AFTL_train.py, but the
PRETRAIN phase adds a weak adversarial domain-confusion loss (DANN-style, via a
gradient-reversal layer) that tries to make the backbone's fused 64-dim feature
NOT reveal which of the 14 source subjects a window came from.

Idea being tested: does pushing the backbone toward subject-invariant features
during pretraining make the later adapter fine-tuning step (on an unseen 15th
subject) more effective, compared to plain joint pretraining (AFTL_train.py)?

Kept WEAK on purpose (small grl_lambda, default 0.1) per request -- a strong
adversarial term risks throwing away the very features that carry emotion
information (subject identity and emotion state are not independent in EEG).

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python AFTL_adversarial_train.py -run_all -pretrain_epochs 30 -finetune_epochs 50 \
        -batch_size 256 -grl_lambda 0.1 -seed 42
"""
import copy
import faulthandler
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

faulthandler.enable()

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Dev\BCI\DBGC-ATFFNet-AFTL")

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import setup_seed
from Trainer.training import train as libeer_train

from model import DAGCN as _DAGCNCore, Adapter  # DBGC-ATFFNet-AFTL/model.py
from DAGCN_train import concat_nested


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DAGCNAdversarial(nn.Module):
    """Same forward(x) -> logits contract as DAGCN_train.DAGCNWrapper (so it drops
    straight into LibEER's Trainer.training.train() for the finetune phase), plus an
    optional forward(x, return_domain=True) -> (logits, domain_logits) used only during
    the adversarial pretrain phase."""

    def __init__(self, channels, feature_dim, num_classes, n_domains, grl_lambda=0.1):
        super().__init__()
        assert channels == 62 and feature_dim == 10 and num_classes == 3
        self.core = _DAGCNCore('seed')
        self.grl_lambda = grl_lambda
        self.domain_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, n_domains)
        )

    def forward(self, x, return_domain=False):
        out, tsne = self.core(x)
        if return_domain:
            reversed_feat = GradientReversalFunction.apply(tsne, self.grl_lambda)
            domain_logits = self.domain_head(reversed_feat)
            return out, domain_logits
        return out


def adversarial_pretrain(model, x, y, domain_labels, device, epochs, lr, batch_size):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(x), torch.Tensor(y), torch.tensor(domain_labels, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_emo_loss, total_dom_loss = 0.0, 0.0, 0.0
        correct, dom_correct, total = 0, 0, 0
        for inputs, targets, domains in loader:
            inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)
            optimizer.zero_grad()
            outputs, domain_logits = model(inputs, return_domain=True)
            emo_loss = criterion(outputs, targets)
            dom_loss = criterion(domain_logits, domains)
            loss = emo_loss + dom_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_emo_loss += emo_loss.item() * inputs.size(0)
            total_dom_loss += dom_loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            tgt = targets.argmax(dim=1) if targets.dim() > 1 else targets
            correct += pred.eq(tgt).sum().item()
            dom_correct += domain_logits.argmax(dim=1).eq(domains).sum().item()
            total += targets.size(0)
        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            n_domains = domain_labels.max() + 1
            chance = 1.0 / n_domains
            print(f"  Pretrain epoch {epoch:3d}/{epochs}: emo_loss={total_emo_loss/total:.4f} "
                  f"emo_acc={correct/total*100:.2f}%  dom_loss={total_dom_loss/total:.4f} "
                  f"dom_acc={dom_correct/total*100:.2f}% (chance={chance*100:.1f}%)", flush=True)
    return model


def main():
    parser = get_args_parser()
    parser.add_argument("-run_all", action="store_true")
    parser.add_argument("-target_subject", type=int, default=None,
                         help="run only this one subject (1-15); merges into the existing result JSON")
    parser.add_argument("-pretrain_epochs", type=int, default=30)
    parser.add_argument("-finetune_epochs", type=int, default=50)
    parser.add_argument("-lr_ft", type=float, default=0.001)
    parser.add_argument("-grl_lambda", type=float, default=0.1, help="adversarial strength (weak by default)")
    args = parser.parse_args()

    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    setting_de = copy.deepcopy(setting)
    setting_de.dataset = "seed_de_lds"
    setting_de.feature_type = "de_lds"
    data_de, label, channels, _fd_de, num_classes = get_data(setting_de)

    setting_psd = copy.deepcopy(setting)
    setting_psd.dataset = "seed_psd_lds"
    setting_psd.feature_type = "psd_lds"
    data_psd, _label_psd, _ch_psd, _fd_psd, _nc_psd = get_data(setting_psd)

    data = concat_nested(data_de, data_psd)
    feature_dim = 10

    m_data, m_label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    n_subjects = len(m_data)
    n_domains = n_subjects - 1  # source subjects only (target excluded)

    if args.target_subject is not None:
        targets = [args.target_subject - 1]
    elif args.run_all:
        targets = range(n_subjects)
    else:
        targets = [0]

    result_path = os.path.join(os.path.dirname(__file__), "AFTL_adversarial_libeer_result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = {int(k): v for k, v in json.load(f).items()}
    else:
        results = {}
    for t in targets:
        subj_num = t + 1
        print(f"\n########## AFTL-adversarial (grl_lambda={args.grl_lambda}) target subject {subj_num} ##########", flush=True)
        setup_seed(args.seed + subj_num)

        # 1. pretrain data: pool ALL trials from the other 14 subjects, with a domain
        #    (source-subject-index) label per window for the adversarial loss
        pretrain_x_list, pretrain_y_list, pretrain_dom_list = [], [], []
        dom_idx = 0
        for s in range(n_subjects):
            if s == t:
                continue
            for trial_x, trial_y in zip(m_data[s], m_label[s]):
                trial_x = np.asarray(trial_x)
                pretrain_x_list.append(trial_x)
                pretrain_y_list.append(np.asarray(trial_y))
                pretrain_dom_list.append(np.full(len(trial_x), dom_idx, dtype=np.int64))
            dom_idx += 1
        pretrain_x = np.concatenate(pretrain_x_list, axis=0)
        pretrain_y = np.concatenate(pretrain_y_list, axis=0)
        pretrain_dom = np.concatenate(pretrain_dom_list, axis=0)
        pretrain_x, _, _ = normalize(pretrain_x, pretrain_x, None, dim='sample', method='z-score')

        # 2. adversarial pretrain
        model = DAGCNAdversarial(channels, feature_dim, num_classes, n_domains=n_domains,
                                  grl_lambda=args.grl_lambda)
        model = adversarial_pretrain(model, pretrain_x, pretrain_y, pretrain_dom, device,
                                      epochs=args.pretrain_epochs, lr=args.lr, batch_size=args.batch_size)

        # 3. leakage-free trial split for target subject: 8 finetune / 7 test trials
        trial_indices = np.arange(15)
        rng = np.random.RandomState(args.seed + subj_num)
        rng.shuffle(trial_indices)
        ft_idx, test_idx = trial_indices[:8], trial_indices[8:]

        ft_x = np.concatenate([np.asarray(m_data[t][i]) for i in ft_idx], axis=0)
        ft_y = np.concatenate([np.asarray(m_label[t][i]) for i in ft_idx], axis=0)
        test_x = np.concatenate([np.asarray(m_data[t][i]) for i in test_idx], axis=0)
        test_y = np.concatenate([np.asarray(m_label[t][i]) for i in test_idx], axis=0)

        ft_x, _, test_x = normalize(ft_x, ft_x, test_x, dim='sample', method='z-score')

        # 4. zero-shot (no finetune) eval, same as AFTL_train.py
        model.eval()
        with torch.no_grad():
            zs_logits = model(torch.Tensor(test_x).to(device))
            zs_pred = zs_logits.argmax(dim=1).cpu().numpy()
        zs_true = test_y.argmax(axis=1) if test_y.ndim > 1 else test_y
        zero_shot_acc = float((zs_pred == zs_true).mean())
        print(f"-> Subject {subj_num} ZERO-SHOT (no finetune) test acc: {zero_shot_acc*100:.2f}%", flush=True)

        # 5. freeze backbone (incl. domain head), unfreeze Adapter modules only
        for p in model.parameters():
            p.requires_grad = False
        adapter_params = 0
        for name, module in model.named_modules():
            if isinstance(module, Adapter):
                for p in module.parameters():
                    p.requires_grad = True
                    adapter_params += p.numel()
        print(f"-> Unfrozen adapter parameters: {adapter_params} (paper Table 8 target: 1456)", flush=True)

        # 6. fine-tune adapters via LibEER's own Trainer.training.train()
        dataset_ft = torch.utils.data.TensorDataset(torch.Tensor(ft_x), torch.Tensor(ft_y))
        dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr_ft, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        output_dir = os.path.join(args.output_dir, "AFTL_adversarial", args.setting or "custom", f"subject{subj_num}")
        os.makedirs(output_dir, exist_ok=True)

        round_metric = libeer_train(
            model=model, dataset_train=dataset_ft, dataset_val=dataset_test, dataset_test=dataset_test,
            device=device, output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
            optimizer=optimizer_ft, batch_size=args.batch_size, epochs=args.finetune_epochs, criterion=criterion,
        )
        print(f"-> Subject {subj_num} AFTL-adversarial test metrics: {round_metric}", flush=True)
        results[subj_num] = {"zero_shot_acc": zero_shot_acc, "finetuned_acc": round_metric['acc']}

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        # explicit cleanup between subjects -- long-running loops on Windows/CUDA seem
        # to destabilize without this (empirically: subject 2 crashed silently, no
        # traceback, in the un-cleaned-up version of this loop)
        del model, optimizer_ft
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY (AFTL-adversarial, grl_lambda={args.grl_lambda}, leakage-free trial split)")
    print("=" * 70)
    for s, r in results.items():
        print(f"Subject {s:>2}: zero-shot={r['zero_shot_acc']*100:6.2f}%   finetuned={r['finetuned_acc']*100:6.2f}%   "
              f"delta={(r['finetuned_acc']-r['zero_shot_acc'])*100:+6.2f}pp")
    if results:
        zs_vals = [r['zero_shot_acc'] for r in results.values()]
        ft_vals = [r['finetuned_acc'] for r in results.values()]
        print("-" * 70)
        print(f"Zero-shot  avg: {np.mean(zs_vals)*100:.2f}%  std: {np.std(zs_vals)*100:.2f}%")
        print(f"Finetuned  avg: {np.mean(ft_vals)*100:.2f}%  std: {np.std(ft_vals)*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
