"""
Runs the DBGC-ATFFNet-AFTL paper's AFTL (Adapter-Finetuned Transfer Learning) protocol
through LibEER's own data pipeline / normalize() utility / Trainer.training.train() loop,
mirroring what DAGCN_train.py already did for the non-transfer baseline.

Protocol (LEAKAGE-FREE trial split, same as DBGC-ATFFNet-AFTL/repro_aftl_trial.py):
  - Pretrain DAGCN('seed') from scratch on the other 14 subjects' full session-1 data
    (all 15 trials each), pooled.
  - Freeze the backbone, unfreeze only the Adapter modules (1,456 params).
  - Split the TARGET subject's 15 trials into 8 (fine-tune) / 7 (test) at the TRIAL
    level (no window-level leakage) -- NOT the paper's random 50/50 window split.
  - Fine-tune the adapters on the 8-trial set, evaluate on the 7-trial held-out set
    using LibEER's Trainer.training.train() (val=test, same limitation already flagged
    for DAGCN_train.py -- LibEER's front-back-style settings have no real val split).

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python AFTL_train.py -run_all -pretrain_epochs 30 -finetune_epochs 50 -batch_size 256 -seed 42
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Dev\BCI\DBGC-ATFFNet-AFTL")

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import setup_seed
from Trainer.training import train as libeer_train

from model import Adapter  # DBGC-ATFFNet-AFTL/model.py
from DAGCN_train import DAGCNWrapper, concat_nested


def pretrain_backbone(model, x, y, device, epochs, lr, batch_size):
    """Plain supervised pretraining loop -- no LibEER Trainer here since there's
    no meaningful val/test split during pretraining (we always keep the final weights,
    then freeze and adapt)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            tgt = targets.argmax(dim=1) if targets.dim() > 1 else targets
            correct += pred.eq(tgt).sum().item()
            total += targets.size(0)
        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(f"  Pretrain epoch {epoch:3d}/{epochs}: loss={total_loss/total:.4f} acc={correct/total*100:.2f}%", flush=True)
    return model


def main():
    parser = get_args_parser()
    parser.add_argument("-run_all", action="store_true")
    parser.add_argument("-pretrain_epochs", type=int, default=30)
    parser.add_argument("-finetune_epochs", type=int, default=50)
    parser.add_argument("-lr_ft", type=float, default=0.001)
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

    # subject-dependent + cross_trail=true -> m_data[subject] = [trial0, trial1, ..., trial14]
    m_data, m_label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    n_subjects = len(m_data)

    targets = range(n_subjects) if args.run_all else [0]
    results = {}
    for t in targets:
        subj_num = t + 1
        print(f"\n########## AFTL (LibEER pipeline) target subject {subj_num} ##########", flush=True)
        setup_seed(args.seed + subj_num)

        # 1. pretrain data: pool ALL trials from the other 14 subjects
        pretrain_x_list, pretrain_y_list = [], []
        for s in range(n_subjects):
            if s == t:
                continue
            for trial_x, trial_y in zip(m_data[s], m_label[s]):
                pretrain_x_list.append(np.asarray(trial_x))
                pretrain_y_list.append(np.asarray(trial_y))
        pretrain_x = np.concatenate(pretrain_x_list, axis=0)
        pretrain_y = np.concatenate(pretrain_y_list, axis=0)
        pretrain_x, _, _ = normalize(pretrain_x, pretrain_x, None, dim='sample', method='z-score')

        # 2. pretrain backbone from scratch
        model = DAGCNWrapper(channels, feature_dim, num_classes)
        model = pretrain_backbone(model, pretrain_x, pretrain_y, device,
                                   epochs=args.pretrain_epochs, lr=args.lr, batch_size=args.batch_size)

        # 3. freeze backbone, unfreeze Adapter modules only
        for p in model.parameters():
            p.requires_grad = False
        adapter_params = 0
        for name, module in model.named_modules():
            if isinstance(module, Adapter):
                for p in module.parameters():
                    p.requires_grad = True
                    adapter_params += p.numel()
        print(f"-> Unfrozen adapter parameters: {adapter_params} (paper Table 8 target: 1456)", flush=True)

        # 4. leakage-free trial split for target subject: 8 finetune / 7 test trials
        trial_indices = np.arange(15)
        rng = np.random.RandomState(args.seed + subj_num)
        rng.shuffle(trial_indices)
        ft_idx, test_idx = trial_indices[:8], trial_indices[8:]

        ft_x = np.concatenate([np.asarray(m_data[t][i]) for i in ft_idx], axis=0)
        ft_y = np.concatenate([np.asarray(m_label[t][i]) for i in ft_idx], axis=0)
        test_x = np.concatenate([np.asarray(m_data[t][i]) for i in test_idx], axis=0)
        test_y = np.concatenate([np.asarray(m_label[t][i]) for i in test_idx], axis=0)

        # normalize using ONLY the fine-tune set's stats (val_data=ft_x duplicate avoids
        # leaking test_x into the scaler fit -- see AFTL_train.py docstring)
        ft_x, _, test_x = normalize(ft_x, ft_x, test_x, dim='sample', method='z-score')

        # 5. fine-tune adapters via LibEER's own Trainer.training.train()
        dataset_ft = torch.utils.data.TensorDataset(torch.Tensor(ft_x), torch.Tensor(ft_y))
        dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr_ft, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        output_dir = os.path.join(args.output_dir, "AFTL", args.setting or "custom", f"subject{subj_num}")
        os.makedirs(output_dir, exist_ok=True)

        round_metric = libeer_train(
            model=model, dataset_train=dataset_ft, dataset_val=dataset_test, dataset_test=dataset_test,
            device=device, output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
            optimizer=optimizer_ft, batch_size=args.batch_size, epochs=args.finetune_epochs, criterion=criterion,
        )
        print(f"-> Subject {subj_num} AFTL test metrics: {round_metric}", flush=True)
        results[subj_num] = round_metric['acc']

        with open(os.path.join(os.path.dirname(__file__), "AFTL_libeer_result.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("FINAL SUMMARY (AFTL, LibEER pipeline, leakage-free trial split)")
    print("=" * 50)
    for s, acc in results.items():
        print(f"Subject {s:>2}: {acc*100:.2f}%")
    if results:
        vals = list(results.values())
        print("-" * 50)
        print(f"Average Accuracy  : {np.mean(vals)*100:.2f}%")
        print(f"Standard Deviation: {np.std(vals)*100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
