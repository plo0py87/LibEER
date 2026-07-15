"""
AFTL few-shot size ablation: how many fine-tune trials does the adapter actually need?

Same pretrain (14 source subjects) as AFTL_train.py, but instead of always using 8
fine-tune trials, this reuses ONE pretrained backbone per subject and re-finetunes
(from that same pretrained starting point, reset each time) with fine-tune set sizes
N in {3, 4, 5, 6, 8}, class-stratified so every N includes at least one positive, one
negative, and one neutral trial. The held-out TEST set is always the SAME fixed 7
trials (same as AFTL_train.py's split) so results are directly comparable across N and
against the existing AFTL_libeer_result.json (N=8 baseline).

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python AFTL_fewshot_size_train.py -run_all -pretrain_epochs 30 -finetune_epochs 50 -seed 42
"""
import copy
import gc
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
from AFTL_train import pretrain_backbone

N_VALUES = [3, 4, 5, 6, 8]


def trial_label_of(trial_y):
    y = np.asarray(trial_y)
    idx = y.argmax(axis=1) if y.ndim > 1 else y
    return int(idx[0])  # constant within a trial


def stratified_select(pool_idx, labels_by_trial, n, rng):
    """pool_idx: list of trial indices (into the subject's 15 trials) eligible for
    fine-tuning. labels_by_trial: dict trial_idx -> class label (0/1/2).
    Returns n trial indices, guaranteed to include >=1 of each class present in the
    pool (as long as n >= number of distinct classes in the pool)."""
    by_class = {}
    for idx in pool_idx:
        by_class.setdefault(labels_by_trial[idx], []).append(idx)

    chosen = []
    for cls, idxs in by_class.items():
        chosen.append(rng.choice(idxs))
    chosen = list(chosen)

    remaining = [i for i in pool_idx if i not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < n and remaining:
        chosen.append(remaining.pop())
    return chosen[:n]


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

    m_data, m_label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    n_subjects = len(m_data)

    result_path = os.path.join(os.path.dirname(__file__), "AFTL_fewshot_size_result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = {int(k): v for k, v in json.load(f).items()}
    else:
        results = {}

    targets = range(n_subjects) if args.run_all else [0]
    for t in targets:
        subj_num = t + 1
        print(f"\n########## Few-shot-size AFTL target subject {subj_num} ##########", flush=True)
        setup_seed(args.seed + subj_num)

        # 1. pretrain data + pretrain backbone (ONCE per subject, reused for every N)
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

        model = DAGCNWrapper(channels, feature_dim, num_classes)
        model = pretrain_backbone(model, pretrain_x, pretrain_y, device,
                                   epochs=args.pretrain_epochs, lr=args.lr, batch_size=args.batch_size)
        base_state = copy.deepcopy(model.state_dict())

        # 2. SAME trial split as AFTL_train.py: first 8 = fine-tune-eligible pool, last 7 = fixed test
        trial_indices = np.arange(15)
        rng = np.random.RandomState(args.seed + subj_num)
        rng.shuffle(trial_indices)
        pool_idx, test_idx = list(trial_indices[:8]), list(trial_indices[8:])

        labels_by_trial = {i: trial_label_of(m_label[t][i]) for i in range(15)}
        pool_classes = sorted(set(labels_by_trial[i] for i in pool_idx))
        print(f"  fine-tune pool classes present: {pool_classes} (0=neg,1=neu,2=pos)", flush=True)

        test_x = np.concatenate([np.asarray(m_data[t][i]) for i in test_idx], axis=0)
        test_y = np.concatenate([np.asarray(m_label[t][i]) for i in test_idx], axis=0)

        subj_results = results.get(subj_num, {})
        for n in N_VALUES:
            select_rng = np.random.RandomState(args.seed + subj_num + n * 1000)
            ft_idx = stratified_select(pool_idx, labels_by_trial, n, select_rng)
            ft_classes = sorted(set(labels_by_trial[i] for i in ft_idx))

            ft_x = np.concatenate([np.asarray(m_data[t][i]) for i in ft_idx], axis=0)
            ft_y = np.concatenate([np.asarray(m_label[t][i]) for i in ft_idx], axis=0)

            ft_x_n, _, test_x_n = normalize(ft_x, ft_x, test_x, dim='sample', method='z-score')

            # reset to the SAME pretrained starting point for every N
            model.load_state_dict(base_state)
            for p in model.parameters():
                p.requires_grad = False
            adapter_params = 0
            for name, module in model.named_modules():
                if isinstance(module, Adapter):
                    for p in module.parameters():
                        p.requires_grad = True
                        adapter_params += p.numel()

            dataset_ft = torch.utils.data.TensorDataset(torch.Tensor(ft_x_n), torch.Tensor(ft_y))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_x_n), torch.Tensor(test_y))
            optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=args.lr_ft, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            output_dir = os.path.join(args.output_dir, "AFTL_fewshot", args.setting or "custom",
                                       f"subject{subj_num}", f"n{n}")
            os.makedirs(output_dir, exist_ok=True)

            round_metric = libeer_train(
                model=model, dataset_train=dataset_ft, dataset_val=dataset_test, dataset_test=dataset_test,
                device=device, output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
                optimizer=optimizer_ft, batch_size=args.batch_size, epochs=args.finetune_epochs, criterion=criterion,
            )
            print(f"  N={n} (classes={ft_classes}, {len(ft_x)} windows): test acc = {round_metric['acc']*100:.2f}%", flush=True)
            subj_results[str(n)] = {"acc": round_metric['acc'], "n_windows": len(ft_x), "classes": ft_classes}

        results[subj_num] = subj_results
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        del model, optimizer_ft
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY (few-shot fine-tune size ablation)")
    print("=" * 80)
    for n in N_VALUES:
        accs = [results[s][str(n)]["acc"] for s in results if str(n) in results[s]]
        if accs:
            print(f"N={n} trials: avg={np.mean(accs)*100:.2f}%  std={np.std(accs)*100:.2f}%  (n_subjects={len(accs)})")


if __name__ == "__main__":
    main()
