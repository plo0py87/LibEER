"""
Runs the DBGC-ATFFNet-AFTL paper's "DAGCN" model (from C:/Dev/BCI/DBGC-ATFFNet-AFTL/model.py)
through LibEER's own data pipeline, split protocol, and Trainer.training.train() loop,
instead of the hand-rolled train.py in the DBGC-ATFFNet-AFTL repo.

Protocol: seed_sub_dependent_front_back_setting (LibEER's own name for the paper's
"Within-Session Subject-Dependent" protocol) -- trial 1-9 train / trial 10-15 test,
per subject, session 1 only, matching what DGCNN_train.py etc. use for their reported
SEED subject-dependent numbers.

The model needs both DE and PSD (5 bands each, concatenated -> 10-dim per channel), but
LibEER's get_data() only loads one feature_type at a time, so we call it twice (de_lds,
psd_lds) and concatenate. LibEER's "front-back" setting has no real validation split
(val falls back to = test), same limitation our own train.py has, so this is an
apples-to-apples comparison on that front, not a stricter protocol.

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python DAGCN_train.py -run_all -epochs 200 -batch_size 128 -lr 0.001 -sessions 1 -seed 42
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

from models.Models import Model  # noqa: F401  (unused, kept for parity w/ other *_train.py scripts)
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import setup_seed
from Trainer.training import train as libeer_train

from DAGCN_model import DAGCN as _DAGCNCore


class DAGCNWrapper(nn.Module):
    """Adapts DBGC-ATFFNet-AFTL's DAGCN (forward returns (logits, fused_feat)) to
    LibEER's model convention: __init__(channels, feature_dim, num_classes), forward(x)->logits."""

    def __init__(self, channels, feature_dim, num_classes):
        super().__init__()
        assert channels == 62 and feature_dim == 10 and num_classes == 3, (
            f"DAGCN only supports SEED (62 channels, 5 DE + 5 PSD = 10 feat dim, 3 classes); "
            f"got channels={channels}, feature_dim={feature_dim}, num_classes={num_classes}"
        )
        self.core = _DAGCNCore('seed')

    def forward(self, x):
        out, _ = self.core(x)
        return out


def concat_nested(data_a, data_b):
    """data_a/data_b: nested list [session][subject][trial] of ndarray (samples,62,5).
    Returns the same nested structure with arrays concatenated along the last axis -> (samples,62,10)."""
    out = []
    for ses_a, ses_b in zip(data_a, data_b):
        ses_out = []
        for sub_a, sub_b in zip(ses_a, ses_b):
            sub_out = []
            for trial_a, trial_b in zip(sub_a, sub_b):
                trial_a = np.asarray(trial_a)
                trial_b = np.asarray(trial_b)
                sub_out.append(np.concatenate([trial_a, trial_b], axis=-1))
            ses_out.append(sub_out)
        out.append(ses_out)
    return out


def main():
    parser = get_args_parser()
    parser.add_argument("-run_all", action="store_true", help="run all 15 subjects and report mean/std")
    args = parser.parse_args()

    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    # 1. load DE_LDS and PSD_LDS separately (LibEER only loads one feature_type per call),
    #    then concatenate along the band axis to get the (62, 10) input DAGCN expects.
    setting_de = copy.deepcopy(setting)
    setting_de.dataset = "seed_de_lds"
    setting_de.feature_type = "de_lds"
    data_de, label, channels, _feat_dim_de, num_classes = get_data(setting_de)

    setting_psd = copy.deepcopy(setting)
    setting_psd.dataset = "seed_psd_lds"
    setting_psd.feature_type = "psd_lds"
    data_psd, _label_psd, _channels_psd, _feat_dim_psd, _num_classes_psd = get_data(setting_psd)

    data = concat_nested(data_de, data_psd)
    feature_dim = 10

    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    dependent_metrics = []
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        subject_round_metrics = []
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            print(f"\n=== Subject {rridx}/15 ===")
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            # DAGCN concatenates DE (z-scored on a totally different scale than raw PSD)
            # so z-score normalization is required -- same as ACRNN_train.py/FBSTCNet_train.py
            # do explicitly in this codebase (DGCNN_train.py doesn't, but DGCNN doesn't mix
            # heterogeneous-scale features the way DAGCN does).
            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method='z-score')

            model = DAGCNWrapper(channels, feature_dim, num_classes)
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            output_dir = os.path.join(args.output_dir, "DAGCN", args.setting or "custom", f"subject{rridx}")
            os.makedirs(output_dir, exist_ok=True)

            round_metric = libeer_train(
                model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test,
                device=device, output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
                optimizer=optimizer, batch_size=args.batch_size, epochs=args.epochs, criterion=criterion,
            )
            print(f"-> Subject {rridx} test metrics: {round_metric}")
            subject_round_metrics.append(round_metric)

        dependent_metrics.append(subject_round_metrics)

        if not args.run_all:
            break

    # Manual mean/std summary (skip utils.sub_result_log: save_res() uses a ':'-containing
    # timestamp filename that crashes on Windows -- known issue, see LibEER-environment memory)
    print("\n" + "=" * 50)
    print("FINAL SUMMARY (DAGCN, LibEER seed_sub_dependent_front_back_setting)")
    print("=" * 50)
    per_subject_acc = {}
    for i, subj_metrics in enumerate(dependent_metrics, 1):
        accs = [m['acc'] for m in subj_metrics]
        mean_acc = float(np.mean(accs))
        per_subject_acc[i] = mean_acc
        print(f"Subject {i:2d}: {mean_acc*100:.2f}%")
    if per_subject_acc:
        vals = list(per_subject_acc.values())
        print("-" * 50)
        print(f"Average Accuracy  : {np.mean(vals)*100:.2f}%")
        print(f"Standard Deviation: {np.std(vals)*100:.2f}%")
    print("=" * 50)

    out_path = os.path.join(os.path.dirname(__file__), "DAGCN_libeer_result.json")
    with open(out_path, "w") as f:
        json.dump(per_subject_acc, f, indent=2)
    print(f"Saved per-subject results to {out_path}")


if __name__ == "__main__":
    main()
