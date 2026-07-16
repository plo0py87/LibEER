"""
Same as DAGCN_train.py, but slices SEED's 62-channel montage down to the
19 channels the CGX Quick-20 headset can record, matching the approach in
DGCNN_quick20_train.py. DAGCN's channel count is baked into a per-dataset
options dict in DAGCN_model.py (chan_num isn't a constructor arg), so a
'seed_quick20': [19,3,5] entry was added there for this.

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python DAGCN_quick20_train.py -run_all -epochs 200 -batch_size 128 -lr 0.001 -sessions 1 -seed 42
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

SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

QUICK20_CHANNEL_NAME = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']

QUICK20_CHANNEL_INDICES = [SEED_CHANNEL_NAME.index(ch) for ch in QUICK20_CHANNEL_NAME]


def select_channels(data, indices):
    if isinstance(data, np.ndarray):
        return np.take(data, indices, axis=-2)
    if isinstance(data, list):
        return [select_channels(d, indices) for d in data]
    return data


class DAGCNQuick20Wrapper(nn.Module):
    """Same adapter as DAGCN_train.py's DAGCNWrapper, but for the 19-channel Quick-20 montage."""

    def __init__(self, channels, feature_dim, num_classes):
        super().__init__()
        assert channels == 19 and feature_dim == 10 and num_classes == 3, (
            f"DAGCN (quick20) only supports 19 channels, 5 DE + 5 PSD = 10 feat dim, 3 classes; "
            f"got channels={channels}, feature_dim={feature_dim}, num_classes={num_classes}"
        )
        self.core = _DAGCNCore('seed_quick20')

    def forward(self, x):
        out, _ = self.core(x)
        return out


def concat_nested(data_a, data_b):
    """data_a/data_b: nested list [session][subject][trial] of ndarray (samples,19,5).
    Returns the same nested structure with arrays concatenated along the last axis -> (samples,19,10)."""
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

    setting_de = copy.deepcopy(setting)
    setting_de.dataset = "seed_de_lds"
    setting_de.feature_type = "de_lds"
    data_de, label, channels, _feat_dim_de, num_classes = get_data(setting_de)
    assert channels == len(SEED_CHANNEL_NAME)
    data_de = select_channels(data_de, QUICK20_CHANNEL_INDICES)

    setting_psd = copy.deepcopy(setting)
    setting_psd.dataset = "seed_psd_lds"
    setting_psd.feature_type = "psd_lds"
    data_psd, _label_psd, _channels_psd, _feat_dim_psd, _num_classes_psd = get_data(setting_psd)
    data_psd = select_channels(data_psd, QUICK20_CHANNEL_INDICES)

    data = concat_nested(data_de, data_psd)
    channels = len(QUICK20_CHANNEL_INDICES)
    feature_dim = 10
    print(f"Quick-20 channel subset: {QUICK20_CHANNEL_NAME} ({channels} channels)")

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

            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method='z-score')

            model = DAGCNQuick20Wrapper(channels, feature_dim, num_classes)
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            output_dir = os.path.join(args.output_dir, "DAGCN_quick20", args.setting or "custom", f"subject{rridx}")
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

    print("\n" + "=" * 50)
    print("FINAL SUMMARY (DAGCN quick20, LibEER seed_sub_dependent_front_back_setting)")
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

    out_path = os.path.join(os.path.dirname(__file__), "DAGCN_quick20_libeer_result.json")
    with open(out_path, "w") as f:
        json.dump(per_subject_acc, f, indent=2)
    print(f"Saved per-subject results to {out_path}")


if __name__ == "__main__":
    main()
