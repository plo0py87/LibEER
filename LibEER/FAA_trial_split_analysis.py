"""
Honest, trial-level train/test evaluation of a simple FAA-threshold classifier,
compared for DE_LDS (smoothed) vs DE_movingAve ("raw", un-smoothed) alpha power,
using the SAME trial 1-9 (train) / trial 10-15 (test) split the DAGCN model uses.

For each subject:
  - train pool  = positive+negative trials among trial 1-9  (3 positive + 3 negative,
                   the SEED trial label sequence is fixed and identical for every
                   subject's session 1, so this split is the same count for everyone)
  - test pool   = positive+negative trials among trial 10-15 (2 positive + 2 negative)
  - pick the best FAA threshold + polarity using ONLY the train pool's labels
  - apply that fixed threshold to the held-out test pool -> honest generalization accuracy

Run for both feature_type='de_lds' (official LDS-smoothed) and feature_type='de'
(LibEER's index-0 feature, which earlier we confirmed maps to SEED's de_movingAve --
i.e. un-smoothed, closer to raw).
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.setting import preset_setting
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, get_split_index

CHANNEL_ORDER = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ",
    "C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7",
    "P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6",
    "PO8","CB1","O1","OZ","O2","CB2",
]
ELECTRODE_PAIRS = [("F3", "F4"), ("F7", "F8"), ("FP1", "FP2")]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
ALPHA_IDX = BAND_NAMES.index("alpha")

DATASET_PATH = "C:/Dev/BCI/EEG_Dataset/SEED/SEED/SEED_EEG"


class Args:
    dataset = "seed_de_lds"
    dataset_path = DATASET_PATH
    low_pass, high_pass = 0.3, 50
    time_window, overlap, sample_length, stride = 1, 0, 1, 1
    seed = 42
    feature_type = "de_lds"
    only_seg = False
    cross_trail = "true"
    experiment_mode = "subject-dependent"
    normalize = True
    split_type = "front-back"
    fold_num = 5
    fold_shuffle = "true"
    front = 9
    sessions = [1]
    pr = None
    sr = None
    bounds = None
    onehot = True
    label_used = None
    keep_dim = False


def best_threshold_and_polarity(faa_train, label_train):
    """Return (threshold, higher_is_positive) maximizing accuracy on the TRAIN pool only."""
    best_acc, best_thresh, best_polarity = -1, None, True
    for thresh in np.unique(faa_train):
        pred_pos = (faa_train > thresh).astype(int)
        acc_higher_pos = (pred_pos == label_train).mean()
        acc_lower_pos = ((1 - pred_pos) == label_train).mean()
        if acc_higher_pos > best_acc:
            best_acc, best_thresh, best_polarity = acc_higher_pos, thresh, True
        if acc_lower_pos > best_acc:
            best_acc, best_thresh, best_polarity = acc_lower_pos, thresh, False
    return best_thresh, best_polarity, best_acc


def apply_threshold(faa, thresh, higher_is_positive):
    pred_pos = (faa > thresh).astype(int)
    return pred_pos if higher_is_positive else (1 - pred_pos)


def run_for_feature_type(feature_type, label_name):
    args = Args()
    args.dataset = f"seed_{feature_type}" if not feature_type.startswith("de") else f"seed_{feature_type}"
    args.dataset = "seed_" + feature_type
    args.feature_type = feature_type
    setting = preset_setting["seed_sub_dependent_front_back_setting"](args)
    setting.dataset = "seed_" + feature_type
    setting.feature_type = feature_type

    data, label, channels, feature_dim, num_classes = get_data(setting)
    m_data, m_label = merge_to_part(data, label, setting)

    pair_indices = [(CHANNEL_ORDER.index(l), CHANNEL_ORDER.index(r)) for l, r in ELECTRODE_PAIRS]

    results = []
    for s_idx, (trials_x, trials_y) in enumerate(zip(m_data, m_label), 1):
        tts = get_split_index(trials_x, trials_y, setting)
        train_trial_idx, test_trial_idx = tts['train'][0], tts['test'][0]

        def trial_label(i):
            y = np.asarray(trials_y[i])
            idx = y.argmax(axis=1) if y.ndim > 1 else y
            return idx[0]  # constant within a trial

        train_pn = [i for i in train_trial_idx if trial_label(i) in (0, 2)]
        test_pn = [i for i in test_trial_idx if trial_label(i) in (0, 2)]

        train_x = np.concatenate([np.asarray(trials_x[i]) for i in train_pn], axis=0)
        train_y_idx = np.concatenate([[trial_label(i)] * len(trials_x[i]) for i in train_pn])
        test_x = np.concatenate([np.asarray(trials_x[i]) for i in test_pn], axis=0)
        test_y_idx = np.concatenate([[trial_label(i)] * len(trials_x[i]) for i in test_pn])

        train_bin = (train_y_idx == 2).astype(int)
        test_bin = (test_y_idx == 2).astype(int)

        row = {"subject": s_idx, "n_train": len(train_bin), "n_test": len(test_bin)}
        for (l_name, r_name), (l_idx, r_idx) in zip(ELECTRODE_PAIRS, pair_indices):
            faa_train = train_x[:, r_idx, ALPHA_IDX] - train_x[:, l_idx, ALPHA_IDX]
            faa_test = test_x[:, r_idx, ALPHA_IDX] - test_x[:, l_idx, ALPHA_IDX]

            thresh, polarity, train_acc = best_threshold_and_polarity(faa_train, train_bin)
            test_pred = apply_threshold(faa_test, thresh, polarity)
            test_acc = (test_pred == test_bin).mean()

            pair_key = f"{l_name}{r_name}"
            row[f"{pair_key}_train_acc"] = float(train_acc)
            row[f"{pair_key}_test_acc"] = float(test_acc)

        results.append(row)
        print(f"[{label_name}] Subject {s_idx:2d}: n_train={row['n_train']:3d} n_test={row['n_test']:3d}  "
              f"F3F4 train={row['F3F4_train_acc']*100:.1f}% test={row['F3F4_test_acc']*100:.1f}%   "
              f"F7F8 test={row['F7F8_test_acc']*100:.1f}%   FP1FP2 test={row['FP1FP2_test_acc']*100:.1f}%",
              flush=True)
    return results


def main():
    all_results = {}
    for feature_type, label_name in [("de_lds", "DE_LDS(smoothed)"), ("de", "DE_movingAve(raw)")]:
        print(f"\n===== {label_name} =====")
        all_results[feature_type] = run_for_feature_type(feature_type, label_name)

    out_dir = os.path.join(os.path.dirname(__file__), "faa_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "faa_trial_split_result.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY: honest trial-level held-out test accuracy, DE_LDS vs DE_movingAve(raw)")
    print("=" * 80)
    for pair in ["F3F4", "F7F8", "FP1FP2"]:
        for feature_type, label_name in [("de_lds", "DE_LDS "), ("de", "DE_raw ")]:
            accs = [r[f"{pair}_test_acc"] for r in all_results[feature_type]]
            print(f"{pair:8s} {label_name}: mean test acc = {np.mean(accs)*100:5.2f}%  (std {np.std(accs)*100:4.2f}%)")


if __name__ == "__main__":
    main()
