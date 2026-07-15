"""
Frontal Alpha Asymmetry (FAA) vs. model prediction accuracy, per window.

FAA (classic Davidson definition) = ln(alpha_power_RIGHT) - ln(alpha_power_LEFT).
Higher/more-positive FAA -> relatively greater LEFT frontal activation (alpha power is
inversely related to cortical activation) -> associated with approach motivation /
positive affect in the classic literature. Lower/negative FAA -> withdrawal / negative
affect.

We don't need to recompute ln(power) from scratch: SEED's official DE feature for a
Gaussian-distributed signal segment is DE = 0.5*ln(2*pi*e*variance) = 0.5*ln(power) + C,
where C is the same additive constant for every channel/window (same window length,
same sampling rate). So:
    DE_alpha(right) - DE_alpha(left) = 0.5 * [ln(power_right) - ln(power_left)]
                                      = 0.5 * FAA_classic
i.e. the DE difference is proportional to the classic FAA (same sign, half magnitude,
constant cancels). We report this DE-based quantity directly and note the 0.5 scaling
-- it doesn't affect any correlation/sign result below.

For each of the 15 subjects' held-out test trials (10-15, no leakage, same split used to
train the LibEER DAGCN checkpoints), and for each of 3 electrode pairs (F3/F4, F7/F8,
Fp1/Fp2), we compute per-window FAA and compare it against:
  1. the window's TRUE label (positive vs negative only -- neutral excluded)
  2. the LOADED DAGCN checkpoint's predicted label / correctness on that window
to see whether FAA is itself predictive of positive/negative emotion in this data, and
whether subjects where FAA is more predictive are also the subjects the model classifies
more accurately.
"""
import copy
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Dev\BCI\DBGC-ATFFNet-AFTL")

from config.setting import preset_setting
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from data_utils.preprocess import normalize
from DAGCN_train import DAGCNWrapper, concat_nested

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
RESULT_DIR = os.path.join(os.path.dirname(__file__), "result", "DAGCN", "seed_sub_dependent_front_back_setting")


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


def main():
    args = Args()
    setting = preset_setting["seed_sub_dependent_front_back_setting"](args)

    setting_de = copy.deepcopy(setting)
    setting_de.dataset, setting_de.feature_type = "seed_de_lds", "de_lds"
    data_de, label, channels, _fd, num_classes = get_data(setting_de)

    setting_psd = copy.deepcopy(setting)
    setting_psd.dataset, setting_psd.feature_type = "seed_psd_lds", "psd_lds"
    data_psd, _l, _c, _fd2, _nc = get_data(setting_psd)

    data = concat_nested(data_de, data_psd)
    m_data, m_label = merge_to_part(data, label, setting)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pair_indices = [(CHANNEL_ORDER.index(l), CHANNEL_ORDER.index(r)) for l, r in ELECTRODE_PAIRS]

    per_subject_rows = []
    all_window_rows = []  # pooled across subjects, for the window-level correlation

    for s_idx, (data_i, label_i) in enumerate(zip(m_data, m_label), 1):
        tts = get_split_index(data_i, label_i, setting)
        train_indexes, test_indexes, val_indexes = tts['train'][0], tts['test'][0], tts['val'][0]

        train_data, train_label, val_data, val_label, test_data, test_label = \
            index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
        if len(val_data) == 0:
            val_data, val_label = test_data, test_label

        raw_test_data = test_data.copy()  # keep RAW (un-normalized) DE for FAA

        # exact same normalization DAGCN_train.py used (fit on train+val, i.e. train+test here)
        _, _, norm_test_data = normalize(train_data, val_data, test_data, dim='sample', method='z-score')

        ckpt_path = os.path.join(RESULT_DIR, f"subject{s_idx}", "checkpoint-bestacc")
        if not os.path.exists(ckpt_path):
            print(f"[skip] no checkpoint for subject {s_idx}")
            continue
        model = DAGCNWrapper(channels, 10, num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        model.eval()

        with torch.no_grad():
            logits = model(torch.Tensor(norm_test_data).to(device))
            pred = logits.argmax(dim=1).cpu().numpy()

        true_label = np.array(test_label)
        true_idx = true_label.argmax(axis=1) if true_label.ndim > 1 else true_label

        # keep only positive (2) / negative (0) windows -- FAA is an approach/withdrawal axis
        mask = (true_idx == 0) | (true_idx == 2)
        raw_pos_neg = raw_test_data[mask]
        pred_pos_neg = pred[mask]
        true_pos_neg = true_idx[mask]  # 0=negative, 2=positive
        binary_true = (true_pos_neg == 2).astype(int)  # 1=positive, 0=negative
        correct = (pred_pos_neg == true_pos_neg).astype(int)

        row = {"subject": s_idx, "n_windows": int(mask.sum())}
        for (l_name, r_name), (l_idx, r_idx) in zip(ELECTRODE_PAIRS, pair_indices):
            faa = raw_pos_neg[:, r_idx, ALPHA_IDX] - raw_pos_neg[:, l_idx, ALPHA_IDX]  # DE(right)-DE(left)

            # 1. does FAA separate positive vs negative windows? point-biserial corr
            r_true, p_true = stats.pointbiserialr(binary_true, faa)
            # 2. simple FAA-threshold classifier accuracy: brute-force best split, either polarity
            best_acc = 0.0
            for thresh in np.unique(faa):
                pred_pos = (faa > thresh).astype(int)
                acc1 = (pred_pos == binary_true).mean()
                acc2 = ((1 - pred_pos) == binary_true).mean()
                best_acc = max(best_acc, acc1, acc2)

            # 3. does |FAA| (asymmetry magnitude) correlate with the model's correctness?
            r_mag_correct, p_mag_correct = stats.pointbiserialr(correct, np.abs(faa))

            pair_key = f"{l_name}{r_name}"
            row[f"{pair_key}_faa_vs_label_r"] = r_true
            row[f"{pair_key}_faa_vs_label_p"] = p_true
            row[f"{pair_key}_faa_threshold_acc"] = best_acc
            row[f"{pair_key}_absfaa_vs_correct_r"] = r_mag_correct
            row[f"{pair_key}_absfaa_vs_correct_p"] = p_mag_correct

            if s_idx == 1:
                pass
            for w in range(len(faa)):
                all_window_rows.append({
                    "subject": s_idx, "pair": pair_key, "faa": float(faa[w]),
                    "true_positive": int(binary_true[w]), "correct": int(correct[w]),
                })

        row["model_acc_pos_neg"] = float(correct.mean())
        per_subject_rows.append(row)
        print(f"Subject {s_idx:2d}: n={row['n_windows']:4d}  model_acc(pos/neg)={row['model_acc_pos_neg']*100:.2f}%  "
              f"F3F4 r={row['F3F4_faa_vs_label_r']:.3f} (p={row['F3F4_faa_vs_label_p']:.3g})  "
              f"F3F4 thresh_acc={row['F3F4_faa_threshold_acc']*100:.2f}%", flush=True)

    # ---- save raw results ----
    out_dir = os.path.join(os.path.dirname(__file__), "faa_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "faa_per_subject.json"), "w") as f:
        json.dump(per_subject_rows, f, indent=2)
    with open(os.path.join(out_dir, "faa_per_window.json"), "w") as f:
        json.dump(all_window_rows, f)

    # ---- subject-level summary: does "how well FAA alone predicts pos/neg" correlate
    #      with "how well the DAGCN model predicts pos/neg", across the 15 subjects? ----
    print("\n" + "=" * 70)
    print("SUBJECT-LEVEL: FAA-threshold-accuracy vs DAGCN model accuracy (pos/neg only)")
    print("=" * 70)
    model_accs = [r["model_acc_pos_neg"] for r in per_subject_rows]
    for l_name, r_name in ELECTRODE_PAIRS:
        pair_key = f"{l_name}{r_name}"
        faa_accs = [r[f"{pair_key}_faa_threshold_acc"] for r in per_subject_rows]
        faa_rs = [r[f"{pair_key}_faa_vs_label_r"] for r in per_subject_rows]
        r_corr, p_corr = stats.pearsonr(faa_accs, model_accs)
        print(f"{pair_key}: mean FAA-vs-label |r|={np.mean(np.abs(faa_rs)):.3f}   "
              f"mean FAA-threshold-acc={np.mean(faa_accs)*100:.2f}%   "
              f"corr(FAA-acc, model-acc) across subjects: r={r_corr:.3f} (p={p_corr:.3g})")
    print(f"\nmean DAGCN model accuracy (pos/neg only, all subjects): {np.mean(model_accs)*100:.2f}%")


if __name__ == "__main__":
    main()
