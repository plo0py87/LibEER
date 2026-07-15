"""
Cross-subject (subject-independent), from-raw-signal, causally-smoothed
version of DAGCN on the 19-channel Quick-20 montage -- the DAGCN analogue
of DGCNN_quick20_realtime_train.py, intended as a pretrained starting
point for fine-tuning on self-recorded Quick-20 data (same rationale as
DGCNN_quick20_realtime_train.py's seed_sub_independent_train_val_test_setting run).

DAGCN needs both DE and PSD (5 bands each, concat -> 10-dim per channel).
Both are recomputed causally from raw signal (no look-ahead anywhere):
  - DE: causal bandpass (Butterworth, lfilter) -> 1s window variance -> DE formula
        (DGCNN_quick20_realtime_train.py's causal_de_features)
  - PSD: causal 1s-window Welch periodogram -> 10*log10(mean PSD in band),
         matching LibEER's own psd_extraction() formula (data_utils/preprocess.py),
         which is already causal per-window -- only the official "_lds" suffix
         (whole-trial Kalman smoother) is non-causal, and we don't use that.
Then a forward-only Kalman filter (stronger smoothing than the DGCNN run:
lower q, higher r) is applied to the concatenated (19, 10) per-second vector.

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python DAGCN_quick20_realtime_train.py -epochs 200 -batch_size 128 -lr 0.001 \
        -setting seed_sub_independent_train_val_test_setting \
        -dataset_path "<...>/SEED_EEG" -kalman_q 0.01 -kalman_r 0.5 -device cuda
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Dev\BCI\DBGC-ATFFNet-AFTL")

from config.setting import preset_setting, set_setting_by_args
from data_utils.preprocess import normalize, label_process
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import setup_seed
from Trainer.training import train as libeer_train

from model import DAGCN as _DAGCNCore  # DBGC-ATFFNet-AFTL/model.py

from DGCNN_quick20_realtime_train import (
    QUICK20_CHANNEL_NAME, QUICK20_CHANNEL_INDICES, BANDS,
    causal_de_features, kalman_smooth_trial, ema_smooth_trial, SEED_RAW_FILES,
)


class DAGCNQuick20Wrapper(nn.Module):
    def __init__(self, channels, feature_dim, num_classes):
        super().__init__()
        assert channels == 19 and feature_dim == 10 and num_classes == 3
        self.core = _DAGCNCore('seed_quick20')

    def forward(self, x):
        out, _ = self.core(x)
        return out


def causal_psd_features(trial_raw, sample_rate, channel_indices):
    """
    Causal per-second PSD, matching LibEER's own psd_extraction() formula
    (data_utils/preprocess.py) -- Welch periodogram over each *individual*
    1s window (no look-ahead into other windows), 10*log10(mean psd in
    each band's frequency-bin range). LibEER's own "psd" (pre-"_lds") is
    already causal this way; only the official "_lds" suffix is not.
    """
    sig = trial_raw[channel_indices, :]  # (19, T)
    num_seconds = sig.shape[1] // sample_rate
    sig = sig[:, :num_seconds * sample_rate]
    psd_out = np.zeros((num_seconds, len(channel_indices), len(BANDS)))
    for t in range(num_seconds):
        window = sig[:, t * sample_rate:(t + 1) * sample_rate]
        f, psd = welch(window, fs=sample_rate, nperseg=sample_rate, window='hamming')
        for b_i, (low, high) in enumerate(BANDS):
            psd_out[t, :, b_i] = np.mean(10 * np.log10(psd[:, low:high + 1] + 1e-20), axis=1)
    return list(psd_out)


def build_dataset(dataset_path, sample_rate, kalman_q, kalman_r):
    """Same streaming-one-file-at-a-time approach as DGCNN_quick20_realtime_train.py's
    build_dataset(), but produces concatenated (19,10) DE+PSD features and applies the
    (stronger) Kalman smoothing to the combined vector."""
    eeg_dir = dataset_path + "/Preprocessed_EEG"
    label = np.array(loadmat(f"{eeg_dir}/label.mat")['label'])
    labels = np.tile(label[0] + 1, (3, 15, 1))

    data = []
    for session_files in SEED_RAW_FILES:
        ses_out = []
        for file in session_files:
            subject_data = loadmat(f"{eeg_dir}/{file}")
            keys = list(subject_data.keys())[3:]
            sub_out = []
            for i in range(15):
                trial_raw = subject_data[keys[i]][:, 1:]
                de = causal_de_features(trial_raw, sample_rate, QUICK20_CHANNEL_INDICES)
                psd = causal_psd_features(trial_raw, sample_rate, QUICK20_CHANNEL_INDICES)
                combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]  # (19,10) per second
                combined = kalman_smooth_trial(combined, kalman_q, kalman_r)
                sub_out.append(combined)
            ses_out.append(sub_out)
            del subject_data
        data.append(ses_out)
    return data, labels


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    data, label = build_dataset(setting.dataset_path, 200, args.kalman_q, args.kalman_r)
    data, label, num_classes = label_process(data=data, label=label, bounds=setting.bounds,
                                              onehot=False, label_used=setting.label_used)
    channels = len(QUICK20_CHANNEL_NAME)
    feature_dim = 10
    print(f"Quick-20 realtime DE+PSD, kalman_q={args.kalman_q}, kalman_r={args.kalman_r} "
          f"(stronger smoothing than the DGCNN run's q=0.05/r=0.2)")

    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    dependent_metrics = []
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        subject_round_metrics = []
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
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
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.LongTensor(np.argmax(train_label, axis=1) if train_label.ndim > 1 else train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.LongTensor(np.argmax(val_label, axis=1) if val_label.ndim > 1 else val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.LongTensor(np.argmax(test_label, axis=1) if test_label.ndim > 1 else test_label))

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "DAGCN_quick20_realtime")

            round_metric = libeer_train(
                model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test,
                device=device, output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
                optimizer=optimizer, batch_size=args.batch_size, epochs=args.epochs, criterion=criterion,
            )
            print(f"round metric: {round_metric}")
            subject_round_metrics.append(round_metric)
        dependent_metrics.append(subject_round_metrics)

    print("final metrics:", dependent_metrics)


if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument('-kalman_q', default=0.01, type=float, help="Kalman filter process noise (lower = more smoothing)")
    args.add_argument('-kalman_r', default=0.5, type=float, help="Kalman filter observation noise (higher = more smoothing)")
    args = args.parse_args()
    main(args)
