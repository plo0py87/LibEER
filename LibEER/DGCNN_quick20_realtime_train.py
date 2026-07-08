import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.preprocess import label_process
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
from models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn

# Full from-raw-signal real-time simulation for CGX Quick-20:
#   raw 200Hz EEG -> causal bandpass (5 SEED bands) -> 1s windowed band-power
#   -> DE -> optional causal smoothing (EMA or forward-only Kalman filter)
# Every step here only ever looks at past samples, so this is what a live
# CGX stream -> feature -> DGCNN pipeline would actually see -- unlike
# LibEER's official DE_LDS (offline, whole-trial Kalman *smoother*, not
# filter) or feeding the pre-extracted DE straight to an EMA (which skips
# re-deriving DE from raw signal).
#
# run this file with:
#   python DGCNN_quick20_realtime_train.py -onehot -batch_size 16 -lr 0.0015 -sessions 1 2 -epochs 80 \
#       -setting seed_sub_dependent_front_back_setting -dataset_path "<...>/SEED_EEG" \
#       -smoothing kalman -kalman_q 0.05 -kalman_r 0.2 -device cuda \
#       > logs/DGCNN_quick20/repro_realtime.log 2>&1

SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

QUICK20_CHANNEL_NAME = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']

QUICK20_CHANNEL_INDICES = [SEED_CHANNEL_NAME.index(ch) for ch in QUICK20_CHANNEL_NAME]

# standard SEED / Zheng & Lu (2015) 5-band split
BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]


def causal_de_features(trial_raw, sample_rate, channel_indices):
    """
    trial_raw: (62, T) raw preprocessed EEG for one trial.
    Returns a list of T//sample_rate per-second (19, 5) DE feature arrays,
    in chronological order, using only causal (forward, IIR) filtering and
    non-overlapping 1s windows -- no look-ahead anywhere.
    """
    sig = trial_raw[channel_indices, :]  # (19, T)
    num_seconds = sig.shape[1] // sample_rate
    sig = sig[:, :num_seconds * sample_rate]
    band_power = np.zeros((num_seconds, len(channel_indices), len(BANDS)))
    for b_i, (low, high) in enumerate(BANDS):
        nyq = 0.5 * sample_rate
        b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype='bandpass')
        filtered = lfilter(b, a, sig, axis=1)  # causal, forward-only IIR filter
        windows = filtered.reshape(len(channel_indices), num_seconds, sample_rate)
        band_power[:, :, b_i] = windows.var(axis=2).T
    de = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-10)
    return list(de)  # chronological list of (19, 5) arrays


def ema_smooth_trial(trial_list, alpha):
    smoothed, running = [], None
    for x in trial_list:
        running = x if running is None else alpha * x + (1 - alpha) * running
        smoothed.append(running)
    return smoothed


def kalman_smooth_trial(trial_list, q, r):
    """
    Scalar forward-only Kalman filter applied independently to every
    (channel, band) feature dimension, run causally sample-by-sample down
    the trial's per-second DE sequence. State model: x_t = x_{t-1} + w
    (w ~ N(0, q)), observation z_t = x_t + v (v ~ N(0, r)). Only ever uses
    z_0..z_t to produce the estimate at time t -- no RTS backward pass.
    """
    smoothed = []
    x_est = p_est = None
    for z in trial_list:
        if x_est is None:
            x_est = z.copy()
            p_est = np.ones_like(z)
        else:
            p_pred = p_est + q
            k = p_pred / (p_pred + r)
            x_est = x_est + k * (z - x_est)
            p_est = (1 - k) * p_pred
        smoothed.append(x_est.copy())
    return smoothed


# same 45 filenames read_seed_raw() uses (data_utils/load_data.py:121-136),
# duplicated here so we can process one file at a time instead of loading
# all 3 sessions x 15 subjects of raw signal into memory simultaneously.
SEED_RAW_FILES = [
    ['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
     '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
     '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
     '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
     '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
    ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
     '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
     '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
     '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
     '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
    ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
     '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
     '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
     '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
     '13_20140610.mat', '14_20140627.mat', '15_20131105.mat'],
]


def build_dataset(dataset_path, sample_rate, smoothing, ema_alpha, kalman_q, kalman_r):
    """
    Stream the 45 Preprocessed_EEG files one at a time: load a single
    subject/session .mat, immediately reduce every trial's raw (62, T)
    signal down to its tiny (num_seconds, 19, 5) causal DE feature array,
    then let the raw signal go out of scope before loading the next file.
    At no point do we hold more than one file's raw signal in memory --
    read_seed_raw() instead loads all 45 files (~7.6GB) up front, which
    OOM'd on this machine.
    """
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
                trial_de = causal_de_features(trial_raw, sample_rate, QUICK20_CHANNEL_INDICES)
                if smoothing == "ema":
                    trial_de = ema_smooth_trial(trial_de, ema_alpha)
                elif smoothing == "kalman":
                    trial_de = kalman_smooth_trial(trial_de, kalman_q, kalman_r)
                sub_out.append(trial_de)
            ses_out.append(sub_out)
            del subject_data  # drop this file's raw signal before the next one
        data.append(ses_out)
    return data, labels


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    data, label = build_dataset(setting.dataset_path, 200, args.smoothing, args.ema_alpha, args.kalman_q, args.kalman_r)
    data, label, num_classes = label_process(data=data, label=label, bounds=setting.bounds, onehot=setting.onehot, label_used=setting.label_used)
    channels = len(QUICK20_CHANNEL_INDICES)
    feature_dim = len(BANDS)
    print(f"Quick-20 realtime-simulated DE, smoothing={args.smoothing} "
          f"(ema_alpha={args.ema_alpha}, kalman_q={args.kalman_q}, kalman_r={args.kalman_r})")

    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []
    dependent_metrics = [[] for _ in range(len(data))]
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
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
            model = Model['DGCNN'](channels, feature_dim, num_classes)
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            criterion = nn.CrossEntropyLoss()
            loss_func = NewSparseL2Regularization(0.01).to(device)
            output_dir = make_output_dir(args, "DGCNN_quick20_realtime")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                  output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                  batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, loss_func=loss_func, loss_param=model)
            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                dependent_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, dependent_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument('-smoothing', default='none', choices=['none', 'ema', 'kalman'], help="causal smoothing applied to the per-second DE sequence")
    args.add_argument('-ema_alpha', default=0.3, type=float, help="EMA smoothing factor (higher = less smoothing)")
    args.add_argument('-kalman_q', default=0.05, type=float, help="Kalman filter process noise")
    args.add_argument('-kalman_r', default=0.2, type=float, help="Kalman filter observation noise")
    args = args.parse_args()
    main(args)
