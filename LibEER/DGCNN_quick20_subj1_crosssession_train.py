import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from models.Models import Model
from models.DGCNN import NewSparseL2Regularization
from Trainer.training import train
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import setup_seed
from DGCNN_quick20_realtime_train import (
    QUICK20_CHANNEL_NAME, BANDS, build_dataset,
)

# Single-subject, cross-session split for SEED subject 1: train on
# sessions 2 and 3, test on session 1 (0-indexed: train sessions [1,2],
# test session [0]). This is a subject-dependent model (only ever sees
# subject 1's own data), unlike DGCNN_quick20_realtime_train.py's
# subject-independent/subject-dependent-front-back settings, so it's a
# separate small script rather than another `-setting` preset.
#
# Uses the same causal from-raw feature pipeline (bandpass -> DE ->
# Kalman/EMA smoothing) as DGCNN_quick20_realtime_train.py, reusing its
# build_dataset() so features stay identical between scripts.
#
# run this file with:
#   python DGCNN_quick20_subj1_crosssession_train.py -batch_size 16 -lr 0.0015 -epochs 80 \
#       -dataset_path "<...>/SEED_EEG" -smoothing kalman -kalman_q 0.05 -kalman_r 0.2 -device cuda \
#       > logs/DGCNN_quick20/subj1_crosssession.log 2>&1

SUBJECT_INDEX = 0       # subject 1 (0-indexed)
TRAIN_SESSION_INDICES = [1, 2]   # sessions 2 and 3
TEST_SESSION_INDICES = [0]       # session 1
VAL_FRACTION = 0.2                # held-out trials from the training sessions, for checkpoint selection


def trials_to_samples(trial_list_of_lists, label):
    """
    trial_list_of_lists: list of trials, each trial a chronological list of
    (19, 5) DE feature arrays (one per second).
    label: the single class label (int) for that trial.
    Returns (X, y): X shape (num_samples, 19, 5), y shape (num_samples,).
    """
    X, y = [], []
    for trial in trial_list_of_lists:
        for sample in trial:
            X.append(sample)
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)


def main(args):
    setup_seed(args.seed)
    data, label = build_dataset(args.dataset_path, 200, args.smoothing, args.ema_alpha, args.kalman_q, args.kalman_r)
    # data[session][subject] -> list of 15 trials, each a list of (19,5) per-second arrays
    # label[session][subject] -> (15,) array of class indices (0/1/2)

    train_trials, train_labels = [], []
    for s in TRAIN_SESSION_INDICES:
        train_trials += data[s][SUBJECT_INDEX]
        train_labels += [int(l) for l in label[s][SUBJECT_INDEX]]

    test_trials = data[TEST_SESSION_INDICES[0]][SUBJECT_INDEX]
    test_labels = [int(l) for l in label[TEST_SESSION_INDICES[0]][SUBJECT_INDEX]]

    # hold out a slice of the training trials (deterministic, no shuffling of
    # SEED's fixed trial order) as validation, for best-checkpoint selection
    n_val = max(1, int(len(train_trials) * VAL_FRACTION))
    val_trials, val_labels = train_trials[-n_val:], train_labels[-n_val:]
    train_trials, train_labels = train_trials[:-n_val], train_labels[:-n_val]

    print(f"subject 1: {len(train_trials)} train trials, {len(val_trials)} val trials, "
          f"{len(test_trials)} test trials (sessions {TRAIN_SESSION_INDICES}->train, "
          f"{TEST_SESSION_INDICES}->test)")

    def build_xy(trials, labels):
        X, y = [], []
        for trial, lbl in zip(trials, labels):
            tx, ty = trials_to_samples([trial], lbl)
            X.append(tx)
            y.append(ty)
        return np.concatenate(X), np.concatenate(y)

    train_data, train_label = build_xy(train_trials, train_labels)
    val_data, val_label = build_xy(val_trials, val_labels)
    test_data, test_label = build_xy(test_trials, test_labels)

    device = torch.device(args.device)
    channels = len(QUICK20_CHANNEL_NAME)
    feature_dim = len(BANDS)
    model = Model['DGCNN'](channels, feature_dim, 3)
    dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_label))
    dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.LongTensor(val_label))
    dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_label))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss_func = NewSparseL2Regularization(0.01).to(device)
    output_dir = make_output_dir(args, "DGCNN_quick20_subj1_crosssession")
    round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test,
                          device=device, output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
                          optimizer=optimizer, batch_size=args.batch_size, epochs=args.epochs, criterion=criterion,
                          loss_func=loss_func, loss_param=model)
    print("final round metric:", round_metric)


if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument('-smoothing', default='kalman', choices=['none', 'ema', 'kalman'])
    args.add_argument('-ema_alpha', default=0.3, type=float)
    args.add_argument('-kalman_q', default=0.05, type=float)
    args.add_argument('-kalman_r', default=0.2, type=float)
    args = args.parse_args()
    main(args)
