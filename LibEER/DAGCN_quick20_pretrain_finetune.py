"""
Leave-one-subject-out pretrain + within-subject fine-tune test for DAGCN
on the 19-channel Quick-20 montage, causal DE+PSD features (from raw
signal, same pipeline as DAGCN_quick20_realtime_train.py), session 1 only.

Protocol (one held-out subject at a time, HELD_OUT_SUBJECT_INDEX below):
  1. Pretrain on the other 14 subjects' session-1 data (all 15 trials
     each), with a held-out slice of those 14 subjects' data as
     validation for checkpoint selection.
  2. Split the held-out subject's own session-1 data the same way LibEER's
     seed_sub_dependent_front_back_setting does: first 9 trials -> that
     subject's fine-tune-train set, last 6 trials -> that subject's test set.
  3. "before" accuracy: evaluate the pretrained (never seen this subject)
     model directly on the held-out subject's 6 test trials.
  4. Fine-tune: continue training the pretrained model on the held-out
     subject's 9 fine-tune-train trials (low LR, few epochs).
  5. "after" accuracy: evaluate the fine-tuned model on the *same* 6 test
     trials as step 3, for a fair before/after comparison.

Usage (from C:/Dev/BCI/LibEER/LibEER, using the LibEER venv):
    python DAGCN_quick20_pretrain_finetune.py -held_out 0 \
        -pretrain_epochs 200 -pretrain_lr 0.001 \
        -finetune_epochs 40 -finetune_lr 0.0001 \
        -dataset_path "<...>/SEED_EEG" -kalman_q 0.01 -kalman_r 0.5 -device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Dev\BCI\DBGC-ATFFNet-AFTL")

from sklearn.preprocessing import StandardScaler
from utils.utils import setup_seed
from Trainer.training import train as libeer_train

from model import DAGCN as _DAGCNCore  # DBGC-ATFFNet-AFTL/model.py

from DGCNN_quick20_realtime_train import QUICK20_CHANNEL_NAME, QUICK20_CHANNEL_INDICES
from DAGCN_quick20_realtime_train import causal_psd_features
from DGCNN_quick20_realtime_train import causal_de_features, kalman_smooth_trial

# session 1 only, all 15 subject files (see data_utils/load_data.py:121-125)
SESSION1_FILES = [
    '1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
    '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
    '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
    '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
    '13_20140527.mat', '14_20140601.mat', '15_20130709.mat',
]

FRONT_TRIALS = 9  # LibEER's seed_sub_dependent_front_back_setting default


class DAGCNQuick20Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _DAGCNCore('seed_quick20')

    def forward(self, x):
        out, _ = self.core(x)
        return out


def build_session1_dataset(dataset_path, sample_rate, kalman_q, kalman_r):
    """
    Returns data[subject] -> list of 15 trials, each a list of (19,10)
    per-second DE+PSD feature arrays (causally smoothed); and
    label[subject] -> (15,) array of class indices (0/1/2). Session 1 only.
    """
    eeg_dir = dataset_path + "/Preprocessed_EEG"
    label_mat = np.array(loadmat(f"{eeg_dir}/label.mat")['label'])
    trial_labels = (label_mat[0] + 1).astype(np.int64)  # (15,)

    data = []
    for file in SESSION1_FILES:
        subject_data = loadmat(f"{eeg_dir}/{file}")
        keys = list(subject_data.keys())[3:]
        sub_out = []
        for i in range(15):
            trial_raw = subject_data[keys[i]][:, 1:]
            de = causal_de_features(trial_raw, sample_rate, QUICK20_CHANNEL_INDICES)
            psd = causal_psd_features(trial_raw, sample_rate, QUICK20_CHANNEL_INDICES)
            combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
            combined = kalman_smooth_trial(combined, kalman_q, kalman_r)
            sub_out.append(combined)
        data.append(sub_out)
        del subject_data
    return data, trial_labels


def trials_to_xy(trials, labels):
    X, y = [], []
    for trial, lbl in zip(trials, labels):
        for sample in trial:
            X.append(sample)
            y.append(int(lbl))
    return np.stack(X), np.array(y, dtype=np.int64)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.Tensor(X).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def train_model(model, train_data, train_label, val_data, val_label, test_data, test_label,
                 device, epochs, lr, batch_size, output_dir):
    dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_label))
    dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.LongTensor(val_label))
    dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_label))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(output_dir, exist_ok=True)
    return libeer_train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test,
                         device=device, output_dir=output_dir, metrics=['acc'], metric_choose='acc',
                         optimizer=optimizer, batch_size=batch_size, epochs=epochs, criterion=criterion)


def main(args):
    setup_seed(args.seed)
    data, trial_labels = build_session1_dataset(args.dataset_path, 200, args.kalman_q, args.kalman_r)
    # data[subject] -> 15 trials; trial_labels -> (15,) shared across all subjects (same trial order)

    held_out = args.held_out
    other_subjects = [s for s in range(15) if s != held_out]
    print(f"held-out subject: {held_out + 1} (0-indexed {held_out}); "
          f"pretraining on subjects {[s+1 for s in other_subjects]}")

    # --- Stage 1: pretrain on the other 14 subjects ---
    pretrain_trials, pretrain_labels = [], []
    for s in other_subjects:
        pretrain_trials += data[s]
        pretrain_labels += list(trial_labels)
    # held out a validation slice (last 2 trials of each of the 14 subjects) for checkpoint selection
    val_trials, val_labels, train_trials, train_labels = [], [], [], []
    for s_i in range(len(other_subjects)):
        subj_trials = data[other_subjects[s_i]]
        val_trials += subj_trials[-2:]
        val_labels += list(trial_labels[-2:])
        train_trials += subj_trials[:-2]
        train_labels += list(trial_labels[:-2])

    pretrain_train_X, pretrain_train_y = trials_to_xy(train_trials, train_labels)
    pretrain_val_X, pretrain_val_y = trials_to_xy(val_trials, val_labels)

    # held-out subject's own data, split LibEER front-back style (9 train / 6 test)
    ho_trials = data[held_out]
    ho_ft_train_trials, ho_ft_train_labels = ho_trials[:FRONT_TRIALS], trial_labels[:FRONT_TRIALS]
    ho_test_trials, ho_test_labels = ho_trials[FRONT_TRIALS:], trial_labels[FRONT_TRIALS:]
    ho_ft_train_X, ho_ft_train_y = trials_to_xy(ho_ft_train_trials, ho_ft_train_labels)
    ho_test_X, ho_test_y = trials_to_xy(ho_test_trials, ho_test_labels)

    # z-score normalize everything with ONE scaler fit on the pretrain population
    # (train+val), so fine-tune-train and test stay in the same feature space the
    # model was pretrained in -- using separately-fit scalers here would silently
    # shift the input distribution between pretraining, fine-tuning, and testing.
    shape = pretrain_train_X.shape
    scaler = StandardScaler().fit(
        np.concatenate([pretrain_train_X, pretrain_val_X], axis=0).reshape(-1, shape[1] * shape[2]))

    def apply_scaler(x):
        flat = scaler.transform(x.reshape(-1, shape[1] * shape[2]))
        return flat.reshape(x.shape)

    pretrain_train_X = apply_scaler(pretrain_train_X)
    pretrain_val_X = apply_scaler(pretrain_val_X)
    ho_ft_train_X = apply_scaler(ho_ft_train_X)
    ho_test_X = apply_scaler(ho_test_X)

    device = torch.device(args.device)

    print(f"\n=== Stage 1: pretrain on 14 subjects "
          f"({len(pretrain_train_X)} train samples, {len(pretrain_val_X)} val samples) ===")
    model = DAGCNQuick20Wrapper()
    pretrain_out_dir = f"result/DAGCN_quick20_pretrain_finetune/held_out_{held_out+1}/pretrain"
    train_model(model, pretrain_train_X, pretrain_train_y, pretrain_val_X, pretrain_val_y,
                ho_test_X, ho_test_y, device, args.pretrain_epochs, args.pretrain_lr, args.batch_size, pretrain_out_dir)

    before_acc = evaluate(model, ho_test_X, ho_test_y, device)
    print(f"\n>>> BEFORE fine-tuning: held-out subject {held_out+1} test acc = {before_acc:.4f}")

    torch.save({'model': model.state_dict()}, os.path.join(pretrain_out_dir, 'checkpoint-pretrained'))

    print(f"\n=== Stage 2: fine-tune on held-out subject {held_out+1}'s own "
          f"{len(ho_ft_train_X)} samples ({FRONT_TRIALS} trials) ===")
    finetune_out_dir = f"result/DAGCN_quick20_pretrain_finetune/held_out_{held_out+1}/finetune"
    # fine-tune-train doubles as its own val here (no separate held-out slice -- too little
    # data in 9 trials to spare more), matching the same fallback LibEER's front-back setting uses
    train_model(model, ho_ft_train_X, ho_ft_train_y, ho_ft_train_X, ho_ft_train_y,
                ho_test_X, ho_test_y, device, args.finetune_epochs, args.finetune_lr, args.batch_size, finetune_out_dir)

    after_acc = evaluate(model, ho_test_X, ho_test_y, device)
    print(f"\n>>> AFTER fine-tuning: held-out subject {held_out+1} test acc = {after_acc:.4f}")

    torch.save({'model': model.state_dict()}, os.path.join(finetune_out_dir, 'checkpoint-finetuned'))

    print("\n" + "=" * 50)
    print(f"SUMMARY (held-out subject {held_out+1}, session 1)")
    print(f"  before fine-tune: {before_acc:.4f}")
    print(f"  after fine-tune : {after_acc:.4f}")
    print(f"  delta           : {after_acc - before_acc:+.4f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-held_out', default=0, type=int, help="0-indexed subject to hold out (0 = subject 1)")
    parser.add_argument('-dataset_path', required=True)
    parser.add_argument('-pretrain_epochs', default=200, type=int)
    parser.add_argument('-pretrain_lr', default=0.001, type=float)
    parser.add_argument('-finetune_epochs', default=40, type=int)
    parser.add_argument('-finetune_lr', default=0.0001, type=float)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-kalman_q', default=0.01, type=float)
    parser.add_argument('-kalman_r', default=0.5, type=float)
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
