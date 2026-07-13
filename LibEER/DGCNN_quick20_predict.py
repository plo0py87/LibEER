import argparse
import numpy as np
import torch

from models.Models import Model
from DGCNN_quick20_realtime_train import (
    QUICK20_CHANNEL_NAME, BANDS, causal_de_features, kalman_smooth_trial, ema_smooth_trial,
)

# Standalone inference for the pretrained Quick-20 DGCNN checkpoint. Not
# used during training -- DGCNN_quick20_realtime_train.py only trains and
# evaluates against SEED. This script is what you run against your own
# recorded CGX Quick-20 signal.
#
# Input: a numpy .npy file containing raw EEG, shape (19, T), sampled at
# 200Hz, with rows in exactly this channel order (must match your CGX
# electrode montage -> reorder your recording to match if needed):
#   FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2
#
# Output: one predicted class (0=negative, 1=neutral, 2=positive) per
# second of input, plus the softmax probabilities.
#
# run this file with:
#   python DGCNN_quick20_predict.py -input my_recording.npy \
#       -checkpoint result/DGCNN_quick20_realtime/seed_sub_independent_train_val_test_setting/checkpoint-bestacc \
#       -smoothing kalman -kalman_q 0.05 -kalman_r 0.2 -device cpu

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def predict(raw_eeg, checkpoint_path, smoothing="kalman", ema_alpha=0.3, kalman_q=0.05, kalman_r=0.2,
            sample_rate=200, device="cpu"):
    """
    raw_eeg: (19, T) numpy array, 200Hz, channel order == QUICK20_CHANNEL_NAME.
    Returns (pred_classes, probs): pred_classes is (num_seconds,) int array,
    probs is (num_seconds, 3) float array.
    """
    assert raw_eeg.shape[0] == len(QUICK20_CHANNEL_NAME), (
        f"expected {len(QUICK20_CHANNEL_NAME)} channels in order {QUICK20_CHANNEL_NAME}, got shape {raw_eeg.shape}")

    channel_indices = list(range(len(QUICK20_CHANNEL_NAME)))  # already in Quick-20 order
    de = causal_de_features(raw_eeg, sample_rate, channel_indices)
    if smoothing == "ema":
        de = ema_smooth_trial(de, ema_alpha)
    elif smoothing == "kalman":
        de = kalman_smooth_trial(de, kalman_q, kalman_r)

    model = Model['DGCNN'](len(QUICK20_CHANNEL_NAME), len(BANDS), 3)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()

    x = torch.Tensor(np.stack(de)).to(device)  # (num_seconds, 19, 5)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_classes = probs.argmax(axis=1)
    return pred_classes, probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True, help="path to a .npy file, shape (19, T), 200Hz, Quick-20 channel order")
    parser.add_argument('-checkpoint', required=True, help="path to a saved checkpoint (dict with 'model' key)")
    parser.add_argument('-smoothing', default='kalman', choices=['none', 'ema', 'kalman'])
    parser.add_argument('-ema_alpha', default=0.3, type=float)
    parser.add_argument('-kalman_q', default=0.05, type=float)
    parser.add_argument('-kalman_r', default=0.2, type=float)
    parser.add_argument('-sample_rate', default=200, type=int)
    parser.add_argument('-device', default='cpu')
    args = parser.parse_args()

    raw_eeg = np.load(args.input)
    pred_classes, probs = predict(raw_eeg, args.checkpoint, args.smoothing, args.ema_alpha,
                                   args.kalman_q, args.kalman_r, args.sample_rate, args.device)
    for t, (c, p) in enumerate(zip(pred_classes, probs)):
        print(f"t={t}s  pred={LABEL_NAMES[c]}  probs(neg/neu/pos)={p.round(3).tolist()}")
