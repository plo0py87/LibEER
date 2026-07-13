# CGX Quick-20 DGCNN pretrained model — usage notes

Repo: https://github.com/plo0py87/LibEER (branch `main`)
Checkpoint path in repo: `LibEER/result/DGCNN_quick20_realtime/seed_sub_independent_train_val_test_setting/checkpoint-bestacc`
Checkpoint commit: `a677216` — "Add cross-subject Quick-20 DGCNN checkpoint (test acc 0.52)"

## What this model is

DGCNN (Song et al., "EEG Emotion Recognition Using Dynamical Graph
Convolutional Neural Networks"), retrained from scratch to take only the
19 EEG channels the CGX Quick-20 dry-electrode headset can record (SEED's
official montage has 62). Predicts 3 classes: 0=negative, 1=neutral,
2=positive, one prediction per second of EEG.

Trained on SEED (15 subjects, session 1) with LibEER's
`seed_sub_independent_train_val_test_setting`: random 9 subjects train /
3 val / 3 test, single split, **not** subject-dependent — so this is a
cross-subject baseline, not fit to any specific person. best_val_acc:
0.54, best_test_acc: 0.52. This is meant to be a **pretrained starting
point to fine-tune** on self-recorded Quick-20 data, not a finished model.

## Preprocessing pipeline (what the model expects as input)

Every step below is causal (only ever looks at past/current samples), so
this matches what a live LSL stream can actually produce — it deliberately
does **not** use LibEER's official DE_LDS feature, which needs the whole
trial's mean up front and can't run online.

1. **Channels**: reduce to exactly these 19, in exactly this order —
   `FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2`.
   If your CGX channel order differs, reorder your recording to match
   before calling the predict script.
2. **Sample rate**: 200Hz. Resample if your CGX stream runs at a
   different rate.
3. **Causal bandpass**: 4th-order Butterworth, one-directional
   (`scipy.signal.lfilter`, not `filtfilt`), 5 bands —
   delta 1-4Hz, theta 4-8Hz, alpha 8-14Hz, beta 14-31Hz, gamma 31-50Hz.
4. **1-second non-overlapping windows**: 200 samples per window.
5. **Differential entropy**: `DE = 0.5 * log(2*pi*e*variance + 1e-10)`,
   variance of the bandpass-filtered signal within each window, per
   channel per band. Output: one `(19, 5)` feature vector per second.
6. **Causal smoothing (Kalman)**: a scalar forward-only Kalman filter
   applied independently to every (channel, band) dimension, run
   sample-by-sample down the per-second DE sequence (`q=0.05, r=0.2`).
   State model: `x_t = x_{t-1} + w`, `z_t = x_t + v`. Only ever uses
   `z_0..z_t` — no backward/RTS pass, no whole-trial statistics.
7. Feed the `(19, 5)` smoothed DE vector for each second into DGCNN.
   `num_electrodes=19, in_channels=5, num_classes=3`; DGCNN's adjacency
   matrix is learned during training, not fixed, so no extra graph setup
   is needed at inference time.

All of this logic lives in `LibEER/LibEER/DGCNN_quick20_realtime_train.py`
(functions `causal_de_features`, `kalman_smooth_trial`,
`QUICK20_CHANNEL_NAME`, `BANDS`) — reuse those functions rather than
reimplementing, so training and inference stay consistent.

## Setup

```bash
git clone https://github.com/plo0py87/LibEER.git
cd LibEER
python -m venv .venv
# activate it, then:
pip install numpy==1.24.3 scipy==1.9.3 scikit-learn==1.4.2 PyYAML==6.0.1 tqdm pandas mne mat73 xmltodict skorch braindecode torch_geometric
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your GPU's CUDA version, or use cpu build
```

(The pinned `requirements.txt` in the repo targets old CUDA builds that
may not match a newer laptop GPU — install a torch build matching your
own CUDA version instead, or CPU-only; this model is tiny, ~313K
parameters, CPU inference is fast enough for 1Hz real-time regardless.)

## Running inference

A standalone predict script is at `LibEER/LibEER/DGCNN_quick20_predict.py`.
It takes a `.npy` file of shape `(19, T)` at 200Hz (channel order as
above) and prints one predicted class per second.

```bash
cd LibEER/LibEER
python DGCNN_quick20_predict.py \
    -input path/to/your_recording.npy \
    -checkpoint result/DGCNN_quick20_realtime/seed_sub_independent_train_val_test_setting/checkpoint-bestacc \
    -smoothing kalman -kalman_q 0.05 -kalman_r 0.2 \
    -device cpu
```

Output per second: predicted label (negative/neutral/positive) and the
3-class softmax probabilities.

To use it from Python directly (e.g. inside an LSL receive loop) import
`predict()` from that file — it takes a `(19, T)` numpy array and the
checkpoint path, and returns `(pred_classes, probs)`.

## Fine-tuning on your own data

Load the checkpoint's `'model'` key as a state dict into a freshly
constructed `Model['DGCNN'](19, 5, 3)`, then continue training on your
own labeled `(19, 5)`-per-second sequences with a low learning rate. The
existing `DGCNN_quick20_realtime_train.py` training loop can be adapted:
swap `build_dataset()` for a loader over your own recordings/labels, and
call `model.load_state_dict(torch.load(checkpoint_path)['model'])` before
constructing the optimizer.

## Known caveats

- This checkpoint is a cross-subject baseline (52% test acc on SEED,
  3-way classification, ~33% chance) — expect it to perform worse
  zero-shot on a genuinely new person/hardware. Fine-tuning on your own
  data is expected to be necessary, not optional.
- The causal DE features here are numerically on a different scale than
  LibEER's official `seed_de_lds` features (~2.5 vs ~20.5) — this is a
  known, unexplained discrepancy versus the official feature extraction,
  not a bug in the causal pipeline. Don't mix the two feature types.
- A repo-wide bug (Windows-only): `utils/store.py`'s final summary
  logging step crashes with `OSError: [Errno 22] Invalid argument` on
  Windows because it uses a colon in a timestamp-based filename. Harmless
  — training/checkpoint saving completes fine before it, just ignore the
  traceback at the very end of a training run.
