import numpy as np

from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
from models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn

# Retrains DGCNN on only the 19 SEED channels that correspond to the CGX
# Quick-20 dry-electrode headset's electrode positions, to see how much
# accuracy is lost by going from the full 62-channel SEED montage down to
# the montage we can actually record with the Quick-20 in the lab.
#
# run this file with (mirrors the documented 62-channel reproduction command):
#   python DGCNN_quick20_train.py -onehot -batch_size 16 -lr 0.0015 -sessions 1 2 -epochs 80 \
#       -setting seed_sub_dependent_front_back_setting -dataset_path <path to SEED_EEG folder> \
#       > logs/DGCNN_quick20/repro.log 2>&1

# SEED's 62-channel order (data_utils/constants/seed.py: SEED_CHANNEL_NAME)
SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

# CGX Quick-20 electrode montage (19 recording sites + A2 reference, not
# itself a data channel here)
QUICK20_CHANNEL_NAME = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']

QUICK20_CHANNEL_INDICES = [SEED_CHANNEL_NAME.index(ch) for ch in QUICK20_CHANNEL_NAME]


def select_channels(data, indices):
    """
    Recursively walk the (possibly ragged) nested-list structure returned by
    get_data() and slice the channel axis (second-to-last axis of every leaf
    array) down to `indices`.
    """
    if isinstance(data, np.ndarray):
        return np.take(data, indices, axis=-2)
    if isinstance(data, list):
        return [select_channels(d, indices) for d in data]
    return data


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    assert channels == len(SEED_CHANNEL_NAME), (
        f"expected the full {len(SEED_CHANNEL_NAME)}-channel SEED montage before slicing, got {channels}")
    data = select_channels(data, QUICK20_CHANNEL_INDICES)
    channels = len(QUICK20_CHANNEL_INDICES)
    print(f"Quick-20 channel subset: {QUICK20_CHANNEL_NAME} ({channels} channels)")

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

            test_sub_label = None

            if setting.experiment_mode == "subject-independent":
                train_data, train_label, val_data, val_label, test_data, test_label = \
                    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, True)
                test_sub_num = len(test_data)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(test_data[i])
                    test_sub_label.extend([i + 1 for j in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

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
            output_dir = make_output_dir(args, "DGCNN_quick20")
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
    args = args.parse_args()
    main(args)
