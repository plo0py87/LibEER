import numpy as np

from models.Models import Model
from config.setting import preset_setting, set_setting_by_args
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

# ST-GCLSTM (Feng et al., IEEE JBHI 2022) — LibEER-native port, best-config
# hyperparameters (gcn_hidden=128, lstm_hidden=64, pre_lstm_dropout=0.8,
# head_dropout=0.8) from the source repo's own reproduction study. See
# models/ST_GCLSTM.py for why the paper's fixed PCC adjacency is replaced
# with DGCNN's learnable global adjacency.
#
# Data format: (B, T, V, F)  — requires sample_length > 1 to form the time sequence.
# Recommended settings mirror the paper's windowing: 10s windows with 50% overlap →
#   -time_window 1  -sample_length 10  -stride 5
#
# Example commands:
#
# SEED sub-dependent (train/val/test split):
#   python ST_GCLSTM_train.py -metrics acc macro-f1 -metric_choose macro-f1 \
#       -setting seed_sub_dependent_train_val_test_setting \
#       -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds \
#       -batch_size 32 -seed 2024 -epochs 80 -lr 0.0015 -onehot \
#       -sample_length 10 -stride 5
#
# SEED-IV sub-dependent:
#   python ST_GCLSTM_train.py -metrics acc macro-f1 -metric_choose macro-f1 \
#       -setting seediv_sub_dependent_train_val_test_setting \
#       -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw \
#       -batch_size 32 -epochs 150 -time_window 1 -feature_type de_lds \
#       -seed 2024 -onehot -sample_length 10 -stride 5


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
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

            # ST_GCLSTM expects (B, T, V, F); sample_length > 1 gives T > 1.
            # feature_dim = F (number of DE bands, typically 5).
            # channels     = V (number of electrodes, typically 62 for SEED).
            model = Model['ST_GCLSTM'](
                num_electrodes=channels,
                in_channels=feature_dim,
                num_classes=num_classes,
            )

            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val   = torch.utils.data.TensorDataset(torch.Tensor(val_data),   torch.Tensor(val_label))
            dataset_test  = torch.utils.data.TensorDataset(torch.Tensor(test_data),  torch.Tensor(test_label))

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            criterion = nn.CrossEntropyLoss()
            # Same L2 regularisation as DGCNN_train.py / DGCNN_LSTM_train.py
            loss_func  = NewSparseL2Regularization(0.01).to(device)
            output_dir = make_output_dir(args, "ST_GCLSTM")

            round_metric = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                batch_size=args.batch_size,
                epochs=args.epochs,
                criterion=criterion,
                test_sub_label=test_sub_label,
                loss_func=loss_func,
                loss_param=model,
            )
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
