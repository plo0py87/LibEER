import torch
import torch.nn as nn
import torch.utils.data

from models.DGCNN import GraphConv, B1ReLU, laplacian


# ST-GCLSTM for EEG Emotion Recognition — LibEER-native port.
# L. Feng et al., "EEG-Based Emotion Recognition Using Spatial-Temporal Graph
# Convolutional LSTM With Attention Mechanism," IEEE JBHI 2022.
#
# The paper's own spatial block (SGCN) mixes channels through a fixed,
# offline-computed per-window Pearson-correlation adjacency matrix, which
# LibEER's data pipeline has no plumbing for (LibEER's Trainer only ever
# calls model(samples) with a single dense feature tensor — see
# Trainer/training.py). The source repo's own leak-free reproduction study
# (ST-GCLSTM/docs/ST-GCLSTM_reproduction_report.md) additionally found that
# dropping that fixed PCC prior in favor of a fully learnable adjacency
# *improves* accuracy (+2-3 points) once the spatial block is trained with a
# adequate learning rate/recipe. So this port replaces SGCN with DGCNN's
# Chebyshev graph conv over a single learnable global adjacency (Song et al.,
# IEEE TAFFC 2020) — self-contained, no extra adjacency tensor required at
# forward time — and keeps the paper's temporal stack (BiLSTM + eq.15-18
# attention) and its 2-layer BatchNorm classifier head (eq.19-20), sized with
# the best hyperparameters found in that repo's config search
# (gcn_hidden=128, lstm_hidden=64, pre_lstm_dropout=0.8, head_dropout=0.8).
#
# Input shape: (B, T, V, F)
#   B = batch size
#   T = sequence length (number of consecutive 1s DE windows, e.g. 10 for a 10s trial)
#   V = num_electrodes (62 for SEED)
#   F = in_channels  (5 DE bands)
#
# Pipeline:
#   1. DGCNN spatial block (shared weights across T):
#        (B*T, V, F) -> GraphConv + dropout + B1ReLU -> (B*T, V, gcn_hidden)
#        flatten -> (B*T, V*gcn_hidden) -> FC(V*gcn_hidden -> fc_dim) + dropout
#        reshape -> (B, T, fc_dim)
#   2. pre_lstm_dropout -> BiLSTM: (B, T, fc_dim) -> (B, T, 2*lstm_hidden), LayerNorm
#   3. Attention (Feng et al. eq.15-17):
#        M = tanh(H)  alpha = softmax(w^T M)  context = sum(alpha * H)  -> (B, 2*lstm_hidden)
#      H* = tanh(context)  (eq.18)
#   4. Classifier head (eq.19-20): Dropout -> FC(256) -> BatchNorm -> LeakyReLU
#      -> Dropout -> FC(64) -> LeakyReLU -> FC(num_classes)


class STGCLSTM(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 62,
        in_channels: int = 5,
        num_classes: int = 3,
        k: int = 2,
        gcn_hidden: int = 128,
        fc_dim: int = 256,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        spatial_dropout: float = 0.5,
        pre_lstm_dropout: float = 0.8,
        head_dropout: float = 0.8,
        head_hidden1: int = 256,
        head_hidden2: int = 64,
    ):
        """
        Args:
            num_electrodes: V, number of EEG channels (62 for SEED, 32 for DEAP).
            in_channels:    F, feature dimension per electrode (5 DE bands).
            num_classes:    number of emotion classes.
            k:              Chebyshev polynomial order (default 2, same as DGCNN).
            gcn_hidden:     output channels of the single GraphConv layer.
            fc_dim:         hidden dim of the spatial FC (BiLSTM input width).
            lstm_hidden:    BiLSTM hidden size per direction.
            lstm_layers:    number of stacked BiLSTM layers.
            spatial_dropout: dropout probability applied after GCN and spatial FC.
            pre_lstm_dropout: dropout applied to the spatial features before the BiLSTM.
            head_dropout:   dropout probability inside the classifier head.
            head_hidden1:   first classifier hidden layer width.
            head_hidden2:   second classifier hidden layer width.
        """
        super().__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.gcn_hidden = gcn_hidden
        self.fc_dim = fc_dim

        # ── Learnable adjacency (shared across all timesteps, same as DGCNN) ──────
        self.adj = nn.Parameter(torch.empty(num_electrodes, num_electrodes))
        self.adj_bias = nn.Parameter(torch.empty(1))
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0.0, std=0.1)
        self.relu = nn.ReLU(inplace=True)

        # ── DGCNN spatial block (one GraphConv layer) ──────────────────────────────
        self.graph_conv = GraphConv(k, in_channels, gcn_hidden)
        self.b_relu = B1ReLU(gcn_hidden)
        self.spatial_dropout = nn.Dropout(p=spatial_dropout)

        # Spatial feature projection (mirrors DGCNN's self.fc)
        self.spatial_fc = nn.Linear(num_electrodes * gcn_hidden, fc_dim, bias=True)
        nn.init.xavier_normal_(self.spatial_fc.weight)
        nn.init.zeros_(self.spatial_fc.bias)

        # ── BiLSTM ────────────────────────────────────────────────────────────────
        self.pre_lstm_drop = nn.Dropout(p=pre_lstm_dropout)
        self.bilstm = nn.LSTM(
            input_size=fc_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=spatial_dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = 2 * lstm_hidden  # bidirectional
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # ── Attention (Feng et al. eq.15-17) ─────────────────────────────────────
        self.attn_w = nn.Linear(lstm_out_dim, 1, bias=False)

        # ── Classifier head (Feng et al. eq.19-20) ──────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(lstm_out_dim, head_hidden1),
            nn.BatchNorm1d(head_hidden1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden1, head_hidden2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(head_hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V, F)
        Returns:
            logits: (B, num_classes)
        """
        B, T, V, F = x.shape

        # Build graph Laplacian (shared across batch and time)
        adj = self.relu(self.adj + self.adj_bias)   # (V, V)
        lap = laplacian(adj)                          # (V, V)

        # ── Apply DGCNN spatial block at every timestep ───────────────────────────
        x_flat = x.reshape(B * T, V, F)
        x_flat = self.graph_conv(x_flat, lap)         # (B*T, V, gcn_hidden)
        x_flat = self.spatial_dropout(x_flat)
        x_flat = self.b_relu(x_flat)                  # (B*T, V, gcn_hidden)

        # Flatten electrodes and project to fc_dim
        x_flat = x_flat.reshape(B * T, -1)            # (B*T, V*gcn_hidden)
        x_flat = self.spatial_dropout(x_flat)
        x_flat = self.spatial_fc(x_flat)               # (B*T, fc_dim)

        # Restore time dimension
        x_seq = x_flat.reshape(B, T, self.fc_dim)     # (B, T, fc_dim)
        x_seq = self.pre_lstm_drop(x_seq)

        # ── BiLSTM over T timesteps ───────────────────────────────────────────────
        H, _ = self.bilstm(x_seq)                      # (B, T, 2*lstm_hidden)
        H = self.lstm_norm(H)

        # ── Attention ─────────────────────────────────────────────────────────────
        M = torch.tanh(H)                              # (B, T, 2*lstm_hidden)
        score = self.attn_w(M)                          # (B, T, 1)
        alpha = torch.softmax(score, dim=1)              # (B, T, 1)
        context = (alpha * H).sum(dim=1)                 # (B, 2*lstm_hidden)

        # H* = tanh(context)  (Feng et al. eq.18)
        h_star = torch.tanh(context)                    # (B, 2*lstm_hidden)

        return self.head(h_star)                        # (B, num_classes)
