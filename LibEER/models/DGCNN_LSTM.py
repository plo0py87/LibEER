import torch
import torch.nn as nn
import torch.utils.data

from models.DGCNN import GraphConv, B1ReLU, laplacian


# DGCNN + BiLSTM + Attention for EEG Emotion Recognition
# Spatial feature extractor follows: T. Song et al., "EEG Emotion Recognition Using Dynamical
# Graph Convolutional Neural Networks," IEEE TAFFC 2020.
# Temporal modeling follows: L. Feng et al., "ST-GCLSTM," IEEE JBHI 2022.
#
# Input shape: (B, T, V, F)
#   B = batch size
#   T = sequence length (number of consecutive 1s DE windows, e.g. 10 for a 10s trial)
#   V = num_electrodes (62 for SEED)
#   F = in_channels  (5 DE bands)
#
# Pipeline:
#   1. DGCNN spatial block (shared weights across T):
#        (B*T, V, F) -> GraphConv + dropout + B1ReLU -> (B*T, V, gcn_out)
#        flatten -> (B*T, V*gcn_out) -> FC(V*gcn_out -> fc_dim) + dropout
#        reshape -> (B, T, fc_dim)
#   2. BiLSTM:  (B, T, fc_dim) -> (B, T, 2*lstm_hidden)
#   3. Attention (Feng et al. eq.15-17):
#        M = tanh(H)  alpha = softmax(w^T M)  context = sum(alpha * H)  -> (B, 2*lstm_hidden)
#   4. Classifier: FC(2*lstm_hidden -> num_classes)


class DGCNN_LSTM(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 62,
        in_channels: int = 5,
        num_classes: int = 3,
        k: int = 2,
        gcn_out: int = 64,
        fc_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            num_electrodes: V, number of EEG channels (62 for SEED, 32 for DEAP).
            in_channels:    F, feature dimension per electrode (5 DE bands).
            num_classes:    number of emotion classes.
            k:              Chebyshev polynomial order (default 2, same as DGCNN).
            gcn_out:        output channels of the single GraphConv layer.
            fc_dim:         hidden dim of the spatial FC (same as DGCNN's fc → 256).
            lstm_hidden:    BiLSTM hidden size per direction.
            lstm_layers:    number of stacked BiLSTM layers.
            dropout_rate:   dropout probability applied after GCN and spatial FC.
        """
        super().__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.gcn_out = gcn_out
        self.fc_dim = fc_dim

        # ── Learnable adjacency (shared across all timesteps, same as DGCNN) ──────
        self.adj = nn.Parameter(torch.empty(num_electrodes, num_electrodes))
        self.adj_bias = nn.Parameter(torch.empty(1))
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0.0, std=0.1)
        self.relu = nn.ReLU(inplace=True)

        # ── DGCNN spatial block (one GraphConv layer, matches DGCNN default) ──────
        self.graph_conv = GraphConv(k, in_channels, gcn_out)
        self.b_relu = B1ReLU(gcn_out)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Spatial feature projection (mirrors DGCNN's self.fc)
        self.spatial_fc = nn.Linear(num_electrodes * gcn_out, fc_dim, bias=True)
        nn.init.xavier_normal_(self.spatial_fc.weight)
        nn.init.zeros_(self.spatial_fc.bias)

        # ── BiLSTM ────────────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=fc_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = 2 * lstm_hidden  # bidirectional

        # ── Attention (Feng et al. eq.15-17) ─────────────────────────────────────
        self.attn_w = nn.Linear(lstm_out_dim, 1, bias=False)

        # ── Classifier ────────────────────────────────────────────────────────────
        self.classifier = nn.Linear(lstm_out_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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
        lap = laplacian(adj)                         # (V, V)

        # ── Apply DGCNN spatial block at every timestep ───────────────────────────
        # Merge batch and time dims so GraphConv sees (B*T, V, F)
        x_flat = x.reshape(B * T, V, F)
        x_flat = self.graph_conv(x_flat, lap)        # (B*T, V, gcn_out)
        x_flat = self.dropout(x_flat)
        x_flat = self.b_relu(x_flat)                 # (B*T, V, gcn_out)

        # Flatten electrodes and project to fc_dim
        x_flat = x_flat.reshape(B * T, -1)           # (B*T, V*gcn_out)
        x_flat = self.dropout(x_flat)
        x_flat = self.spatial_fc(x_flat)             # (B*T, fc_dim)
        x_flat = self.dropout(x_flat)

        # Restore time dimension
        x_seq = x_flat.reshape(B, T, self.fc_dim)   # (B, T, fc_dim)

        # ── BiLSTM over T timesteps ───────────────────────────────────────────────
        H, _ = self.bilstm(x_seq)                    # (B, T, 2*lstm_hidden)

        # ── Attention ─────────────────────────────────────────────────────────────
        # M = tanh(H_t)
        M = torch.tanh(H)                            # (B, T, 2*lstm_hidden)
        # alpha = softmax(w^T M)
        score = self.attn_w(M)                       # (B, T, 1)
        alpha = torch.softmax(score, dim=1)          # (B, T, 1)
        # context = sum_t(alpha_t * H_t)
        context = (alpha * H).sum(dim=1)             # (B, 2*lstm_hidden)

        # H* = tanh(context)  (Feng et al. eq.18)
        h_star = torch.tanh(context)                 # (B, 2*lstm_hidden)

        return self.classifier(h_star)               # (B, num_classes)
