import torch
import torch.nn as nn
import torch.utils.data

from models.DGCNN import GraphConv, B1ReLU, laplacian


# DGCNN + Per-Electrode BiLSTM + Attention (node-wise temporal variant)
#
# Key difference from DGCNN_LSTM.py (V1):
#   V1: flatten all electrodes first (FC 3968→256), then one shared LSTM over time.
#   V2: skip the spatial-compression FC; each of the V electrodes gets its own
#       independent LSTM lane (shared weights) over T timesteps, preserving
#       per-node temporal dynamics.  The classifier head is the only place that
#       compresses across electrodes.
#
# Input shape: (B, T, V, F)
#   B = batch, T = time steps (e.g. 10 for a 10s trial), V = electrodes, F = DE bands
#
# Pipeline:
#   1. DGCNN spatial block (shared across T, no spatial FC):
#        (B*T, V, F) → GraphConv + dropout + B1ReLU → (B*T, V, gcn_out)
#        reshape → (B, T, V, gcn_out)
#   2. Per-electrode BiLSTM (weights shared across V):
#        permute → (B*V, T, gcn_out)
#        BiLSTM  → (B*V, T, 2*lstm_hidden)
#   3. Temporal attention (per electrode, Feng et al. eq.15-17):
#        M = tanh(H)  alpha = softmax(w^T M)  context = Σ alpha*H
#        → (B*V, 2*lstm_hidden) → reshape (B, V, 2*lstm_hidden)
#   4. H* = tanh(context)
#   5. Classifier head (compress here):
#        flatten → (B, V*2*lstm_hidden)
#        FC(V*2H → head_dim) + act → FC(head_dim → num_classes)


class DGCNN_NodeLSTM(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 62,
        in_channels: int = 5,
        num_classes: int = 3,
        k: int = 2,
        gcn_out: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout_rate: float = 0.5,
        head_dim: int = 256,
    ):
        """
        Args:
            num_electrodes: V, number of EEG channels (62 for SEED, 32 for DEAP).
            in_channels:    F, feature dimension per electrode (5 DE bands).
            num_classes:    number of emotion classes.
            k:              Chebyshev polynomial order (default 2, same as DGCNN).
            gcn_out:        output channels of the GraphConv layer.
                            Also the LSTM input size (no compression FC before LSTM).
            lstm_hidden:    BiLSTM hidden size per direction.
            lstm_layers:    number of stacked BiLSTM layers.
            dropout_rate:   dropout probability after GCN and inside the head.
            head_dim:       intermediate dimension of the two-layer classifier head
                            (this is where electrode-axis compression happens).
        """
        super().__init__()
        self.num_electrodes = num_electrodes
        self.gcn_out = gcn_out
        lstm_out_dim = 2 * lstm_hidden

        # ── Learnable adjacency (same as DGCNN) ──────────────────────────────────
        self.adj = nn.Parameter(torch.empty(num_electrodes, num_electrodes))
        self.adj_bias = nn.Parameter(torch.empty(1))
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0.0, std=0.1)
        self.relu = nn.ReLU(inplace=True)

        # ── DGCNN spatial block (no spatial FC — output stays per-electrode) ──────
        self.graph_conv = GraphConv(k, in_channels, gcn_out)
        self.b_relu = B1ReLU(gcn_out)
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── Per-electrode BiLSTM (shared weights across all V electrodes) ─────────
        # Input is gcn_out per step (no FC compression before LSTM).
        self.bilstm = nn.LSTM(
            input_size=gcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        # ── Temporal attention (shared across electrodes, Feng et al. eq.15-17) ──
        self.attn_w = nn.Linear(lstm_out_dim, 1, bias=False)

        # ── Classifier head (electrode compression happens here) ──────────────────
        # Flatten (V, 2H) → head_dim → num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_electrodes * lstm_out_dim, head_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(head_dim, num_classes, bias=True),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V, F)
        Returns:
            logits: (B, num_classes)
        """
        B, T, V, F = x.shape

        # Build shared graph Laplacian
        adj = self.relu(self.adj + self.adj_bias)   # (V, V)
        lap = laplacian(adj)                         # (V, V)

        # ── DGCNN spatial block at every timestep ─────────────────────────────────
        x_flat = x.reshape(B * T, V, F)
        x_flat = self.graph_conv(x_flat, lap)        # (B*T, V, gcn_out)
        x_flat = self.dropout(x_flat)
        x_flat = self.b_relu(x_flat)                 # (B*T, V, gcn_out)

        # Restore (B, T, V, gcn_out) — no electrode flattening
        x_seq = x_flat.reshape(B, T, V, self.gcn_out)

        # ── Per-electrode BiLSTM ───────────────────────────────────────────────────
        # Each electrode is treated as an independent sample with a T-length sequence.
        # Permute to (B, V, T, gcn_out) → merge B and V → (B*V, T, gcn_out)
        x_node = x_seq.permute(0, 2, 1, 3).reshape(B * V, T, self.gcn_out)
        H, _ = self.bilstm(x_node)                  # (B*V, T, 2*lstm_hidden)

        # ── Temporal attention (per electrode) ────────────────────────────────────
        M = torch.tanh(H)                            # (B*V, T, 2*lstm_hidden)
        score = self.attn_w(M)                       # (B*V, T, 1)
        alpha = torch.softmax(score, dim=1)          # (B*V, T, 1)
        context = (alpha * H).sum(dim=1)             # (B*V, 2*lstm_hidden)

        # H* = tanh(context)  (Feng et al. eq.18), then restore electrode axis
        h_star = torch.tanh(context)                 # (B*V, 2*lstm_hidden)
        h_star = h_star.reshape(B, V, -1)            # (B, V, 2*lstm_hidden)

        # ── Classifier head: flatten (V, 2H) and compress ────────────────────────
        return self.head(h_star.flatten(start_dim=1))   # (B, num_classes)
