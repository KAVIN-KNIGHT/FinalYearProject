"""PyTorch LSTM architecture for LEO satellite temporal congestion prediction and 128-dim node temporal embedding extraction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class LEOLSTMModel(nn.Module):
    """2-layer LSTM model predicting target congestion_score(t+1) from historical sequence X(t-W+1..t).

    Produces 128-dimensional temporal node embeddings from the final LSTM hidden state
    for downstream fusion with GAT spatial embeddings and PPO reinforcement learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 2-layer causal LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Prediction head: Linear(128, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input sequence tensor of shape [batch_size, sequence_length, input_dim].

        Returns:
            Tuple of (pred_congestion, temporal_embedding):
                - pred_congestion: Predicted congestion_score(t+1) of shape [batch_size, 1].
                - temporal_embedding: Final LSTM hidden state embedding of shape [batch_size, 128].
        """
        # lstm_out shape: [batch_size, sequence_length, hidden_dim]
        # (h_n, c_n): h_n shape: [num_layers, batch_size, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Extract final timestep hidden state for prediction and temporal embedding
        # h_last shape: [batch_size, hidden_dim]
        h_last = lstm_out[:, -1, :]

        pred_congestion = self.head(h_last)
        return pred_congestion, h_last
