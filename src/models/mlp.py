"""Feature-only baselines: a logistic-regression (num_layers=1) and an MLP.

These ignore the graph structure, isolating the contribution of the features
(raw vs FCA) from the contribution of message passing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron. ``num_layers=1`` reduces to logistic regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        if num_layers <= 1:
            self.lins.append(nn.Linear(in_dim, num_classes))
        else:
            self.lins.append(nn.Linear(in_dim, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels) if norm else nn.Identity())
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.norms.append(nn.BatchNorm1d(hidden_channels) if norm else nn.Identity())
            self.lins.append(nn.Linear(hidden_channels, num_classes))

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()
