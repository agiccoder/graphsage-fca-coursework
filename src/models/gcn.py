"""A standard GCN baseline (optional comparison model)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm: str | bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_channels] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            use_norm = norm not in (False, None, "none") and i < num_layers - 1
            self.norms.append(nn.BatchNorm1d(dims[i + 1]) if use_norm else nn.Identity())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
