"""Configurable GraphSAGE encoder + node classifier (PyTorch Geometric)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import JumpingKnowledge, SAGEConv


def _norm_layer(kind: str | bool, dim: int) -> nn.Module:
    if kind in (False, None, "none"):
        return nn.Identity()
    if kind in (True, "batch", "batchnorm"):
        return nn.BatchNorm1d(dim)
    if kind in ("layer", "layernorm"):
        return nn.LayerNorm(dim)
    return nn.Identity()


class GraphSAGE(nn.Module):
    """GraphSAGE with configurable depth, width, aggregation and tricks.

    Parameters
    ----------
    aggr      : "mean" | "max" | "lstm" (passed to SAGEConv).
    project   : if True, project neighbour features before aggregation. With
                ``aggr="max"`` this reproduces the paper's pool/projected-max.
    norm      : False | "batch" | "layer".
    residual  : add a skip connection across hidden layers.
    jk        : "cat" | "max" | "lstm" jumping-knowledge over layer outputs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
        project: bool = False,
        norm: str | bool = False,
        residual: bool = False,
        jk: str | bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.jk_mode = jk if jk and jk != "none" else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_channels] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr, project=project))
            self.norms.append(_norm_layer(norm, hidden_channels))

        if self.jk_mode:
            self.jk = JumpingKnowledge(self.jk_mode, hidden_channels, num_layers)
            clf_in = hidden_channels * num_layers if self.jk_mode == "cat" else hidden_channels
        else:
            self.jk = None
            clf_in = hidden_channels
        self.classifier = nn.Linear(clf_in, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        xs = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h.shape == x.shape:
                h = h + x
            x = h
            xs.append(x)
        x = self.jk(xs) if self.jk is not None else x
        return self.classifier(x)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()
        if self.jk is not None and hasattr(self.jk, "reset_parameters"):
            self.jk.reset_parameters()
