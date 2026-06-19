"""The single internal graph representation used across the whole pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GraphData:
    """Unified container for an attributed graph node-classification task.

    Every loader normalises its dataset to this format so that the FCA, model,
    training and evaluation code never needs to know the original source.
    """

    name: str
    edge_index: torch.Tensor  # LongTensor [2, E]
    x_raw: torch.Tensor       # FloatTensor [N, F]
    y: torch.Tensor           # LongTensor [N]
    train_mask: torch.Tensor  # BoolTensor [N]
    val_mask: torch.Tensor    # BoolTensor [N]
    test_mask: torch.Tensor   # BoolTensor [N]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ props
    @property
    def num_nodes(self) -> int:
        return int(self.x_raw.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def num_features(self) -> int:
        return int(self.x_raw.size(1))

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    # ----------------------------------------------------------------- helpers
    def degrees(self) -> torch.Tensor:
        """Per-node degree computed from ``edge_index`` (treats it as stored)."""
        deg = torch.zeros(self.num_nodes, dtype=torch.long)
        deg.index_add_(0, self.edge_index[0], torch.ones(self.num_edges, dtype=torch.long))
        return deg

    def to(self, device: torch.device | str) -> "GraphData":
        """Move all tensors to ``device`` (returns self for chaining)."""
        self.edge_index = self.edge_index.to(device)
        self.x_raw = self.x_raw.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train_mask.sum().item()),
            "val": int(self.val_mask.sum().item()),
            "test": int(self.test_mask.sum().item()),
        }


def masks_from_indices(
    num_nodes: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build boolean masks of length ``num_nodes`` from index tensors."""
    def _mask(idx: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[idx.long()] = True
        return m

    return _mask(train_idx), _mask(val_idx), _mask(test_idx)
