"""Official dataset loaders, all normalised to :class:`GraphData`.

Supported (canonical names):
    core     : cora, citeseer, pubmed                 (Planetoid)
    optional : roman_empire, amazon_ratings           (HeterophilousGraphDataset)
    optional : ogbn_arxiv                             (OGB, requires `ogb`)

Name matching is tolerant: "Roman-Empire", "roman empire" and "roman_empire"
all resolve to the same loader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from ..utils.logging import get_logger
from ..utils.paths import DATASETS_DIR
from .types import GraphData, masks_from_indices

logger = get_logger("data.loaders")

_PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
_HETERO = {"roman_empire": "Roman-empire", "amazon_ratings": "Amazon-ratings"}
_OGBN = {"ogbn_arxiv": "ogbn-arxiv"}


def normalize_name(name: str) -> str:
    """Canonicalise a dataset name (lowercase, spaces/dashes -> underscores)."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def list_datasets() -> dict[str, list[str]]:
    return {
        "core": list(_PLANETOID),
        "heterophilous": list(_HETERO),
        "ogb": list(_OGBN),
    }


def load_dataset(
    name: str,
    root: str | Path | None = None,
    *,
    split_idx: int = 0,
    planetoid_split: str = "public",
    to_undirected: bool | None = None,
    **_: object,
) -> GraphData:
    """Load ``name`` and return a normalised :class:`GraphData`.

    Parameters
    ----------
    split_idx       : which split column to use for heterophilous datasets.
    planetoid_split : "public" | "full" | "geom-gcn" for Planetoid datasets.
    to_undirected   : force symmetrise edges; defaults to a per-dataset choice.
    """
    root = Path(root) if root is not None else DATASETS_DIR
    key = normalize_name(name)
    if key in _PLANETOID:
        return _load_planetoid(key, root, planetoid_split, to_undirected)
    if key in _HETERO:
        return _load_hetero(key, root, split_idx, to_undirected)
    if key in _OGBN:
        return _load_ogbn_arxiv(root, to_undirected)
    raise ValueError(
        f"Unknown dataset '{name}'. Known: "
        f"{sorted(set(_PLANETOID) | set(_HETERO) | set(_OGBN))}"
    )


# --------------------------------------------------------------------- helpers
def _maybe_undirected(edge_index: torch.Tensor, num_nodes: int, flag: bool | None,
                      default: bool) -> torch.Tensor:
    use = default if flag is None else flag
    if not use:
        return edge_index
    from torch_geometric.utils import to_undirected
    return to_undirected(edge_index, num_nodes=num_nodes)


def _finalize(name: str, x: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor,
              train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
              extra_meta: dict | None = None) -> GraphData:
    x = x.float()
    y = y.long().view(-1)
    edge_index = edge_index.long()
    meta = {
        "name": name,
        "num_nodes": int(x.size(0)),
        "num_edges": int(edge_index.size(1)),
        "num_features": int(x.size(1)),
        "num_classes": int(y.max().item()) + 1,
    }
    if extra_meta:
        meta.update(extra_meta)
    data = GraphData(
        name=name,
        edge_index=edge_index,
        x_raw=x,
        y=y,
        train_mask=train_mask.bool(),
        val_mask=val_mask.bool(),
        test_mask=test_mask.bool(),
        metadata=meta,
    )
    logger.info(
        "Loaded %s: N=%d E=%d F=%d C=%d | splits=%s",
        name, data.num_nodes, data.num_edges, data.num_features,
        data.num_classes, data.split_sizes(),
    )
    return data


def _load_planetoid(key: str, root: Path, split: str,
                    to_undirected: bool | None) -> GraphData:
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root=str(root / "Planetoid"), name=_PLANETOID[key], split=split)
    d = ds[0]
    edge_index = _maybe_undirected(d.edge_index, d.num_nodes, to_undirected, default=False)
    return _finalize(
        key, d.x, d.y, edge_index, d.train_mask, d.val_mask, d.test_mask,
        extra_meta={"source": "Planetoid", "planetoid_split": split},
    )


def _select_split_column(mask: torch.Tensor, split_idx: int) -> torch.Tensor:
    """Heterophilous datasets ship masks of shape [N, num_splits]."""
    if mask.dim() == 2:
        split_idx = min(split_idx, mask.size(1) - 1)
        return mask[:, split_idx]
    return mask


def _load_hetero(key: str, root: Path, split_idx: int,
                 to_undirected: bool | None) -> GraphData:
    from torch_geometric.datasets import HeterophilousGraphDataset

    ds = HeterophilousGraphDataset(root=str(root / "Heterophilous"), name=_HETERO[key])
    d = ds[0]
    edge_index = _maybe_undirected(d.edge_index, d.num_nodes, to_undirected, default=True)
    train = _select_split_column(d.train_mask, split_idx)
    val = _select_split_column(d.val_mask, split_idx)
    test = _select_split_column(d.test_mask, split_idx)
    return _finalize(
        key, d.x, d.y, edge_index, train, val, test,
        extra_meta={"source": "HeterophilousGraphDataset", "split_idx": split_idx},
    )


def _load_ogbn_arxiv(root: Path, to_undirected: bool | None) -> GraphData:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ogbn-arxiv requires the `ogb` package. Install it with `pip install ogb`."
        ) from exc

    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "OGB"))
    d = ds[0]
    split = ds.get_idx_split()
    train_mask, val_mask, test_mask = masks_from_indices(
        d.num_nodes, split["train"], split["valid"], split["test"]
    )
    edge_index = _maybe_undirected(d.edge_index, d.num_nodes, to_undirected, default=True)
    return _finalize(
        "ogbn_arxiv", d.x, d.y, edge_index, train_mask, val_mask, test_mask,
        extra_meta={"source": "OGB", "split": "time"},
    )
