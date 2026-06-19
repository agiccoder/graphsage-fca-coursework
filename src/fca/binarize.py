"""Build a binary formal context (objects x attributes) from node features.

A formal context is the triple (G, M, I): objects G = nodes, attributes M =
binarised features, incidence I = which node has which attribute. Continuous /
dense features must be scaled to a binary form first -- this is a normal part of
the FCA pipeline, not a limitation.

Supported modes
---------------
binary_nonzero        : attribute_f = (x_f != 0)            (ideal for bag-of-words)
quantile_binarization : attribute_{f,q} = (x_f >= quantile_q(x_f))   (one col / quantile)
interval_scaling      : nominal scaling into per-feature quantile bins
quantile_global       : attribute_f = (x_f >= quantile_f(q))         (one col / feature)
topk_per_node         : attribute_f = (x_f in top-k of its row)      (row-local)
quantile_topk         : quantile_global AND topk_per_node
graph_smoothed_topk   : top-k per node on graph-smoothed features
optional_smoothed     : graph-smooth features (A_hat @ x) then binarise as above

Richer-scaling note
-------------------
``quantile_global`` / ``quantile_topk`` may compute their per-column thresholds on
the *training nodes only* when a ``train_mask`` is supplied to
:func:`build_formal_context` (preferred, avoids feature-distribution leakage).
Labels never participate in binarisation. ``topk_per_node`` is purely row-local and
leakage-free by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class AttributeMeta:
    """Human-readable description of one binary attribute (context column)."""

    index: int
    feature: int
    op: str            # "nonzero" | ">=" | "in"
    threshold: float
    name: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "feature": self.feature,
            "op": self.op,
            "threshold": round(float(self.threshold), 6),
            "name": self.name,
        }


@dataclass
class FormalContext:
    """Boolean incidence matrix plus per-attribute metadata."""

    incidence: np.ndarray  # bool [N, M]
    attributes: list[AttributeMeta]
    mode: str
    params: dict = field(default_factory=dict)

    @property
    def num_objects(self) -> int:
        return int(self.incidence.shape[0])

    @property
    def num_attributes(self) -> int:
        return int(self.incidence.shape[1])

    def attribute_names(self) -> list[str]:
        return [a.name for a in self.attributes]


# ----------------------------------------------------------------- smoothing
def _normalized_adj_smooth(x: np.ndarray, edge_index: torch.Tensor,
                           num_nodes: int, hops: int = 1) -> np.ndarray:
    """Return symmetric-normalised neighbour-averaged features (with self loops)."""
    from torch_geometric.utils import add_self_loops, degree

    ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    dinv_sqrt = deg.pow(-0.5)
    norm = dinv_sqrt[row] * dinv_sqrt[col]
    xt = torch.from_numpy(x).float()
    for _ in range(max(1, hops)):
        out = torch.zeros_like(xt)
        msg = xt[col] * norm.unsqueeze(1)
        out.index_add_(0, row, msg)
        xt = out
    return xt.numpy()


# --------------------------------------------------------------- binarisers
def _binarize_nonzero(x: np.ndarray) -> tuple[np.ndarray, list[AttributeMeta]]:
    inc = x != 0
    attrs = [
        AttributeMeta(index=j, feature=j, op="nonzero", threshold=0.0, name=f"f{j}!=0")
        for j in range(x.shape[1])
    ]
    return inc, attrs


def _binarize_quantile(x: np.ndarray, quantiles: list[float]
                       ) -> tuple[np.ndarray, list[AttributeMeta]]:
    cols: list[np.ndarray] = []
    attrs: list[AttributeMeta] = []
    for f in range(x.shape[1]):
        xf = x[:, f]
        for q in quantiles:
            thr = float(np.quantile(xf, q))
            col = xf >= thr
            frac = col.mean()
            if frac <= 0.0 or frac >= 1.0:  # degenerate attribute, skip
                continue
            attrs.append(AttributeMeta(len(cols), f, ">=", thr, f"f{f}>=q{q:g}"))
            cols.append(col)
    if not cols:
        return np.zeros((x.shape[0], 0), dtype=bool), []
    inc = np.stack(cols, axis=1)
    for j, a in enumerate(attrs):
        a.index = j
    return inc, attrs


def _binarize_intervals(x: np.ndarray, n_bins: int
                        ) -> tuple[np.ndarray, list[AttributeMeta]]:
    cols: list[np.ndarray] = []
    attrs: list[AttributeMeta] = []
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    for f in range(x.shape[1]):
        xf = x[:, f]
        edges = np.unique(np.quantile(xf, qs))
        if edges.size < 2:
            continue
        for b in range(edges.size - 1):
            lo, hi = edges[b], edges[b + 1]
            last = b == edges.size - 2
            col = (xf >= lo) & ((xf <= hi) if last else (xf < hi))
            frac = col.mean()
            if frac <= 0.0 or frac >= 1.0:
                continue
            attrs.append(AttributeMeta(len(cols), f, "in", float(lo),
                                       f"f{f}in[{lo:.3g},{hi:.3g}{']' if last else ')'}"))
            cols.append(col)
    if not cols:
        return np.zeros((x.shape[0], 0), dtype=bool), []
    inc = np.stack(cols, axis=1)
    for j, a in enumerate(attrs):
        a.index = j
    return inc, attrs


# ----------------------------------------------------- richer scaling modes
def _column_thresholds(x: np.ndarray, q: float,
                       train_idx: Optional[np.ndarray]) -> np.ndarray:
    """Per-column quantile thresholds, computed on train rows when available.

    Using only training rows for the feature distribution avoids
    feature-distribution leakage (labels are never used here). Falls back to all
    rows when ``train_idx`` is None/empty.
    """
    src = x[train_idx] if (train_idx is not None and train_idx.size) else x
    return np.quantile(src, q, axis=0)


def _binarize_quantile_global(x: np.ndarray, q: float,
                              train_idx: Optional[np.ndarray]
                              ) -> tuple[np.ndarray, list[AttributeMeta]]:
    """One attribute per feature: 1 iff value >= the column's q-quantile."""
    thr = _column_thresholds(x, q, train_idx)
    cols: list[np.ndarray] = []
    attrs: list[AttributeMeta] = []
    for f in range(x.shape[1]):
        col = x[:, f] >= thr[f]
        frac = col.mean()
        if frac <= 0.0 or frac >= 1.0:  # degenerate -> useless attribute
            continue
        attrs.append(AttributeMeta(len(cols), f, ">=", float(thr[f]),
                                   f"f{f}>=q{q:g}"))
        cols.append(col)
    if not cols:
        return np.zeros((x.shape[0], 0), dtype=bool), []
    return np.stack(cols, axis=1), attrs


def _topk_row_mask(x: np.ndarray, k: int) -> np.ndarray:
    """Boolean [N, F] keeping the top-``k`` positive entries of each row.

    Ties are broken by argpartition order. Zero/negative entries are never kept
    (so a row with fewer than k positives activates fewer than k attributes).
    """
    n, f = x.shape
    k = max(1, min(int(k), f))
    mask = np.zeros((n, f), dtype=bool)
    if f == 0:
        return mask
    # Indices of the k largest values per row.
    part = np.argpartition(-x, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    mask[rows, part.ravel()] = True
    mask &= x > 0.0  # only genuinely active features count
    return mask


def _binarize_topk_per_node(x: np.ndarray, k: int
                            ) -> tuple[np.ndarray, list[AttributeMeta]]:
    """Attribute f active for node i iff x_if is among row i's top-k positives."""
    inc = _topk_row_mask(x, k)
    keep = np.flatnonzero(inc.any(axis=0))  # drop never-active columns
    if keep.size == 0:
        return np.zeros((x.shape[0], 0), dtype=bool), []
    inc = np.ascontiguousarray(inc[:, keep])
    attrs = [AttributeMeta(j, int(f), "topk", float(k), f"f{int(f)}@top{k}")
             for j, f in enumerate(keep)]
    return inc, attrs


def _binarize_quantile_topk(x: np.ndarray, q: float, k: int,
                            train_idx: Optional[np.ndarray]
                            ) -> tuple[np.ndarray, list[AttributeMeta]]:
    """Conjunction of column-wise quantile gate and row-wise top-k gate."""
    thr = _column_thresholds(x, q, train_idx)
    above = x >= thr[None, :]
    topk = _topk_row_mask(x, k)
    inc_full = above & topk
    keep = np.flatnonzero(inc_full.any(axis=0) & ~inc_full.all(axis=0))
    if keep.size == 0:
        return np.zeros((x.shape[0], 0), dtype=bool), []
    inc = np.ascontiguousarray(inc_full[:, keep])
    attrs = [AttributeMeta(j, int(f), ">=&topk", float(thr[f]),
                           f"f{int(f)}>=q{q:g}&top{k}")
             for j, f in enumerate(keep)]
    return inc, attrs


# --------------------------------------------------------------------- API
def build_formal_context(
    x: torch.Tensor,
    mode: str = "binary_nonzero",
    params: Optional[dict] = None,
    edge_index: Optional[torch.Tensor] = None,
    train_mask: Optional[object] = None,
) -> FormalContext:
    """Construct a :class:`FormalContext` from a feature matrix.

    Parameters
    ----------
    x          : FloatTensor [N, F] node features.
    mode       : binary_nonzero | quantile_binarization | interval_scaling |
                 quantile_global | topk_per_node | quantile_topk |
                 graph_smoothed_topk | optional_smoothed_features.
    params     : mode-specific options, e.g. ``{"quantiles": [0.5, 0.8]}``,
                 ``{"n_bins": 4}``, ``{"quantile": 0.75}``, ``{"topk": 10}``,
                 ``{"quantile": 0.75, "topk": 10}``,
                 ``{"smooth_alpha": 0.5, "topk": 10}``, or for smoothing
                 ``{"hops": 1, "base_mode": "quantile_binarization",
                 "quantiles": [0.7]}``.
    edge_index : required for ``optional_smoothed_features`` and
                 ``graph_smoothed_topk``.
    train_mask : optional boolean / index array. When given, quantile thresholds
                 for ``quantile_global`` / ``quantile_topk`` are computed on the
                 training rows only (avoids feature-distribution leakage).
    """
    params = dict(params or {})
    xn = x.detach().cpu().numpy().astype(np.float32)

    train_idx: Optional[np.ndarray] = None
    if train_mask is not None:
        _to_np = getattr(train_mask, "detach", None)
        raw = _to_np().cpu().numpy() if _to_np is not None else train_mask
        tm = np.asarray(raw)
        train_idx = np.flatnonzero(tm) if tm.dtype == bool else tm.astype(int)
        params.setdefault("threshold_scope", "train")
    else:
        params.setdefault("threshold_scope", "all")

    if mode == "optional_smoothed_features":
        if edge_index is None:
            raise ValueError("optional_smoothed_features requires edge_index.")
        xn = _normalized_adj_smooth(xn, edge_index, xn.shape[0],
                                    hops=int(params.get("hops", 1)))
        mode_eff = params.get("base_mode", "quantile_binarization")
    elif mode == "graph_smoothed_topk":
        if edge_index is None:
            raise ValueError("graph_smoothed_topk requires edge_index.")
        alpha = float(params.get("smooth_alpha", 0.5))
        smoothed = _normalized_adj_smooth(xn, edge_index, xn.shape[0],
                                          hops=int(params.get("hops", 1)))
        xn = alpha * xn + (1.0 - alpha) * smoothed
        mode_eff = "topk_per_node"
    else:
        mode_eff = mode

    if mode_eff == "binary_nonzero":
        inc, attrs = _binarize_nonzero(xn)
    elif mode_eff == "quantile_binarization":
        inc, attrs = _binarize_quantile(xn, params.get("quantiles", [0.5]))
    elif mode_eff == "interval_scaling":
        inc, attrs = _binarize_intervals(xn, int(params.get("n_bins", 4)))
    elif mode_eff == "quantile_global":
        inc, attrs = _binarize_quantile_global(
            xn, float(params.get("quantile", 0.75)), train_idx)
    elif mode_eff == "topk_per_node":
        inc, attrs = _binarize_topk_per_node(xn, int(params.get("topk", 10)))
    elif mode_eff == "quantile_topk":
        inc, attrs = _binarize_quantile_topk(
            xn, float(params.get("quantile", 0.75)),
            int(params.get("topk", 10)), train_idx)
    else:
        raise ValueError(f"Unknown binarization mode: {mode_eff}")

    return FormalContext(incidence=np.ascontiguousarray(inc, dtype=bool),
                         attributes=attrs, mode=mode, params=params)
