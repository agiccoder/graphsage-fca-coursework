"""Interval pattern-structure features for numeric node attributes.

This module implements a compact, experiment-oriented pattern-structure extension
for the existing GraphSAGE+FCA pipeline. Instead of first reducing numeric node
features to a binary context, it builds interval descriptions directly on the raw
feature values:

    pattern = (features, lows, highs)
    extent  = nodes whose values lie inside every selected feature interval

The implementation is deliberately conservative and leakage-aware:

* interval boundaries are computed from training rows when a train mask is given;
* labels are used only by supervised scorers and only on train rows;
* candidate generation is bounded by max_features/object_sample to keep runtime
  tractable on citation-network datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class IntervalPattern:
    """A numeric pattern represented by per-feature intervals."""

    pattern_id: int
    features: tuple[int, ...]
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    support: int
    extent_size: int
    intent_size: int
    selection_score: float = 0.0
    coverage: float = 0.0
    dominant_class: int = -1
    purity: float = 0.0
    lift: float = 0.0
    attributes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "concept_id": self.pattern_id,
            "support": self.support,
            "extent_size": self.extent_size,
            "intent_size": self.intent_size,
            "selection_score": round(float(self.selection_score), 6),
            "coverage": round(float(self.coverage), 6),
            "dominant_class": self.dominant_class,
            "purity": round(float(self.purity), 6),
            "lift": round(float(self.lift), 6),
            "attributes": ";".join(self.attributes),
            "intent_indices": ";".join(map(str, self.features)),
            "lows": ";".join(f"{v:.6g}" for v in self.lows),
            "highs": ";".join(f"{v:.6g}" for v in self.highs),
            "pattern_type": "interval",
        }


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def _train_indices(train_mask: Optional[object], n: int) -> np.ndarray:
    if train_mask is None:
        return np.arange(n)
    detach = getattr(train_mask, "detach", None)
    raw = detach().cpu().numpy() if detach is not None else train_mask
    arr = np.asarray(raw)
    if arr.dtype == bool:
        idx = np.flatnonzero(arr)
    else:
        idx = arr.astype(int)
    return idx if idx.size else np.arange(n)


def _select_features(x: np.ndarray, train_idx: np.ndarray, max_features: int,
                     rank: str) -> np.ndarray:
    src = x[train_idx]
    positive = (src > 0).mean(axis=0)
    variance = src.var(axis=0)
    if rank == "variance":
        score = variance
    elif rank == "balanced":
        score = positive * (1.0 - positive)
    else:
        score = positive
    keep = np.flatnonzero((positive > 0.0) & (variance >= 0.0))
    if keep.size == 0:
        return np.arange(min(max_features, x.shape[1]))
    order = np.argsort(-score[keep])
    return np.sort(keep[order][:max_features])


def _quantile_edges(x: np.ndarray, train_idx: np.ndarray, features: np.ndarray,
                    n_bins: int) -> dict[int, np.ndarray]:
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out: dict[int, np.ndarray] = {}
    for f in features.tolist():
        vals = x[train_idx, f]
        edges = np.unique(np.quantile(vals, qs))
        if edges.size < 2:
            vals_all = x[:, f]
            edges = np.unique(np.quantile(vals_all, qs))
        if edges.size >= 2:
            out[int(f)] = edges.astype(np.float32)
    return out


def _interval_for_value(value: float, edges: np.ndarray) -> tuple[float, float]:
    pos = int(np.searchsorted(edges, value, side="right") - 1)
    pos = max(0, min(pos, edges.size - 2))
    lo = float(edges[pos])
    hi = float(edges[pos + 1])
    if lo == hi:
        hi = lo + 1e-12
    return lo, hi


def _extent(x: np.ndarray, features: tuple[int, ...], lows: tuple[float, ...],
            highs: tuple[float, ...]) -> np.ndarray:
    mask = np.ones(x.shape[0], dtype=bool)
    for f, lo, hi in zip(features, lows, highs):
        col = x[:, f]
        mask &= (col >= lo) & (col <= hi)
    return mask


def _pattern_key(features: tuple[int, ...], lows: tuple[float, ...],
                 highs: tuple[float, ...]) -> tuple:
    return (
        features,
        tuple(round(float(v), 8) for v in lows),
        tuple(round(float(v), 8) for v in highs),
    )


def mine_interval_patterns(
    x_tensor: torch.Tensor,
    train_mask: Optional[object] = None,
    params: Optional[dict] = None,
    seed: int = 0,
) -> tuple[list[IntervalPattern], dict]:
    """Generate bounded interval-pattern candidates from node descriptions."""
    params = dict(params or {})
    x = _to_numpy(x_tensor)
    n, _ = x.shape
    rng = np.random.default_rng(seed)
    train_idx = _train_indices(train_mask, n)

    max_features = int(params.get("max_features", 512))
    feature_rank = str(params.get("feature_rank", "support"))
    features = _select_features(x, train_idx, max_features, feature_rank)

    n_bins = int(params.get("n_bins", 4))
    edges = _quantile_edges(x, train_idx, features, n_bins)
    features = np.array([f for f in features.tolist() if int(f) in edges], dtype=int)
    if features.size == 0:
        return [], {"pattern_num_candidates": 0, "pattern_num_features": 0}

    object_sample = int(params.get("object_sample", min(2000, n)))
    sample_idx = np.arange(n) if object_sample >= n else rng.choice(n, size=object_sample, replace=False)
    intent_size = int(params.get("intent_size", 2))
    intent_size = max(1, min(intent_size, features.size))
    min_support = float(params.get("min_support", 0.01))
    max_support = float(params.get("max_support", 0.6))
    lo_sup = int(np.ceil(min_support * n))
    hi_sup = int(np.floor(max_support * n))

    candidates: dict[tuple, IntervalPattern] = {}
    for obj in sample_idx.tolist():
        vals = x[obj, features]
        positive = vals > 0.0
        if not positive.any():
            continue
        avail = features[positive]
        avals = vals[positive]
        top = np.argsort(-avals)[:intent_size]
        chosen = tuple(sorted(int(f) for f in avail[top].tolist()))
        lows: list[float] = []
        highs: list[float] = []
        for f in chosen:
            lo, hi = _interval_for_value(float(x[obj, f]), edges[f])
            lows.append(lo)
            highs.append(hi)
        key = _pattern_key(chosen, tuple(lows), tuple(highs))
        if key in candidates:
            continue
        ext = _extent(x, chosen, tuple(lows), tuple(highs))
        support = int(ext.sum())
        if support < max(lo_sup, 1) or support > hi_sup:
            continue
        attrs = [f"f{f}in[{lo:.3g},{hi:.3g}]" for f, lo, hi in zip(chosen, lows, highs)]
        pid = len(candidates)
        candidates[key] = IntervalPattern(
            pattern_id=pid,
            features=chosen,
            lows=tuple(lows),
            highs=tuple(highs),
            support=support,
            extent_size=support,
            intent_size=len(chosen),
            coverage=support / n,
            attributes=attrs,
        )

    meta = {
        "pattern_num_candidates": len(candidates),
        "pattern_num_features": int(features.size),
        "pattern_n_bins": n_bins,
        "pattern_intent_size": intent_size,
    }
    return list(candidates.values()), meta


def compute_pattern_extents(patterns: list[IntervalPattern], x_tensor: torch.Tensor) -> np.ndarray:
    x = _to_numpy(x_tensor)
    out = np.zeros((x.shape[0], len(patterns)), dtype=bool)
    for j, p in enumerate(patterns):
        out[:, j] = _extent(x, p.features, p.lows, p.highs)
    return out


def _train_class_stats(ext_col: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                       num_classes: int) -> tuple[np.ndarray, int]:
    sel = ext_col & train_mask
    count = int(sel.sum())
    if count == 0:
        return np.zeros(num_classes), 0
    counts = np.bincount(y[sel], minlength=num_classes).astype(np.float64)
    return counts, count


def score_patterns(
    patterns: list[IntervalPattern],
    x_tensor: torch.Tensor,
    scorer: str = "support",
    y: Optional[np.ndarray] = None,
    train_mask: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    params: Optional[dict] = None,
) -> list[IntervalPattern]:
    """Score interval patterns. Supervised scorers use train labels only."""
    params = dict(params or {})
    ext = compute_pattern_extents(patterns, x_tensor)
    y_arr: Optional[np.ndarray] = y
    train_arr: Optional[np.ndarray] = train_mask
    n_classes: Optional[int] = num_classes
    if scorer in {"target_entropy", "lift"}:
        if y_arr is None or train_arr is None or n_classes is None:
            raise ValueError(f"Scorer '{scorer}' needs y, train_mask and num_classes.")
    if scorer == "support":
        scores = ext.sum(axis=0).astype(np.float64)
    elif scorer == "area":
        sizes = np.array([p.intent_size for p in patterns], dtype=np.float64)
        scores = ext.sum(axis=0).astype(np.float64) * np.maximum(sizes, 1.0)
    elif scorer == "target_entropy":
        assert y_arr is not None and train_arr is not None and n_classes is not None
        scores = np.zeros(len(patterns), dtype=np.float64)
        log_c = np.log(max(int(n_classes), 2))
        min_count = int(params.get("min_train_count", 3))
        for j in range(len(patterns)):
            counts, count = _train_class_stats(ext[:, j], y_arr, train_arr, int(n_classes))
            if count < min_count:
                continue
            p = counts / counts.sum()
            nz = p[p > 0]
            h = float(-(nz * np.log(nz)).sum()) / log_c
            scores[j] = (1.0 - h) * np.sqrt(count)
    elif scorer == "lift":
        assert y_arr is not None and train_arr is not None and n_classes is not None
        scores = np.zeros(len(patterns), dtype=np.float64)
        prior = np.bincount(y_arr[train_arr], minlength=int(n_classes)).astype(np.float64)
        prior = prior / max(prior.sum(), 1.0)
        min_count = int(params.get("min_train_count", 3))
        for j in range(len(patterns)):
            counts, count = _train_class_stats(ext[:, j], y_arr, train_arr, int(n_classes))
            if count < min_count:
                continue
            p = counts / counts.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(prior > 0, p / prior, 0.0)
            scores[j] = float(np.max(ratio)) * np.sqrt(count)
    else:
        raise ValueError("Unknown pattern scorer: %s" % scorer)
    for pat, score in zip(patterns, scores):
        pat.selection_score = float(score)
    return patterns


def annotate_patterns(
    patterns: list[IntervalPattern],
    x_tensor: torch.Tensor,
    y: np.ndarray,
    train_mask: np.ndarray,
    num_classes: int,
) -> list[IntervalPattern]:
    ext = compute_pattern_extents(patterns, x_tensor)
    prior = np.bincount(y[train_mask], minlength=num_classes).astype(np.float64)
    prior = prior / max(prior.sum(), 1.0)
    for j, pat in enumerate(patterns):
        counts, count = _train_class_stats(ext[:, j], y, train_mask, num_classes)
        if count == 0:
            continue
        p = counts / counts.sum()
        dom = int(np.argmax(p))
        pat.dominant_class = dom
        pat.purity = float(p[dom])
        pat.lift = float(p[dom] / prior[dom]) if prior[dom] > 0 else 0.0
    return patterns


def select_top_patterns(patterns: list[IntervalPattern], k: int) -> list[IntervalPattern]:
    ranked = sorted(patterns, key=lambda p: p.selection_score, reverse=True)[:k]
    for i, pat in enumerate(ranked):
        pat.pattern_id = i
    return ranked


def build_pattern_membership(
    patterns: list[IntervalPattern],
    x_tensor: torch.Tensor,
    mode: str = "hard",
) -> np.ndarray:
    """Return [N, K] interval-pattern membership features."""
    x = _to_numpy(x_tensor)
    if not patterns:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    hard = compute_pattern_extents(patterns, x_tensor).astype(np.float32)
    if mode == "hard":
        return hard
    if mode != "soft":
        raise ValueError("Pattern membership must be 'hard' or 'soft'.")
    out = np.zeros_like(hard, dtype=np.float32)
    eps = 1e-12
    for j, pat in enumerate(patterns):
        sims = np.ones(x.shape[0], dtype=np.float32)
        for f, lo, hi in zip(pat.features, pat.lows, pat.highs):
            center = 0.5 * (lo + hi)
            radius = max(0.5 * (hi - lo), eps)
            dist = np.abs(x[:, f] - center) / radius
            sims *= np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
        out[:, j] = sims
    return out


def pattern_coverage(patterns: list[IntervalPattern], membership: np.ndarray,
                     meta: Optional[dict] = None) -> dict:
    meta = dict(meta or {})
    if not patterns:
        return {
            "num_concepts": 0,
            "node_coverage": 0.0,
            "avg_concepts_per_node": 0.0,
            "sparsity": 1.0,
            "mean_intent_size": 0.0,
            "median_intent_size": 0.0,
            "n_single_attr": 0,
            "n_multi_attr": 0,
            "mean_extent_size": 0.0,
            **meta,
        }
    active = membership > 0
    per_node = active.sum(axis=1)
    sizes = np.array([p.intent_size for p in patterns])
    extents = np.array([p.extent_size for p in patterns])
    return {
        "num_concepts": len(patterns),
        "node_coverage": float((per_node > 0).mean()),
        "avg_concepts_per_node": float(per_node.mean()),
        "sparsity": float(1.0 - active.mean()),
        "mean_intent_size": round(float(sizes.mean()), 4),
        "median_intent_size": float(np.median(sizes)),
        "n_single_attr": int((sizes == 1).sum()),
        "n_multi_attr": int((sizes >= 2).sum()),
        "mean_extent_size": round(float(extents.mean()), 4),
        **meta,
    }


def patterns_dataframe(patterns: list[IntervalPattern]) -> pd.DataFrame:
    return pd.DataFrame([p.to_row() for p in patterns])
