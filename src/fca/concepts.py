"""Concept representation, extent computation, scoring and selection.

Important (no label leakage): the *unsupervised* scorers (support, area,
stability) never touch labels. The *supervised* scorers (target_entropy, lift)
and :func:`class_association` use labels ONLY on the training mask, so test
labels never influence which concepts are selected or how features are built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .binarize import FormalContext


@dataclass
class Concept:
    """A formal concept (extent, intent) with bookkeeping for selection/analysis."""

    concept_id: int
    intent: tuple[int, ...]            # attribute indices into the context
    support: int                       # |extent| (number of objects)
    intent_size: int
    extent_size: int
    selection_score: float = 0.0
    attributes: list[str] = field(default_factory=list)  # human-readable intent
    coverage: float = 0.0              # support / N
    # Class association computed on TRAIN nodes only (leakage-safe), -1 if unknown
    dominant_class: int = -1
    purity: float = 0.0
    lift: float = 0.0

    def to_row(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "support": self.support,
            "extent_size": self.extent_size,
            "intent_size": self.intent_size,
            "selection_score": round(float(self.selection_score), 6),
            "coverage": round(float(self.coverage), 6),
            "dominant_class": self.dominant_class,
            "purity": round(float(self.purity), 6),
            "lift": round(float(self.lift), 6),
            "attributes": ";".join(self.attributes),
            "intent_indices": ";".join(map(str, self.intent)),
        }


# --------------------------------------------------------------------- extents
def compute_extents(concepts: list[Concept], context: FormalContext) -> np.ndarray:
    """Return a boolean extent matrix of shape [N, K] (node in concept k)."""
    B = context.incidence
    n = B.shape[0]
    k = len(concepts)
    out = np.ones((n, k), dtype=bool)
    for j, c in enumerate(concepts):
        if c.intent:
            out[:, j] = B[:, list(c.intent)].all(axis=1)
    return out


# --------------------------------------------------------------------- scorers
def _score_support(ext: np.ndarray, concepts, **_) -> np.ndarray:
    return ext.sum(axis=0).astype(np.float64)


def _score_area(ext: np.ndarray, concepts, **_) -> np.ndarray:
    sizes = np.array([c.intent_size for c in concepts], dtype=np.float64)
    return ext.sum(axis=0).astype(np.float64) * np.maximum(sizes, 1.0)


def _score_stability(ext: np.ndarray, concepts, context: FormalContext,
                     params: dict, rng: np.random.Generator, **_) -> np.ndarray:
    """Monte-Carlo approximation of intensional stability (unsupervised)."""
    B = context.incidence
    n_samples = int(params.get("stability_samples", 50))
    max_ext = int(params.get("stability_max_extent", 256))
    scores = np.zeros(len(concepts), dtype=np.float64)
    for j, c in enumerate(concepts):
        objs = np.flatnonzero(ext[:, j])
        if objs.size == 0:
            continue
        if objs.size > max_ext:
            objs = rng.choice(objs, size=max_ext, replace=False)
        intent_set = set(c.intent)
        stable = 0
        for _ in range(n_samples):
            keep = objs[rng.random(objs.size) < 0.5]
            if keep.size == 0:
                keep = objs[rng.integers(0, objs.size, size=1)]
            sub_intent = np.flatnonzero(B[keep].all(axis=0))
            if set(sub_intent.tolist()) == intent_set:
                stable += 1
        scores[j] = stable / n_samples
    return scores


def _train_class_stats(ext_col: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                       num_classes: int) -> tuple[np.ndarray, int]:
    sel = ext_col & train_mask
    count = int(sel.sum())
    if count == 0:
        return np.zeros(num_classes), 0
    counts = np.bincount(y[sel], minlength=num_classes).astype(np.float64)
    return counts, count


def _score_target_entropy(ext, concepts, y, train_mask, num_classes, params, **_):
    scores = np.zeros(len(concepts), dtype=np.float64)
    log_c = np.log(max(num_classes, 2))
    min_count = int(params.get("min_train_count", 3))
    for j in range(len(concepts)):
        counts, count = _train_class_stats(ext[:, j], y, train_mask, num_classes)
        if count < min_count:
            continue
        p = counts / counts.sum()
        nz = p[p > 0]
        h = float(-(nz * np.log(nz)).sum()) / log_c
        scores[j] = (1.0 - h) * np.sqrt(count)
    return scores


def _score_lift(ext, concepts, y, train_mask, num_classes, params, **_):
    scores = np.zeros(len(concepts), dtype=np.float64)
    prior = np.bincount(y[train_mask], minlength=num_classes).astype(np.float64)
    prior = prior / max(prior.sum(), 1.0)
    min_count = int(params.get("min_train_count", 3))
    for j in range(len(concepts)):
        counts, count = _train_class_stats(ext[:, j], y, train_mask, num_classes)
        if count < min_count:
            continue
        p = counts / counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(prior > 0, p / prior, 0.0)
        scores[j] = float(np.max(ratio)) * np.sqrt(count)
    return scores


SCORERS: dict[str, Callable] = {
    "support": _score_support,
    "area": _score_area,
    "stability": _score_stability,
    "target_entropy": _score_target_entropy,
    "lift": _score_lift,
}

SUPERVISED_SCORERS = {"target_entropy", "lift"}


def score_concepts(
    concepts: list[Concept],
    context: FormalContext,
    scorer: str = "support",
    y: Optional[np.ndarray] = None,
    train_mask: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    params: Optional[dict] = None,
    seed: int = 0,
) -> list[Concept]:
    """Compute ``selection_score`` for every concept using ``scorer``."""
    if scorer not in SCORERS:
        raise ValueError(f"Unknown scorer '{scorer}'. Available: {list(SCORERS)}")
    params = dict(params or {})
    ext = compute_extents(concepts, context)
    if scorer in SUPERVISED_SCORERS:
        if y is None or train_mask is None or num_classes is None:
            raise ValueError(f"Scorer '{scorer}' needs y, train_mask, num_classes.")
    scores = SCORERS[scorer](
        ext=ext, concepts=concepts, context=context, y=y, train_mask=train_mask,
        num_classes=num_classes, params=params, rng=np.random.default_rng(seed),
    )
    for c, s in zip(concepts, scores):
        c.selection_score = float(s)
    return concepts


def class_association(
    concepts: list[Concept],
    context: FormalContext,
    y: np.ndarray,
    train_mask: np.ndarray,
    num_classes: int,
) -> list[Concept]:
    """Annotate concepts with dominant class / purity / lift on TRAIN nodes only."""
    ext = compute_extents(concepts, context)
    prior = np.bincount(y[train_mask], minlength=num_classes).astype(np.float64)
    prior = prior / max(prior.sum(), 1.0)
    for j, c in enumerate(concepts):
        counts, count = _train_class_stats(ext[:, j], y, train_mask, num_classes)
        if count == 0:
            continue
        p = counts / counts.sum()
        dom = int(np.argmax(p))
        c.dominant_class = dom
        c.purity = float(p[dom])
        c.lift = float(p[dom] / prior[dom]) if prior[dom] > 0 else 0.0
    return concepts


def select_top_k(concepts: list[Concept], k: int) -> list[Concept]:
    """Return the top-``k`` concepts by ``selection_score`` (ids reassigned 0..k-1)."""
    ranked = sorted(concepts, key=lambda c: c.selection_score, reverse=True)[:k]
    for i, c in enumerate(ranked):
        c.concept_id = i
    return ranked
