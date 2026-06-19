"""Turn selected concepts into node feature matrices.

Two membership modes (per the brief):
* **hard**  -- 1 if the node is in the concept's extent (has all intent attrs).
* **soft**  -- intent-overlap score = fraction of the intent's attributes the
  node possesses (a graded membership in [0, 1]).

Grouping (FCA_GROUP) clusters concepts by intent overlap and aggregates their
membership, giving a compact, coverage-oriented representation.
"""
from __future__ import annotations

import numpy as np

from .binarize import FormalContext
from .concepts import Concept, compute_extents


def build_membership(concepts: list[Concept], context: FormalContext,
                     mode: str = "hard") -> np.ndarray:
    """Return a [N, K] float membership matrix for ``concepts``."""
    if not concepts:
        return np.zeros((context.num_objects, 0), dtype=np.float32)
    if mode == "hard":
        return compute_extents(concepts, context).astype(np.float32)
    if mode == "soft":
        B = context.incidence
        n = B.shape[0]
        out = np.ones((n, len(concepts)), dtype=np.float32)
        for j, c in enumerate(concepts):
            if c.intent:
                cols = list(c.intent)
                out[:, j] = B[:, cols].sum(axis=1) / float(len(cols))
        return out
    raise ValueError(f"Unknown membership mode '{mode}' (use 'hard' or 'soft').")


def _intent_jaccard_distances(concepts: list[Concept]) -> np.ndarray:
    sets = [set(c.intent) for c in concepts]
    k = len(sets)
    d = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = sets[i], sets[j]
            union = len(a | b)
            sim = (len(a & b) / union) if union else 0.0
            d[i, j] = d[j, i] = 1.0 - sim
    return d


def group_concepts(concepts: list[Concept], num_groups: int) -> np.ndarray:
    """Assign each concept to one of ``num_groups`` clusters by intent overlap."""
    k = len(concepts)
    if k == 0:
        return np.zeros(0, dtype=int)
    if num_groups >= k:
        return np.arange(k)
    from sklearn.cluster import AgglomerativeClustering

    dist = _intent_jaccard_distances(concepts)
    model = AgglomerativeClustering(
        n_clusters=num_groups, metric="precomputed", linkage="average"
    )
    return model.fit_predict(dist)


def group_features(membership: np.ndarray, groups: np.ndarray,
                   num_groups: int, agg: str = "max") -> np.ndarray:
    """Aggregate concept membership within each group -> [N, num_groups]."""
    n = membership.shape[0]
    g = int(groups.max()) + 1 if groups.size else 0
    g = max(g, num_groups if num_groups > 0 else g)
    out = np.zeros((n, g), dtype=np.float32)
    for gi in range(g):
        cols = np.flatnonzero(groups == gi)
        if cols.size == 0:
            continue
        block = membership[:, cols]
        out[:, gi] = block.max(axis=1) if agg == "max" else block.mean(axis=1)
    return out
