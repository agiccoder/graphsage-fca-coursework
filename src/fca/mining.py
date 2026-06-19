"""Concept mining with a self-contained, tractable fallback.

The fallback generates genuine closed concepts from two complementary sources:

* **Attribute concepts** -- the closure of each single attribute, derived in one
  shot from the co-occurrence matrix ``CO = B^T B``: the intent of attribute *a*
  is ``{c : CO[a, c] == CO[a, a]}``. There are at most ``M`` of these and they
  are cheap and interpretable.
* **Object concepts** -- the closure of (a sample of) individual node rows, which
  adds finer, data-specific concepts.

Both are real formal concepts. ``caspailleur`` can be plugged in via
``backend="caspailleur"`` when installed; otherwise the fallback is used.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging import get_logger
from .binarize import FormalContext
from .concepts import Concept

logger = get_logger("fca.mining")


# ---------------------------------------------------------------- prefiltering
def prefilter_attributes(
    context: FormalContext,
    min_support: float = 0.01,
    max_support: float = 0.5,
    max_attributes: int = 512,
    rank: str = "support",
) -> FormalContext:
    """Drop too-rare / too-frequent attributes and cap the attribute count.

    Filtering before mining keeps the concept lattice tractable and removes
    attributes that cannot yield discriminative concepts.
    """
    B = context.incidence
    n, m = B.shape
    if m == 0:
        return context
    frac = B.mean(axis=0)
    keep = np.flatnonzero((frac >= min_support) & (frac <= max_support))
    if keep.size == 0:  # window too tight -> fall back to the least frequent ones
        keep = np.argsort(frac)[: max_attributes]

    if rank == "balanced":
        order = np.argsort(np.abs(frac[keep] - 0.5))
    else:  # "support": keep the most frequent within the window
        order = np.argsort(-frac[keep])
    keep = keep[order][:max_attributes]
    keep = np.sort(keep)

    new_inc = np.ascontiguousarray(B[:, keep])
    new_attrs = []
    for new_idx, old_idx in enumerate(keep.tolist()):
        a = context.attributes[old_idx]
        new_attrs.append(type(a)(new_idx, a.feature, a.op, a.threshold, a.name))
    logger.info("Prefilter attributes: %d -> %d (window=[%.3f, %.3f])",
                m, new_inc.shape[1], min_support, max_support)
    return FormalContext(new_inc, new_attrs, context.mode, context.params)


# ------------------------------------------------------------------- mining
def _attribute_concept_intents(B: np.ndarray) -> list[frozenset[int]]:
    """Closure of each single attribute, derived from the co-occurrence matrix."""
    bf = B.astype(np.float32)
    co = bf.T @ bf  # [M, M], co[a, c] = |objects having both a and c|
    diag = np.diag(co)
    intents = []
    for a in range(B.shape[1]):
        if diag[a] == 0:
            continue
        intent = np.flatnonzero(co[a] >= diag[a] - 0.5)  # == support(a)
        intents.append(frozenset(intent.tolist()))
    return intents


def _object_concept_intents(B: np.ndarray, sample: int,
                            rng: np.random.Generator) -> list[frozenset[int]]:
    """Closure of a sample of object rows."""
    n = B.shape[0]
    idx = np.arange(n) if sample >= n else rng.choice(n, size=sample, replace=False)
    intents = []
    for o in idx:
        a = np.flatnonzero(B[o])
        if a.size == 0:
            continue
        extent = B[:, a].all(axis=1)
        intent = np.flatnonzero(B[extent].all(axis=0))
        intents.append(frozenset(intent.tolist()))
    return intents


def mine_concepts(
    context: FormalContext,
    strategy: str = "both",
    min_support: float = 0.0,
    max_support: float = 1.0,
    object_sample: int = 2000,
    backend: str = "fallback",
    seed: int = 0,
) -> list[Concept]:
    """Generate deduplicated closed concepts as a list of :class:`Concept`.

    ``strategy`` is one of {attribute, object, both}. Concepts are filtered to a
    support window ``[min_support, max_support]`` (fractions of N).
    """
    B = context.incidence
    n, m = B.shape
    if m == 0 or n == 0:
        return []

    if backend == "caspailleur":
        try:
            return _mine_caspailleur(context, min_support, max_support)
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("caspailleur backend failed (%s); using fallback.", exc)

    rng = np.random.default_rng(seed)
    intents: set[frozenset[int]] = set()
    if strategy in ("attribute", "both"):
        intents.update(_attribute_concept_intents(B))
    if strategy in ("object", "both"):
        intents.update(_object_concept_intents(B, object_sample, rng))

    lo = int(np.ceil(min_support * n))
    hi = int(np.floor(max_support * n))
    concepts: list[Concept] = []
    cid = 0
    for intent in intents:
        if not intent:
            continue
        cols = list(intent)
        extent = B[:, cols].all(axis=1)
        support = int(extent.sum())
        if support < max(lo, 1) or support > hi:
            continue
        concepts.append(Concept(
            concept_id=cid,
            intent=tuple(sorted(intent)),
            support=support,
            intent_size=len(intent),
            extent_size=support,
            coverage=support / n,
            attributes=[context.attributes[i].name for i in sorted(intent)],
        ))
        cid += 1
    logger.info("Mined %d unique concepts (strategy=%s, candidates=%d).",
                len(concepts), strategy, len(intents))
    return concepts


def _mine_caspailleur(context: FormalContext, min_support: float,
                      max_support: float) -> list[Concept]:  # pragma: no cover
    """Optional adapter for the `caspailleur` package (best-effort)."""
    import caspailleur as csp  # type: ignore

    B = context.incidence
    n = B.shape[0]
    # caspailleur exposes intents/extents over a boolean frame; APIs vary by
    # version, so we use the stable `mine_concepts`-style entry if present.
    intents_extents = csp.iter_concepts(B) if hasattr(csp, "iter_concepts") else None
    if intents_extents is None:
        raise NotImplementedError("Installed caspailleur lacks a known concept API.")
    concepts: list[Concept] = []
    lo, hi = int(np.ceil(min_support * n)), int(np.floor(max_support * n))
    for cid, (extent, intent) in enumerate(intents_extents):
        intent = tuple(sorted(int(i) for i in intent))
        support = len(extent)
        if not intent or support < max(lo, 1) or support > hi:
            continue
        concepts.append(Concept(cid, intent, support, len(intent), support,
                                coverage=support / n,
                                attributes=[context.attributes[i].name for i in intent]))
    return concepts
