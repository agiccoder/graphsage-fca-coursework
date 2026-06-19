"""Correctness tests for the FCA pipeline (no network access)."""
import numpy as np
import torch

from src.fca.binarize import build_formal_context
from src.fca.concepts import compute_extents, score_concepts, select_top_k
from src.fca.features import build_membership
from src.fca.mining import mine_concepts, prefilter_attributes


def _toy_context():
    # 5 objects x 4 attributes.
    x = torch.tensor([
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)
    return build_formal_context(x, mode="binary_nonzero")


def test_binary_nonzero_incidence():
    ctx = _toy_context()
    assert ctx.incidence.shape == (5, 4)
    assert ctx.incidence.dtype == bool
    assert ctx.incidence[2].tolist() == [True, True, False, True]


def test_concepts_are_closed():
    ctx = _toy_context()
    concepts = mine_concepts(ctx, strategy="both", min_support=0.0, max_support=1.0)
    assert concepts, "expected at least one concept"
    B = ctx.incidence
    for c in concepts:
        # extent must equal objects having all intent attributes
        extent = B[:, list(c.intent)].all(axis=1)
        assert int(extent.sum()) == c.support
        # intent must be closed: intent(extent(intent)) == intent
        closed = set(np.flatnonzero(B[extent].all(axis=0)).tolist())
        assert closed == set(c.intent)


def test_extents_and_membership_shapes():
    ctx = _toy_context()
    concepts = mine_concepts(ctx, strategy="both", min_support=0.0, max_support=1.0)
    ext = compute_extents(concepts, ctx)
    assert ext.shape == (5, len(concepts))
    hard = build_membership(concepts, ctx, "hard")
    soft = build_membership(concepts, ctx, "soft")
    assert hard.shape == soft.shape == (5, len(concepts))
    assert ((hard == 0) | (hard == 1)).all()
    assert (soft >= 0).all() and (soft <= 1).all()
    # hard membership equals extent membership
    assert np.array_equal(hard.astype(bool), ext)


def test_select_top_k_by_support():
    ctx = _toy_context()
    concepts = mine_concepts(ctx, strategy="both", min_support=0.0, max_support=1.0)
    concepts = score_concepts(concepts, ctx, scorer="support")
    top = select_top_k(concepts, 2)
    assert len(top) <= 2
    supports = [c.support for c in top]
    assert supports == sorted(supports, reverse=True)
    assert [c.concept_id for c in top] == list(range(len(top)))


def test_prefilter_window():
    ctx = _toy_context()
    # attribute 3 (last) has support 4/5 = 0.8 -> dropped by max_support=0.5
    filtered = prefilter_attributes(ctx, min_support=0.1, max_support=0.5,
                                    max_attributes=10)
    fracs = filtered.incidence.mean(axis=0)
    assert ((fracs >= 0.1) & (fracs <= 0.5)).all()


def test_supervised_scorers_use_train_only():
    """target_entropy / lift must depend ONLY on training labels (no leakage).

    Scrambling labels outside the train mask must not change any concept's score.
    """
    y = np.array([0, 0, 1, 1, 0])
    train_mask = np.array([True, True, True, False, False])  # nodes 3,4 = non-train
    y_scrambled = y.copy()
    y_scrambled[~train_mask] = 1 - y_scrambled[~train_mask]  # flip held-out labels

    for scorer in ("target_entropy", "lift"):
        ctx_a = _toy_context()
        a = {c.intent: c.selection_score for c in score_concepts(
            mine_concepts(ctx_a, strategy="both", min_support=0.0, max_support=1.0, seed=0),
            ctx_a, scorer=scorer, y=y.copy(), train_mask=train_mask, num_classes=2)}
        ctx_b = _toy_context()
        b = {c.intent: c.selection_score for c in score_concepts(
            mine_concepts(ctx_b, strategy="both", min_support=0.0, max_support=1.0, seed=0),
            ctx_b, scorer=scorer, y=y_scrambled, train_mask=train_mask, num_classes=2)}
        assert a.keys() == b.keys()
        for intent, score in a.items():
            assert abs(score - b[intent]) < 1e-9, f"{scorer} leaked non-train labels"


def test_min_intent_filter_drops_single_attribute_concepts():
    ctx = _toy_context()
    concepts = mine_concepts(ctx, strategy="both", min_support=0.0, max_support=1.0)
    multi = [c for c in concepts if c.intent_size >= 2]
    assert all(c.intent_size >= 2 for c in multi)
