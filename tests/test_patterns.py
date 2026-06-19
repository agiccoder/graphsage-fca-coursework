import numpy as np
import torch

from src.fca.patterns import (
    build_pattern_membership,
    mine_interval_patterns,
    score_patterns,
    select_top_patterns,
)


def test_interval_patterns_build_membership_shape():
    x = torch.tensor([
        [1.0, 0.0, 0.2],
        [0.9, 0.0, 0.3],
        [0.0, 1.0, 0.8],
        [0.1, 0.8, 0.7],
    ])
    train_mask = torch.tensor([True, True, False, False])
    patterns, meta = mine_interval_patterns(
        x,
        train_mask=train_mask,
        params={"n_bins": 2, "intent_size": 1, "object_sample": 4, "max_features": 3, "min_support": 0.0, "max_support": 1.0},
        seed=0,
    )
    patterns = score_patterns(patterns, x, scorer="support")
    patterns = select_top_patterns(patterns, 2)
    membership = build_pattern_membership(patterns, x, mode="hard")
    assert membership.shape == (4, len(patterns))
    assert meta["pattern_num_features"] > 0


def test_target_entropy_uses_train_labels_only_smoke():
    x = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.0],
        [0.0, 1.0],
        [0.0, 0.8],
    ])
    train_mask = torch.tensor([True, True, False, False])
    y = np.array([0, 0, 1, 1])
    patterns, _ = mine_interval_patterns(
        x,
        train_mask=train_mask,
        params={"n_bins": 2, "intent_size": 1, "object_sample": 4, "max_features": 2, "min_support": 0.0, "max_support": 1.0},
        seed=1,
    )
    scored = score_patterns(
        patterns,
        x,
        scorer="target_entropy",
        y=y,
        train_mask=train_mask.numpy(),
        num_classes=2,
        params={"min_train_count": 1},
    )
    assert len(scored) == len(patterns)
    assert all(np.isfinite(p.selection_score) for p in scored)
