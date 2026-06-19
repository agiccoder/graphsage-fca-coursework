"""Tests for aggregation identity, duplicate detection and report filters.

These are pure-pandas tests (no torch / torch_geometric) covering the Task A/B/D
fixes: binarisation is part of the config identity, duplicate runs are surfaced,
and the report's strict ``_filter`` selects only comparable rows.
"""
import numpy as np
import pandas as pd

from src.eval.aggregate import (KEY_COLS, aggregate, detect_duplicates)
from src.eval import report as R


def _seed_rows(experiment, n_seeds=5, binarize_mode="binary_nonzero", acc=0.80,
               membership="hard", scorer="support", k=128, min_intent=1,
               dataset="cora", model="graphsage", variant="fca_feat"):
    rows = []
    for s in range(n_seeds):
        rows.append({
            "experiment": experiment, "dataset": dataset, "model": model,
            "variant": variant, "seed": s, "added_dim": k, "hidden_channels": 128,
            "num_layers": 2, "dropout": 0.5, "aggr": "mean", "lr": 0.005,
            "weight_decay": 5e-4, "k_concepts": k, "scorer": scorer,
            "membership": membership, "min_intent_size": min_intent,
            "binarize_mode": binarize_mode, "binarize_params": "{}",
            "mean_intent_size": 1.0, "node_coverage": 0.99, "num_concepts": k,
            "test_accuracy": acc + 0.001 * s, "test_macro_f1": acc - 0.01,
            "val_accuracy": acc, "val_macro_f1": acc - 0.01,
            "low_degree_accuracy": acc - 0.05,
        })
    return rows


def test_binarize_mode_in_key_cols():
    assert "binarize_mode" in KEY_COLS
    assert "binarize_params" in KEY_COLS


def test_different_binarize_modes_are_distinct_configs():
    """Two FCA runs differing only in binarisation must NOT merge (Task B)."""
    df = pd.DataFrame(
        _seed_rows("soft_run", membership="soft", binarize_mode="binary_nonzero")
        + _seed_rows("quantile_run", membership="soft", binarize_mode="quantile_binarization")
    )
    main = aggregate(df)
    fca = main[main["variant"] == "fca_feat"]
    # Without the binarisation key these would collapse to ONE 10-seed row.
    assert len(fca) == 2, "binarisation modes must yield two separate configs"
    assert set(fca["n_seeds"]) == {5}


def test_detect_duplicates_flags_name_collision():
    """Same identity (ignoring binarize) under two names with repeated seeds."""
    # Force a collision by giving both the SAME binarize_mode.
    df = pd.DataFrame(
        _seed_rows("run_a", binarize_mode="binary_nonzero", membership="soft")
        + _seed_rows("run_b", binarize_mode="binary_nonzero", membership="soft")
    )
    dup = detect_duplicates(df)
    assert len(dup) == 1
    row = dup.iloc[0]
    assert "name_collision" in row["issue"]
    assert "repeated_seed" in row["issue"]
    assert row["n_unique_seeds"] == 5 and row["n_rows"] == 10


def test_detect_duplicates_clean_when_binarize_differs():
    df = pd.DataFrame(
        _seed_rows("soft_run", binarize_mode="binary_nonzero", membership="soft")
        + _seed_rows("quantile_run", binarize_mode="quantile_binarization", membership="soft")
    )
    dup = detect_duplicates(df)
    assert dup.empty, "distinct binarisations must not be flagged as duplicates"


def test_report_filter_excludes_non_graphsage_and_constraints():
    """report._filter must drop MLP rows and honour scalar/list constraints."""
    df = pd.DataFrame(
        _seed_rows("sage_support", model="graphsage", scorer="support")
        + _seed_rows("mlp_fca", model="mlp", scorer="support")
        + _seed_rows("sage_lift", model="graphsage", scorer="lift")
    )
    main = aggregate(df)
    # canonical filter: graphsage + support only
    sel = R._filter(main, "cora", **R.CANON_FCA)
    assert (sel["model"] == "graphsage").all()
    assert (sel["scorer"].astype(str) == "support").all()
    assert "mlp" not in set(sel["model"])
    assert len(sel) == 1  # only the graphsage/support config survives


def test_report_filter_numeric_min_intent():
    df = pd.DataFrame(
        _seed_rows("intent1", min_intent=1) + _seed_rows("intent2", min_intent=2)
    )
    main = aggregate(df)
    only1 = R._filter(main, "cora", model=R.GRAPHSAGE_MODELS, variant="fca_feat",
                      min_intent_size=1)
    assert len(only1) == 1
    assert R._num(only1.iloc[0]["min_intent_size"]) == 1.0
