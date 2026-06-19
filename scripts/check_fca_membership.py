"""Hard-vs-soft membership sanity check (Task C / Problem 3).

    python scripts/check_fca_membership.py

For each core dataset this rebuilds the *selected* FCA concepts under the
canonical comparable configuration (support scorer, K=128, single-attribute
concepts allowed) and compares the **hard** and **soft** membership matrices.

Why this exists
---------------
``soft`` membership is the fraction of a concept's intent attributes a node
possesses. ``hard`` membership is 1 iff the node has *all* of them. For a concept
whose intent is a **single** attribute these are mathematically identical
(fraction over one item == that item's boolean value). Several datasets (notably
CiteSeer) select almost exclusively single-attribute concepts, so a "hard vs
soft" ablation there is comparing a config with itself — the byte-identical result
rows are expected, not a bug. This script makes that explicit and quantitative.

Outputs
-------
    results/membership_sanity.csv   one row per dataset with:
        n_concepts, n_single_attr, frac_single_attr,
        max_abs_diff, mean_abs_diff, frac_cols_identical, identical (bool),
        expected_identical (bool, == all concepts single-attribute)

A row is *consistent* when ``identical == expected_identical``; any mismatch is a
genuine bug in the membership construction and is logged as a warning.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.fca.features import build_membership
from src.fca.integrate import _build_concepts
from src.utils.config import get, load_config, set_in
from src.utils.io import save_dataframe
from src.utils.logging import get_logger
from src.utils.paths import CONFIGS_DIR, DATASETS_DIR, RESULTS_DIR

logger = get_logger("check.membership")

# Numerical tolerance: hard/soft are float32; treat sub-1e-6 gaps as identical.
ATOL = 1e-6
CORE_DATASETS = ["cora", "citeseer", "pubmed"]


def base_config(dataset: str, kind: str = "fca") -> dict:
    """Load ``{dataset}_sage_{kind}.yaml`` (or compose a minimal fallback).

    Inlined from ``scripts/_ablation_common`` so this diagnostic does NOT import
    the training loop (and therefore does not require ``torch_geometric`` just to
    read a config).
    """
    path = CONFIGS_DIR / "experiments" / f"{dataset}_sage_{kind}.yaml"
    if path.exists():
        return load_config(path)
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    set_in(cfg, "dataset.name", dataset)
    return cfg


def _load_data(cfg: dict):
    ds_name = get(cfg, "dataset.name")
    return load_dataset(
        ds_name,
        root=get(cfg, "dataset.root", str(DATASETS_DIR)),
        planetoid_split=get(cfg, "dataset.planetoid_split", "public"),
        split_idx=int(get(cfg, "dataset.split_idx", 0)),
        to_undirected=get(cfg, "dataset.to_undirected", None),
    )


def check_dataset(dataset: str, seed: int = 0) -> dict | None:
    cfg = base_config(dataset, "fca")
    fca_cfg = get(cfg, "features.fca", {})
    data = _load_data(cfg)
    context, concepts = _build_concepts(data, fca_cfg, seed)
    if not concepts:
        logger.warning("[%s] no concepts selected; skipping.", dataset)
        return None

    hard = build_membership(concepts, context, mode="hard")
    soft = build_membership(concepts, context, mode="soft")

    intent_sizes = np.array([c.intent_size for c in concepts])
    n_single = int((intent_sizes == 1).sum())
    diff = np.abs(hard - soft)
    # A column (concept) is identical if hard==soft for every node.
    col_identical = (diff.max(axis=0) <= ATOL) if diff.size else np.array([], bool)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0

    identical = max_abs <= ATOL
    expected_identical = bool((intent_sizes == 1).all())
    consistent = identical == expected_identical

    # Sanity invariant: every single-attribute concept MUST have hard==soft.
    single_cols = np.flatnonzero(intent_sizes == 1)
    single_ok = bool(np.all(col_identical[single_cols])) if single_cols.size else True
    if not single_ok:
        logger.error("[%s] BUG: single-attribute concept with hard != soft!", dataset)
    if not consistent:
        logger.warning("[%s] hard/soft identity (%s) != expected from intents (%s).",
                       dataset, identical, expected_identical)
    else:
        logger.info("[%s] consistent: identical=%s (single-attr %d/%d).",
                    dataset, identical, n_single, len(concepts))

    return {
        "dataset": dataset,
        "n_concepts": len(concepts),
        "n_single_attr": n_single,
        "frac_single_attr": round(n_single / len(concepts), 4),
        "mean_intent_size": round(float(intent_sizes.mean()), 4),
        "max_abs_diff": round(max_abs, 8),
        "mean_abs_diff": round(mean_abs, 8),
        "frac_cols_identical": round(float(col_identical.mean()), 4) if col_identical.size else 1.0,
        "identical": identical,
        "expected_identical": expected_identical,
        "single_attr_invariant_ok": single_ok,
        "consistent": consistent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard-vs-soft membership sanity check.")
    ap.add_argument("--datasets", nargs="*", default=CORE_DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        try:
            rec = check_dataset(ds, seed=args.seed)
        except Exception as exc:  # one bad dataset must not sink the rest
            logger.exception("[%s] failed: %s", ds, exc)
            continue
        if rec is not None:
            rows.append(rec)

    if not rows:
        logger.warning("No datasets produced membership results.")
        return
    df = pd.DataFrame(rows)
    save_dataframe(df, RESULTS_DIR / "membership_sanity.csv")
    logger.info("Wrote results/membership_sanity.csv (%d datasets).", len(df))
    if not bool(df["single_attr_invariant_ok"].all()):
        raise SystemExit("Membership invariant violated (see results/membership_sanity.csv).")


if __name__ == "__main__":
    main()
