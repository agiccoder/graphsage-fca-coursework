"""Shared helpers for ablation runner scripts.

Each runner loads a per-dataset *base* YAML (which holds all fixed
hyperparameters) and overrides only the single swept axis (K / membership /
scorer / SVD dim). The swept axis is the loop variable; everything else stays in
YAML, so runs remain reproducible and are not hardcoded.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import copy

from src.train.run import run_experiment
from src.utils.config import load_config, set_in
from src.utils.logging import get_logger
from src.utils.paths import CONFIGS_DIR

logger = get_logger("ablation")

CORE_DATASETS = ["cora", "citeseer", "pubmed"]


def base_config(dataset: str, kind: str) -> dict:
    """Load ``{dataset}_sage_{kind}.yaml`` (kind in {raw, fca, svd})."""
    path = CONFIGS_DIR / "experiments" / f"{dataset}_sage_{kind}.yaml"
    if path.exists():
        return load_config(path)
    # Fallback: compose from base + dataset partial.
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    set_in(cfg, "dataset.name", dataset)
    return cfg


def run_variant(base: dict, experiment: str, overrides: dict,
                seeds=None, save_models: bool = False) -> None:
    """Deep-copy ``base``, apply overrides, name it and run across seeds."""
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        set_in(cfg, key, val)
    cfg["experiment"] = experiment
    logger.info("--- %s | %s ---", experiment, overrides)
    run_experiment(cfg, seeds=seeds, save_models=save_models)
