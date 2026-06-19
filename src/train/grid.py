"""Hyperparameter grid sweep over a base experiment config.

Usage
-----
    python -m src.train.grid --config configs/experiments/cora_sage_fca.yaml \
                             --grid configs/grids/default.yaml

The grid file is a mapping of dotted keys to value lists, e.g.::

    grid:
      model.hidden_channels: [64, 128]
      model.num_layers: [2, 3]
      train.lr: [0.001, 0.005]
      features.fca.k_concepts: [64, 128]

Every combination is merged onto the base config and run across all seeds.
Results accumulate in results/per_seed_results.csv (differentiated by columns).
"""
from __future__ import annotations

import argparse
import copy
import itertools

import yaml

from ..utils.config import get, load_config, set_in
from ..utils.io import save_json
from ..utils.logging import get_logger
from ..utils.paths import RUNS_DIR, resolve
from .run import experiment_name, run_experiment

logger = get_logger("train.grid")


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a {dotted_key: [values]} mapping."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, vals)) for vals in combos]


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid sweep for GraphSAGE+FCA.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_config(args.config)
    with resolve(args.grid).open("r", encoding="utf-8") as f:
        grid_spec = (yaml.safe_load(f) or {}).get("grid", {})
    combos = expand_grid(grid_spec)
    if args.max_runs:
        combos = combos[: args.max_runs]

    base_exp = experiment_name(base)
    manifest = []
    logger.info("Grid '%s': %d combinations x seeds.", base_exp, len(combos))
    if args.dry_run:
        for i, combo in enumerate(combos):
            logger.info("[%d] %s", i, combo)
        return

    for i, combo in enumerate(combos):
        cfg = copy.deepcopy(base)
        for key, val in combo.items():
            set_in(cfg, key, val)
        cfg["experiment"] = f"{base_exp}__g{i:03d}"
        manifest.append({"index": i, "experiment": cfg["experiment"], "combo": combo})
        logger.info("=== grid run %d/%d: %s ===", i + 1, len(combos), combo)
        run_experiment(cfg, seeds=args.seeds)

    save_json({"base": base_exp, "runs": manifest},
              RUNS_DIR / f"{base_exp}_grid" / "manifest.json")


if __name__ == "__main__":
    main()
