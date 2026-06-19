"""Run the core experiment battery, then aggregate + report + figures.

    python scripts/run_core.py                      # full core battery, all seeds
    python scripts/run_core.py --seeds 0 1 2 --set train.epochs=150

This is a convenience wrapper around `python -m src.train.run` for each core
config; every experiment is still independently reproducible from its YAML.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import make_figures  # scripts/make_figures.py

from src.eval.aggregate import run as aggregate_run
from src.eval.report import build_report
from src.train.run import run_experiment
from src.utils.config import load_config
from src.utils.io import save_text
from src.utils.logging import get_logger
from src.utils.paths import CONFIGS_DIR, REPORTS_DIR

logger = get_logger("run_core")

CORE_EXPERIMENTS = [
    # Cora: full baseline + FCA + controls
    "cora_logreg_raw", "cora_mlp_raw", "cora_mlp_fca", "cora_gcn_raw",
    "cora_sage_raw", "cora_sage_pool_raw", "cora_sage_fca", "cora_sage_fca_group",
    "cora_sage_svd",
    # CiteSeer
    "citeseer_mlp_raw", "citeseer_sage_raw", "citeseer_sage_fca", "citeseer_sage_svd",
    # PubMed
    "pubmed_mlp_raw", "pubmed_sage_raw", "pubmed_sage_fca", "pubmed_sage_svd",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the core battery + analysis.")
    ap.add_argument("--experiments", nargs="*", default=CORE_EXPERIMENTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    for stem in args.experiments:
        path = CONFIGS_DIR / "experiments" / f"{stem}.yaml"
        logger.info("######## %s ########", stem)
        cfg = load_config(path, overrides=args.overrides)
        run_experiment(cfg, seeds=args.seeds)

    aggregate_run()
    save_text(build_report(), REPORTS_DIR / "experiment_summary.md")
    if not args.skip_figures:
        make_figures.main()
    logger.info("Core battery complete. See results/ , reports/ and figures/.")


if __name__ == "__main__":
    main()
