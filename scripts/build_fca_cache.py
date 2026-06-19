"""Precompute FCA concepts + features for a config, without training.

    python scripts/build_fca_cache.py --config configs/experiments/cora_sage_fca.yaml

Writes artifacts/concepts/<dataset>_concepts.csv and
artifacts/features/<dataset>_x_fca.pt so feature engineering can be inspected
independently of model training.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from src.data import load_dataset
from src.fca import build_features
from src.utils.config import get, load_config
from src.utils.logging import get_logger
from src.utils.paths import DATASETS_DIR

logger = get_logger("build_fca_cache")


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute FCA features for a config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    ds_name = get(cfg, "dataset.name")
    data = load_dataset(ds_name, root=get(cfg, "dataset.root", str(DATASETS_DIR)),
                        planetoid_split=get(cfg, "dataset.planetoid_split", "public"),
                        split_idx=int(get(cfg, "dataset.split_idx", 0)),
                        to_undirected=get(cfg, "dataset.to_undirected", None))
    bundle = build_features(data, get(cfg, "features", {"variant": "raw"}),
                            seed=int(get(cfg, "features.seed", 0)), save_as=ds_name)
    logger.info("Saved FCA cache for %s | variant=%s | concepts=%d | coverage=%.3f",
                ds_name, bundle.variant, bundle.coverage.get("num_concepts", 0),
                bundle.coverage.get("node_coverage", 0.0))


if __name__ == "__main__":
    main()
