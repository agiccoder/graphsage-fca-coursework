"""Download datasets and write their JSON/Markdown summaries.

    python scripts/prepare_data.py                      # core datasets
    python scripts/prepare_data.py --datasets cora pubmed roman_empire
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from src.data import load_dataset, write_summary
from src.utils.logging import get_logger
from src.utils.paths import DATASETS_DIR, RESULTS_DIR

logger = get_logger("prepare_data")
CORE = ["cora", "citeseer", "pubmed"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + summarize datasets.")
    ap.add_argument("--datasets", nargs="*", default=CORE)
    ap.add_argument("--root", default=str(DATASETS_DIR))
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "dataset_summaries"
    for name in args.datasets:
        try:
            data = load_dataset(name, root=args.root)
            summary = write_summary(data, out_dir)
            logger.info("%s: %s", name, {k: summary[k] for k in
                        ("num_nodes", "num_edges", "num_features", "num_classes")})
        except Exception as exc:  # keep going on optional/missing datasets
            logger.warning("Could not load '%s': %s", name, exc)


if __name__ == "__main__":
    main()
