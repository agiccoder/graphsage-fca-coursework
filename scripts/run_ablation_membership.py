"""Hard vs soft concept membership ablation (H5 / Task D).

    python scripts/run_ablation_membership.py --datasets cora citeseer pubmed \
                                              --k 128 --memberships hard soft --seeds 0 1 2 3 4
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from _ablation_common import CORE_DATASETS, base_config, run_variant
from src.utils.logging import get_logger

logger = get_logger("ablation.membership")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard vs soft membership ablation.")
    ap.add_argument("--datasets", nargs="*", default=CORE_DATASETS)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--memberships", nargs="*", default=["hard", "soft"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    for ds in args.datasets:
        fca_base = base_config(ds, "fca")
        for m in args.memberships:
            run_variant(fca_base, f"{ds}_sage_fca_k{args.k}_support_{m}",
                        {"features.variant": "fca_feat", "features.fca.k_concepts": args.k,
                         "features.fca.scorer": "support", "features.fca.membership": m},
                        seeds=args.seeds)
    logger.info("Membership ablation complete.")


if __name__ == "__main__":
    main()
