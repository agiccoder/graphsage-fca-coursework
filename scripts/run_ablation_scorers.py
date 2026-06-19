"""Concept-scorer ablation: support vs lift vs target_entropy (H4 / Task E).

    python scripts/run_ablation_scorers.py --datasets cora citeseer \
                                           --k 128 --scorers support lift target_entropy --seeds 0 1 2 3 4

Supervised scorers (lift, target_entropy) use TRAINING labels only; this is
enforced by tests/test_fca.py::test_supervised_scorers_use_train_only.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from _ablation_common import base_config, run_variant
from src.utils.logging import get_logger

logger = get_logger("ablation.scorers")


def main() -> None:
    ap = argparse.ArgumentParser(description="Concept-scorer ablation.")
    ap.add_argument("--datasets", nargs="*", default=["cora", "citeseer"])
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--scorers", nargs="*", default=["support", "lift", "target_entropy"])
    ap.add_argument("--membership", default="hard")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    for ds in args.datasets:
        fca_base = base_config(ds, "fca")
        for s in args.scorers:
            run_variant(fca_base, f"{ds}_sage_fca_k{args.k}_{s}_{args.membership}",
                        {"features.variant": "fca_feat", "features.fca.k_concepts": args.k,
                         "features.fca.scorer": s, "features.fca.membership": args.membership},
                        seeds=args.seeds)
    logger.info("Scorer ablation complete.")


if __name__ == "__main__":
    main()
