"""K-sweep for FCA features with K-matched SVD controls (Tasks B + C).

    python scripts/run_ablation_k.py --datasets cora citeseer pubmed \
                                     --ks 32 64 128 256 --seeds 0 1 2 3 4

For each (dataset, K) it runs:
    GraphSAGE + fca_feat (support, hard)   at K
    GraphSAGE + svd_control                at the same added dimension K
and optionally GraphSAGE + fca_group. Results accumulate in
results/per_seed_results.csv; aggregate by added_dim for the matched comparison.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from _ablation_common import CORE_DATASETS, base_config, run_variant
from src.utils.logging import get_logger

logger = get_logger("ablation.k")


def main() -> None:
    ap = argparse.ArgumentParser(description="FCA K-sweep + matched SVD controls.")
    ap.add_argument("--datasets", nargs="*", default=CORE_DATASETS)
    ap.add_argument("--ks", nargs="*", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--with-group", action="store_true", help="also run fca_group")
    ap.add_argument("--no-svd", action="store_true", help="skip matched SVD controls")
    args = ap.parse_args()

    for ds in args.datasets:
        fca_base = base_config(ds, "fca")
        svd_base = base_config(ds, "svd")
        for k in args.ks:
            run_variant(fca_base, f"{ds}_sage_fca_k{k}_support_hard",
                        {"features.variant": "fca_feat", "features.fca.k_concepts": k,
                         "features.fca.scorer": "support", "features.fca.membership": "hard",
                         "features.fca.min_intent_size": 1}, seeds=args.seeds)
            if args.with_group:
                run_variant(fca_base, f"{ds}_sage_fcagroup_k{k}",
                            {"features.variant": "fca_group", "features.fca.k_concepts": k,
                             "features.fca.num_groups": k}, seeds=args.seeds)
            if not args.no_svd:
                run_variant(svd_base, f"{ds}_sage_svd_k{k}",
                            {"features.variant": "svd_control",
                             "features.svd.n_components": k}, seeds=args.seeds)
    logger.info("K-sweep complete.")


if __name__ == "__main__":
    main()
