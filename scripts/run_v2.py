"""v2 master battery: clean main + K/SVD sweep + mechanism ablations + analyses.

    python scripts/run_v2.py --fresh                 # full study, 5 seeds
    python scripts/run_v2.py --quick                 # fast smoke (1 seed, cora+citeseer)

Stages are designed to be DISJOINT: no (config, seed) pair is ever run twice, so
the per-seed aggregation stays correct even though tables span multiple stages.

  Stage 1  baselines & variant configs (logreg/mlp/gcn/sage-raw/pool/group/intent2)
  Stage 2  FCA_FEAT(support,hard) K-sweep + K-matched SVD controls   (K = 32..256)
  Stage 3  mechanism: membership=soft, scorer in {lift,target_entropy}, PubMed diag
  Stage 4  aggregate -> report -> figures -> degree buckets -> concept stats
  Stage 5  multi-intent (min_intent_size=2) K-sweep across datasets (H6 / Task D)

Stage 5 is disjoint from stage 2 because every stage-2 FCA run pins
``min_intent_size=1`` whereas stage 5 pins ``min_intent_size=2`` — different config
identities, so they never collapse into one seed group. Multi-attribute concepts
are scarce, so stage 5 sweeps deliberately *small* K values.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import shutil

from _ablation_common import base_config, run_variant
import make_figures
import analyze_degree_buckets
import analyze_concepts

from src.eval.aggregate import run as aggregate_run
from src.eval.report import build_report
from src.train.run import run_experiment
from src.utils.config import load_config
from src.utils.io import save_text
from src.utils.logging import get_logger
from src.utils.paths import CONFIGS_DIR, REPORTS_DIR, RESULTS_DIR

logger = get_logger("run_v2")

# Stage 1: baseline + distinct-variant configs (NO fca_feat/svd at K — those are stage 2).
STAGE1 = {
    "cora": ["cora_logreg_raw", "cora_mlp_raw", "cora_mlp_fca", "cora_gcn_raw",
             "cora_sage_raw", "cora_sage_pool_raw", "cora_sage_fca_group",
             "cora_sage_fca_intent2"],
    "citeseer": ["citeseer_mlp_raw", "citeseer_sage_raw"],
    "pubmed": ["pubmed_mlp_raw", "pubmed_sage_raw",
               "pubmed_sage_fca_soft", "pubmed_sage_fca_quantile"],
}


def archive_old_results() -> None:
    dst = RESULTS_DIR / "_archive_v1"
    dst.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in ("per_seed_results.csv", "main_results.csv", "deltas.csv",
                 "ranking_by_dataset.csv"):
        src = RESULTS_DIR / name
        if src.exists():
            shutil.move(str(src), str(dst / name))
            moved.append(name)
    logger.info("Archived old results -> %s (%s)", dst, ", ".join(moved) or "nothing")


def _safe_experiment(label: str, fn) -> None:
    """Run one experiment; log and continue on failure so one bad config does
    not abort the whole multi-hour battery."""
    try:
        fn()
    except Exception:  # noqa: BLE001 - intentional: isolate per-experiment failures
        logger.exception("experiment '%s' FAILED; continuing with the rest.", label)


def run_stage1(datasets, seeds) -> None:
    for ds in datasets:
        for stem in STAGE1.get(ds, []):
            path = CONFIGS_DIR / "experiments" / f"{stem}.yaml"
            if not path.exists():
                logger.warning("missing config %s; skipping", stem)
                continue
            logger.info("######## stage1: %s ########", stem)
            _safe_experiment(stem, lambda p=path: run_experiment(load_config(p), seeds=seeds))


def run_stage2(datasets, ks, seeds) -> None:
    for ds in datasets:
        fca_base = base_config(ds, "fca")
        svd_base = base_config(ds, "svd")
        for k in ks:
            _safe_experiment(f"{ds}_fca_k{k}", lambda b=fca_base, k=k: run_variant(
                b, f"{ds}_sage_fca_k{k}_support_hard",
                {"features.variant": "fca_feat", "features.fca.k_concepts": k,
                 "features.fca.scorer": "support", "features.fca.membership": "hard",
                 "features.fca.min_intent_size": 1}, seeds=seeds))
            _safe_experiment(f"{ds}_svd_k{k}", lambda b=svd_base, k=k: run_variant(
                b, f"{ds}_sage_svd_k{k}",
                {"features.variant": "svd_control", "features.svd.n_components": k}, seeds=seeds))


def run_stage3(datasets, k, seeds, scorer_datasets) -> None:
    # membership=soft (hard already covered in stage 2); skip pubmed (diag config covers it).
    for ds in datasets:
        if ds == "pubmed":
            continue
        _safe_experiment(f"{ds}_soft", lambda ds=ds: run_variant(
            base_config(ds, "fca"), f"{ds}_sage_fca_k{k}_support_soft",
            {"features.variant": "fca_feat", "features.fca.k_concepts": k,
             "features.fca.scorer": "support", "features.fca.membership": "soft"}, seeds=seeds))
    # supervised scorers (support already covered in stage 2)
    for ds in scorer_datasets:
        for s in ("lift", "target_entropy"):
            _safe_experiment(f"{ds}_{s}", lambda ds=ds, s=s: run_variant(
                base_config(ds, "fca"), f"{ds}_sage_fca_k{k}_{s}_hard",
                {"features.variant": "fca_feat", "features.fca.k_concepts": k,
                 "features.fca.scorer": s, "features.fca.membership": "hard"}, seeds=seeds))


def run_stage5(datasets, ks, seeds) -> None:
    """Multi-intent (min_intent_size=2) FCA_FEAT K-sweep (H6 / Task D).

    Same canonical axes as stage 2 (GraphSAGE, support, hard) EXCEPT the intent
    filter is >=2, so concepts are genuinely conceptual (multi-attribute) rather
    than single-feature selectors. Disjoint from stage 2 by min_intent_size."""
    for ds in datasets:
        fca_base = base_config(ds, "fca")
        for k in ks:
            _safe_experiment(f"{ds}_fca_intent2_k{k}", lambda b=fca_base, k=k, ds=ds: run_variant(
                b, f"{ds}_sage_fca_intent2_k{k}",
                {"features.variant": "fca_feat", "features.fca.k_concepts": k,
                 "features.fca.scorer": "support", "features.fca.membership": "hard",
                 "features.fca.min_intent_size": 2}, seeds=seeds))


def run_analyses() -> None:
    aggregate_run()
    save_text(build_report(), REPORTS_DIR / "experiment_summary.md")
    make_figures.main()
    analyze_degree_buckets.main()
    analyze_concepts.main()
    logger.info("Analyses complete. See results/, reports/, figures/.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full v2 study.")
    ap.add_argument("--datasets", nargs="*", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--ks", nargs="*", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--ks-intent2", nargs="*", type=int, default=[12, 32, 64],
                    help="K values for the multi-intent (intent>=2) sweep; kept "
                         "small (multi-attribute concepts are scarce) and away from "
                         "K=128 which the cora_sage_fca_intent2 stage-1 config covers.")
    ap.add_argument("--k-main", type=int, default=128, help="K for mechanism ablations")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--fresh", action="store_true", help="archive old results first")
    ap.add_argument("--quick", action="store_true", help="fast smoke configuration")
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    if args.quick:
        args.datasets = ["cora", "citeseer"]
        args.ks = [32, 128]
        args.ks_intent2 = [12, 32]
        args.seeds = [0]

    if args.fresh:
        archive_old_results()

    scorer_ds = [d for d in args.datasets if d in ("cora", "citeseer")]
    if 1 in args.stages:
        run_stage1(args.datasets, args.seeds)
    if 2 in args.stages:
        run_stage2(args.datasets, args.ks, args.seeds)
    if 3 in args.stages:
        run_stage3(args.datasets, args.k_main, args.seeds, scorer_ds)
    if 5 in args.stages:
        run_stage5(args.datasets, args.ks_intent2, args.seeds)
    # Stage 4 (analyses) always runs last so it sees every produced seed row.
    if 4 in args.stages:
        run_analyses()


if __name__ == "__main__":
    main()
