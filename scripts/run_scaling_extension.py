"""Phase 2a — compact downstream training for the promising richer-scaling modes.

This runs a *small* GraphSAGE grid for the scaling modes that passed the Phase 1
structural diagnostics (see ``reports/scaling_diagnostics_review.md``), then writes
``reports/scaling_extension_summary.md`` comparing each richer-FCA config against
its three honest references:

    * raw GraphSAGE                          (same dataset)
    * binary_nonzero FCA at the same K       (same dataset, K, scorer, membership)
    * K-matched SVD control                  (same dataset, added_dim == K)

Phase 2a grid (TZ §7.2, compact):
    K = 128 only
    scorer in {support, target_entropy}
    membership = hard
    seeds = 0..4
    selected modes:
        cora      quantile_global  q=0.90
        citeseer  quantile_global  q=0.90
        pubmed    quantile_global  q=0.75
        pubmed    topk_per_node    topk=10
    => 4 dataset-mode combos x 2 scorers = 8 configs x 5 seeds = 40 runs.

Usage
-----
    # full Phase-2a training (needs torch_geometric):
    python scripts/run_scaling_extension.py --seeds 0 1 2 3 4

    # only (re)generate the report from existing results:
    python scripts/run_scaling_extension.py --report-only

Guardrails
----------
* Existing v2 rows are **never overwritten**: results are *appended* to
  ``results/per_seed_results.csv`` and each run carries an explicit experiment name
  prefixed ``scaling_ext__`` so it is trivially filterable and never collides with
  v2 experiment identities.
* The binarisation identity (``binarize_mode`` + ``binarize_params``) is recorded
  per row by ``src.train.run._row``, so aggregation already keeps richer-scaling
  runs distinct from ``binary_nonzero`` runs (no inflated seed groups).
* Success is **not** claimed unless a richer-FCA config beats the K-matched SVD
  control OR clearly improves on binary_nonzero FCA at the same K.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.io import save_text
from src.utils.logging import get_logger
from src.utils.paths import REPORTS_DIR, RESULTS_DIR

logger = get_logger("scaling.ext")

CORE_DATASETS = ["cora", "citeseer", "pubmed"]
PER_SEED = RESULTS_DIR / "per_seed_results.csv"
REPORT_PATH = REPORTS_DIR / "scaling_extension_summary.md"

# Phase-2a selection (mode + params) that survived Phase 1 structural diagnostics.
# Each entry: (dataset, binarize_mode, params, short_tag).
SELECTED: list[tuple[str, str, dict, str]] = [
    ("cora",     "quantile_global", {"quantile": 0.90}, "qglobal_q090"),
    ("citeseer", "quantile_global", {"quantile": 0.90}, "qglobal_q090"),
    ("pubmed",   "quantile_global", {"quantile": 0.75}, "qglobal_q075"),
    ("pubmed",   "topk_per_node",   {"topk": 10},       "topk_t10"),
]
SCORERS = ["support", "target_entropy"]
K = 128
MEMBERSHIP = "hard"
EXP_PREFIX = "scaling_ext__"

# ---- Phase 2b: narrow K-sweep around the modes that showed signal in Phase 2a.
# Cora is deliberately excluded (both scorers were worse than SVD *and* binary in
# Phase 2a, see reports/scaling_phase2b_decision.md). Each entry mirrors SELECTED.
PHASE2B_SELECTED: list[tuple[str, str, dict, str]] = [
    ("citeseer", "quantile_global", {"quantile": 0.90}, "qglobal_q090"),
    ("pubmed",   "quantile_global", {"quantile": 0.75}, "qglobal_q075"),
    ("pubmed",   "topk_per_node",   {"topk": 10},       "topk_t10"),
]
PHASE2B_KS = [64, 256]
PHASE2B_REPORT_PATH = REPORTS_DIR / "scaling_phase2b_summary.md"

# ---- Phase 3 (optional, §11): graph-smoothed top-k on PubMed + Cora.
# Justified because PubMed still trails SVD but topk_per_node shows partial gains.
PHASE3_SELECTED: list[tuple[str, str, dict, str]] = [
    ("pubmed", "graph_smoothed_topk", {"smooth_alpha": 0.5, "topk": 10}, "gsmooth_t10"),
    ("cora",   "graph_smoothed_topk", {"smooth_alpha": 0.5, "topk": 10}, "gsmooth_t10"),
]
PHASE3_SCORERS = ["support"]
PHASE3_K = 128
PHASE3_REPORT_PATH = REPORTS_DIR / "scaling_phase3_summary.md"

# Tolerance below which a delta is treated as "tie" rather than win/loss.
TIE_EPS = 0.002  # 0.2 accuracy points


# --------------------------------------------------------------------------- run
def _run_one(base_config, run_variant, ds, mode, params, tag, k, scorer, seeds):
    """Train a single richer-FCA config at one (k, scorer)."""
    fca_base = base_config(ds, "fca")
    exp = f"{EXP_PREFIX}{ds}_{tag}_k{k}_{scorer}_{MEMBERSHIP}"
    overrides = {
        "features.variant": "fca_feat",
        "features.fca.k_concepts": k,
        "features.fca.scorer": scorer,
        "features.fca.membership": MEMBERSHIP,
        "features.fca.min_intent_size": 1,
        "features.fca.binarize_mode": mode,
        "features.fca.binarize_params": dict(params),
    }
    logger.info("RUN %s | mode=%s params=%s scorer=%s k=%d", exp, mode, params, scorer, k)
    run_variant(fca_base, exp, overrides, seeds=seeds)


def _run_training(seeds: list[int] | None) -> None:
    """Execute the compact Phase-2a grid. Imports the training stack lazily so the
    report-only path works on hosts without ``torch_geometric``."""
    from _ablation_common import base_config, run_variant

    for ds, mode, params, tag in SELECTED:
        for scorer in SCORERS:
            _run_one(base_config, run_variant, ds, mode, params, tag, K, scorer, seeds)
    logger.info("Phase 2a training complete (%d configs).", len(SELECTED) * len(SCORERS))


def _run_training_phase2b(seeds: list[int] | None,
                          datasets: list[str] | None,
                          ks: list[int] | None) -> None:
    """Execute the Phase-2b K-sweep around the promising modes (see
    ``reports/scaling_phase2b_decision.md``). Lazily imports the training stack."""
    from _ablation_common import base_config, run_variant

    ks = ks or PHASE2B_KS
    sel = [s for s in PHASE2B_SELECTED if (datasets is None or s[0] in datasets)]
    n = 0
    for ds, mode, params, tag in sel:
        for k in ks:
            for scorer in SCORERS:
                _run_one(base_config, run_variant, ds, mode, params, tag, k, scorer, seeds)
                n += 1
    logger.info("Phase 2b training complete (%d configs, ks=%s).", n, ks)


def _run_training_phase3(seeds: list[int] | None,
                         datasets: list[str] | None) -> None:
    """Execute the optional Phase-3 graph-smoothed top-k grid (§11)."""
    from _ablation_common import base_config, run_variant

    sel = [s for s in PHASE3_SELECTED if (datasets is None or s[0] in datasets)]
    n = 0
    for ds, mode, params, tag in sel:
        for scorer in PHASE3_SCORERS:
            _run_one(base_config, run_variant, ds, mode, params, tag,
                     PHASE3_K, scorer, seeds)
            n += 1
    logger.info("Phase 3 training complete (%d configs).", n)


# ------------------------------------------------------------------------ report
def _load() -> pd.DataFrame:
    if not PER_SEED.exists():
        raise FileNotFoundError(f"{PER_SEED} not found; run training first.")
    df = pd.read_csv(PER_SEED)
    # Old rows predate the binarize_mode column -> treat missing as binary_nonzero.
    if "binarize_mode" not in df.columns:
        df["binarize_mode"] = np.nan
    df["binarize_mode"] = df["binarize_mode"].fillna("binary_nonzero")
    if "min_intent_size" not in df.columns:
        df["min_intent_size"] = np.nan
    return df


def _agg(sub: pd.DataFrame) -> tuple[float, float, int]:
    """(mean, std(ddof=0), n) of test_accuracy for a per-seed slice."""
    if sub.empty:
        return float("nan"), float("nan"), 0
    acc = sub["test_accuracy"].astype(float)
    return float(acc.mean()), float(acc.std(ddof=0)), int(len(sub))


def _is_sage(df: pd.DataFrame) -> pd.Series:
    return df["model"].isin(["graphsage", "sage"])


def _richer_row(df: pd.DataFrame, ds: str, mode: str, scorer: str,
                k: int = K) -> pd.DataFrame:
    m = (
        (df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
        & (df["k_concepts"] == k) & (df["scorer"] == scorer)
        & (df["membership"] == MEMBERSHIP) & (df["binarize_mode"] == mode)
    )
    return df[m]


def _binary_ref(df: pd.DataFrame, ds: str, scorer: str, k: int = K) -> pd.DataFrame:
    """binary_nonzero FCA at the same K/scorer/membership; min_intent_size in {1, na}."""
    mii = df["min_intent_size"].isna() | (df["min_intent_size"] == 1)
    m = (
        (df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
        & (df["k_concepts"] == k) & (df["scorer"] == scorer)
        & (df["membership"] == MEMBERSHIP)
        & (df["binarize_mode"] == "binary_nonzero") & mii
    )
    return df[m]


def _raw_ref(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    m = (df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "raw")
    if "aggr" in df.columns:
        pref = df[m & (df["aggr"].astype(str) == "mean")]
        if not pref.empty:
            return pref
    return df[m]


def _svd_ref(df: pd.DataFrame, ds: str, k: int = K) -> pd.DataFrame:
    m = ((df["dataset"] == ds) & (df["variant"] == "svd_control")
         & (np.round(pd.to_numeric(df["added_dim"], errors="coerce")) == k))
    return df[m]


def _verdict(d_svd: float, d_bin: float) -> str:
    """Classify a richer-FCA config against SVD and binary_nonzero (acc deltas)."""
    beats_svd = not np.isnan(d_svd) and d_svd > TIE_EPS
    beats_bin = not np.isnan(d_bin) and d_bin > TIE_EPS
    if beats_svd and beats_bin:
        return "SUCCESS (beats SVD & binary_nonzero)"
    if beats_svd:
        return "SUCCESS (beats K-matched SVD)"
    if beats_bin:
        return "PARTIAL (improves binary_nonzero, not SVD)"
    if (not np.isnan(d_svd) and d_svd < -TIE_EPS) or (not np.isnan(d_bin) and d_bin < -TIE_EPS):
        return "NO (worse)"
    return "NEUTRAL (tie)"


def _fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "—"
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def _fmt_d(d: float) -> str:
    if np.isnan(d):
        return "—"
    return f"{d:+.4f}"


def build_report() -> str:
    df = _load()
    lines: list[str] = []
    lines.append("# Richer FCA Scaling — Phase 2a downstream results\n")
    lines.append(
        "_Compact GraphSAGE check (K=128, membership=hard, seeds 0–4) for the "
        "scaling modes that passed Phase 1 structural diagnostics "
        "([`scaling_diagnostics_review.md`](scaling_diagnostics_review.md)). Each "
        "richer-FCA config is compared against raw GraphSAGE, **binary_nonzero FCA "
        "at the same K/scorer**, and the **K-matched SVD control** "
        "(added_dim=128). Success is claimed only when a config beats K-matched SVD "
        "or clearly improves binary_nonzero FCA (tie band ±"
        f"{TIE_EPS:.3f} accuracy). Source: "
        "[`results/per_seed_results.csv`](../results/per_seed_results.csv)._\n"
    )

    any_rows = False
    summary_rows: list[dict] = []

    for ds, mode, params, tag in SELECTED:
        raw_m, raw_s, raw_n = _agg(_raw_ref(df, ds))
        svd_m, svd_s, svd_n = _agg(_svd_ref(df, ds))
        pstr = json.dumps(params, sort_keys=True)
        lines.append(f"\n## {ds} — `{mode}` {pstr}\n")
        lines.append("| scorer | richer FCA | binary_nonzero FCA | K-SVD (128) | raw SAGE | Δ vs SVD | Δ vs binary | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for scorer in SCORERS:
            r_m, r_s, r_n = _agg(_richer_row(df, ds, mode, scorer))
            b_m, b_s, b_n = _agg(_binary_ref(df, ds, scorer))
            d_svd = r_m - svd_m if (r_n and svd_n) else float("nan")
            d_bin = r_m - b_m if (r_n and b_n) else float("nan")
            verdict = _verdict(d_svd, d_bin) if r_n else "NOT RUN"
            if r_n:
                any_rows = True
            lines.append(
                f"| {scorer} | {_fmt(r_m, r_s, r_n)} | {_fmt(b_m, b_s, b_n)} | "
                f"{_fmt(svd_m, svd_s, svd_n)} | {_fmt(raw_m, raw_s, raw_n)} | "
                f"{_fmt_d(d_svd)} | {_fmt_d(d_bin)} | {verdict} |"
            )
            summary_rows.append({
                "dataset": ds, "mode": mode, "params": pstr, "scorer": scorer,
                "richer_acc": r_m, "binary_acc": b_m, "svd_acc": svd_m,
                "raw_acc": raw_m, "d_vs_svd": d_svd, "d_vs_binary": d_bin,
                "verdict": verdict, "n": r_n,
            })

    # ----- per-dataset interpretation -----
    lines.append("\n## Per-dataset interpretation\n")
    if not any_rows:
        lines.append(
            "> **No Phase-2a training rows found yet.** The grid has not been "
            "executed on this host (training requires `torch_geometric`, which is "
            "absent). Run `python scripts/run_scaling_extension.py --seeds 0 1 2 3 4` "
            "on a torch_geometric-enabled host, then re-run with `--report-only` to "
            "populate this section. Reference rows shown above are the existing v2 "
            "raw / SVD / binary_nonzero results.\n"
        )
    else:
        sdf = pd.DataFrame([r for r in summary_rows if r["n"]])
        for ds in sdf["dataset"].unique():
            d = sdf[sdf["dataset"] == ds]
            best = d.loc[d["richer_acc"].idxmax()]
            lines.append(
                f"- **{ds}**: best richer-FCA = `{best['mode']}` {best['params']} "
                f"/ {best['scorer']} at {best['richer_acc']:.4f} "
                f"(Δ vs SVD {_fmt_d(best['d_vs_svd'])}, Δ vs binary "
                f"{_fmt_d(best['d_vs_binary'])}) → {best['verdict']}."
            )

    # ----- overall honest verdict -----
    lines.append("\n## Overall verdict (Phase 2a)\n")
    if not any_rows:
        lines.append(
            "Pending execution. **No success is claimed.** Per the reporting rules, "
            "the v2 conclusion (FCA does not robustly beat K-matched SVD) stands "
            "until Phase-2a numbers exist.\n"
        )
    else:
        sdf = pd.DataFrame([r for r in summary_rows if r["n"]])
        n_succ = int(sdf["verdict"].str.startswith("SUCCESS").sum())
        n_part = int(sdf["verdict"].str.startswith("PARTIAL").sum())
        if n_succ:
            lines.append(
                f"**{n_succ}/{len(sdf)} configs beat the K-matched SVD control.** "
                "Richer conceptual scaling improves downstream GraphSAGE on at least "
                "one dataset — this is the first evidence that the structural gains "
                "from Phase 1 translate into accuracy. Phase 2b (K∈{64,256}) is "
                "warranted for the winning dataset/mode(s).\n"
            )
        elif n_part:
            lines.append(
                f"**No config beats K-matched SVD**, but {n_part}/{len(sdf)} improve "
                "on binary_nonzero FCA at the same K. Richer scaling helps *within* "
                "the FCA family but does not yet overturn the SVD baseline. Phase 2b "
                "is optional; prioritise the dataset with the largest binary→richer "
                "gain.\n"
            )
        else:
            lines.append(
                "**No config beats either K-matched SVD or binary_nonzero FCA.** "
                "Despite curing the structural degeneracy (Phase 1), richer scaling "
                "does not improve downstream accuracy here. The v2 conclusion stands; "
                "Phase 2b is **not** recommended.\n"
            )

    return "\n".join(lines) + "\n"


def build_report_phase2b() -> str:
    """Phase 2b K-sweep summary: each promising mode at K in {64, 128, 256}."""
    df = _load()
    ks = sorted({K, *PHASE2B_KS})
    lines: list[str] = []
    lines.append("# Richer FCA Scaling — Phase 2b K-sweep results\n")
    lines.append(
        "_Narrow K-sensitivity check around the modes that showed signal in "
        "Phase 2a (see [`scaling_phase2b_decision.md`](scaling_phase2b_decision.md)). "
        "K128 rows are the Phase 2a results, shown here for continuity. Each richer-"
        "FCA config is compared against the **K-matched SVD control** and "
        "`binary_nonzero` FCA at the same K/scorer. Cora is excluded (no Phase 2a "
        f"signal). Tie band ±{TIE_EPS:.3f} accuracy. Source: "
        "[`results/per_seed_results.csv`](../results/per_seed_results.csv)._\n"
    )

    any_rows = False
    best_per_combo: list[dict] = []

    for ds, mode, params, tag in PHASE2B_SELECTED:
        pstr = json.dumps(params, sort_keys=True)
        raw_m, raw_s, raw_n = _agg(_raw_ref(df, ds))
        lines.append(f"\n## {ds} — `{mode}` {pstr}\n")
        lines.append(
            f"_raw SAGE = {_fmt(raw_m, raw_s, raw_n)}_\n"
        )
        lines.append("| K | scorer | richer FCA | binary_nonzero FCA | K-SVD | Δ vs SVD | Δ vs binary | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k in ks:
            svd_m, svd_s, svd_n = _agg(_svd_ref(df, ds, k))
            for scorer in SCORERS:
                r_m, r_s, r_n = _agg(_richer_row(df, ds, mode, scorer, k))
                b_m, b_s, b_n = _agg(_binary_ref(df, ds, scorer, k))
                d_svd = r_m - svd_m if (r_n and svd_n) else float("nan")
                d_bin = r_m - b_m if (r_n and b_n) else float("nan")
                verdict = _verdict(d_svd, d_bin) if r_n else "NOT RUN"
                if r_n:
                    any_rows = True
                    best_per_combo.append({
                        "dataset": ds, "mode": mode, "params": pstr, "k": k,
                        "scorer": scorer, "richer_acc": r_m, "svd_acc": svd_m,
                        "binary_acc": b_m, "d_vs_svd": d_svd, "d_vs_binary": d_bin,
                        "verdict": verdict,
                    })
                lines.append(
                    f"| {k} | {scorer} | {_fmt(r_m, r_s, r_n)} | {_fmt(b_m, b_s, b_n)} | "
                    f"{_fmt(svd_m, svd_s, svd_n)} | {_fmt_d(d_svd)} | {_fmt_d(d_bin)} | {verdict} |"
                )

    lines.append("\n## Phase 2b verdict\n")
    if not any_rows:
        lines.append(
            "> **No Phase-2b rows found.** Run "
            "`python scripts/run_scaling_extension.py --phase2b --seeds 0 1 2 3 4` "
            "on a torch_geometric-enabled host, then re-run with "
            "`--phase2b --report-only`.\n"
        )
    else:
        bdf = pd.DataFrame(best_per_combo)
        n_succ = int(bdf["verdict"].str.startswith("SUCCESS").sum())
        n_part = int(bdf["verdict"].str.startswith("PARTIAL").sum())
        b = best_per_combo[int(bdf["d_vs_svd"].astype(float).idxmax())]
        lines.append(
            f"- Best richer config across the sweep: **{b['dataset']} "
            f"`{b['mode']}` {b['params']} / {b['scorer']} K{int(b['k'])}** "
            f"at {float(b['richer_acc']):.4f} (Δ vs SVD {_fmt_d(float(b['d_vs_svd']))}, "
            f"Δ vs binary {_fmt_d(float(b['d_vs_binary']))})."
        )
        lines.append(
            f"- Across the K-sweep, **{n_succ}** config(s) beat K-matched SVD and "
            f"**{n_part}** improve binary_nonzero without beating SVD.\n"
        )
    return "\n".join(lines) + "\n"


def build_report_phase3() -> str:
    """Optional Phase 3 graph-smoothed top-k summary (§11)."""
    df = _load()
    lines: list[str] = []
    lines.append("# Richer FCA Scaling — Phase 3 (graph-smoothed top-k)\n")
    lines.append(
        "_Optional graph-aware scaling (§11): each node's features are blended with "
        "its 1-hop neighbourhood mean (`smooth_alpha=0.5`) before top-k="
        "10 binarisation, K=128, scorer=support, seeds 0–4. Compared against the "
        "K-matched SVD control, `binary_nonzero` FCA, and the best non-smoothed "
        f"richer mode for that dataset. Tie band ±{TIE_EPS:.3f} accuracy. Source: "
        "[`results/per_seed_results.csv`](../results/per_seed_results.csv)._\n"
    )
    any_rows = False
    for ds, mode, params, tag in PHASE3_SELECTED:
        pstr = json.dumps(params, sort_keys=True)
        raw_m, raw_s, raw_n = _agg(_raw_ref(df, ds))
        svd_m, svd_s, svd_n = _agg(_svd_ref(df, ds, PHASE3_K))
        lines.append(f"\n## {ds} — `{mode}` {pstr}\n")
        lines.append("| scorer | smoothed FCA | binary_nonzero FCA | K-SVD (128) | raw SAGE | Δ vs SVD | Δ vs binary | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for scorer in PHASE3_SCORERS:
            r_m, r_s, r_n = _agg(_richer_row(df, ds, mode, scorer, PHASE3_K))
            b_m, b_s, b_n = _agg(_binary_ref(df, ds, scorer, PHASE3_K))
            d_svd = r_m - svd_m if (r_n and svd_n) else float("nan")
            d_bin = r_m - b_m if (r_n and b_n) else float("nan")
            verdict = _verdict(d_svd, d_bin) if r_n else "NOT RUN"
            if r_n:
                any_rows = True
            lines.append(
                f"| {scorer} | {_fmt(r_m, r_s, r_n)} | {_fmt(b_m, b_s, b_n)} | "
                f"{_fmt(svd_m, svd_s, svd_n)} | {_fmt(raw_m, raw_s, raw_n)} | "
                f"{_fmt_d(d_svd)} | {_fmt_d(d_bin)} | {verdict} |"
            )
    lines.append("\n## Phase 3 verdict\n")
    if not any_rows:
        lines.append(
            "> **No Phase-3 rows found.** Run "
            "`python scripts/run_scaling_extension.py --phase3 --seeds 0 1 2 3 4`.\n"
        )
    else:
        lines.append(
            "Graph smoothing is evaluated only as an exploratory probe; see "
            "[`richer_scaling_final_findings.md`](richer_scaling_final_findings.md) "
            "for the consolidated interpretation.\n"
        )
    return "\n".join(lines) + "\n"


def _write_report(phase2b: bool = False, phase3: bool = False) -> None:
    if phase2b:
        save_text(build_report_phase2b(), PHASE2B_REPORT_PATH)
        logger.info("Wrote %s", PHASE2B_REPORT_PATH)
    if phase3:
        save_text(build_report_phase3(), PHASE3_REPORT_PATH)
        logger.info("Wrote %s", PHASE3_REPORT_PATH)
    text = build_report()
    save_text(text, REPORT_PATH)
    logger.info("Wrote %s", REPORT_PATH)


def main() -> None:
    ap = argparse.ArgumentParser(description="Richer-scaling training + report (Phase 2a / 2b / 3).")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--report-only", action="store_true",
                    help="Skip training; just (re)build the summary from existing CSV.")
    ap.add_argument("--phase2b", action="store_true",
                    help="Run the Phase 2b K-sweep (K in {64,256}) around promising modes.")
    ap.add_argument("--phase3", action="store_true",
                    help="Run the optional Phase 3 graph_smoothed_topk grid (pubmed+cora).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict Phase 2b/3 to these datasets (default: all selected).")
    ap.add_argument("--ks", nargs="*", type=int, default=None,
                    help="Override Phase 2b K values (default: 64 256).")
    args = ap.parse_args()

    if not args.report_only:
        if args.phase2b:
            _run_training_phase2b(args.seeds, args.datasets, args.ks)
        elif args.phase3:
            _run_training_phase3(args.seeds, args.datasets)
        else:
            _run_training(args.seeds)
    _write_report(phase2b=args.phase2b, phase3=args.phase3)


if __name__ == "__main__":
    main()
