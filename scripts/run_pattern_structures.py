"""Run interval pattern-structure experiments and build reports/figures.

Feature variant: ``fca_pattern`` = raw node features concatenated with interval
pattern-membership features.

Usage
-----
    python scripts/run_pattern_structures.py --phase baseline --seeds 0 1
    python scripts/run_pattern_structures.py --phase extended --seeds 0 1 2 3 4
    python scripts/run_pattern_structures.py --phase sweep --seeds 0 1 2 3 4
    python scripts/run_pattern_structures.py --phase cora_graph --seeds 0 1 2 3 4
    python scripts/run_pattern_structures.py --report-only
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import save_text
from src.utils.logging import get_logger
from src.utils.paths import FIGURES_DIR, REPORTS_DIR, RESULTS_DIR

logger = get_logger("pattern.structures")

PER_SEED = RESULTS_DIR / "per_seed_results.csv"
REPORT_PATH = REPORTS_DIR / "pattern_structures_summary.md"
FIGURE_PATH = FIGURES_DIR / "pattern_delta_vs_svd.png"
EXP_PREFIX = "pattern_struct__"
BASELINE_DATASETS = ["cora", "citeseer", "pubmed"]
EXTENDED_DATASETS = ["citeseer", "pubmed"]
SCORERS = ["support", "target_entropy"]
BASE_PATTERN_PARAMS = {
    "n_bins": 4,
    "intent_size": 2,
    "object_sample": 2000,
    "max_features": 512,
    "feature_rank": "support",
    "min_support": 0.01,
    "max_support": 0.6,
}
PATTERN_VARIANTS = {
    "baseline_hard": {
        "tag": "interval_qbins4_i2_hard",
        "membership": "hard",
        "params": dict(BASE_PATTERN_PARAMS),
        "datasets": BASELINE_DATASETS,
        "phase": "baseline",
        "ks": [128],
    },
    "soft": {
        "tag": "interval_qbins4_i2_soft",
        "membership": "soft",
        "params": dict(BASE_PATTERN_PARAMS),
        "datasets": EXTENDED_DATASETS,
        "phase": "extended",
        "ks": [128],
    },
    "graph_hard": {
        "tag": "gaware_qbins4_i2_hard",
        "membership": "hard",
        "params": {**BASE_PATTERN_PARAMS, "source": "graph_smoothed", "smooth_alpha": 0.5, "hops": 1},
        "datasets": EXTENDED_DATASETS,
        "phase": "extended",
        "ks": [128],
    },
    "graph_soft": {
        "tag": "gaware_qbins4_i2_soft",
        "membership": "soft",
        "params": {**BASE_PATTERN_PARAMS, "source": "graph_smoothed", "smooth_alpha": 0.5, "hops": 1},
        "datasets": EXTENDED_DATASETS,
        "phase": "extended",
        "ks": [128],
    },
    "graph_soft_sweep": {
        "tag": "gaware_qbins4_i2_soft",
        "membership": "soft",
        "params": {**BASE_PATTERN_PARAMS, "source": "graph_smoothed", "smooth_alpha": 0.5, "hops": 1},
        "datasets": EXTENDED_DATASETS,
        "phase": "sweep",
        "ks": [64, 256],
    },
    "cora_graph_soft": {
        "tag": "gaware_qbins4_i2_soft",
        "membership": "soft",
        "params": {**BASE_PATTERN_PARAMS, "source": "graph_smoothed", "smooth_alpha": 0.5, "hops": 1},
        "datasets": ["cora"],
        "phase": "cora_graph",
        "ks": [128],
    },
}
TIE_EPS = 0.002


def _run_one(base_config, run_variant, ds: str, scorer: str, variant_key: str,
             k: int, seeds: list[int] | None) -> None:
    spec = PATTERN_VARIANTS[variant_key]
    base = base_config(ds, "fca")
    tag = str(spec["tag"])
    membership = str(spec["membership"])
    params = dict(spec["params"])
    exp = f"{EXP_PREFIX}{ds}_{tag}_k{k}_{scorer}_{membership}"
    overrides = {
        "features.variant": "fca_pattern",
        "features.fca.k_concepts": k,
        "features.fca.scorer": scorer,
        "features.fca.membership": membership,
        "features.fca.pattern_params": params,
        "features.fca.min_intent_size": 1,
        "features.fca.binarize_mode": "pattern_interval",
        "features.fca.binarize_params": params,
    }
    logger.info("RUN %s | scorer=%s | k=%d | params=%s", exp, scorer, k, params)
    run_variant(base, exp, overrides, seeds=seeds)


def run_training(seeds: list[int] | None, phase: str) -> None:
    from _ablation_common import base_config, run_variant

    selected = [k for k, v in PATTERN_VARIANTS.items() if v["phase"] == phase]
    for variant_key in selected:
        spec = PATTERN_VARIANTS[variant_key]
        for ds in spec["datasets"]:
            for k in spec["ks"]:
                for scorer in SCORERS:
                    _run_one(base_config, run_variant, ds, scorer, variant_key, int(k), seeds)


def _load() -> pd.DataFrame:
    if not PER_SEED.exists():
        raise FileNotFoundError(f"{PER_SEED} not found")
    df = pd.read_csv(PER_SEED)
    for col in ["binarize_mode", "membership", "scorer", "experiment", "k_concepts"]:
        if col not in df.columns:
            df[col] = np.nan
    df["binarize_mode"] = df["binarize_mode"].fillna("binary_nonzero")
    return df


def _is_sage(df: pd.DataFrame) -> pd.Series:
    return df["model"].isin(["graphsage", "sage"])


def _agg(sub: pd.DataFrame, col: str = "test_accuracy") -> tuple[float, float, int]:
    if sub.empty:
        return float("nan"), float("nan"), 0
    val = sub[col].astype(float)
    return float(val.mean()), float(val.std(ddof=0)), int(len(sub))


def _raw_ref(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    m = (df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "raw")
    if "aggr" in df.columns:
        pref = df[m & (df["aggr"].astype(str) == "mean")]
        if not pref.empty:
            return pref
    return df[m]


def _svd_ref(df: pd.DataFrame, ds: str, k: int) -> pd.DataFrame:
    return df[
        (df["dataset"] == ds) & (df["variant"] == "svd_control")
        & (np.round(pd.to_numeric(df["added_dim"], errors="coerce")) == k)
    ]


def _binary_ref(df: pd.DataFrame, ds: str, scorer: str, membership: str, k: int) -> pd.DataFrame:
    mii = df["min_intent_size"].isna() | (df["min_intent_size"] == 1)
    return df[
        (df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
        & (df["k_concepts"] == k) & (df["scorer"] == scorer)
        & (df["membership"] == membership)
        & (df["binarize_mode"] == "binary_nonzero") & mii
    ]


def _richer_ref(df: pd.DataFrame, ds: str, k: int) -> pd.DataFrame:
    return df[
        (df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
        & (df["k_concepts"] == k) & (df["membership"] == "hard")
        & (df["binarize_mode"].isin(["quantile_global", "topk_per_node", "graph_smoothed_topk"]))
    ]


def _pattern_groups(df: pd.DataFrame) -> pd.DataFrame:
    pat = df[
        _is_sage(df) & (df["variant"] == "fca_pattern")
        & df["experiment"].astype(str).str.startswith(EXP_PREFIX)
    ].copy()
    if pat.empty:
        return pat
    pat["pattern_tag"] = pat["experiment"].astype(str).str.replace(rf"^{EXP_PREFIX}", "", regex=True)
    pat["pattern_tag"] = pat["pattern_tag"].str.replace(rf"^(cora|citeseer|pubmed)_", "", regex=True)
    pat["pattern_tag"] = pat["pattern_tag"].str.replace(r"_k\d+_(support|target_entropy)_(hard|soft)$", "", regex=True)
    pat["k_concepts"] = pd.to_numeric(pat["k_concepts"], errors="coerce").round().astype("Int64")
    return pat


def _fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "—"
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def _delta(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return a - b


def _fmt_d(v: float) -> str:
    if np.isnan(v):
        return "—"
    return f"{v:+.4f}"


def _bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0,
                     n_boot: int = 10000) -> tuple[float, float, float, float]:
    """Bootstrap mean(a-b), 95% CI and one-sided P(delta <= 0).

    If seed ids overlap, use paired differences by seed. Otherwise resample group
    means independently. Returns (mean_delta, ci_low, ci_high, p_le_zero).
    """
    if a.empty or b.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    aa = a[["seed", "test_accuracy"]].copy()
    bb = b[["seed", "test_accuracy"]].copy()
    aa["seed"] = pd.to_numeric(aa["seed"], errors="coerce")
    bb["seed"] = pd.to_numeric(bb["seed"], errors="coerce")
    paired = aa.merge(bb, on="seed", suffixes=("_pattern", "_svd"))
    rng = np.random.default_rng(seed)
    if len(paired) >= 2:
        diffs = (paired["test_accuracy_pattern"] - paired["test_accuracy_svd"]).to_numpy(float)
        boot = rng.choice(diffs, size=(n_boot, diffs.size), replace=True).mean(axis=1)
        mean = float(diffs.mean())
    else:
        av = aa["test_accuracy"].to_numpy(float)
        bv = bb["test_accuracy"].to_numpy(float)
        boot = (rng.choice(av, size=(n_boot, av.size), replace=True).mean(axis=1)
                - rng.choice(bv, size=(n_boot, bv.size), replace=True).mean(axis=1))
        mean = float(av.mean() - bv.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = float((boot <= 0.0).mean())
    return mean, float(lo), float(hi), p


def _verdict(d_svd: float, d_binary: float) -> str:
    if not np.isnan(d_svd) and d_svd > TIE_EPS:
        return "SUCCESS vs K-matched SVD"
    if not np.isnan(d_binary) and d_binary > TIE_EPS:
        return "PARTIAL vs binary FCA"
    if not np.isnan(d_svd) and d_svd < -TIE_EPS:
        return "NO vs SVD"
    return "NEUTRAL"


def _collect_rows(df: pd.DataFrame) -> list[dict]:
    pat = _pattern_groups(df)
    rows: list[dict] = []
    if pat.empty:
        return rows
    group_cols = ["dataset", "pattern_tag", "k_concepts", "scorer", "membership"]
    for (ds, tag, k, scorer, membership), sub in pat.groupby(group_cols, dropna=False):
        ds_s = str(ds)
        tag_s = str(tag)
        scorer_s = str(scorer)
        membership_s = str(membership)
        k_i = int(str(k))
        raw_m, raw_s, raw_n = _agg(_raw_ref(df, ds_s))
        svd = _svd_ref(df, ds_s, k_i)
        svd_m, svd_s, svd_n = _agg(svd)
        bin_m, bin_s, bin_n = _agg(_binary_ref(df, ds_s, scorer_s, membership_s, k_i))
        richer = _richer_ref(df, ds_s, k_i)
        richer_m, richer_s, richer_n = _agg(richer) if not richer.empty else (float("nan"), float("nan"), 0)
        p_m, p_s, p_n = _agg(sub)
        pf_m, pf_s, pf_n = _agg(sub, "test_macro_f1")
        d_svd = _delta(p_m, svd_m)
        d_bin = _delta(p_m, bin_m)
        boot_mean, ci_lo, ci_hi, p_le_zero = _bootstrap_delta(sub, svd)
        rows.append({
            "dataset": ds_s,
            "pattern_variant": tag_s,
            "K": k_i,
            "scorer": scorer_s,
            "membership": membership_s,
            "pattern_acc": p_m,
            "pattern_acc_std": p_s,
            "pattern_macro_f1": pf_m,
            "pattern_macro_f1_std": pf_s,
            "raw_acc": raw_m,
            "binary_acc": bin_m,
            "svd_acc": svd_m,
            "richer_acc": richer_m,
            "delta_vs_svd": d_svd,
            "delta_vs_binary": d_bin,
            "boot_ci_low": ci_lo,
            "boot_ci_high": ci_hi,
            "boot_p_le_zero": p_le_zero,
            "n": p_n,
            "verdict": _verdict(d_svd, d_bin),
        })
    return rows


def plot_pattern_delta(rows: list[dict]) -> None:
    """Plot the best scorer per dataset/pattern variant at K=128.

    Keeping both scorers for every variant made the figure unreadable. The report
    table still contains all rows; the paper figure shows the best accuracy row for
    each dataset × pattern-variant pair.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    focus = df[(df["K"] == 128) & (df["pattern_variant"].isin([
        "interval_qbins4_i2", "interval_qbins4_i2_soft", "gaware_qbins4_i2_soft"
    ]))].copy()
    if focus.empty:
        return
    idx = focus.groupby(["dataset", "pattern_variant"], dropna=False)["pattern_acc"].idxmax()
    focus = focus.loc[idx].copy()
    name_map = {
        "interval_qbins4_i2": "interval hard",
        "interval_qbins4_i2_soft": "interval soft",
        "gaware_qbins4_i2_soft": "graph-aware soft",
    }
    focus["label"] = focus["dataset"].str.replace("citeseer", "CiteSeer").str.replace("pubmed", "PubMed").str.replace("cora", "Cora")
    focus["label"] += "\n" + focus["pattern_variant"].map(name_map)
    focus["label"] += "\n" + focus["scorer"].astype(str)
    focus = focus.sort_values(["dataset", "pattern_variant"])
    colors = ["#C44E52" if d < -TIE_EPS else "#55A868" if d > TIE_EPS else "#999999"
              for d in focus["delta_vs_svd"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(focus))
    ax.bar(x, focus["delta_vs_svd"], color=colors)
    ax.axhline(0, color="#333", linewidth=1)
    ax.axhline(TIE_EPS, color="#777", linestyle="--", linewidth=0.8)
    ax.axhline(-TIE_EPS, color="#777", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(focus["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy delta vs K-matched SVD")
    ax.set_title("Pattern-derived features: best K=128 variant vs SVD")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", FIGURE_PATH)


def build_report() -> str:
    df = _load()
    rows = _collect_rows(df)
    plot_pattern_delta(rows)
    lines: list[str] = []
    lines.append("# Pattern structures — interval-pattern experiments\n")
    lines.append(
        "_Exploratory GraphSAGE runs for `fca_pattern`: raw features are concatenated "
        "with interval pattern-membership features. Pattern intervals are built from "
        "train-quantile bins; supervised scoring uses train labels only. Extended rows "
        "also test soft membership and graph-smoothed pattern sources. Source: "
        "[`results/per_seed_results.csv`](../results/per_seed_results.csv)._\n"
    )
    lines.append(f"Tie band for verdicts: ±{TIE_EPS:.3f} accuracy. Bootstrap CI is for pattern minus K-matched SVD.\n")
    lines.append("| dataset | pattern variant | K | scorer | membership | pattern acc | pattern macro-F1 | raw acc | SVD acc | Δ vs SVD | bootstrap 95% CI | P(Δ≤0) | verdict |")
    lines.append("|---|---|---:|---|---|---|---|---|---|---|---|---:|---|")
    if not rows:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | NOT RUN |")
    else:
        for r in sorted(rows, key=lambda x: (x["dataset"], x["pattern_variant"], x["K"], x["scorer"])):
            ci = "—" if np.isnan(r["boot_ci_low"]) else f"[{r['boot_ci_low']:+.4f}, {r['boot_ci_high']:+.4f}]"
            p = "—" if np.isnan(r["boot_p_le_zero"]) else f"{r['boot_p_le_zero']:.3f}"
            lines.append(
                f"| {r['dataset']} | {r['pattern_variant']} | {r['K']} | {r['scorer']} | {r['membership']} | "
                f"{_fmt(r['pattern_acc'], r['pattern_acc_std'], r['n'])} | "
                f"{_fmt(r['pattern_macro_f1'], r['pattern_macro_f1_std'], r['n'])} | "
                f"{r['raw_acc']:.4f} | {r['svd_acc']:.4f} | {_fmt_d(r['delta_vs_svd'])} | {ci} | {p} | {r['verdict']} |"
            )
    lines.append("\n## Interpretation\n")
    if rows:
        sdf = pd.DataFrame(rows)
        for ds in sorted(sdf["dataset"].unique()):
            d = sdf[sdf["dataset"] == ds]
            best = d.loc[d["pattern_acc"].idxmax()]
            lines.append(
                f"- **{ds}**: best pattern run is `{best['pattern_variant']}` / K={int(best['K'])} / "
                f"`{best['scorer']}` / `{best['membership']}` with accuracy {best['pattern_acc']:.4f}; "
                f"Δ vs SVD {_fmt_d(best['delta_vs_svd'])}, bootstrap CI "
                f"[{best['boot_ci_low']:+.4f}, {best['boot_ci_high']:+.4f}], "
                f"P(Δ≤0)={best['boot_p_le_zero']:.3f}. Verdict: {best['verdict']}."
            )
    else:
        lines.append("No pattern-structure training rows found yet. Run the script without `--report-only`.\n")
    lines.append("\n## Write-up guidance\n")
    lines.append(
        "Treat small positive deltas cautiously when the bootstrap CI crosses zero. "
        "For the coursework text, use `competitive with SVD` unless the CI is clearly "
        "positive. The figure `figures/pattern_delta_vs_svd.png` visualises K=128 "
        "pattern variants against the K-matched SVD control.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run interval pattern-structure experiments.")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--phase", choices=["baseline", "extended", "sweep", "cora_graph"], default="extended")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run_training(args.seeds, args.phase)
    report = build_report()
    save_text(report, REPORT_PATH)
    print(report)
    logger.info("Wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()
