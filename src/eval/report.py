"""Generate ``reports/experiment_summary.md`` -- the paper-ready summary.

Sections: main results, deltas (vs SAGE-raw / MLP-raw / K-matched SVD), K-sweep,
hard-vs-soft membership, concept-scorer ablation, degree-bucket analysis, concept
statistics, top concepts per class, and an explicit PubMed failure analysis.
All FCA tables are auto-skipped when the corresponding runs are absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import save_text
from ..utils.logging import get_logger
from ..utils.paths import CONCEPTS_DIR, REPORTS_DIR
from .aggregate import run as aggregate_run

logger = get_logger("eval.report")


def _fmt(mean, std=None) -> str:
    if mean is None or pd.isna(mean):
        return "-"
    return f"{mean:.4f} ± {std:.4f}" if std is not None and not pd.isna(std) else f"{mean:.4f}"


def _signed(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{v:+.4f}"


def _num(v):
    """Coerce a possibly-'na' key value to float, else None."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fca_feat(main: pd.DataFrame, ds: str) -> pd.DataFrame:
    return main[(main["dataset"] == ds) & (main["variant"] == "fca_feat")].copy()


# Canonical "comparable" FCA configuration: a single-attribute, hard-membership,
# support-scored GraphSAGE run. Ablations vary exactly one of these axes; mixing
# them (e.g. soft + lift + intent>=2 at one K) is what produced the contaminated
# K-sweep / scorer / membership tables (Task A).
GRAPHSAGE_MODELS = ("graphsage", "sage")
CANON_FCA = dict(model=GRAPHSAGE_MODELS, variant="fca_feat",
                 scorer="support", membership="hard", min_intent_size=1)


def _filter(main: pd.DataFrame, ds: str, **constraints) -> pd.DataFrame:
    """Select rows for one dataset matching every ``column == value`` constraint.

    Values may be a scalar (string / numeric, with ``na`` and float coercion) or
    a list/tuple/set (membership test). Numeric constraints compare via :func:`_num`
    so that object-typed key columns (``min_intent_size`` = ``1.0`` vs ``"na"``)
    match cleanly.
    """
    sub = main[main["dataset"] == ds].copy()
    for col, want in constraints.items():
        if col not in sub.columns:
            return sub.iloc[0:0]
        if isinstance(want, (list, tuple, set)):
            sub = sub[sub[col].astype(str).isin([str(w) for w in want])]
        elif isinstance(want, bool):
            sub = sub[sub[col] == want]
        elif isinstance(want, (int, float)):
            sub = sub[sub[col].map(_num) == float(want)]
        else:
            sub = sub[sub[col].astype(str) == str(want)]
    return sub


# ------------------------------------------------------------------- sections
def _results_table(main: pd.DataFrame, ds: str) -> str:
    grp = main[main["dataset"] == ds].sort_values("test_accuracy_mean", ascending=False)
    lines = ["| model | variant | +dim | K | scorer | memb | test acc | macro-F1 | low-deg acc | seeds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in grp.iterrows():
        k = _num(r.get("k_concepts"))
        lines.append(
            f"| {r['model']} | {r['variant']} | {int(r.get('added_dim') or 0)} | "
            f"{'' if k is None else int(k)} | {r.get('scorer') if r.get('scorer') not in (None,'na') else ''} | "
            f"{r.get('membership') if r.get('membership') not in (None,'na') else ''} | "
            f"{_fmt(r['test_accuracy_mean'], r['test_accuracy_std'])} | "
            f"{_fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'])} | "
            f"{_fmt(r.get('low_degree_accuracy_mean'), r.get('low_degree_accuracy_std'))} | "
            f"{int(r['n_seeds'])} |")
    return "\n".join(lines)


def _delta_table(deltas: pd.DataFrame, ds: str) -> str:
    grp = deltas[deltas["dataset"] == ds].sort_values("d_acc_vs_sage_raw", ascending=False)
    lines = ["| model | variant | +dim | Δacc vs SAGE(raw) | Δmacro-F1 vs SAGE(raw) | Δacc vs MLP(raw) | Δacc vs SVD(K-matched) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in grp.iterrows():
        lines.append(
            f"| {r['model']} | {r['variant']} | {int(r.get('added_dim') or 0)} | "
            f"{_signed(r['d_acc_vs_sage_raw'])} | {_signed(r['d_mf1_vs_sage_raw'])} | "
            f"{_signed(r['d_acc_vs_mlp_raw'])} | {_signed(r['d_acc_vs_svd_kmatched'])} |")
    return "\n".join(lines)


def _ksweep_table(main: pd.DataFrame, ds: str) -> str:
    """FCA_FEAT vs K-matched SVD across added dimension.

    The FCA arm is pinned to the canonical comparable config (GraphSAGE, support
    scorer, hard membership, single-attribute concepts) so each K row holds *one*
    run per arm. Averaging incomparable rows here is exactly what inflated/deflated
    the per-K accuracies before (Task A1).
    """
    fca = _filter(main, ds, **CANON_FCA)
    svd = _filter(main, ds, model=GRAPHSAGE_MODELS, variant="svd_control")
    if fca["added_dim"].nunique() < 2 and svd.empty:
        return ""
    lines = [f"### K-sweep — `{ds}` (FCA_FEAT vs K-matched SVD)", "",
             "_FCA arm: GraphSAGE · support · hard · intent=1 (single comparable config per K)._", "",
             "| K (+dim) | FCA acc | FCA macro-F1 | SVD acc | SVD macro-F1 |",
             "|---|---|---|---|---|"]
    dims = sorted(set(fca["added_dim"].dropna()) | set(svd["added_dim"].dropna()))
    for d in dims:
        f = fca[fca["added_dim"] == d]
        s = svd[svd["added_dim"] == d]
        fa = _fmt(f["test_accuracy_mean"].mean()) if len(f) else "-"
        ff = _fmt(f["test_macro_f1_mean"].mean()) if len(f) else "-"
        sa = _fmt(s["test_accuracy_mean"].mean()) if len(s) else "-"
        sf = _fmt(s["test_macro_f1_mean"].mean()) if len(s) else "-"
        lines.append(f"| {int(d)} | {fa} | {ff} | {sa} | {sf} |")
    return "\n".join(lines)


# Axis ablations: vary exactly one axis while pinning all other canonical axes
# AND the concept count K, so the table compares like with like (Task A1).
_AXIS_PINS = {
    "membership": dict(model=GRAPHSAGE_MODELS, variant="fca_feat",
                       scorer="support", min_intent_size=1, k_concepts=128),
    "scorer": dict(model=GRAPHSAGE_MODELS, variant="fca_feat",
                   membership="hard", min_intent_size=1, k_concepts=128),
}


def _axis_table(main: pd.DataFrame, ds: str, axis: str, title: str) -> str:
    """FCA_FEAT ablation over one axis, with every other axis (and K) pinned."""
    pins = _AXIS_PINS.get(axis, dict(model=GRAPHSAGE_MODELS, variant="fca_feat"))
    fca = _filter(main, ds, **pins)
    fca = fca[fca[axis].notna() & (fca[axis] != "na")]
    if fca[axis].nunique() < 2:
        return ""
    k_note = pins.get("k_concepts")
    note = f"_GraphSAGE · K={k_note} · all other concept axes fixed._" if k_note else ""
    lines = [f"### {title} — `{ds}`", ""]
    if note:
        lines += [note, ""]
    lines += [f"| {axis} | K | test acc | macro-F1 | low-deg acc |",
              "|---|---|---|---|---|"]
    for val, sub in fca.groupby(axis):
        for _, r in sub.sort_values("test_accuracy_mean", ascending=False).iterrows():
            k = _num(r.get("k_concepts"))
            lines.append(f"| {val} | {'' if k is None else int(k)} | "
                         f"{_fmt(r['test_accuracy_mean'], r['test_accuracy_std'])} | "
                         f"{_fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'])} | "
                         f"{_fmt(r.get('low_degree_accuracy_mean'))} |")
    return "\n".join(lines)


def _multi_intent_table(main: pd.DataFrame, ds: str) -> str:
    """Single-attribute (intent=1) vs multi-attribute (intent>=2) concepts (Task D).

    Probes H6: are genuinely conceptual, multi-attribute concepts useful, or is the
    FCA layer effectively single-feature selection? Both arms are pinned to
    GraphSAGE · support · hard so only the intent filter differs.
    """
    pinned = dict(model=GRAPHSAGE_MODELS, variant="fca_feat",
                  scorer="support", membership="hard")
    single = _filter(main, ds, min_intent_size=1, **pinned)
    multi = _filter(main, ds, min_intent_size=2, **pinned)
    if single.empty or multi.empty:
        return ""
    lines = [f"### Multi-intent concepts — `{ds}` (intent=1 vs intent≥2)", "",
             "_GraphSAGE · support · hard. Same pipeline; only the minimum intent "
             "size differs. ``mean intent`` and ``#concepts`` show how scarce "
             "multi-attribute structure is._", "",
             "| min intent | K | #concepts | mean intent | node cov | test acc | macro-F1 |",
             "|---|---|---|---|---|---|---|"]
    for label, sub in (("1", single), ("≥2", multi)):
        r = sub.sort_values("test_accuracy_mean", ascending=False).iloc[0]
        k = _num(r.get("k_concepts"))
        lines.append(
            f"| {label} | {'' if k is None else int(k)} | "
            f"{_fmt(r.get('num_concepts_mean'))} | {_fmt(r.get('mean_intent_size_mean'))} | "
            f"{_fmt(r.get('node_coverage_mean'))} | "
            f"{_fmt(r['test_accuracy_mean'], r['test_accuracy_std'])} | "
            f"{_fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'])} |")
    # Explicit trade-off line: accuracy delta and coverage delta.
    rs = single.sort_values("test_accuracy_mean", ascending=False).iloc[0]
    rm = multi.sort_values("test_accuracy_mean", ascending=False).iloc[0]
    d_acc = rm["test_accuracy_mean"] - rs["test_accuracy_mean"]
    d_cov = _num(rm.get("node_coverage_mean"))
    cov_s = _num(rs.get("node_coverage_mean"))
    cov_note = (f" Node coverage moves {_signed((d_cov - cov_s)) if (d_cov is not None and cov_s is not None) else '-'}"
                if d_cov is not None and cov_s is not None else "")
    lines += ["", f"_intent≥2 vs intent=1: Δacc = **{_signed(d_acc)}**.{cov_note} "
              "(coverage/performance trade-off)._"]
    return "\n".join(lines)


def _degree_bucket_table(main: pd.DataFrame, ds: str) -> str:
    grp = main[main["dataset"] == ds]
    if "bucket_low_accuracy_mean" not in grp.columns or grp["bucket_low_accuracy_mean"].isna().all():
        return ""
    sage_raw = grp[(grp["model"].isin(["graphsage", "sage"])) & (grp["variant"] == "raw")
                   & (grp["aggr"].astype(str) == "mean")]
    fca = grp[grp["variant"] == "fca_feat"].sort_values("test_accuracy_mean", ascending=False)
    svd = grp[grp["variant"] == "svd_control"].sort_values("test_accuracy_mean", ascending=False)
    picks = []
    if len(sage_raw):
        picks.append(("SAGE(raw)", sage_raw.iloc[0]))
    if len(fca):
        picks.append(("SAGE+FCA(best)", fca.iloc[0]))
    if len(svd):
        picks.append(("SAGE+SVD(best)", svd.iloc[0]))
    if not picks:
        return ""
    lines = [f"### Degree-bucket accuracy — `{ds}` (tertile split)", "",
             "| model | low-deg | medium-deg | high-deg |", "|---|---|---|---|"]
    for name, r in picks:
        lines.append(f"| {name} | {_fmt(r.get('bucket_low_accuracy_mean'))} | "
                     f"{_fmt(r.get('bucket_medium_accuracy_mean'))} | "
                     f"{_fmt(r.get('bucket_high_accuracy_mean'))} |")
    return "\n".join(lines)


def _concept_stats_table(main: pd.DataFrame) -> str:
    fca = main[main["variant"] == "fca_feat"]
    if fca.empty or "mean_intent_size_mean" not in fca.columns:
        return ""
    lines = ["## Concept statistics (FCA_FEAT)", "",
             "| dataset | K (+dim) | #concepts | mean intent | node coverage |",
             "|---|---|---|---|---|"]
    for _, r in fca.sort_values(["dataset", "added_dim"]).iterrows():
        lines.append(f"| {r['dataset']} | {int(r.get('added_dim') or 0)} | "
                     f"{_fmt(r.get('num_concepts_mean'))} | {_fmt(r.get('mean_intent_size_mean'))} | "
                     f"{_fmt(r.get('node_coverage_mean'))} |")
    return "\n".join(lines)


def _top_concepts_section(ds: str, top_n: int = 3) -> str:
    path = CONCEPTS_DIR / f"{ds}_concepts.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if "dominant_class" not in df.columns or df.empty:
        return ""
    df = df[df["dominant_class"] >= 0]  # drop concepts with no TRAIN coverage (class -1)
    if df.empty:
        return ""
    lines = [f"### Top concepts per class — `{ds}`", "",
             "| class | support | purity | lift | intent_size | intent (attributes) |",
             "|---|---|---|---|---|---|"]
    for cls in sorted(df["dominant_class"].unique()):
        sub = df[df["dominant_class"] == cls].sort_values(
            ["purity", "lift", "support"], ascending=False).head(top_n)
        for _, r in sub.iterrows():
            attrs = str(r.get("attributes", ""))[:70]
            lines.append(f"| {int(cls)} | {int(r['support'])} | {r['purity']:.3f} | "
                         f"{r['lift']:.3f} | {int(r.get('intent_size', 0))} | {attrs} |")
    return "\n".join(lines)


def _pubmed_failure(main: pd.DataFrame, deltas: pd.DataFrame) -> str:
    if "pubmed" not in set(main["dataset"]):
        return ""
    d = deltas[(deltas["dataset"] == "pubmed") & (deltas["variant"] == "fca_feat")]
    if d.empty:
        return ""
    best = d.sort_values("d_acc_vs_sage_raw", ascending=False).iloc[0]
    drop = best["d_acc_vs_sage_raw"]
    svd_delta = best.get("d_acc_vs_svd_kmatched")
    verdict = ("FCA still underperforms raw GraphSAGE on PubMed"
               if drop < 0 else "FCA recovers to near raw GraphSAGE on PubMed")

    out = ["## PubMed failure analysis", "",
           f"- Best FCA_FEAT vs SAGE(raw): **{_signed(drop)}** accuracy. {verdict}.",
           f"- vs K-matched SVD control: **{_signed(svd_delta)}**.",
           "- PubMed features are dense TF-IDF; `binary_nonzero` discretisation is "
           "coarse and yields almost exclusively single-attribute concepts with very "
           "large extents (low discriminative power). The diagnostics below isolate "
           "whether the drop is caused by the *discretisation* or by FCA itself; the "
           "negative result is reported, not hidden — it bounds where the FCA "
           "construction is appropriate.", ""]

    # --- Diagnostic comparison across binarisation / membership variants ---
    pm = main[(main["dataset"] == "pubmed") & (main["variant"] == "fca_feat")].copy()
    sage_raw = main[(main["dataset"] == "pubmed")
                    & (main["model"].isin(list(GRAPHSAGE_MODELS)))
                    & (main["variant"] == "raw")]
    raw_acc = sage_raw["test_accuracy_mean"].max() if len(sage_raw) else None
    has_binmode = "binarize_mode_mean" not in pm.columns and "binarize_mode" in pm.columns
    if not pm.empty:
        out += ["**FCA diagnostics on PubMed (discretisation × membership)**", "",
                "| binarize | membership | K | mean intent | mean extent | node cov | test acc | Δ vs raw |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in pm.sort_values("test_accuracy_mean", ascending=False).iterrows():
            binm = r.get("binarize_mode") if has_binmode else "—"
            binm = "binary_nonzero" if binm in (None, "na", float("nan")) or pd.isna(binm) else binm
            memb = r.get("membership") if r.get("membership") not in (None, "na") else ""
            k = _num(r.get("k_concepts"))
            dacc = (r["test_accuracy_mean"] - raw_acc) if raw_acc is not None else None
            out.append(
                f"| {binm} | {memb} | {'' if k is None else int(k)} | "
                f"{_fmt(r.get('mean_intent_size_mean'))} | "
                f"{_fmt(r.get('mean_extent_size_mean'))} | "
                f"{_fmt(r.get('node_coverage_mean'))} | "
                f"{_fmt(r['test_accuracy_mean'], r['test_accuracy_std'])} | "
                f"{_signed(dacc) if dacc is not None else '-'} |")
        out += ["",
                "_Interpretation: if quantile/interval scaling or soft membership do "
                "not close the gap, the limitation is the bag-of-words formal context "
                "(huge low-purity extents), not the membership encoding. A near-zero "
                "intent≥2 count (see concept statistics) confirms PubMed offers little "
                "multi-attribute conceptual structure under this binarisation._"]
    return "\n".join(out)


# --------------------------------------------------------------------- driver
def build_report() -> str:
    res = aggregate_run()
    main, deltas = res["main"], res["deltas"]
    datasets = list(main["dataset"].unique())

    parts = [
        "# Experiment summary — GraphSAGE + FCA node classification", "",
        "Auto-generated from `results/per_seed_results.csv`. Cells are `mean ± std` "
        "over seeds. Configs are de-duplicated by hyperparameters; the SVD control is "
        "matched by added dimensionality (SVD-K vs FCA-K).", "",
        f"Datasets covered: {', '.join(map(str, datasets))}.", "",
    ]
    for ds in datasets:
        parts += [f"## Dataset: `{ds}`", "", "**Results**", "", _results_table(main, ds),
                  "", "**Deltas (FCA / controls vs baselines)**", "", _delta_table(deltas, ds), ""]
        for sect in (_ksweep_table(main, ds),
                     _axis_table(main, ds, "membership", "Hard vs soft membership"),
                     _axis_table(main, ds, "scorer", "Concept-scorer ablation"),
                     _multi_intent_table(main, ds),
                     _degree_bucket_table(main, ds),
                     _top_concepts_section(ds)):
            if sect:
                parts += [sect, ""]

    for sect in (_concept_stats_table(main), _pubmed_failure(main, deltas)):
        if sect:
            parts += [sect, ""]

    parts += [
        "## Notes", "",
        "- Supervised concept scorers (`target_entropy`, `lift`) and the "
        "class-association columns use **training labels only** — no validation/test "
        "leakage (enforced by a unit test).",
        "- A positive Δ vs SAGE(raw) supports the main hypothesis; a positive Δ vs the "
        "**K-matched** SVD control indicates the gain is specific to FCA structure, "
        "not merely added dimensionality.",
        "- Concepts with no training-set coverage have `dominant_class = -1` and are "
        "excluded from the top-concept tables.",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the experiment summary report.")
    ap.add_argument("--out", default=str(REPORTS_DIR / "experiment_summary.md"))
    args = ap.parse_args()
    save_text(build_report(), Path(args.out))
    logger.info("Wrote report -> %s", args.out)


if __name__ == "__main__":
    main()
