"""Generate the final paper tables (CSV + Markdown) for the coursework write-up.

Deterministic: reads only the existing result CSVs / dataset summaries and emits

    results/final_table_dataset_summary.csv
    results/final_table_main_v2.csv
    results/final_table_structural_diagnostics.csv
    results/final_table_richer_scaling.csv
    results/final_table_best_configs.csv
    reports/final_tables.md

No training is performed. All accuracies are seed-averaged (seeds 0–4) with
population std (ddof=0). Strict filters are used so richer-scaling rows are never
merged with binary_nonzero, and only canonical experiments are used for baselines.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd

from src.utils.paths import RESULTS_DIR, REPORTS_DIR

PER_SEED = RESULTS_DIR / "per_seed_results.csv"
DIAG = RESULTS_DIR / "scaling_diagnostics.csv"
SUMM_DIR = RESULTS_DIR / "dataset_summaries"
DATASETS = ["cora", "citeseer", "pubmed"]
K_MAIN = 128
MEMBERSHIP = "hard"
TIE = 0.002


# --------------------------------------------------------------------------- io
def _load() -> pd.DataFrame:
    df = pd.read_csv(PER_SEED)
    if "binarize_mode" not in df.columns:
        df["binarize_mode"] = np.nan
    df["binarize_mode"] = df["binarize_mode"].fillna("binary_nonzero")
    # binary_nonzero rows carry NaN binarize_params; pandas groupby drops NaN
    # keys, which would silently exclude all binary configs from Table 5.
    if "binarize_params" not in df.columns:
        df["binarize_params"] = np.nan
    df["binarize_params"] = df["binarize_params"].fillna("default")
    if "min_intent_size" not in df.columns:
        df["min_intent_size"] = np.nan
    return df


def _is_sage(df):
    return df["model"].isin(["graphsage", "sage"])


def _ms(sub: pd.DataFrame, col: str) -> tuple[float, float, int]:
    if sub.empty:
        return float("nan"), float("nan"), 0
    v = sub[col].astype(float)
    return float(v.mean()), float(v.std(ddof=0)), int(len(v))


def _fmt(m, s, n):
    if n == 0 or np.isnan(m):
        return "—"
    return f"{m:.4f} ± {s:.4f}"


# ----------------------------------------------------------------- Table 1
def table1() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        j = json.loads((SUMM_DIR / f"{ds}_summary.json").read_text())
        sp = j.get("split_sizes", {})
        rows.append({
            "dataset": ds,
            "num_nodes": j["num_nodes"],
            "num_edges": j["num_edges"],
            "num_features": j["num_features"],
            "num_classes": j["num_classes"],
            "train": sp.get("train"), "val": sp.get("val"), "test": sp.get("test"),
            "split": j.get("planetoid_split", "public"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Table 2
def _baseline_row(df, ds, experiment, model, variant, *, added_dim=None,
                  k=None, scorer=None, label_variant=None):
    sub = df[(df["dataset"] == ds) & (df["experiment"] == experiment)]
    if sub.empty:
        return None
    acc = _ms(sub, "test_accuracy")
    mf1 = _ms(sub, "test_macro_f1")
    low = _ms(sub, "low_degree_accuracy")
    return {
        "dataset": ds, "model": model, "variant": label_variant or variant,
        "K_added_dim": added_dim if added_dim is not None else "",
        "scorer": scorer or "",
        "test_acc_mean": acc[0], "test_acc_std": acc[1],
        "macro_f1_mean": mf1[0], "macro_f1_std": mf1[1],
        "low_degree_acc_mean": low[0], "n_seeds": acc[2],
    }


def table2(df) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        # raw GraphSAGE (canonical experiment only)
        rows.append(_baseline_row(df, ds, f"{ds}_sage_raw", "graphsage", "raw"))
        # MLP raw
        rows.append(_baseline_row(df, ds, f"{ds}_mlp_raw", "mlp", "raw"))
        # GCN raw (cora only)
        rows.append(_baseline_row(df, ds, f"{ds}_gcn_raw", "gcn", "raw"))
        # binary_nonzero FCA at K=128, support (canonical FCA experiment)
        mii = df["min_intent_size"].isna() | (df["min_intent_size"] == 1)
        b = df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
               & (df["k_concepts"] == K_MAIN) & (df["scorer"] == "support")
               & (df["membership"] == MEMBERSHIP)
               & (df["binarize_mode"] == "binary_nonzero") & mii]
        if not b.empty:
            acc = _ms(b, "test_accuracy"); mf1 = _ms(b, "test_macro_f1"); low = _ms(b, "low_degree_accuracy")
            rows.append({"dataset": ds, "model": "graphsage", "variant": "fca_binary_nonzero",
                         "K_added_dim": K_MAIN, "scorer": "support",
                         "test_acc_mean": acc[0], "test_acc_std": acc[1],
                         "macro_f1_mean": mf1[0], "macro_f1_std": mf1[1],
                         "low_degree_acc_mean": low[0], "n_seeds": acc[2]})
        # K-matched SVD (added_dim=128)
        ad = pd.to_numeric(df["added_dim"], errors="coerce").round()
        s = df[(df["dataset"] == ds) & (df["variant"] == "svd_control") & (ad == K_MAIN)]
        if not s.empty:
            acc = _ms(s, "test_accuracy"); mf1 = _ms(s, "test_macro_f1"); low = _ms(s, "low_degree_accuracy")
            rows.append({"dataset": ds, "model": "graphsage", "variant": "svd_control",
                         "K_added_dim": K_MAIN, "scorer": "",
                         "test_acc_mean": acc[0], "test_acc_std": acc[1],
                         "macro_f1_mean": mf1[0], "macro_f1_std": mf1[1],
                         "low_degree_acc_mean": low[0], "n_seeds": acc[2]})
    return pd.DataFrame([r for r in rows if r is not None])


# ----------------------------------------------------------------- Table 3
def table3() -> pd.DataFrame:
    d = pd.read_csv(DIAG)

    def want(r):
        m = r["binarize_mode"]
        if m == "binary_nonzero":
            return True
        if m == "quantile_global" and r["dataset"] == "cora" and r["quantile"] == 0.90:
            return True
        if m == "quantile_global" and r["dataset"] == "citeseer" and r["quantile"] == 0.90:
            return True
        if m == "quantile_global" and r["dataset"] == "pubmed" and r["quantile"] == 0.75:
            return True
        if m == "topk_per_node" and r["topk"] == 10:
            return True
        return False

    sub = d[d.apply(want, axis=1)].drop_duplicates(
        ["dataset", "binarize_mode", "quantile", "topk"]).copy()

    def params(r):
        if r["binarize_mode"] == "quantile_global":
            return f"q={r['quantile']:.2f}"
        if r["binarize_mode"] == "topk_per_node":
            return f"topk={int(r['topk'])}"
        return "—"

    sub["params"] = sub.apply(params, axis=1)
    out = sub[["dataset", "binarize_mode", "params", "mean_intent_size",
               "frac_multi_attr", "node_coverage", "mean_extent_size",
               "median_extent_size"]].rename(columns={"binarize_mode": "scaling_mode"})
    order = {"cora": 0, "citeseer": 1, "pubmed": 2}
    mo = {"binary_nonzero": 0, "quantile_global": 1, "topk_per_node": 2}
    out = out.sort_values(by=["dataset", "scaling_mode"],
                          key=lambda c: c.map(order) if c.name == "dataset" else c.map(mo))
    return out.reset_index(drop=True)


# ----------------------------------------------------------------- Table 4
RICHER = [
    ("cora",     "quantile_global", "q=0.90",  [128]),
    ("citeseer", "quantile_global", "q=0.90",  [128, 256]),
    ("pubmed",   "quantile_global", "q=0.75",  [128]),
    ("pubmed",   "topk_per_node",   "topk=10", [64, 128, 256]),
    ("pubmed",   "graph_smoothed_topk", "alpha=0.5,topk=10", [128]),
    ("cora",     "graph_smoothed_topk", "alpha=0.5,topk=10", [128]),
]
SCORERS = ["support", "target_entropy"]


def _richer(df, ds, mode, scorer, k):
    return df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
              & (df["k_concepts"] == k) & (df["scorer"] == scorer)
              & (df["membership"] == MEMBERSHIP) & (df["binarize_mode"] == mode)]


def _binary(df, ds, scorer, k):
    mii = df["min_intent_size"].isna() | (df["min_intent_size"] == 1)
    return df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
              & (df["k_concepts"] == k) & (df["scorer"] == scorer)
              & (df["membership"] == MEMBERSHIP)
              & (df["binarize_mode"] == "binary_nonzero") & mii]


def _svd(df, ds, k):
    ad = pd.to_numeric(df["added_dim"], errors="coerce").round()
    return df[(df["dataset"] == ds) & (df["variant"] == "svd_control") & (ad == k)]


def _raw(df, ds):
    return df[(df["dataset"] == ds) & (df["experiment"] == f"{ds}_sage_raw")]


def _verdict(d_svd, d_bin):
    bs = (not np.isnan(d_svd)) and d_svd > TIE
    bb = (not np.isnan(d_bin)) and d_bin > TIE
    if bs and bb:
        return "SUCCESS (beats SVD & binary)"
    if bs:
        return "SUCCESS (beats SVD)"
    if bb:
        return "PARTIAL (beats binary, not SVD)"
    if (not np.isnan(d_svd) and d_svd < -TIE) or (not np.isnan(d_bin) and d_bin < -TIE):
        return "NO (worse)"
    return "NEUTRAL (tie)"


def table4(df) -> pd.DataFrame:
    rows = []
    for ds, mode, pstr, ks in RICHER:
        raw_m, raw_s, raw_n = _ms(_raw(df, ds), "test_accuracy")
        for k in ks:
            svd_m, svd_s, _ = _ms(_svd(df, ds, k), "test_accuracy")
            for scorer in SCORERS:
                r = _richer(df, ds, mode, scorer, k)
                if r.empty:
                    continue
                r_m, r_s, r_n = _ms(r, "test_accuracy")
                b_m, b_s, b_n = _ms(_binary(df, ds, scorer, k), "test_accuracy")
                d_svd = r_m - svd_m if not np.isnan(svd_m) else float("nan")
                d_bin = r_m - b_m if b_n else float("nan")
                rows.append({
                    "dataset": ds, "mode": mode, "params": pstr, "scorer": scorer, "K": k,
                    "richer_acc_mean": r_m, "richer_acc_std": r_s,
                    "binary_acc_mean": b_m, "binary_acc_std": b_s,
                    "svd_acc_mean": svd_m, "svd_acc_std": svd_s,
                    "raw_acc_mean": raw_m, "raw_acc_std": raw_s,
                    "delta_vs_binary": d_bin, "delta_vs_svd": d_svd,
                    "verdict": _verdict(d_svd, d_bin),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Table 5
def table5(df, t4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    interp = {
        "cora": "Richer scaling does not help; best FCA is binary_nonzero + target_entropy (≈ ties SVD, below it).",
        "citeseer": "Richer quantile_global beats both K-matched SVD and raw GraphSAGE (K=128 and 256).",
        "pubmed": "Richer scaling strongly repairs binary_nonzero but stays below SVD/raw — softened negative case.",
    }
    for ds in DATASETS:
        # best FCA config across ALL fca_feat rows for this dataset (any mode/scorer/K)
        f = df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")]
        # aggregate by config
        grp = f.groupby(["binarize_mode", "binarize_params", "k_concepts", "scorer"])
        best_key, best_acc, best_std = None, -1, 0
        for key, g in grp:
            m = g["test_accuracy"].astype(float).mean()
            if m > best_acc:
                best_acc, best_std, best_key = m, g["test_accuracy"].astype(float).std(ddof=0), key
        if best_key is None:
            # No fca_feat rows for this dataset — emit an empty/NaN config row.
            rows.append({
                "dataset": ds, "best_fca_config": "n/a",
                "best_fca_acc_mean": float("nan"), "best_fca_acc_std": float("nan"),
                "raw_sage_acc": _ms(_raw(df, ds), "test_accuracy")[0],
                "best_svd_acc": float("nan"), "best_svd_k": None,
                "interpretation": interp[ds],
            })
            continue
        bmode, bparams, bk, bscorer = best_key
        raw_m, _, _ = _ms(_raw(df, ds), "test_accuracy")
        # best SVD across added_dim
        sv = df[(df["dataset"] == ds) & (df["variant"] == "svd_control")]
        svg = sv.groupby(pd.to_numeric(sv["added_dim"], errors="coerce").round())["test_accuracy"].mean()
        best_svd = float(svg.max()) if len(svg) else float("nan")
        best_svd_k = int(svg.idxmax()) if len(svg) else None
        cfg = f"{bmode} {bparams} K{int(bk)} {bscorer}"
        rows.append({
            "dataset": ds, "best_fca_config": cfg,
            "best_fca_acc_mean": best_acc, "best_fca_acc_std": best_std,
            "raw_sage_acc": raw_m,
            "best_svd_acc": best_svd, "best_svd_k": best_svd_k,
            "interpretation": interp[ds],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- markdown
def _md(df: pd.DataFrame, floatfmt=4) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "—" if pd.isna(x) else f"{x:.{floatfmt}f}")
        else:
            d[c] = d[c].map(lambda x: "—" if (isinstance(x, float) and pd.isna(x)) else x)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.values]
    return "\n".join([head, sep] + body)


def main() -> None:
    df = _load()
    t1 = table1()
    t2 = table2(df)
    t3 = table3()
    t4 = table4(df)
    t5 = table5(df, t4)

    t1.to_csv(RESULTS_DIR / "final_table_dataset_summary.csv", index=False)
    t2.to_csv(RESULTS_DIR / "final_table_main_v2.csv", index=False)
    t3.to_csv(RESULTS_DIR / "final_table_structural_diagnostics.csv", index=False)
    t4.to_csv(RESULTS_DIR / "final_table_richer_scaling.csv", index=False)
    t5.to_csv(RESULTS_DIR / "final_table_best_configs.csv", index=False)

    md = []
    md.append("# Final paper tables\n")
    md.append("_Auto-generated by [`scripts/make_final_tables.py`](../scripts/make_final_tables.py); "
              "seed-averaged over seeds 0–4 (population std). Strict filters: richer-scaling "
              "rows are kept separate from `binary_nonzero` by `binarize_mode`/`binarize_params`; "
              "baselines use canonical `{dataset}_sage_raw` / `_mlp_raw` / `_gcn_raw` experiments. "
              "Do not hand-edit — regenerate instead._\n")

    md.append("\n## Table 1 — Dataset summary\n")
    md.append(_md(t1, floatfmt=0))
    md.append("\n## Table 2 — Main v2 results (K=128 where applicable)\n")
    md.append("_`fca_binary_nonzero` = naive FCA; `svd_control` = dimensionality-matched SVD. "
              "low_degree_acc = accuracy on the lowest-degree tertile._\n")
    md.append(_md(t2))
    md.append("\n## Table 3 — Structural diagnostics of formal contexts\n")
    md.append(_md(t3))
    md.append("\n## Table 4 — Richer scaling downstream results\n")
    md.append("_`—` in the binary column means no `binary_nonzero` reference exists at that "
              "K/scorer (target_entropy was only swept at K=128); the SVD comparison still holds._\n")
    md.append(_md(t4))
    md.append("\n## Table 5 — Best configuration per dataset\n")
    md.append(_md(t5))
    md.append("")

    (REPORTS_DIR / "final_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Wrote 5 CSVs + reports/final_tables.md")
    for name, t in [("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5)]:
        print(f"  {name}: {len(t)} rows")


if __name__ == "__main__":
    main()
