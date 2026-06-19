"""Aggregate per-seed results into mean/std tables, rankings and deltas.

Reads ``results/per_seed_results.csv`` and writes:
    results/main_results.csv        (one row per *config*: mean/std over seeds)
    results/ranking_by_dataset.csv  (configs ranked within each dataset)
    results/deltas.csv              (delta vs SAGE(raw), MLP(raw), K-matched SVD)
    results/duplicate_runs.csv      (config identities with repeated seeds /
                                     multiple experiment names — should be empty)

Configs are identified by their hyperparameters (NOT by experiment name), so two
runs of the same config under different names merge into one row instead of
producing phantom duplicates (Task A1). The SVD control is matched by *added
dimensionality* so FCA-K is compared to SVD-K, not to an arbitrary SVD (Task B).

The config identity now includes the *binarisation* (mode + params hash): two FCA
runs that share every hyperparameter but discretise raw features differently are
distinct experiments and must NOT be merged into one inflated seed group (Task B).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..utils.io import save_dataframe
from ..utils.logging import get_logger
from ..utils.paths import RESULTS_DIR

logger = get_logger("eval.aggregate")

# Config identity: everything that makes two runs "the same experiment".
# `experiment` is deliberately excluded; `added_dim` distinguishes SVD-32/64/...
# `binarize_mode`/`binarize_params` distinguish e.g. binary_nonzero vs quantile.
KEY_COLS = ["dataset", "model", "variant", "hidden_channels", "num_layers",
            "dropout", "aggr", "lr", "weight_decay", "k_concepts", "scorer",
            "membership", "min_intent_size", "binarize_mode", "binarize_params",
            "added_dim"]
METRIC_COLS = ["test_accuracy", "test_macro_f1", "val_accuracy", "val_macro_f1",
               "low_degree_accuracy", "low_degree_macro_f1",
               "bucket_low_accuracy", "bucket_medium_accuracy", "bucket_high_accuracy",
               "bucket_low_macro_f1", "bucket_medium_macro_f1", "bucket_high_macro_f1",
               "train_time_sec", "best_epoch", "total_dim", "num_concepts",
               "node_coverage", "mean_intent_size", "mean_extent_size"]


def load_per_seed(path=None) -> pd.DataFrame:
    path = path or (RESULTS_DIR / "per_seed_results.csv")
    df = pd.read_csv(path)
    for c in KEY_COLS + METRIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by configuration and compute per-metric mean/std and seed count."""
    work = df.copy()
    # NaN keys (e.g. scorer for raw rows) become the literal "na" so they group.
    work[KEY_COLS] = work[KEY_COLS].astype(object).where(df[KEY_COLS].notna(), "na")
    metrics = [m for m in METRIC_COLS if m in work.columns]
    g = work.groupby(KEY_COLS, dropna=False)
    mean = g[metrics].mean().add_suffix("_mean")
    std = g[metrics].std(ddof=0).add_suffix("_std")
    n = g.size().rename("n_seeds")
    label = g["experiment"].first() if "experiment" in work.columns else None
    out = pd.concat([mean, std, n] + ([label] if label is not None else []), axis=1)
    out = out.reset_index()
    # `added_dim` came in as an object key; restore a numeric copy for matching.
    out["added_dim"] = pd.to_numeric(out["added_dim"], errors="coerce")
    return out.sort_values(["dataset", "test_accuracy_mean"],
                           ascending=[True, False]).reset_index(drop=True)


def _baseline(grp: pd.DataFrame, *, models, variant, prefer_aggr=None) -> pd.Series | None:
    sub = grp[(grp["variant"] == variant) & (grp["model"].isin(models))]
    if prefer_aggr is not None:
        pref = sub[sub["aggr"].astype(str) == prefer_aggr]
        if len(pref):
            sub = pref
    if not len(sub):
        return None
    return sub.sort_values("test_accuracy_mean", ascending=False).iloc[0]


def _svd_matched(grp: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    """SVD control with the same added dimensionality as ``row`` (K-matched)."""
    if pd.isna(row.get("added_dim")):
        return None
    svd = grp[(grp["variant"] == "svd_control")
              & (np.round(grp["added_dim"]) == round(float(row["added_dim"])))]
    return svd.iloc[0] if len(svd) else None


def _delta(row, base, col) -> float:
    if base is None or col not in row or pd.isna(row[col]):
        return float("nan")
    return round(float(row[col] - base[col]), 6)


def compute_deltas(main: pd.DataFrame) -> pd.DataFrame:
    """Delta columns vs SAGE(raw,mean), MLP(raw) and the K-matched SVD control."""
    rows = []
    for dataset, grp in main.groupby("dataset"):
        sage_raw = _baseline(grp, models=["graphsage", "sage"], variant="raw",
                             prefer_aggr="mean")
        mlp_raw = _baseline(grp, models=["mlp", "logreg"], variant="raw")
        for _, r in grp.iterrows():
            svd_k = _svd_matched(grp, r)
            rows.append({
                "dataset": dataset, "experiment": r.get("experiment"),
                "model": r["model"], "variant": r["variant"],
                "added_dim": r.get("added_dim"), "k_concepts": r.get("k_concepts"),
                "scorer": r.get("scorer"), "membership": r.get("membership"),
                "test_accuracy_mean": r["test_accuracy_mean"],
                "test_macro_f1_mean": r["test_macro_f1_mean"],
                "d_acc_vs_sage_raw": _delta(r, sage_raw, "test_accuracy_mean"),
                "d_mf1_vs_sage_raw": _delta(r, sage_raw, "test_macro_f1_mean"),
                "d_acc_vs_mlp_raw": _delta(r, mlp_raw, "test_accuracy_mean"),
                "d_mf1_vs_mlp_raw": _delta(r, mlp_raw, "test_macro_f1_mean"),
                "d_acc_vs_svd_kmatched": _delta(r, svd_k, "test_accuracy_mean"),
                "d_mf1_vs_svd_kmatched": _delta(r, svd_k, "test_macro_f1_mean"),
                "svd_matched_added_dim": None if svd_k is None else int(svd_k["added_dim"]),
            })
    return pd.DataFrame(rows)


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag config identities whose per-seed rows look duplicated.

    Two failure modes are reported (Task B):

    * **repeated seeds** — the same config (by ``KEY_COLS``) has the *same* seed
      logged more than once, i.e. a run was appended twice. ``n_rows`` then exceeds
      ``n_unique_seeds``.
    * **name collision** — one config identity is produced by *several* experiment
      names. Before binarisation was part of the key this silently merged distinct
      experiments (e.g. ``pubmed_sage_fca_soft`` + ``pubmed_sage_fca_quantile``)
      into a single 10-seed group. With the key fixed this should be empty; if not,
      it surfaces genuinely redundant configs.

    Returns a DataFrame (possibly empty) written to ``results/duplicate_runs.csv``.
    """
    work = df.copy()
    for c in KEY_COLS:
        if c not in work.columns:
            work[c] = np.nan
    work[KEY_COLS] = work[KEY_COLS].astype(object).where(work[KEY_COLS].notna(), "na")
    out = []
    for key, grp in work.groupby(KEY_COLS, dropna=False):
        seeds = grp["seed"] if "seed" in grp.columns else pd.Series(range(len(grp)))
        n_rows = len(grp)
        n_unique_seeds = int(seeds.nunique())
        experiments = sorted(map(str, grp["experiment"].unique())) if "experiment" in grp.columns else []
        repeated_seed = n_rows > n_unique_seeds
        name_collision = len(experiments) > 1
        if not (repeated_seed or name_collision):
            continue
        rec = dict(zip(KEY_COLS, key if isinstance(key, tuple) else (key,)))
        rec.update({
            "n_rows": n_rows,
            "n_unique_seeds": n_unique_seeds,
            "experiments": "; ".join(experiments),
            "issue": ", ".join(
                ([f"repeated_seed(+{n_rows - n_unique_seeds})"] if repeated_seed else [])
                + (["name_collision"] if name_collision else [])),
        })
        out.append(rec)
    res = pd.DataFrame(out)
    if res.empty:
        logger.info("Duplicate check: no duplicated configs.")
    else:
        logger.warning("Duplicate check: %d suspect config(s) -> duplicate_runs.csv", len(res))
    return res


def ranking_by_dataset(main: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, grp in main.groupby("dataset"):
        g = grp.sort_values("test_accuracy_mean", ascending=False).copy()
        g["rank"] = range(1, len(g) + 1)
        cols = ["rank", "dataset", "experiment", "model", "variant", "added_dim",
                "k_concepts", "scorer", "membership", "test_accuracy_mean",
                "test_accuracy_std", "test_macro_f1_mean", "test_macro_f1_std", "n_seeds"]
        out.append(g[[c for c in cols if c in g.columns]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(per_seed_path=None) -> dict:
    df = load_per_seed(per_seed_path)
    main = aggregate(df)
    deltas = compute_deltas(main)
    ranking = ranking_by_dataset(main)
    duplicates = detect_duplicates(df)
    save_dataframe(main, RESULTS_DIR / "main_results.csv")
    save_dataframe(deltas, RESULTS_DIR / "deltas.csv")
    save_dataframe(ranking, RESULTS_DIR / "ranking_by_dataset.csv")
    save_dataframe(duplicates, RESULTS_DIR / "duplicate_runs.csv")
    logger.info("Aggregated %d runs -> %d configs across %d datasets.",
                len(df), len(main), main["dataset"].nunique())
    return {"main": main, "deltas": deltas, "ranking": ranking,
            "duplicates": duplicates}


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-seed results.")
    ap.add_argument("--per-seed", default=None, help="Path to per_seed_results.csv")
    args = ap.parse_args()
    run(args.per_seed)


if __name__ == "__main__":
    main()
