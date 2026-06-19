"""Degree-bucket structural analysis (H3 / Task F).

    python scripts/analyze_degree_buckets.py

Reads the aggregated results and writes:
    results/degree_bucket_results.csv          (long format: one row per bucket)
    figures/degree_bucket_accuracy.png         (raw vs FCA vs SVD per bucket)
    figures/degree_bucket_delta_vs_raw.png     (FCA/SVD minus raw, per bucket)
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.aggregate import run as aggregate_run
from src.utils.io import save_dataframe
from src.utils.logging import get_logger
from src.utils.paths import FIGURES_DIR, RESULTS_DIR

logger = get_logger("analyze.degree")
BUCKETS = ["low", "medium", "high"]


def _long_format(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in main.iterrows():
        if pd.isna(r.get("bucket_low_accuracy_mean")):
            continue
        for b in BUCKETS:
            rows.append({
                "dataset": r["dataset"], "experiment": r.get("experiment"),
                "model": r["model"], "variant": r["variant"],
                "added_dim": r.get("added_dim"),
                "bucket": b,
                "accuracy_mean": r.get(f"bucket_{b}_accuracy_mean"),
                "accuracy_std": r.get(f"bucket_{b}_accuracy_std"),
                "macro_f1_mean": r.get(f"bucket_{b}_macro_f1_mean"),
            })
    return pd.DataFrame(rows)


def _pick(main: pd.DataFrame, ds: str):
    grp = main[main["dataset"] == ds]
    raw = grp[(grp["model"].isin(["graphsage", "sage"])) & (grp["variant"] == "raw")
              & (grp["aggr"].astype(str) == "mean")]
    fca = grp[grp["variant"] == "fca_feat"].sort_values("test_accuracy_mean", ascending=False)
    svd = grp[grp["variant"] == "svd_control"].sort_values("test_accuracy_mean", ascending=False)
    out = {}
    if len(raw):
        out["SAGE(raw)"] = raw.iloc[0]
    if len(fca):
        out["SAGE+FCA"] = fca.iloc[0]
    if len(svd):
        out["SAGE+SVD"] = svd.iloc[0]
    return out


def plot_accuracy(main: pd.DataFrame) -> None:
    datasets = list(main["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5, 4 * len(datasets)), 4),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        picks = _pick(main, ds)
        width = 0.8 / max(len(picks), 1)
        for i, (name, r) in enumerate(picks.items()):
            vals = [r.get(f"bucket_{b}_accuracy_mean") for b in BUCKETS]
            ax.bar(np.arange(3) + i * width, vals, width, label=name)
        ax.set_xticks(np.arange(3) + width)
        ax.set_xticklabels(BUCKETS)
        ax.set_title(ds)
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=7)
    fig.suptitle("Accuracy by node-degree bucket")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "degree_bucket_accuracy.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote degree_bucket_accuracy.png")


def plot_delta(main: pd.DataFrame) -> None:
    datasets = list(main["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5, 4 * len(datasets)), 4),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        picks = _pick(main, ds)
        if "SAGE(raw)" not in picks:
            continue
        raw = picks["SAGE(raw)"]
        others = {k: v for k, v in picks.items() if k != "SAGE(raw)"}
        width = 0.8 / max(len(others), 1)
        for i, (name, r) in enumerate(others.items()):
            deltas = [(r.get(f"bucket_{b}_accuracy_mean") or 0) -
                      (raw.get(f"bucket_{b}_accuracy_mean") or 0) for b in BUCKETS]
            ax.bar(np.arange(3) + i * width, deltas, width, label=f"{name} − raw")
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_xticks(np.arange(3) + width / 2)
        ax.set_xticklabels(BUCKETS)
        ax.set_title(ds)
        ax.set_ylabel("Δ accuracy vs raw")
        ax.legend(fontsize=7)
    fig.suptitle("Δ accuracy vs SAGE(raw) by degree bucket")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "degree_bucket_delta_vs_raw.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote degree_bucket_delta_vs_raw.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    main_df = aggregate_run()["main"]
    long = _long_format(main_df)
    if long.empty:
        logger.warning("No degree-bucket metrics found; run experiments first.")
        return
    save_dataframe(long, RESULTS_DIR / "degree_bucket_results.csv")
    plot_accuracy(main_df)
    plot_delta(main_df)


if __name__ == "__main__":
    main()
