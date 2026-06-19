"""Figures for the richer-FCA scaling extension (Phase 2a / 2b / 3).

Reads ``results/per_seed_results.csv`` and writes three figures into ``figures/``:

    scaling_phase2a_accuracy.png  — grouped bars: richer FCA vs binary / SVD / raw
    scaling_delta_vs_binary.png   — Δ(richer − binary_nonzero) per config
    scaling_delta_vs_svd.png      — Δ(richer − K-matched SVD) per config

All accuracies are seed-averaged. Only the K=128 Phase-2a configurations are
shown (one bar group per dataset/mode/scorer) so the figures stay readable; the
full K-sweep lives in the markdown tables.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import RESULTS_DIR, FIGURES_DIR

PER_SEED = RESULTS_DIR / "per_seed_results.csv"
K = 128
MEMBERSHIP = "hard"

# (dataset, binarize_mode, label) for the Phase-2a configs at K=128.
CONFIGS = [
    ("cora",     "quantile_global",    "cora\nqglobal q0.9"),
    ("citeseer", "quantile_global",    "citeseer\nqglobal q0.9"),
    ("pubmed",   "quantile_global",    "pubmed\nqglobal q0.75"),
    ("pubmed",   "topk_per_node",      "pubmed\ntopk10"),
    ("pubmed",   "graph_smoothed_topk","pubmed\ngsmooth"),
    ("cora",     "graph_smoothed_topk","cora\ngsmooth"),
]
SCORER = "support"  # primary scorer (exists for every config incl. Phase 3)


def _load() -> pd.DataFrame:
    df = pd.read_csv(PER_SEED)
    if "binarize_mode" not in df.columns:
        df["binarize_mode"] = np.nan
    df["binarize_mode"] = df["binarize_mode"].fillna("binary_nonzero")
    if "min_intent_size" not in df.columns:
        df["min_intent_size"] = np.nan
    return df


def _is_sage(df):
    return df["model"].isin(["graphsage", "sage"])


def _mean(df, **f) -> float:
    sub = df
    for k, v in f.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return float("nan")
    return float(sub["test_accuracy"].astype(float).mean())


def _richer(df, ds, mode, scorer=SCORER) -> float:
    sub = df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
             & (df["k_concepts"] == K) & (df["scorer"] == scorer)
             & (df["membership"] == MEMBERSHIP) & (df["binarize_mode"] == mode)]
    return float(sub["test_accuracy"].astype(float).mean()) if not sub.empty else float("nan")


def _binary(df, ds, scorer=SCORER) -> float:
    mii = df["min_intent_size"].isna() | (df["min_intent_size"] == 1)
    sub = df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "fca_feat")
             & (df["k_concepts"] == K) & (df["scorer"] == scorer)
             & (df["membership"] == MEMBERSHIP)
             & (df["binarize_mode"] == "binary_nonzero") & mii]
    return float(sub["test_accuracy"].astype(float).mean()) if not sub.empty else float("nan")


def _svd(df, ds) -> float:
    ad = pd.to_numeric(df["added_dim"], errors="coerce").round()
    sub = df[(df["dataset"] == ds) & (df["variant"] == "svd_control") & (ad == K)]
    return float(sub["test_accuracy"].astype(float).mean()) if not sub.empty else float("nan")


def _raw(df, ds) -> float:
    sub = df[(df["dataset"] == ds) & _is_sage(df) & (df["variant"] == "raw")]
    if "aggr" in df.columns:
        pref = sub[sub["aggr"].astype(str) == "mean"]
        if not pref.empty:
            sub = pref
    return float(sub["test_accuracy"].astype(float).mean()) if not sub.empty else float("nan")


def main() -> None:
    df = _load()
    labels, richer, binary, svd, raw = [], [], [], [], []
    for ds, mode, label in CONFIGS:
        r = _richer(df, ds, mode)
        if np.isnan(r):
            continue
        labels.append(label)
        richer.append(r)
        binary.append(_binary(df, ds))
        svd.append(_svd(df, ds))
        raw.append(_raw(df, ds))

    x = np.arange(len(labels))
    w = 0.2

    # --- Figure 1: grouped accuracy bars ---
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5 * w, richer, w, label="richer FCA", color="#2b8cbe")
    ax.bar(x - 0.5 * w, binary, w, label="binary_nonzero FCA", color="#a6bddb")
    ax.bar(x + 0.5 * w, svd, w, label="K-SVD (128)", color="#fdae6b")
    ax.bar(x + 1.5 * w, raw, w, label="raw SAGE", color="#bdbdbd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.65, 0.83)
    ax.set_title("Richer FCA scaling vs controls (K=128, scorer=support)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_phase2a_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: delta vs binary ---
    d_bin = np.array(richer) - np.array(binary)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2ca25f" if d > 0 else "#de2d26" for d in d_bin]
    ax.bar(x, d_bin, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ accuracy (richer − binary_nonzero)")
    ax.set_title("Richer scaling improvement over binary_nonzero FCA")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_delta_vs_binary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: delta vs SVD ---
    d_svd = np.array(richer) - np.array(svd)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2ca25f" if d > 0 else "#de2d26" for d in d_svd]
    ax.bar(x, d_svd, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ accuracy (richer − K-SVD)")
    ax.set_title("Richer scaling vs K-matched SVD control (success = above 0)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_delta_vs_svd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Wrote 3 figures to", FIGURES_DIR)
    for lab, r, b, s in zip(labels, richer, binary, svd):
        print(f"  {lab.replace(chr(10),' '):28s} richer={r:.4f} dbin={r-b:+.4f} dsvd={r-s:+.4f}")


if __name__ == "__main__":
    main()
