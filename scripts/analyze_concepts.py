"""Concept-structure analysis (Task G1 + G3).

    python scripts/analyze_concepts.py

Reads artifacts/concepts/<dataset>_concepts.csv and writes:
    results/concept_statistics.csv               (intent/extent size, coverage)
    results/top_concepts_clean.csv               (per class, no class -1)
    figures/concept_intent_size_distribution.png
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import save_dataframe
from src.utils.logging import get_logger
from src.utils.paths import CONCEPTS_DIR, FIGURES_DIR, RESULTS_DIR

logger = get_logger("analyze.concepts")


def _dataset_concept_files():
    return sorted(CONCEPTS_DIR.glob("*_concepts.csv"))


def _coverage(dataset: str) -> float:
    meta = CONCEPTS_DIR / f"{dataset}_fca_meta.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("coverage", {}).get("node_coverage", float("nan"))
    return float("nan")


def concept_statistics() -> pd.DataFrame:
    rows = []
    for f in _dataset_concept_files():
        ds = f.stem.replace("_concepts", "")
        df = pd.read_csv(f)
        if df.empty:
            continue
        intent = df["intent_size"].to_numpy()
        extent = df["extent_size"].to_numpy()
        rows.append({
            "dataset": ds,
            "num_concepts": len(df),
            "mean_intent_size": round(float(intent.mean()), 3),
            "median_intent_size": float(np.median(intent)),
            "max_intent_size": int(intent.max()),
            "n_single_attr": int((intent == 1).sum()),
            "n_multi_attr": int((intent >= 2).sum()),
            "frac_multi_attr": round(float((intent >= 2).mean()), 3),
            "mean_extent_size": round(float(extent.mean()), 1),
            "node_coverage": round(float(_coverage(ds)), 4),
        })
    return pd.DataFrame(rows)


def top_concepts_clean(top_n: int = 5) -> pd.DataFrame:
    rows = []
    cols = ["support", "purity", "lift", "intent_size", "extent_size",
            "selection_score", "attributes"]
    for f in _dataset_concept_files():
        ds = f.stem.replace("_concepts", "")
        df = pd.read_csv(f)
        if "dominant_class" not in df.columns:
            continue
        df = df[df["dominant_class"] >= 0]
        for cls in sorted(df["dominant_class"].unique()):
            sub = df[df["dominant_class"] == cls].sort_values(
                ["purity", "lift", "support"], ascending=False).head(top_n)
            for _, r in sub.iterrows():
                rows.append({"dataset": ds, "class": int(cls),
                             **{c: r.get(c) for c in cols}})
    return pd.DataFrame(rows)


def plot_intent_distribution() -> None:
    files = _dataset_concept_files()
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(max(5, 4 * len(files)), 3.5),
                             squeeze=False)
    for ax, f in zip(axes[0], files):
        ds = f.stem.replace("_concepts", "")
        df = pd.read_csv(f)
        intent = df["intent_size"].to_numpy()
        ax.hist(intent, bins=range(1, int(intent.max()) + 2), color="#4C72B0",
                align="left", rwidth=0.85)
        ax.set_title(ds)
        ax.set_xlabel("intent size (# attributes)")
        ax.set_ylabel("# concepts")
    fig.suptitle("Selected-concept intent-size distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "concept_intent_size_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote concept_intent_size_distribution.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stats = concept_statistics()
    if stats.empty:
        logger.warning("No concept CSVs found; build FCA features first.")
        return
    save_dataframe(stats, RESULTS_DIR / "concept_statistics.csv")
    save_dataframe(top_concepts_clean(), RESULTS_DIR / "top_concepts_clean.csv")
    plot_intent_distribution()
    logger.info("Concept analysis complete: %d datasets.", len(stats))


if __name__ == "__main__":
    main()
