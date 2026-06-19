"""Generate all paper figures from the results CSVs and FCA artifacts.

    python scripts/make_figures.py

Produces (when the underlying data exists):
    figures/bar_accuracy.png        figures/bar_macro_f1.png
    figures/ablation_k_concepts.png figures/model_diagram.png
    figures/lattice_toy.png         figures/confusion_<exp>.png (best runs)
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils.logging import get_logger
from src.utils.paths import FIGURES_DIR, RESULTS_DIR

logger = get_logger("figures")

# Canonical comparable FCA arm (mirrors src/eval/report.py): GraphSAGE encoder,
# support scorer, hard membership, single-attribute concepts. Ablations vary one
# axis only; everything else stays pinned so figures compare like with like.
GRAPHSAGE_MODELS = ("graphsage", "sage")


def _filter(df: pd.DataFrame, **constraints) -> pd.DataFrame:
    """Row filter on per-seed data: column == value (scalars, lists, or numerics)."""
    sub = df
    for col, want in constraints.items():
        if col not in sub.columns:
            return sub.iloc[0:0]
        if isinstance(want, (list, tuple, set)):
            sub = sub[sub[col].astype(str).isin([str(w) for w in want])]
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            sub = sub[pd.to_numeric(sub[col], errors="coerce") == float(want)]
        else:
            sub = sub[sub[col].astype(str) == str(want)]
    return sub


def _config_label(row) -> str:
    return f"{row['model']}/{row['variant']}"


def plot_bars(main: pd.DataFrame, metric: str, out_name: str, title: str) -> None:
    """Readable bar chart from the cleaned final main table.

    The old figure used ``results/main_results.csv`` directly, which contains many
    ablation rows with identical labels such as ``graphsage/fca_feat`` and
    ``graphsage/svd_control``. For the paper figure we keep only the canonical
    GraphSAGE rows used in the main results table: raw, binary FCA, and K-matched
    SVD. This avoids duplicate-looking columns and matches Table 2 in the text.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if metric == "test_accuracy" and "test_acc_mean" in main.columns:
        mean_col, std_col = "test_acc_mean", "test_acc_std"
    if metric == "test_macro_f1" and "macro_f1_mean" in main.columns:
        mean_col, std_col = "macro_f1_mean", "macro_f1_std"
    if mean_col not in main.columns:
        logger.info("Metric %s unavailable for %s; skipping.", metric, out_name)
        return

    df = main.copy()
    keep = (
        (df["model"].astype(str) == "graphsage")
        & (df["variant"].astype(str).isin(["raw", "fca_binary_nonzero", "svd_control"]))
    )
    df = df[keep].copy()
    if df.empty:
        logger.info("No cleaned GraphSAGE rows for %s; skipping.", out_name)
        return
    label_map = {
        "raw": "Raw GraphSAGE",
        "fca_binary_nonzero": "Binary FCA",
        "svd_control": "SVD control",
    }
    order = ["raw", "fca_binary_nonzero", "svd_control"]
    datasets = [d for d in ["cora", "citeseer", "pubmed"] if d in set(df["dataset"])]
    x = np.arange(len(datasets))
    width = 0.24
    colors = {"raw": "#55A868", "fca_binary_nonzero": "#4C72B0", "svd_control": "#C44E52"}

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for offset_i, variant in enumerate(order):
        vals, errs = [], []
        for ds in datasets:
            row = df[(df["dataset"] == ds) & (df["variant"] == variant)]
            vals.append(float(row[mean_col].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row[std_col].iloc[0]) if (not row.empty and std_col in row.columns) else 0.0)
        ax.bar(x + (offset_i - 1) * width, vals, width, yerr=errs,
               label=label_map[variant], color=colors[variant], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() if d != "citeseer" else "CiteSeer" for d in datasets])
    ax.set_ylabel("Accuracy" if "accuracy" in metric else "Macro-F1")
    ax.set_ylim(0.0, 0.9 if "accuracy" in metric else 0.85)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def plot_k_ablation(per_seed: pd.DataFrame, metric: str, out_name: str) -> None:
    """One subplot per dataset; FCA_FEAT vs K-matched SVD across added dimension.

    The FCA arm is restricted to the canonical comparable config (GraphSAGE,
    support, hard, single-attribute) so each K point averages only over seeds of
    *one* config, not over a mix of scorers/membership/intent sizes.
    """
    base = per_seed[per_seed["model"].astype(str).isin(GRAPHSAGE_MODELS)].copy()
    fca = _filter(base, variant="fca_feat", scorer="support",
                  membership="hard", min_intent_size=1)
    svd = _filter(base, variant="svd_control")
    df = pd.concat([fca, svd], ignore_index=True).dropna(subset=["added_dim"])
    if df.empty or metric not in df.columns:
        logger.info("No K-sweep data for %s; skipping.", out_name)
        return
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5, 4 * len(datasets)), 4),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["dataset"] == ds]
        if sub["added_dim"].nunique() < 2:
            ax.set_visible(False)
            continue
        for variant, style in [("fca_feat", "o-"), ("svd_control", "s--")]:
            v = sub[sub["variant"] == variant]
            if v.empty:
                continue
            agg = v.groupby("added_dim")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["added_dim"], agg["mean"], yerr=agg["std"].fillna(0),
                        fmt=style, capsize=3,
                        label="FCA" if variant == "fca_feat" else "SVD (matched)")
        ax.set_xlabel("added dimension K")
        ax.set_ylabel(metric)
        ax.set_title(ds)
        ax.legend(fontsize=8)
    fig.suptitle(f"Effect of concept count K — {metric} (GraphSAGE · support · hard · intent=1)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def plot_scorer_ablation(per_seed: pd.DataFrame, out_name: str) -> None:
    """Grouped bars: test accuracy per concept scorer, per dataset.

    Pinned to GraphSAGE / hard membership / single-attribute / K=128 so the only
    varying axis is the scorer.
    """
    df = _filter(per_seed, model=GRAPHSAGE_MODELS, variant="fca_feat",
                 membership="hard", min_intent_size=1, k_concepts=128)
    df = df[df["scorer"].notna()].copy()
    if df.empty or df["scorer"].nunique() < 2:
        logger.info("No scorer-ablation data; skipping %s.", out_name)
        return
    datasets = sorted(df["dataset"].unique())
    scorers = sorted(df["scorer"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(datasets) * len(scorers)), 4))
    width = 0.8 / len(scorers)
    for i, s in enumerate(scorers):
        means = [df[(df["dataset"] == ds) & (df["scorer"] == s)]["test_accuracy"].mean()
                 for ds in datasets]
        ax.bar(np.arange(len(datasets)) + i * width, means, width, label=s)
    ax.set_xticks(np.arange(len(datasets)) + width * (len(scorers) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("test accuracy")
    ax.set_title("Concept-scorer ablation (GraphSAGE · hard · intent=1 · K=128)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def plot_membership_ablation(per_seed: pd.DataFrame, out_name: str) -> None:
    """Grouped bars: hard vs soft membership accuracy, per dataset (K=128).

    Pinned to GraphSAGE / support / single-attribute / K=128. NOTE: for datasets
    whose selected concepts are all single-attribute (intent=1), hard and soft are
    mathematically identical — the bars will coincide, which is the expected sanity
    outcome (see scripts/check_fca_membership.py).
    """
    df = _filter(per_seed, model=GRAPHSAGE_MODELS, variant="fca_feat",
                 scorer="support", min_intent_size=1, k_concepts=128)
    df = df[df["membership"].notna()].copy()
    if df.empty or df["membership"].nunique() < 2:
        logger.info("No membership-ablation data; skipping %s.", out_name)
        return
    datasets = sorted(df["dataset"].unique())
    modes = sorted(df["membership"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(datasets) * len(modes)), 4))
    width = 0.8 / len(modes)
    for i, m in enumerate(modes):
        means = [df[(df["dataset"] == ds) & (df["membership"] == m)]["test_accuracy"].mean()
                 for ds in datasets]
        ax.bar(np.arange(len(datasets)) + i * width, means, width, label=m)
    ax.set_xticks(np.arange(len(datasets)) + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("test accuracy")
    ax.set_title("Hard vs soft membership (GraphSAGE · support · intent=1 · K=128)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def plot_degree_bucket_delta(per_seed: pd.DataFrame, out_name: str) -> None:
    """Per-dataset Δ accuracy (best FCA − SAGE raw) across degree tertiles."""
    buckets = ["bucket_low_accuracy", "bucket_medium_accuracy", "bucket_high_accuracy"]
    if not all(b in per_seed.columns for b in buckets):
        logger.info("No degree-bucket columns; skipping %s.", out_name)
        return
    base = per_seed[per_seed["model"].astype(str).isin(GRAPHSAGE_MODELS)].copy()
    datasets = sorted(base["dataset"].unique())
    rows = {}
    for ds in datasets:
        raw = _filter(base[base["dataset"] == ds], variant="raw", aggr="mean")
        fca = _filter(base[base["dataset"] == ds], variant="fca_feat",
                      scorer="support", membership="hard", min_intent_size=1)
        if raw.empty or fca.empty:
            continue
        # pick the best FCA config by overall test accuracy
        best_dim = (fca.groupby("added_dim")["test_accuracy"].mean().idxmax()
                    if fca["added_dim"].notna().any() else None)
        fca_best = fca[fca["added_dim"] == best_dim] if best_dim is not None else fca
        rows[ds] = [fca_best[b].mean() - raw[b].mean() for b in buckets]
    if not rows:
        logger.info("No comparable raw/FCA pairs; skipping %s.", out_name)
        return
    labels = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(rows) * 3), 4))
    width = 0.8 / 3
    dss = list(rows.keys())
    for i, lbl in enumerate(labels):
        vals = [rows[ds][i] for ds in dss]
        ax.bar(np.arange(len(dss)) + i * width, vals, width, label=f"{lbl}-degree")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(np.arange(len(dss)) + width)
    ax.set_xticklabels(dss)
    ax.set_ylabel("Δ accuracy (FCA − raw)")
    ax.set_title("Degree-bucket Δ accuracy vs SAGE(raw)  (best FCA · support · hard · intent=1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def plot_intent_size_distribution(out_name: str = "concept_intent_size_distribution.png") -> None:
    """Histogram of selected-concept intent sizes per dataset (from concept CSVs)."""
    from src.utils.paths import CONCEPTS_DIR
    frames = {}
    for path in sorted(CONCEPTS_DIR.glob("*_concepts.csv")):
        ds = path.stem.replace("_concepts", "")
        df = pd.read_csv(path)
        if "intent_size" in df.columns and not df.empty:
            frames[ds] = df["intent_size"].dropna().astype(int)
    if not frames:
        logger.info("No concept CSVs with intent_size; skipping %s.", out_name)
        return
    datasets = list(frames.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5, 4 * len(datasets)), 4),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sizes = frames[ds]
        max_s = int(sizes.max())
        ax.hist(sizes, bins=range(1, max_s + 2), align="left", color="#4C72B0",
                edgecolor="white")
        ax.set_xlabel("intent size (# attributes)")
        ax.set_ylabel("# concepts")
        ax.set_title(f"{ds}  (single-attr: {int((sizes == 1).mean() * 100)}%)")
        ax.set_xticks(range(1, max_s + 1))
    fig.suptitle("Selected-concept intent-size distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def model_diagram(out_name: str = "model_diagram.png") -> None:
    """Readable two-row pipeline diagram for the paper text.

    The previous version was a single long row and became almost unreadable when
    scaled to page width. This layout separates the raw-feature path from the
    FCA/pattern-derived feature-construction path and joins them at concatenation.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axis("off")

    boxes = {
        "raw": (0.06, 0.68, 0.20, 0.16, "Node features\n$x$"),
        "scale": (0.06, 0.28, 0.20, 0.16, "FCA / pattern\ndescription"),
        "mine": (0.34, 0.28, 0.22, 0.16, "Select concepts\n/ patterns"),
        "membership": (0.64, 0.28, 0.24, 0.16, "Membership\nfeatures $z$"),
        "concat": (0.34, 0.68, 0.22, 0.16, "Concatenate\n$[x \\Vert z]$"),
        "sage": (0.64, 0.68, 0.24, 0.16, "GraphSAGE\nclassifier"),
    }

    def add_box(key: str, color: str = "#EAF0F8") -> None:
        x, y, w, h, text = boxes[key]
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                             linewidth=1.2, edgecolor="#333", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def center_right(key: str) -> tuple[float, float]:
        x, y, w, h, _ = boxes[key]
        return x + w, y + h / 2

    def center_left(key: str) -> tuple[float, float]:
        x, y, _, h, _ = boxes[key]
        return x, y + h / 2

    def top_center(key: str) -> tuple[float, float]:
        x, y, w, h, _ = boxes[key]
        return x + w / 2, y + h

    def bottom_center(key: str) -> tuple[float, float]:
        x, y, w, _, _ = boxes[key]
        return x + w / 2, y

    for key in boxes:
        add_box(key, "#EAF0F8" if key in {"raw", "concat", "sage"} else "#F4EBDD")

    arrows = [
        (center_right("raw"), center_left("concat")),
        (center_right("scale"), center_left("mine")),
        (center_right("mine"), center_left("membership")),
        (top_center("membership"), bottom_center("concat")),
        (center_right("concat"), center_left("sage")),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color="#333",
                                     connectionstyle="arc3,rad=0.0"))

    ax.text(0.16, 0.50, "construct extra\nfeatures", ha="center", va="center", fontsize=9, color="#555")
    ax.text(0.45, 0.91, "model input", ha="center", va="center", fontsize=9, color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.12, 0.96)
    ax.set_title("Raw + FCA/pattern-derived features for GraphSAGE", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def _all_concepts(B: np.ndarray):
    """Brute-force all formal concepts of a tiny context (for visualisation)."""
    from itertools import combinations
    m = B.shape[1]
    concepts = set()
    for r in range(m + 1):
        for cols in combinations(range(m), r):
            cols = list(cols)
            extent = np.ones(B.shape[0], bool) if not cols else B[:, cols].all(1)
            intent = tuple(np.flatnonzero(B[extent].all(0)).tolist()) if extent.any() else tuple(range(m))
            concepts.add((tuple(np.flatnonzero(extent).tolist()), intent))
    return sorted(concepts, key=lambda c: (len(c[1]), c[1]))


def lattice_toy(out_name: str = "lattice_toy.png") -> None:
    objects = ["cat", "dog", "bat", "whale", "shark"]
    attrs = ["fur", "fly", "aquatic", "milk"]
    B = np.array([
        [1, 0, 0, 1],  # cat
        [1, 0, 0, 1],  # dog
        [1, 1, 0, 1],  # bat
        [0, 0, 1, 1],  # whale
        [0, 0, 1, 0],  # shark
    ], dtype=bool)
    concepts = _all_concepts(B)
    idx = {c: i for i, c in enumerate(concepts)}
    levels: dict[int, list[int]] = {}
    for c in concepts:
        levels.setdefault(len(c[1]), []).append(idx[c])
    pos = {}
    for lvl, ids in levels.items():
        for j, i in enumerate(ids):
            pos[i] = (j - (len(ids) - 1) / 2.0, -lvl)

    def covers(a, b) -> bool:  # a is below b: extent(a) ⊂ extent(b)
        ea, eb = set(concepts[a][0]), set(concepts[b][0])
        if not ea < eb:
            return False
        for k in range(len(concepts)):
            if k in (a, b):
                continue
            ek = set(concepts[k][0])
            if ea < ek < eb:
                return False
        return True

    fig, ax = plt.subplots(figsize=(7, 6))
    for a in range(len(concepts)):
        for b in range(len(concepts)):
            if a != b and covers(a, b):
                xa, ya = pos[a]
                xb, yb = pos[b]
                ax.plot([xa, xb], [ya, yb], color="#999", zorder=1)
    for c, i in idx.items():
        x, y = pos[i]
        ax.scatter([x], [y], s=350, color="#4C72B0", zorder=2)
        intent_lbl = ",".join(attrs[k] for k in c[1]) or "⊤"
        ext_lbl = ",".join(objects[k] for k in c[0]) or "⊥"
        ax.annotate(f"{{{intent_lbl}}}\n[{ext_lbl}]", (x, y), fontsize=7,
                    ha="center", va="center", color="white", zorder=3)
    ax.axis("off")
    ax.set_title("Toy concept lattice (intent {attributes} / [extent objects])")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_name)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_diagram()
    lattice_toy()
    plot_intent_size_distribution()
    main_path = RESULTS_DIR / "final_table_main_v2.csv"
    per_seed_path = RESULTS_DIR / "per_seed_results.csv"
    if main_path.exists():
        main = pd.read_csv(main_path)
        plot_bars(main, "test_accuracy", "bar_accuracy.png", "Test accuracy by model/variant")
        plot_bars(main, "test_macro_f1", "bar_macro_f1.png", "Macro-F1 by model/variant")
    else:
        logger.info("results/final_table_main_v2.csv missing; run final table generation first for bar charts.")
    if per_seed_path.exists():
        per_seed = pd.read_csv(per_seed_path)
        plot_k_ablation(per_seed, "test_accuracy", "ablation_k_concepts_accuracy.png")
        plot_k_ablation(per_seed, "test_macro_f1", "ablation_k_concepts_macro_f1.png")
        plot_k_ablation(per_seed, "test_accuracy", "ablation_k_concepts.png")  # legacy alias
        plot_scorer_ablation(per_seed, "concept_scorer_ablation.png")
        plot_membership_ablation(per_seed, "membership_ablation_accuracy.png")
        plot_degree_bucket_delta(per_seed, "degree_bucket_delta_vs_raw.png")


if __name__ == "__main__":
    main()
