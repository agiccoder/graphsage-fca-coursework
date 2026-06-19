"""Phase 1 diagnostics for richer FCA conceptual scaling (no neural training).

    python scripts/analyze_scaling_modes.py \
        --datasets cora citeseer pubmed \
        --modes binary_nonzero quantile_global topk_per_node quantile_topk \
        --k 128

For every (dataset x scaling-mode) this rebuilds the formal context and the
*selected* concepts (binarize -> prefilter -> mine -> score=support -> top-k),
then records structural diagnostics. **No GraphSAGE is trained** — this mirrors
the torch-light pattern of ``scripts/check_fca_membership.py`` so we can judge
whether richer scaling produces denser, more multi-attribute concepts before
spending any training budget (TZ section 7.1).

Output
------
    results/scaling_diagnostics.csv   one row per (dataset, mode, params) with the
    columns mandated by TZ section 6.

Acceptance (TZ 7.1): at least one mode per dataset should yield
``frac_multi_attr > 0.05`` OR a clearly higher mean intent size than
``binary_nonzero``. If none do, that is itself reported as evidence that richer
scaling does not cure context degeneracy.

Leakage note
------------
Quantile thresholds for ``quantile_global`` / ``quantile_topk`` are computed on
TRAIN rows only (passed via ``train_mask``). ``mean_purity`` / ``mean_lift`` use
``class_association`` which is TRAIN-only. Test labels never participate.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.fca.binarize import build_formal_context
from src.fca.concepts import (class_association, compute_extents,
                              score_concepts, select_top_k)
from src.fca.mining import mine_concepts, prefilter_attributes
from src.utils.config import get, load_config, set_in
from src.utils.io import save_dataframe
from src.utils.logging import get_logger
from src.utils.paths import CONFIGS_DIR, DATASETS_DIR, RESULTS_DIR

logger = get_logger("analyze.scaling")

CORE_DATASETS = ["cora", "citeseer", "pubmed"]

# (mode, params) grid for Phase 1. quantile/topk recorded as explicit columns.
DEFAULT_GRID: list[tuple[str, dict]] = [
    ("binary_nonzero", {}),
    ("quantile_global", {"quantile": 0.75}),
    ("quantile_global", {"quantile": 0.90}),
    ("topk_per_node", {"topk": 10}),
    ("topk_per_node", {"topk": 20}),
    ("quantile_topk", {"quantile": 0.75, "topk": 10}),
    ("quantile_topk", {"quantile": 0.90, "topk": 10}),
]


def base_config(dataset: str, kind: str = "fca") -> dict:
    """Load ``{dataset}_sage_{kind}.yaml`` or compose a minimal fallback.

    Inlined (not imported from ``_ablation_common``) so this diagnostic does not
    pull in the training loop / torch_geometric just to read a config.
    """
    path = CONFIGS_DIR / "experiments" / f"{dataset}_sage_{kind}.yaml"
    if path.exists():
        return load_config(path)
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    set_in(cfg, "dataset.name", dataset)
    return cfg


def _load_data(cfg: dict):
    """Load the dataset, preferring the canonical ``load_dataset`` path.

    On hosts where ``torch_geometric`` is unavailable but ``torch`` and the raw
    Planetoid ``ind.*`` files are cached, fall back to a self-contained parser so
    the (purely structural) diagnostics can still run. The fallback reproduces the
    standard Kipf/PyG ``public`` split exactly and is only used for cora / citeseer
    / pubmed.
    """
    name = get(cfg, "dataset.name")
    root = get(cfg, "dataset.root", str(DATASETS_DIR))
    try:
        return load_dataset(
            name,
            root=root,
            planetoid_split=get(cfg, "dataset.planetoid_split", "public"),
            split_idx=int(get(cfg, "dataset.split_idx", 0)),
            to_undirected=get(cfg, "dataset.to_undirected", None),
        )
    except ModuleNotFoundError as exc:
        if "torch_geometric" not in str(exc):
            raise
        logger.warning("torch_geometric unavailable; using torch-free Planetoid "
                       "loader for %s (structural diagnostics only).", name)
        return _load_planetoid_torchfree(name, root)


def _load_planetoid_torchfree(name: str, root: str):
    """Parse cached ``ind.*`` Planetoid files into a :class:`GraphData`.

    Uses the canonical Kipf parsing (scipy sparse + networkx) and the standard
    ``public`` split: train = first ``len(y)`` labeled nodes, val = next 500,
    test = ``test.index`` range. CiteSeer's isolated test nodes are zero-padded.
    """
    import pickle
    from pathlib import Path

    import networkx as nx
    import scipy.sparse as sp
    import torch

    from src.data.types import GraphData, masks_from_indices

    pyg_dir = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name.lower()]
    key = name.lower()
    raw = Path(root) / "Planetoid" / pyg_dir / "raw"

    def _load(part: str):
        with open(raw / f"ind.{key}.{part}", "rb") as f:
            return pickle.load(f, encoding="latin1")

    x, y, tx, ty, allx, ally, graph = (
        _load("x"), _load("y"), _load("tx"), _load("ty"),
        _load("allx"), _load("ally"), _load("graph"),
    )
    test_idx_reorder = [int(line.strip())
                        for line in open(raw / f"ind.{key}.test.index")]
    test_idx_range = np.sort(test_idx_reorder)

    if key == "citeseer":  # isolated nodes -> zero-pad to a contiguous range
        full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1]))
        tx_ext[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_ext
        ty_ext = np.zeros((len(full), y.shape[1]))
        ty_ext[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = list(test_idx_range)
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
    x_raw = torch.tensor(np.asarray(features.todense()), dtype=torch.float)
    y_t = torch.tensor(np.asarray(labels).argmax(axis=1), dtype=torch.long)
    n = x_raw.size(0)
    train_mask, val_mask, test_mask = masks_from_indices(
        n, torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(idx_test))
    return GraphData(name=key, edge_index=edge_index, x_raw=x_raw, y=y_t,
                     train_mask=train_mask, val_mask=val_mask,
                     test_mask=test_mask, metadata={"loader": "torchfree_planetoid"})


def _build_concepts_for_mode(data, fca_cfg: dict, mode: str, mode_params: dict,
                             k: int, seed: int):
    """binarize(mode) -> prefilter -> mine -> score(support) -> top-k."""
    context = build_formal_context(
        data.x_raw, mode=mode, params=dict(mode_params),
        edge_index=data.edge_index, train_mask=data.train_mask,
    )
    if context.num_attributes == 0:
        return context, []
    context = prefilter_attributes(
        context,
        min_support=get(fca_cfg, "min_support", 0.01),
        max_support=get(fca_cfg, "max_support", 0.5),
        max_attributes=int(get(fca_cfg, "max_attributes", 512)),
        rank=get(fca_cfg, "attr_rank", "support"),
    )
    concepts = mine_concepts(
        context,
        strategy=get(fca_cfg, "strategy", "both"),
        min_support=get(fca_cfg, "concept_min_support", get(fca_cfg, "min_support", 0.01)),
        max_support=get(fca_cfg, "concept_max_support", get(fca_cfg, "max_support", 0.6)),
        object_sample=int(get(fca_cfg, "object_sample", 2000)),
        backend=get(fca_cfg, "backend", "fallback"),
        seed=seed,
    )
    if not concepts:
        return context, []
    concepts = score_concepts(concepts, context, scorer="support", seed=seed)
    y = data.y.cpu().numpy()
    train = data.train_mask.cpu().numpy()
    concepts = class_association(concepts, context, y, train, data.num_classes)
    concepts = select_top_k(concepts, k)
    return context, concepts


def diagnose(dataset: str, mode: str, mode_params: dict, k: int,
             seed: int) -> dict | None:
    cfg = base_config(dataset, "fca")
    fca_cfg = get(cfg, "features.fca", {})
    data = _load_data(cfg)
    context, concepts = _build_concepts_for_mode(data, fca_cfg, mode,
                                                 mode_params, k, seed)
    num_nodes = int(data.num_nodes)
    num_features = int(data.num_features)

    # Context density / active-attrs measured on the *pre-selection* context.
    B = context.incidence
    if B.size:
        active = B.sum(axis=1)
        ctx_density = float(B.mean())
        mean_active = float(active.mean())
        median_active = float(np.median(active))
    else:
        ctx_density = mean_active = median_active = 0.0

    if not concepts:
        logger.warning("[%s/%s %s] no concepts; recording degenerate row.",
                       dataset, mode, mode_params)
        row = _empty_row(dataset, mode, mode_params, num_nodes, num_features,
                         ctx_density, mean_active, median_active)
        return row

    intent = np.array([c.intent_size for c in concepts])
    extent = np.array([c.extent_size for c in concepts])
    purity = np.array([c.purity for c in concepts])
    lift = np.array([c.lift for c in concepts])
    ext = compute_extents(concepts, context)
    node_cov = float((ext.sum(axis=1) > 0).mean())

    logger.info("[%s/%s %s] concepts=%d mean_intent=%.3f frac_multi=%.3f cov=%.3f",
                dataset, mode, mode_params, len(concepts), intent.mean(),
                (intent >= 2).mean(), node_cov)

    return {
        "dataset": dataset,
        "binarize_mode": mode,
        "quantile": mode_params.get("quantile", ""),
        "topk": mode_params.get("topk", ""),
        "smooth_alpha": mode_params.get("smooth_alpha", ""),
        "num_nodes": num_nodes,
        "num_features": num_features,
        "context_density": round(ctx_density, 6),
        "mean_active_attrs_per_node": round(mean_active, 4),
        "median_active_attrs_per_node": round(median_active, 4),
        "num_concepts": len(concepts),
        "mean_intent_size": round(float(intent.mean()), 4),
        "median_intent_size": float(np.median(intent)),
        "max_intent_size": int(intent.max()),
        "n_single_attr": int((intent == 1).sum()),
        "n_multi_attr": int((intent >= 2).sum()),
        "frac_multi_attr": round(float((intent >= 2).mean()), 4),
        "mean_extent_size": round(float(extent.mean()), 4),
        "median_extent_size": float(np.median(extent)),
        "node_coverage": round(node_cov, 4),
        "mean_purity": round(float(purity.mean()), 4),
        "mean_lift": round(float(lift.mean()), 4),
    }


def _empty_row(dataset, mode, mode_params, num_nodes, num_features,
               ctx_density, mean_active, median_active) -> dict:
    return {
        "dataset": dataset, "binarize_mode": mode,
        "quantile": mode_params.get("quantile", ""),
        "topk": mode_params.get("topk", ""),
        "smooth_alpha": mode_params.get("smooth_alpha", ""),
        "num_nodes": num_nodes, "num_features": num_features,
        "context_density": round(ctx_density, 6),
        "mean_active_attrs_per_node": round(mean_active, 4),
        "median_active_attrs_per_node": round(median_active, 4),
        "num_concepts": 0, "mean_intent_size": 0.0, "median_intent_size": 0.0,
        "max_intent_size": 0, "n_single_attr": 0, "n_multi_attr": 0,
        "frac_multi_attr": 0.0, "mean_extent_size": 0.0,
        "median_extent_size": 0.0, "node_coverage": 0.0,
        "mean_purity": 0.0, "mean_lift": 0.0,
    }


def _select_grid(modes: list[str] | None) -> list[tuple[str, dict]]:
    if not modes:
        return DEFAULT_GRID
    wanted = set(modes)
    grid = [(m, p) for (m, p) in DEFAULT_GRID if m in wanted]
    # Allow modes named but absent from DEFAULT_GRID (e.g. graph_smoothed_topk).
    for m in modes:
        if m == "graph_smoothed_topk" and not any(g[0] == m for g in grid):
            grid.append((m, {"smooth_alpha": 0.5, "topk": 10}))
    return grid


def _acceptance(df: pd.DataFrame) -> None:
    """Log the TZ 7.1 acceptance verdict per dataset."""
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        base = sub[sub["binarize_mode"] == "binary_nonzero"]
        base_intent = float(base["mean_intent_size"].iloc[0]) if not base.empty else 0.0
        rich = sub[sub["binarize_mode"] != "binary_nonzero"]
        passes = rich[(rich["frac_multi_attr"] > 0.05) |
                      (rich["mean_intent_size"] > base_intent + 1e-6)]
        if not passes.empty:
            best = passes.sort_values("frac_multi_attr", ascending=False).iloc[0]
            logger.info("[%s] ACCEPT: %s (frac_multi=%.3f, mean_intent=%.3f vs base %.3f)",
                        ds, best["binarize_mode"], best["frac_multi_attr"],
                        best["mean_intent_size"], base_intent)
        else:
            logger.warning("[%s] NO richer mode beats binary_nonzero structure "
                           "(base mean_intent=%.3f) -> scaling does not cure degeneracy.",
                           ds, base_intent)


def main() -> None:
    ap = argparse.ArgumentParser(description="FCA richer-scaling diagnostics (no training).")
    ap.add_argument("--datasets", nargs="*", default=CORE_DATASETS)
    ap.add_argument("--modes", nargs="*", default=None,
                    help="Subset of modes; default = full Phase-1 grid.")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    grid = _select_grid(args.modes)
    rows = []
    for ds in args.datasets:
        for mode, params in grid:
            try:
                rec = diagnose(ds, mode, params, args.k, args.seed)
            except Exception as exc:  # one bad combo must not sink the rest
                logger.exception("[%s/%s %s] failed: %s", ds, mode, params, exc)
                continue
            if rec is not None:
                rows.append(rec)

    if not rows:
        logger.warning("No diagnostics produced.")
        return
    df = pd.DataFrame(rows)
    save_dataframe(df, RESULTS_DIR / "scaling_diagnostics.csv")
    logger.info("Wrote results/scaling_diagnostics.csv (%d rows).", len(df))
    _acceptance(df)


if __name__ == "__main__":
    main()
