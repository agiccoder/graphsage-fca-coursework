"""Integration variants: assemble model-ready features from raw + FCA concepts.

Variants
--------
raw            : x_raw only (baseline).
fca_feat       : [x_raw || hard/soft concept membership]   (FCA_FEAT).
fca_group      : [x_raw || grouped concept features]        (FCA_GROUP).
fca_only       : concept membership only (ablation).
fca_pattern    : [x_raw || interval pattern membership]     (pattern structures).
svd_control    : [x_raw || TruncatedSVD(x_raw, k)]          (equal-dim control).
fca_aug_graph  : add concept nodes + node->concept edges    (FCA_AUG_GRAPH).

All FCA computation is label-free except optionally the *scorer*, which -- when
supervised -- is restricted to the training mask. Test labels never participate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from ..utils.config import get
from ..utils.io import save_dataframe, save_json, save_torch
from ..utils.logging import get_logger
from ..utils.paths import CONCEPTS_DIR, FEATURES_DIR
from .binarize import FormalContext, _normalized_adj_smooth, build_formal_context
from .concepts import (Concept, class_association, compute_extents,
                       score_concepts, select_top_k)
from .features import build_membership, group_concepts, group_features
from .mining import mine_concepts, prefilter_attributes
from .patterns import (annotate_patterns, build_pattern_membership,
                       mine_interval_patterns, pattern_coverage,
                       patterns_dataframe, score_patterns, select_top_patterns)

logger = get_logger("fca.integrate")


@dataclass
class FeatureBundle:
    """Everything the training pipeline needs for one feature variant."""

    variant: str
    x: torch.Tensor                       # [num_nodes, D] model input features
    edge_index: torch.Tensor              # possibly augmented
    num_eval_nodes: int                   # original node count (for masks/labels)
    raw_dim: int
    added_dim: int
    concepts: list[Any] = field(default_factory=list)
    concepts_df: Optional[pd.DataFrame] = None
    membership: Optional[np.ndarray] = None
    context: Optional[FormalContext] = None
    coverage: dict = field(default_factory=dict)

    def save(self, dataset_name: str) -> None:
        """Persist required artifacts (concept CSV + FCA feature tensor)."""
        if self.concepts_df is not None:
            save_dataframe(self.concepts_df, CONCEPTS_DIR / f"{dataset_name}_concepts.csv")
        payload = {
            "variant": self.variant,
            "x": self.x.cpu(),
            "raw_dim": self.raw_dim,
            "added_dim": self.added_dim,
            "membership": None if self.membership is None else torch.from_numpy(self.membership),
            "coverage": self.coverage,
        }
        save_torch(payload, FEATURES_DIR / f"{dataset_name}_x_fca.pt")
        save_json({"variant": self.variant, "raw_dim": self.raw_dim,
                   "added_dim": self.added_dim, "coverage": self.coverage},
                  CONCEPTS_DIR / f"{dataset_name}_fca_meta.json")


def _coverage_stats(concepts: list[Concept], context: FormalContext) -> dict:
    if not concepts:
        return {"num_concepts": 0, "node_coverage": 0.0,
                "avg_concepts_per_node": 0.0, "sparsity": 1.0,
                "mean_intent_size": 0.0, "median_intent_size": 0.0,
                "n_single_attr": 0, "n_multi_attr": 0, "mean_extent_size": 0.0}
    ext = compute_extents(concepts, context)
    per_node = ext.sum(axis=1)
    intent_sizes = np.array([c.intent_size for c in concepts])
    extent_sizes = np.array([c.extent_size for c in concepts])
    return {
        "num_concepts": len(concepts),
        "node_coverage": float((per_node > 0).mean()),
        "avg_concepts_per_node": float(per_node.mean()),
        "sparsity": float(1.0 - ext.mean()),
        "mean_intent_size": round(float(intent_sizes.mean()), 4),
        "median_intent_size": float(np.median(intent_sizes)),
        "n_single_attr": int((intent_sizes == 1).sum()),
        "n_multi_attr": int((intent_sizes >= 2).sum()),
        "mean_extent_size": round(float(extent_sizes.mean()), 4),
    }


def _build_concepts(data, fca_cfg: dict, seed: int):
    """Run binarize -> prefilter -> mine -> score -> annotate -> select."""
    context = build_formal_context(
        data.x_raw,
        mode=get(fca_cfg, "binarize_mode", "binary_nonzero"),
        params=get(fca_cfg, "binarize_params", {}),
        edge_index=data.edge_index,
        train_mask=data.train_mask,  # train-only quantile thresholds (leakage-safe)
    )
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
    min_intent = int(get(fca_cfg, "min_intent_size", 1))
    if min_intent > 1:
        before = len(concepts)
        concepts = [c for c in concepts if c.intent_size >= min_intent]
        logger.info("min_intent_size=%d filtered concepts %d -> %d.",
                    min_intent, before, len(concepts))
    if not concepts:
        logger.warning("No concepts after mining/min_intent_size; FCA features empty.")
        return context, []

    y_np = data.y.cpu().numpy()
    train_np = data.train_mask.cpu().numpy()
    num_classes = data.num_classes
    concepts = score_concepts(
        concepts, context,
        scorer=get(fca_cfg, "scorer", "support"),
        y=y_np, train_mask=train_np, num_classes=num_classes,
        params=get(fca_cfg, "scorer_params", {}), seed=seed,
    )
    # Always annotate class association (TRAIN-only) for the interpretability CSV.
    concepts = class_association(concepts, context, y_np, train_np, num_classes)
    concepts = select_top_k(concepts, int(get(fca_cfg, "k_concepts", 128)))
    logger.info("Selected %d concepts (scorer=%s).", len(concepts),
                get(fca_cfg, "scorer", "support"))
    return context, concepts


def _concepts_dataframe(concepts: list[Concept]) -> pd.DataFrame:
    return pd.DataFrame([c.to_row() for c in concepts])


def _augment_graph(data, concepts, context):
    """Add one node per concept and node<->concept edges (FCA_AUG_GRAPH)."""
    n, f = data.num_nodes, data.num_features
    k = len(concepts)
    ext = compute_extents(concepts, context)  # [N, K] bool
    raw = data.x_raw
    concept_feats = torch.zeros((k, f), dtype=raw.dtype)
    src, dst = [], []
    for j in range(k):
        members = np.flatnonzero(ext[:, j])
        if members.size:
            concept_feats[j] = raw[members].mean(dim=0)
        cnode = n + j
        for u in members.tolist():
            src += [u, cnode]
            dst += [cnode, u]
    x_aug = torch.cat([raw, concept_feats], dim=0)
    extra = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.cat([data.edge_index, extra], dim=1)
    return x_aug, edge_index


def build_features(data, features_cfg: dict, seed: int = 0,
                   save_as: Optional[str] = None) -> FeatureBundle:
    """Build a :class:`FeatureBundle` for the configured variant."""
    variant = get(features_cfg, "variant", "raw")
    raw = data.x_raw
    raw_dim = data.num_features
    n = data.num_nodes

    if variant == "raw":
        return FeatureBundle("raw", raw, data.edge_index, n, raw_dim, 0)

    if variant == "svd_control":
        from sklearn.decomposition import TruncatedSVD

        k = get(features_cfg, "svd.n_components", None) or get(features_cfg, "fca.k_concepts", 128)
        k = int(min(k, raw_dim - 1))
        svd = TruncatedSVD(n_components=k, random_state=seed)
        comp = svd.fit_transform(raw.cpu().numpy()).astype(np.float32)
        x = torch.cat([raw, torch.from_numpy(comp)], dim=1)
        return FeatureBundle("svd_control", x, data.edge_index, n, raw_dim, k,
                             coverage={"svd_explained_var": float(svd.explained_variance_ratio_.sum())})

    # ---- Pattern-structure variant ----
    fca_cfg = get(features_cfg, "fca", {})
    if variant == "fca_pattern":
        pattern_params = dict(get(fca_cfg, "pattern_params", {}))
        pattern_params.setdefault("min_support", get(fca_cfg, "concept_min_support", get(fca_cfg, "min_support", 0.01)))
        pattern_params.setdefault("max_support", get(fca_cfg, "concept_max_support", get(fca_cfg, "max_support", 0.6)))
        pattern_params.setdefault("max_features", int(get(fca_cfg, "max_attributes", 512)))
        pattern_source = str(pattern_params.get("source", "raw"))
        pattern_x = data.x_raw
        if pattern_source == "graph_smoothed":
            raw_np = raw.detach().cpu().numpy().astype(np.float32)
            smoothed = _normalized_adj_smooth(
                raw_np, data.edge_index, data.num_nodes,
                hops=int(pattern_params.get("hops", 1)))
            alpha = float(pattern_params.get("smooth_alpha", 0.5))
            pattern_np = alpha * raw_np + (1.0 - alpha) * smoothed
            pattern_x = torch.from_numpy(pattern_np.astype(np.float32))
        patterns, meta = mine_interval_patterns(
            pattern_x, train_mask=data.train_mask, params=pattern_params, seed=seed)
        meta["pattern_source"] = pattern_source
        y_np = data.y.cpu().numpy()
        train_np = data.train_mask.cpu().numpy()
        patterns = score_patterns(
            patterns, pattern_x, scorer=get(fca_cfg, "scorer", "support"),
            y=y_np, train_mask=train_np, num_classes=data.num_classes,
            params=get(fca_cfg, "scorer_params", {}))
        patterns = annotate_patterns(patterns, pattern_x, y_np, train_np, data.num_classes)
        patterns = select_top_patterns(patterns, int(get(fca_cfg, "k_concepts", 128)))
        membership = build_pattern_membership(
            patterns, pattern_x, mode=get(fca_cfg, "membership", "hard"))
        block = torch.from_numpy(membership)
        x = torch.cat([raw, block], dim=1)
        coverage = pattern_coverage(patterns, membership, meta)
        concepts_df = patterns_dataframe(patterns) if patterns else None
        bundle = FeatureBundle("fca_pattern", x, data.edge_index, n, raw_dim,
                               membership.shape[1], patterns, concepts_df,
                               membership, None, coverage)
        logger.info("Variant=%s | x shape=%s | added_dim=%d | coverage=%.3f",
                    variant, tuple(bundle.x.shape), bundle.added_dim,
                    coverage.get("node_coverage", 0.0))
        if save_as:
            bundle.save(save_as)
        return bundle

    # ---- FCA variants ----
    context, concepts = _build_concepts(data, fca_cfg, seed)
    membership = build_membership(concepts, context, mode=get(fca_cfg, "membership", "hard"))
    coverage = _coverage_stats(concepts, context)
    concepts_df = _concepts_dataframe(concepts) if concepts else None

    if variant in ("fca_feat", "fca_only"):
        block = torch.from_numpy(membership)
        x = block if variant == "fca_only" else torch.cat([raw, block], dim=1)
        added = membership.shape[1]
        bundle = FeatureBundle(variant, x, data.edge_index, n, raw_dim, added,
                               concepts, concepts_df, membership, context, coverage)
    elif variant == "fca_group":
        num_groups = int(get(fca_cfg, "num_groups", 32))
        groups = group_concepts(concepts, num_groups)
        gfeat = group_features(membership, groups, num_groups,
                               agg=get(fca_cfg, "group_agg", "max"))
        x = torch.cat([raw, torch.from_numpy(gfeat)], dim=1)
        coverage["num_groups"] = int(gfeat.shape[1])
        bundle = FeatureBundle("fca_group", x, data.edge_index, n, raw_dim,
                               gfeat.shape[1], concepts, concepts_df, membership,
                               context, coverage)
    elif variant == "fca_aug_graph":
        x_aug, edge_index = _augment_graph(data, concepts, context)
        bundle = FeatureBundle("fca_aug_graph", x_aug, edge_index, n, raw_dim, 0,
                               concepts, concepts_df, membership, context, coverage)
    else:
        raise ValueError(f"Unknown feature variant '{variant}'.")

    logger.info("Variant=%s | x shape=%s | added_dim=%d | coverage=%.3f",
                variant, tuple(bundle.x.shape), bundle.added_dim,
                coverage.get("node_coverage", 0.0))
    if save_as:
        bundle.save(save_as)
    return bundle
