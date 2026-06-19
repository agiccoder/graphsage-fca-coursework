"""Formal Concept Analysis (FCA / АФП) preprocessing for node features.

Pipeline:
    binarize ->  build_formal_context()      (objects x attributes incidence)
    mine     ->  mine_concepts()             (closed concepts; fallback miner)
    select   ->  score + select_top_k()      (support / stability / target)
    features ->  build_membership()          (hard / soft concept features)
    integrate->  build_features()            (raw | fca_feat | fca_group | svd_control)
"""
from .binarize import AttributeMeta, FormalContext, build_formal_context
from .concepts import Concept, SCORERS, score_concepts, select_top_k, class_association
from .mining import mine_concepts, prefilter_attributes
from .features import build_membership, group_concepts, group_features
from .integrate import FeatureBundle, build_features

__all__ = [
    "AttributeMeta",
    "FormalContext",
    "build_formal_context",
    "Concept",
    "SCORERS",
    "score_concepts",
    "select_top_k",
    "class_association",
    "mine_concepts",
    "prefilter_attributes",
    "build_membership",
    "group_concepts",
    "group_features",
    "FeatureBundle",
    "build_features",
]
