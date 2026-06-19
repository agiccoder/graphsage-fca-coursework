"""Evaluation: metrics, aggregation across seeds, and report generation."""
from .metrics import (compute_metrics, subgroup_metrics, evaluate_logits,
                      degree_bucket_metrics)
from .aggregate import aggregate, compute_deltas, ranking_by_dataset, run as aggregate_results
from .report import build_report

__all__ = [
    "compute_metrics",
    "subgroup_metrics",
    "evaluate_logits",
    "degree_bucket_metrics",
    "aggregate",
    "compute_deltas",
    "ranking_by_dataset",
    "aggregate_results",
    "build_report",
]
