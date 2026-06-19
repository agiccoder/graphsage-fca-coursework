"""Classification metrics: accuracy, macro-F1, per-class F1, confusion, subgroups."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int) -> dict:
    """Return accuracy, macro-F1, per-class F1 and the confusion matrix."""
    labels = list(range(num_classes))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "per_class_f1": [round(float(v), 6) for v in per_class],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, degrees: np.ndarray,
                     num_classes: int, low_degree_quantile: float = 0.2) -> dict:
    """Accuracy / macro-F1 on the low-degree subgroup (robustness probe)."""
    if degrees.size == 0:
        return {}
    thr = np.quantile(degrees, low_degree_quantile)
    low = degrees <= thr
    if low.sum() == 0:
        return {}
    m = compute_metrics(y_true[low], y_pred[low], num_classes)
    return {
        "low_degree_threshold": float(thr),
        "low_degree_count": int(low.sum()),
        "low_degree_accuracy": m["accuracy"],
        "low_degree_macro_f1": m["macro_f1"],
    }


def degree_bucket_metrics(y_true: np.ndarray, y_pred: np.ndarray, degrees: np.ndarray,
                          num_classes: int, scheme: str = "tertile") -> dict:
    """Per-bucket accuracy / macro-F1 over low / medium / high degree nodes.

    ``scheme`` is ``"tertile"`` (data-driven 33/66 percentiles of the evaluated
    nodes' degrees) or ``"fixed"`` (deg<=2, 3..5, >5). Thresholds are reported so
    the buckets are reproducible and identical across models on the same split.
    """
    if degrees.size == 0:
        return {}
    if scheme == "fixed":
        q_low, q_high = 2.0, 5.0
        masks = {
            "low": degrees <= 2,
            "medium": (degrees >= 3) & (degrees <= 5),
            "high": degrees > 5,
        }
    else:
        q_low, q_high = (float(v) for v in np.quantile(degrees, [1 / 3, 2 / 3]))
        masks = {
            "low": degrees <= q_low,
            "medium": (degrees > q_low) & (degrees <= q_high),
            "high": degrees > q_high,
        }
    out: dict = {
        "degree_bucket_scheme": scheme,
        "degree_thr_low": round(float(q_low), 3),
        "degree_thr_high": round(float(q_high), 3),
    }
    for name, m in masks.items():
        n = int(m.sum())
        out[f"bucket_{name}_count"] = n
        if n == 0:
            out[f"bucket_{name}_accuracy"] = float("nan")
            out[f"bucket_{name}_macro_f1"] = float("nan")
            continue
        mm = compute_metrics(y_true[m], y_pred[m], num_classes)
        out[f"bucket_{name}_accuracy"] = mm["accuracy"]
        out[f"bucket_{name}_macro_f1"] = mm["macro_f1"]
    return out


@torch.no_grad()
def evaluate_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                    num_classes: int, degrees: Optional[torch.Tensor] = None,
                    with_subgroup: bool = False, with_buckets: bool = False,
                    bucket_scheme: str = "tertile") -> dict:
    """Evaluate model logits on the nodes selected by ``mask``."""
    pred = logits[mask].argmax(dim=1).cpu().numpy()
    true = y[mask].cpu().numpy()
    out = compute_metrics(true, pred, num_classes)
    if degrees is not None and (with_subgroup or with_buckets):
        deg_np = degrees[mask].cpu().numpy()
        if with_subgroup:
            out.update(subgroup_metrics(true, pred, deg_np, num_classes))
        if with_buckets:
            out.update(degree_bucket_metrics(true, pred, deg_np, num_classes, bucket_scheme))
    return out
