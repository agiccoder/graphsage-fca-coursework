"""Dataset descriptive statistics, exported as JSON and Markdown."""
from __future__ import annotations

from pathlib import Path

import torch

from ..utils.io import save_json, save_text
from .types import GraphData


def _edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of edges connecting two nodes of the same class."""
    if edge_index.size(1) == 0:
        return float("nan")
    same = (y[edge_index[0]] == y[edge_index[1]]).float().mean().item()
    return round(same, 4)


def summarize(data: GraphData) -> dict:
    """Compute a descriptive summary dict for a dataset (no model involved)."""
    deg = data.degrees().float()
    x = data.x_raw
    nonzero_frac = float((x != 0).float().mean().item())
    is_binary = bool(torch.isin(x.unique(), torch.tensor([0.0, 1.0])).all().item())
    counts = torch.bincount(data.y, minlength=data.num_classes)
    class_dist = {int(c): int(n) for c, n in enumerate(counts.tolist())}

    return {
        **data.metadata,
        "avg_degree": round(deg.mean().item(), 3),
        "median_degree": float(deg.median().item()),
        "max_degree": int(deg.max().item()),
        "min_degree": int(deg.min().item()),
        "feature_nonzero_fraction": round(nonzero_frac, 5),
        "feature_is_binary": is_binary,
        "edge_homophily": _edge_homophily(data.edge_index, data.y),
        "class_distribution": class_dist,
        "split_sizes": data.split_sizes(),
    }


def _to_markdown(summary: dict) -> str:
    lines = [f"# Dataset summary: `{summary.get('name', '?')}`", ""]
    skip = {"class_distribution", "split_sizes", "name"}
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    for k, v in summary.items():
        if k in skip:
            continue
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Split sizes")
    for k, v in summary.get("split_sizes", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Class distribution")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for c, n in summary.get("class_distribution", {}).items():
        lines.append(f"| {c} | {n} |")
    return "\n".join(lines) + "\n"


def write_summary(data: GraphData, out_dir: str | Path) -> dict:
    """Write ``<name>_summary.json`` and ``.md`` into ``out_dir``; return summary."""
    out_dir = Path(out_dir)
    summary = summarize(data)
    save_json(summary, out_dir / f"{data.name}_summary.json")
    save_text(_to_markdown(summary), out_dir / f"{data.name}_summary.md")
    return summary
