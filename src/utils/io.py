"""Small IO helpers for JSON / CSV / torch tensors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of ``path`` and return ``path`` as a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (torch.Tensor,)):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_torch(obj: Any, path: str | Path) -> None:
    p = ensure_parent(path)
    torch.save(obj, p)


def load_torch(path: str | Path, map_location: str = "cpu") -> Any:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    p = ensure_parent(path)
    df.to_csv(p, index=index)


def append_rows_csv(rows: Iterable[dict], path: str | Path) -> pd.DataFrame:
    """Append ``rows`` to a CSV file, creating it (with header) if absent.

    Returns the full dataframe written to disk. Columns are unioned across the
    existing file and the new rows so heterogeneous experiments can share a file.
    """
    p = ensure_parent(path)
    new_df = pd.DataFrame(list(rows))
    if p.exists():
        old = pd.read_csv(p)
        combined = pd.concat([old, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    combined.to_csv(p, index=False)
    return combined


def save_text(text: str, path: str | Path) -> None:
    p = ensure_parent(path)
    p.write_text(text, encoding="utf-8")
