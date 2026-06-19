"""YAML config loading with ``extends`` inheritance and dotted overrides.

A config file may declare ``extends: <path>`` (string or list) to inherit from
one or more base configs. Children are deep-merged on top of parents, so an
experiment file only needs to specify what differs from ``configs/base.yaml``.

CLI overrides use dotted keys, e.g. ``--set model.hidden_channels=256``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import CONFIGS_DIR, PROJECT_ROOT


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level.")
    return data


def _resolve_path(p: str | Path, base_dir: Path) -> Path:
    cand = Path(p)
    if cand.is_absolute():
        return cand
    for root in (base_dir, PROJECT_ROOT, CONFIGS_DIR):
        if (root / cand).exists():
            return (root / cand).resolve()
    # Fall back to project-root relative even if missing (clearer error later).
    return (PROJECT_ROOT / cand).resolve()


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Load a config, resolving ``extends`` chains and applying CLI overrides."""
    path = _resolve_path(path, PROJECT_ROOT)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = _load_yaml(path)

    parents = raw.pop("extends", None)
    merged: dict = {}
    if parents:
        if isinstance(parents, (str, Path)):
            parents = [parents]
        for parent in parents:
            parent_path = _resolve_path(parent, path.parent)
            merged = deep_merge(merged, load_config(parent_path))
    merged = deep_merge(merged, raw)

    if overrides:
        apply_overrides(merged, overrides)
    return merged


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested value via dotted path, returning ``default`` if absent."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_in(cfg: dict, dotted: str, value: Any) -> None:
    """Set a nested value via dotted path, creating intermediate dicts."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> None:
    """Apply ``key.path=value`` overrides; values are parsed as YAML scalars."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.path=value")
        key, raw_val = item.split("=", 1)
        value = yaml.safe_load(raw_val)
        set_in(cfg, key.strip(), value)
