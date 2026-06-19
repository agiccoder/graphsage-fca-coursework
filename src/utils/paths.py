"""Canonical project paths.

All paths are derived from the repository root so the code is location
independent and works the same on Windows / Linux / macOS.
"""
from __future__ import annotations

from pathlib import Path

# src/utils/paths.py -> parents[2] == repository root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
DATASETS_DIR: Path = ARTIFACTS_DIR / "datasets"
CONCEPTS_DIR: Path = ARTIFACTS_DIR / "concepts"
FEATURES_DIR: Path = ARTIFACTS_DIR / "features"
RUNS_DIR: Path = ARTIFACTS_DIR / "runs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def ensure_dirs() -> None:
    """Create all standard output directories if they do not yet exist."""
    for d in (
        DATASETS_DIR,
        CONCEPTS_DIR,
        FEATURES_DIR,
        RUNS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
