"""Dependency-light test runner (works without pytest).

    python tests/run_tests.py

Discovers ``test_*`` functions in the sibling test modules, runs them, and
exits non-zero on the first failure. Useful where the local pytest install is
broken; in a clean environment ``python -m pytest tests/`` works too.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODULES = ["test_fca", "test_train", "test_aggregate", "test_binarize_modes",
           "test_patterns"]


def _load_module(mod_name: str):
    path = TEST_DIR / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    passed = failed = 0
    for mod_name in MODULES:
        mod = _load_module(mod_name)
        for name in sorted(vars(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            try:
                fn()
                print(f"PASS {mod_name}.{name}")
                passed += 1
            except Exception:  # noqa: BLE001
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
