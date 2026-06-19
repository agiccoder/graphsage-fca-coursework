"""Put the repository root on sys.path so scripts can `import src...`.

Import this first in any standalone script: ``import _bootstrap  # noqa``.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
