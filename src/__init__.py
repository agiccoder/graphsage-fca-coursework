"""GraphSAGE + FCA node-classification research package.

See README.md for the high-level architecture. The public entry points are:
    python -m src.train.run   --config configs/experiments/<name>.yaml
    python -m src.train.grid  --config configs/experiments/<name>.yaml --grid configs/grids/<grid>.yaml
    python -m src.eval.aggregate
"""

__version__ = "0.1.0"
