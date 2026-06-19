# Notebooks

This folder is for exploratory analysis and thesis-figure polishing. The core
pipeline is script-driven (see the repo `README.md`), so notebooks are optional.

Suggested starting points (run from the repo root):

```python
import sys; sys.path.insert(0, "..")          # if running inside notebooks/
import pandas as pd

main = pd.read_csv("../results/main_results.csv")
deltas = pd.read_csv("../results/deltas.csv")
concepts = pd.read_csv("../artifacts/concepts/cora_concepts.csv")

# inspect FCA features and concept membership
import torch
bundle = torch.load("../artifacts/features/cora_x_fca.pt", weights_only=False)
print(bundle["variant"], bundle["x"].shape, bundle["coverage"])
```

To regenerate everything: `python scripts/run_core.py` then
`python scripts/make_figures.py`.
