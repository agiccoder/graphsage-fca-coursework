"""Model factory: build a model from a config dict + data dimensions."""
from __future__ import annotations

from typing import Callable

from torch import nn

from ..utils.config import get
from .gcn import GCN
from .mlp import MLP
from .sage import GraphSAGE

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "logreg": lambda **kw: MLP(num_layers=1, **{k: v for k, v in kw.items() if k != "num_layers"}),
    "mlp": MLP,
    "graphsage": GraphSAGE,
    "sage": GraphSAGE,
    "gcn": GCN,
}

# Which models consume the graph structure (need edge_index at train time).
GRAPH_MODELS = {"graphsage", "sage", "gcn"}


def build_model(model_cfg: dict, in_dim: int, num_classes: int) -> nn.Module:
    """Instantiate a model from ``model_cfg`` for the given input dimensionality."""
    name = get(model_cfg, "name", "graphsage").lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    common = dict(
        in_dim=in_dim,
        hidden_channels=int(get(model_cfg, "hidden_channels", 128)),
        num_classes=num_classes,
        num_layers=int(get(model_cfg, "num_layers", 2)),
        dropout=float(get(model_cfg, "dropout", 0.5)),
    )
    if name in ("mlp", "logreg"):
        common["norm"] = bool(get(model_cfg, "norm", False))
        return MODEL_REGISTRY[name](**common)
    if name == "gcn":
        common["norm"] = get(model_cfg, "norm", False)
        return GCN(**common)

    # GraphSAGE family
    common.update(
        aggr=get(model_cfg, "aggr", "mean"),
        project=bool(get(model_cfg, "project", False)),
        norm=get(model_cfg, "norm", False),
        residual=bool(get(model_cfg, "residual", False)),
        jk=get(model_cfg, "jk", False),
    )
    return GraphSAGE(**common)


def is_graph_model(model_cfg: dict) -> bool:
    return get(model_cfg, "name", "graphsage").lower() in GRAPH_MODELS
