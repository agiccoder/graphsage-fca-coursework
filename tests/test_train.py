"""End-to-end training test on a synthetic graph (no downloads)."""
import numpy as np
import torch

from src.data.types import GraphData
from src.fca import build_features
from src.train.loop import train_one_run


def _synthetic_graph(n_per_class: int = 100, num_classes: int = 3,
                     dim: int = 20, seed: int = 0) -> GraphData:
    rng = np.random.default_rng(seed)
    n = n_per_class * num_classes
    y = np.repeat(np.arange(num_classes), n_per_class)
    # class-correlated binary features (good for both raw and FCA)
    x = np.zeros((n, dim), dtype=np.float32)
    for c in range(num_classes):
        block = slice(c * (dim // num_classes), (c + 1) * (dim // num_classes))
        rows = y == c
        x[rows, block] = (rng.random((rows.sum(), dim // num_classes)) < 0.7)
    x += (rng.random((n, dim)) < 0.05)  # noise
    # intra-class edges
    src, dst = [], []
    for c in range(num_classes):
        idx = np.flatnonzero(y == c)
        for _ in range(len(idx) * 4):
            a, b = rng.choice(idx, 2, replace=False)
            src += [a, b]; dst += [b, a]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    perm = rng.permutation(n)
    train = torch.zeros(n, dtype=torch.bool); train[perm[: n // 2]] = True
    val = torch.zeros(n, dtype=torch.bool); val[perm[n // 2: 3 * n // 4]] = True
    test = torch.zeros(n, dtype=torch.bool); test[perm[3 * n // 4:]] = True
    return GraphData("synthetic", edge_index, torch.from_numpy(x),
                     torch.from_numpy(y).long(), train, val, test,
                     metadata={"name": "synthetic"})


_CFG = {
    "model": {"name": "graphsage", "hidden_channels": 32, "num_layers": 2,
              "dropout": 0.3, "aggr": "mean"},
    "train": {"lr": 0.01, "weight_decay": 5e-4, "epochs": 60, "patience": 20,
              "monitor": "accuracy", "deterministic": True},
    "eval": {"subgroup": True},
}


def test_train_raw_graphsage_learns():
    data = _synthetic_graph()
    bundle = build_features(data, {"variant": "raw"})
    res = train_one_run(data, bundle, _CFG, seed=0, device=torch.device("cpu"))
    assert res.test["accuracy"] > 0.7
    assert 0.0 <= res.test["macro_f1"] <= 1.0
    assert res.best_state is not None


def test_train_fca_feat_runs():
    data = _synthetic_graph()
    fcfg = {"variant": "fca_feat", "save_artifacts": False,
            "fca": {"binarize_mode": "binary_nonzero", "k_concepts": 16,
                    "scorer": "support", "membership": "hard",
                    "min_support": 0.05, "max_support": 0.7}}
    bundle = build_features(data, fcfg, seed=0)
    assert bundle.added_dim > 0
    assert bundle.x.shape[1] == data.num_features + bundle.added_dim
    res = train_one_run(data, bundle, _CFG, seed=0, device=torch.device("cpu"))
    assert res.test["accuracy"] > 0.6
