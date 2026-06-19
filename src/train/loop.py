"""Single training run: full-batch (default) or neighbour-sampled mini-batch.

A "run" trains one model on one feature variant with one seed, applies early
stopping on a validation metric, restores the best checkpoint and reports
test-set metrics plus per-epoch learning curves.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..eval.metrics import compute_metrics, evaluate_logits
from ..models.build import build_model, is_graph_model
from ..utils.config import get
from ..utils.logging import get_logger
from ..utils.seed import set_seed

logger = get_logger("train.loop")


@dataclass
class RunResult:
    seed: int
    best_epoch: int
    epochs_ran: int
    train_time_sec: float
    val: dict
    test: dict
    curves: list[dict] = field(default_factory=list)
    best_state: dict | None = None


def _monitor_value(metrics: dict, monitor: str) -> float:
    return metrics.get(monitor.replace("val_", ""), metrics.get("accuracy", 0.0))


def train_one_run(data, bundle, cfg: dict, seed: int,
                  device: torch.device) -> RunResult:
    """Train one model/variant/seed and return metrics + learning curves."""
    set_seed(seed, deterministic=bool(get(cfg, "train.deterministic", True)))

    x = bundle.x.to(device)
    edge_index = bundle.edge_index.to(device)
    y = data.y.to(device)
    num_eval = bundle.num_eval_nodes
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    degrees = data.degrees().to(device)
    num_classes = data.num_classes

    model = build_model(cfg["model"], in_dim=x.size(1), num_classes=num_classes).to(device)
    use_graph = is_graph_model(cfg["model"])

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(get(cfg, "train.lr", 5e-3)),
        weight_decay=float(get(cfg, "train.weight_decay", 5e-4)),
    )

    epochs = int(get(cfg, "train.epochs", 300))
    patience = int(get(cfg, "train.patience", 50))
    monitor = get(cfg, "train.monitor", "accuracy")

    def forward() -> torch.Tensor:
        out = model(x, edge_index) if use_graph else model(x)
        return out[:num_eval]

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_left = patience
    curves: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = forward()
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = forward()
            tr = compute_metrics(y[train_mask].cpu().numpy(),
                                 logits[train_mask].argmax(1).cpu().numpy(), num_classes)
            va = compute_metrics(y[val_mask].cpu().numpy(),
                                 logits[val_mask].argmax(1).cpu().numpy(), num_classes)
            te = compute_metrics(y[test_mask].cpu().numpy(),
                                 logits[test_mask].argmax(1).cpu().numpy(), num_classes)
        curves.append({
            "epoch": epoch, "train_loss": float(loss.item()),
            "train_acc": tr["accuracy"], "val_acc": va["accuracy"],
            "val_macro_f1": va["macro_f1"], "test_acc": te["accuracy"],
            "test_macro_f1": te["macro_f1"],
        })

        score = _monitor_value(va, monitor)
        if score > best_val:
            best_val, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = forward()
    val_metrics = evaluate_logits(logits, y, val_mask, num_classes)
    with_sub = bool(get(cfg, "eval.subgroup", True))
    with_buckets = bool(get(cfg, "eval.degree_buckets", True))
    test_metrics = evaluate_logits(
        logits, y, test_mask, num_classes, degrees=degrees,
        with_subgroup=with_sub, with_buckets=with_buckets,
        bucket_scheme=get(cfg, "eval.degree_bucket_scheme", "tertile"),
    )

    logger.info("seed=%d | best_epoch=%d | val_acc=%.4f | test_acc=%.4f | test_mf1=%.4f | %.1fs",
                seed, best_epoch, val_metrics["accuracy"], test_metrics["accuracy"],
                test_metrics["macro_f1"], train_time)
    cpu_state = {k: v.detach().cpu() for k, v in best_state.items()}
    return RunResult(seed, best_epoch, len(curves), train_time,
                     val_metrics, test_metrics, curves, best_state=cpu_state)
