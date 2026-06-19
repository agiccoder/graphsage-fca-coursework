"""Run one experiment (a config) across seeds.

Usage
-----
    python -m src.train.run --config configs/experiments/cora_sage_fca.yaml
    python -m src.train.run --config <cfg> --set model.hidden_channels=64 --seeds 0 1 2

Outputs
-------
    results/per_seed_results.csv           (appended; one row per seed)
    artifacts/runs/<exp>/seed<k>_curves.csv, seed<k>_model.pt
    artifacts/runs/<exp>/summary.json, confusion_best.json
    artifacts/concepts/<dataset>_concepts.csv, artifacts/features/<dataset>_x_fca.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset, write_summary
from ..fca import build_features
from ..utils.config import get, load_config
from ..utils.io import append_rows_csv, save_dataframe, save_json
from ..utils.logging import get_logger
from ..utils.paths import DATASETS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs
from ..utils.seed import get_device
from .loop import train_one_run

logger = get_logger("train.run")


def experiment_name(cfg: dict) -> str:
    if get(cfg, "experiment"):
        return str(cfg["experiment"])
    return "_".join([
        str(get(cfg, "dataset.name", "data")),
        str(get(cfg, "model.name", "model")),
        str(get(cfg, "features.variant", "raw")),
    ])


def _row(cfg: dict, bundle, result) -> dict:
    # FCA-specific metadata is only meaningful for fca_* variants; raw/svd rows
    # must leave these fields empty so result tables are not misleading (Task A3).
    is_fca = bundle.variant.startswith("fca")
    fca = get(cfg, "features.fca", {}) if is_fca else {}
    cov = bundle.coverage
    t = result.test
    row = {
        "experiment": experiment_name(cfg),
        "dataset": get(cfg, "dataset.name"),
        "model": get(cfg, "model.name"),
        "variant": bundle.variant,
        "feature_variant": bundle.variant,
        "seed": result.seed,
        "raw_dim": bundle.raw_dim,
        "added_dim": bundle.added_dim,
        "total_dim": int(bundle.x.shape[1]),
        "hidden_channels": get(cfg, "model.hidden_channels"),
        "num_layers": get(cfg, "model.num_layers"),
        "dropout": get(cfg, "model.dropout"),
        "aggr": get(cfg, "model.aggr") if get(cfg, "model.name") in ("graphsage", "sage") else None,
        "lr": get(cfg, "train.lr"),
        "weight_decay": get(cfg, "train.weight_decay"),
        "k_concepts": get(fca, "k_concepts") if is_fca else None,
        "scorer": get(fca, "scorer") if is_fca else None,
        "membership": get(fca, "membership") if is_fca else None,
        "min_intent_size": int(get(fca, "min_intent_size", 1)) if is_fca else None,
        # Binarisation identity: two FCA runs that differ ONLY in how raw features
        # are discretised (e.g. binary_nonzero vs quantile_binarization) are NOT
        # the same config. Recording these makes them distinguishable during
        # aggregation so they no longer collapse into one inflated seed group (Task B).
        "binarize_mode": get(fca, "binarize_mode", "binary_nonzero") if is_fca else None,
        "binarize_params": (json.dumps(get(fca, "binarize_params", {}), sort_keys=True)
                            if is_fca else None),
        "mean_intent_size": cov.get("mean_intent_size") if is_fca else None,
        "svd_components": bundle.added_dim if bundle.variant == "svd_control" else None,
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "train_time_sec": round(result.train_time_sec, 3),
        "val_accuracy": round(result.val["accuracy"], 6),
        "val_macro_f1": round(result.val["macro_f1"], 6),
        "test_accuracy": round(t["accuracy"], 6),
        "test_macro_f1": round(t["macro_f1"], 6),
        "low_degree_accuracy": t.get("low_degree_accuracy"),
        "low_degree_macro_f1": t.get("low_degree_macro_f1"),
        "num_concepts": cov.get("num_concepts", 0) if is_fca else None,
        "node_coverage": round(cov.get("node_coverage", 0.0), 6) if is_fca else None,
        "mean_extent_size": cov.get("mean_extent_size") if is_fca else None,
        "test_per_class_f1": json.dumps(t.get("per_class_f1", [])),
    }
    # Degree-bucket metrics (low / medium / high) for the structural analysis.
    for b in ("low", "medium", "high"):
        row[f"bucket_{b}_accuracy"] = t.get(f"bucket_{b}_accuracy")
        row[f"bucket_{b}_macro_f1"] = t.get(f"bucket_{b}_macro_f1")
        row[f"bucket_{b}_count"] = t.get(f"bucket_{b}_count")
    row["degree_thr_low"] = t.get("degree_thr_low")
    row["degree_thr_high"] = t.get("degree_thr_high")
    return row


def run_experiment(cfg: dict, seeds: list[int] | None = None,
                   save_models: bool = True) -> pd.DataFrame:
    ensure_dirs()
    device = get_device(get(cfg, "device", "auto"))
    exp = experiment_name(cfg)
    run_dir = RUNS_DIR / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment '%s' on device=%s", exp, device)

    # ---- data ----
    ds_name = get(cfg, "dataset.name")
    data = load_dataset(
        ds_name,
        root=get(cfg, "dataset.root", str(DATASETS_DIR)),
        planetoid_split=get(cfg, "dataset.planetoid_split", "public"),
        split_idx=int(get(cfg, "dataset.split_idx", 0)),
        to_undirected=get(cfg, "dataset.to_undirected", None),
    )
    write_summary(data, RESULTS_DIR / "dataset_summaries")

    # ---- features (built once, fixed feature seed -> only model init varies) ----
    feature_seed = int(get(cfg, "features.seed", 0))
    bundle = build_features(data, get(cfg, "features", {"variant": "raw"}),
                            seed=feature_seed,
                            save_as=ds_name if get(cfg, "features.save_artifacts", True) else None)

    # ---- seeds ----
    seeds = seeds or list(get(cfg, "seeds", [0, 1, 2, 3, 4]))
    rows, results = [], []
    for seed in seeds:
        result = train_one_run(data, bundle, cfg, seed, device)
        rows.append(_row(cfg, bundle, result))
        results.append(result)
        save_dataframe(pd.DataFrame(result.curves), run_dir / f"seed{seed}_curves.csv")
        if save_models and result.best_state is not None:
            import torch
            torch.save(result.best_state, run_dir / f"seed{seed}_model.pt")

    df = pd.DataFrame(rows)
    # Append to the global per-seed results and write a run-local copy.
    append_rows_csv(rows, RESULTS_DIR / "per_seed_results.csv")
    save_dataframe(df, run_dir / "per_seed.csv")

    # Confusion matrix of the best (highest val acc) seed for the report.
    best_i = int(np.argmax([r.val["accuracy"] for r in results]))
    save_json({"seed": results[best_i].seed,
               "confusion_matrix": results[best_i].test["confusion_matrix"]},
              run_dir / "confusion_best.json")

    summary = {
        "experiment": exp,
        "test_accuracy_mean": float(df["test_accuracy"].mean()),
        "test_accuracy_std": float(df["test_accuracy"].std(ddof=0)),
        "test_macro_f1_mean": float(df["test_macro_f1"].mean()),
        "test_macro_f1_std": float(df["test_macro_f1"].std(ddof=0)),
        "seeds": seeds,
        "coverage": bundle.coverage,
    }
    save_json(summary, run_dir / "summary.json")
    logger.info("DONE %s | acc %.4f +/- %.4f | macro-F1 %.4f +/- %.4f",
                exp, summary["test_accuracy_mean"], summary["test_accuracy_std"],
                summary["test_macro_f1_mean"], summary["test_macro_f1_std"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a GraphSAGE+FCA experiment.")
    ap.add_argument("--config", required=True, help="Path to an experiment YAML.")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="Dotted overrides, e.g. model.hidden_channels=64")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="Override the seed list.")
    ap.add_argument("--no-save-models", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    run_experiment(cfg, seeds=args.seeds, save_models=not args.no_save_models)


if __name__ == "__main__":
    main()
