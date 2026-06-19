# Reproducibility appendix

_Exact commands and environment assumptions to regenerate every result in the
paper. All commands are run from the repository root._

---

## 1. Environment

- **Python 3.9.6** (use `python3`; there is no `python` alias and no `pytest`).
- Key packages: `torch`, `torch_geometric` **2.6.1**, `scikit-learn`, `numpy`,
  `pandas`, `matplotlib`. See [`requirements.txt`](../requirements.txt).
- Datasets download automatically (Planetoid) into `artifacts/datasets/` on first
  run.
- All runs use **seeds 0, 1, 2, 3, 4** and are deterministic (`train.deterministic:
  true`, full-batch).

```bash
python3 -m pip install -r requirements.txt
```

## 2. Data preparation / dataset summaries

```bash
python3 scripts/prepare_data.py          # download + cache datasets
# dataset summary JSON/MD live in results/dataset_summaries/
```

## 3. Main v2 battery (baselines + binary_nonzero FCA + SVD + ablations)

This reproduces Table 2 and the mechanism ablations (K-sweep, scorer, membership,
degree buckets). Final results are already in
`results/per_seed_results.csv`.

```bash
python3 scripts/run_v2.py                 # full v2 battery, seeds 0-4
```

## 4. Richer FCA scaling extension

### Phase 1 — structural diagnostics (no neural training → Table 3)

```bash
python3 scripts/analyze_scaling_modes.py  # writes results/scaling_diagnostics.csv
```

### Phase 2a — downstream training at K=128

```bash
python3 scripts/run_scaling_extension.py --seeds 0 1 2 3 4
```

### Phase 2b — K-sweep (K=64 / 256) confirming CiteSeer robustness

```bash
python3 scripts/run_scaling_extension.py --phase2b --seeds 0 1 2 3 4
```

### Phase 3 — graph-smoothed top-k (Cora + PubMed)

```bash
python3 scripts/run_scaling_extension.py --phase3 --seeds 0 1 2 3 4
```

### Regenerate only the scaling reports (no training)

```bash
python3 scripts/run_scaling_extension.py --report-only
python3 scripts/run_scaling_extension.py --report-only --phase2b
python3 scripts/run_scaling_extension.py --report-only --phase3
```

## 5. Figures

```bash
python3 scripts/make_figures.py           # main paper figures
python3 scripts/make_scaling_figures.py   # scaling_phase2a + delta_vs_binary + delta_vs_svd
```

## 6. Final paper tables (deterministic, no training)

Regenerates the 5 CSVs and `reports/final_tables.md` from the existing per-seed CSV:

```bash
python3 scripts/make_final_tables.py
```

Outputs:
- `results/final_table_dataset_summary.csv`
- `results/final_table_main_v2.csv`
- `results/final_table_structural_diagnostics.csv`
- `results/final_table_richer_scaling.csv`
- `results/final_table_best_configs.csv`
- `reports/final_tables.md`

## 7. Tests

There is **no `pytest`**. Run the dependency-light suite directly:

```bash
python3 tests/run_tests.py
```

(21 tests pass: FCA correctness incl. leakage guards, binarize modes, training
smoke tests, aggregation/duplicate detection.)

## 8. Result-file map

| Artifact | Produced by | Used for |
|---|---|---|
| `results/per_seed_results.csv` | `run_v2.py`, `run_scaling_extension.py` | all tables |
| `results/scaling_diagnostics.csv` | `analyze_scaling_modes.py` | Table 3 |
| `results/final_table_*.csv` | `make_final_tables.py` | Tables 1–5 |
| `results/dataset_summaries/*.json` | `prepare_data.py` | Table 1 |
| `figures/*.png` | `make_figures.py`, `make_scaling_figures.py` | paper figures |

## 9. Reproducibility notes

- Reported numbers are seed means with population std (ddof = 0) over seeds 0–4.
- `binary_nonzero` rows carry `binarize_mode = NaN` / `binarize_params = NaN` in the
  CSV for historical reasons; the table generator normalises these to
  `binary_nonzero` / `default` so they are not silently dropped by `groupby`.
- Backups of the per-seed CSV at each milestone:
  `per_seed_results.backup_pre_scaling_ext.csv`,
  `per_seed_results.backup_pre_phase2b.csv`.
- Re-running the training stages overwrites/append to
  `results/per_seed_results.csv`; the deterministic table/figure generators are
  safe to re-run any number of times.
