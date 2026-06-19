# Minimal public repository manifest

Repository: `https://github.com/agiccoder/graphsage-fca-coursework`

## Keep in public repository

### Root

```text
README.md
requirements.txt
.gitignore
coursework_graphsage_fca_full_current (1).tex
```

Optional if final and up-to-date:

```text
Classification using GraphSAGE and FCA.pdf
```

### Source code and configs

```text
configs/
src/
scripts/
tests/
```

### Small reproducibility artifacts

```text
artifacts/concepts/
```

### Results to keep

```text
results/final_table_main_v2.csv
results/final_table_dataset_summary.csv
results/final_table_structural_diagnostics.csv
results/final_table_richer_scaling.csv
results/final_table_best_configs.csv
results/per_seed_results.csv
results/scaling_diagnostics.csv
results/top_concepts_clean.csv
results/concept_statistics.csv
results/degree_bucket_results.csv
results/deltas.csv
results/ranking_by_dataset.csv
results/dataset_summaries/
```

### Figures to keep

```text
figures/model_diagram.png
figures/bar_accuracy.png
figures/scaling_delta_vs_svd.png
figures/pattern_delta_vs_svd.png
```

Optional:

```text
figures/bar_macro_f1.png
figures/scaling_delta_vs_binary.png
figures/scaling_phase2a_accuracy.png
```

### Reports to keep

```text
reports/final_tables.md
reports/reproducibility_appendix.md
reports/pattern_structures_summary.md
```

Optional:

```text
reports/final_sanity_check.md
reports/richer_scaling_final_findings.md
reports/final_figure_inventory.md
```

## Remove from public repository

```text
plans/
reports/coursework_gost_and_weakness_audit.md
reports/experiment_summary.md
reports/experimental_findings.md
reports/final_claims_for_text.md
reports/final_experiment_status.md
reports/final_text_page_audit.md
reports/methodology_for_text.md
reports/scaling_diagnostics_review.md
reports/scaling_extension_summary.md
reports/scaling_phase2b_decision.md
reports/scaling_phase2b_summary.md
reports/scaling_phase3_summary.md
results/_archive_v1/
results/duplicate_runs.csv
results/duplicate_runs_scaling.csv
results/main_results.csv
results/per_seed_results.backup_pre_phase2b.csv
results/per_seed_results.backup_pre_scaling_ext.csv
figures/ablation_k_concepts.png
figures/concept_intent_size_distribution.png
```

## Never add

```text
.claude/
_pdf_dump.txt
claude_code_*.md
deep-research-report (1).md
Оформление_курсовых_работ_по_ГОСТ_рекомендации.docx
artifacts/datasets/
artifacts/features/
artifacts/runs/
*.pt
results/_scratch/
.env
.env.*
```
