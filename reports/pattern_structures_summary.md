# Pattern structures — interval-pattern experiments

_Exploratory GraphSAGE runs for `fca_pattern`: raw features are concatenated with interval pattern-membership features. Pattern intervals are built from train-quantile bins; supervised scoring uses train labels only. Extended rows also test soft membership and graph-smoothed pattern sources. Source: [`results/per_seed_results.csv`](../results/per_seed_results.csv)._

Tie band for verdicts: ±0.002 accuracy. Bootstrap CI is for pattern minus K-matched SVD.

| dataset | pattern variant | K | scorer | membership | pattern acc | pattern macro-F1 | raw acc | SVD acc | Δ vs SVD | bootstrap 95% CI | P(Δ≤0) | verdict |
|---|---|---:|---|---|---|---|---|---|---|---|---:|---|
| citeseer | gaware_qbins4_i2_hard | 128 | support | hard | 0.6282 ± 0.0088 (n=5) | 0.5952 ± 0.0118 (n=5) | 0.6832 | 0.6924 | -0.0642 | [-0.0686, -0.0602] | 1.000 | NO vs SVD |
| citeseer | gaware_qbins4_i2_hard | 128 | target_entropy | hard | 0.6596 ± 0.0119 (n=5) | 0.6190 ± 0.0098 (n=5) | 0.6832 | 0.6924 | -0.0328 | [-0.0460, -0.0222] | 1.000 | NO vs SVD |
| citeseer | gaware_qbins4_i2_soft | 64 | support | soft | 0.6854 ± 0.0151 (n=5) | 0.6452 ± 0.0100 (n=5) | 0.6832 | 0.7116 | -0.0262 | [-0.0474, -0.0050] | 1.000 | NO vs SVD |
| citeseer | gaware_qbins4_i2_soft | 64 | target_entropy | soft | 0.6942 ± 0.0198 (n=5) | 0.6529 ± 0.0146 (n=5) | 0.6832 | 0.7116 | -0.0174 | [-0.0418, +0.0054] | 0.915 | NO vs SVD |
| citeseer | gaware_qbins4_i2_soft | 128 | support | soft | 0.6950 ± 0.0101 (n=5) | 0.6561 ± 0.0056 (n=5) | 0.6832 | 0.6924 | +0.0026 | [-0.0086, +0.0138] | 0.345 | SUCCESS vs K-matched SVD |
| citeseer | gaware_qbins4_i2_soft | 128 | target_entropy | soft | 0.6988 ± 0.0099 (n=5) | 0.6597 ± 0.0066 (n=5) | 0.6832 | 0.6924 | +0.0064 | [-0.0038, +0.0178] | 0.119 | SUCCESS vs K-matched SVD |
| citeseer | gaware_qbins4_i2_soft | 256 | support | soft | 0.6842 ± 0.0084 (n=5) | 0.6521 ± 0.0073 (n=5) | 0.6832 | 0.6870 | -0.0028 | [-0.0122, +0.0060] | 0.712 | NO vs SVD |
| citeseer | gaware_qbins4_i2_soft | 256 | target_entropy | soft | 0.6936 ± 0.0111 (n=5) | 0.6572 ± 0.0106 (n=5) | 0.6832 | 0.6870 | +0.0066 | [+0.0008, +0.0124] | 0.011 | SUCCESS vs K-matched SVD |
| citeseer | interval_qbins4_i2 | 128 | support | hard | 0.6800 ± 0.0010 (n=2) | 0.6473 ± 0.0058 (n=2) | 0.6832 | 0.6924 | -0.0124 | [-0.0230, -0.0150] | 1.000 | NO vs SVD |
| citeseer | interval_qbins4_i2 | 128 | target_entropy | hard | 0.6800 ± 0.0010 (n=2) | 0.6473 ± 0.0058 (n=2) | 0.6832 | 0.6924 | -0.0124 | [-0.0230, -0.0150] | 1.000 | NO vs SVD |
| citeseer | interval_qbins4_i2_soft | 128 | support | soft | 0.6914 ± 0.0115 (n=5) | 0.6500 ± 0.0083 (n=5) | 0.6832 | 0.6924 | -0.0010 | [-0.0168, +0.0148] | 0.548 | NEUTRAL |
| citeseer | interval_qbins4_i2_soft | 128 | target_entropy | soft | 0.6914 ± 0.0115 (n=5) | 0.6500 ± 0.0083 (n=5) | 0.6832 | 0.6924 | -0.0010 | [-0.0168, +0.0148] | 0.548 | NEUTRAL |
| cora | gaware_qbins4_i2_soft | 128 | support | soft | 0.8002 ± 0.0100 (n=5) | 0.7909 ± 0.0135 (n=5) | 0.8008 | 0.8116 | -0.0114 | [-0.0200, -0.0014] | 0.992 | NO vs SVD |
| cora | gaware_qbins4_i2_soft | 128 | target_entropy | soft | 0.8006 ± 0.0120 (n=5) | 0.7912 ± 0.0111 (n=5) | 0.8008 | 0.8116 | -0.0110 | [-0.0214, +0.0004] | 0.968 | NO vs SVD |
| cora | interval_qbins4_i2 | 128 | support | hard | 0.7990 ± 0.0090 (n=2) | 0.7927 ± 0.0053 (n=2) | 0.8008 | 0.8116 | -0.0126 | [-0.0280, -0.0030] | 1.000 | NO vs SVD |
| cora | interval_qbins4_i2 | 128 | target_entropy | hard | 0.7990 ± 0.0090 (n=2) | 0.7927 ± 0.0053 (n=2) | 0.8008 | 0.8116 | -0.0126 | [-0.0280, -0.0030] | 1.000 | NO vs SVD |
| pubmed | gaware_qbins4_i2_hard | 128 | support | hard | 0.6168 ± 0.0110 (n=5) | 0.6197 ± 0.0108 (n=5) | 0.7658 | 0.7616 | -0.1448 | [-0.1534, -0.1362] | 1.000 | NO vs SVD |
| pubmed | gaware_qbins4_i2_hard | 128 | target_entropy | hard | 0.7298 ± 0.0055 (n=5) | 0.7357 ± 0.0047 (n=5) | 0.7658 | 0.7616 | -0.0318 | [-0.0356, -0.0262] | 1.000 | NO vs SVD |
| pubmed | gaware_qbins4_i2_soft | 64 | support | soft | 0.7636 ± 0.0042 (n=5) | 0.7569 ± 0.0025 (n=5) | 0.7658 | 0.7630 | +0.0006 | [-0.0082, +0.0086] | 0.418 | NEUTRAL |
| pubmed | gaware_qbins4_i2_soft | 64 | target_entropy | soft | 0.7560 ± 0.0043 (n=5) | 0.7505 ± 0.0041 (n=5) | 0.7658 | 0.7630 | -0.0070 | [-0.0182, +0.0002] | 0.963 | NO vs SVD |
| pubmed | gaware_qbins4_i2_soft | 128 | support | soft | 0.7648 ± 0.0058 (n=5) | 0.7570 ± 0.0058 (n=5) | 0.7658 | 0.7616 | +0.0032 | [-0.0024, +0.0098] | 0.162 | SUCCESS vs K-matched SVD |
| pubmed | gaware_qbins4_i2_soft | 128 | target_entropy | soft | 0.7672 ± 0.0056 (n=5) | 0.7623 ± 0.0062 (n=5) | 0.7658 | 0.7616 | +0.0056 | [+0.0014, +0.0104] | 0.002 | SUCCESS vs K-matched SVD |
| pubmed | gaware_qbins4_i2_soft | 256 | support | soft | 0.7626 ± 0.0045 (n=5) | 0.7581 ± 0.0048 (n=5) | 0.7658 | 0.7614 | +0.0012 | [-0.0018, +0.0042] | 0.329 | NEUTRAL |
| pubmed | gaware_qbins4_i2_soft | 256 | target_entropy | soft | 0.7794 ± 0.0026 (n=5) | 0.7729 ± 0.0029 (n=5) | 0.7658 | 0.7614 | +0.0180 | [+0.0126, +0.0228] | 0.000 | SUCCESS vs K-matched SVD |
| pubmed | interval_qbins4_i2 | 128 | support | hard | 0.6325 ± 0.0025 (n=2) | 0.6303 ± 0.0031 (n=2) | 0.7658 | 0.7616 | -0.1291 | [-0.1310, -0.1240] | 1.000 | NO vs SVD |
| pubmed | interval_qbins4_i2 | 128 | target_entropy | hard | 0.6890 ± 0.0010 (n=2) | 0.6964 ± 0.0006 (n=2) | 0.7658 | 0.7616 | -0.0726 | [-0.0710, -0.0710] | 1.000 | NO vs SVD |
| pubmed | interval_qbins4_i2_soft | 128 | support | soft | 0.7596 ± 0.0052 (n=5) | 0.7572 ± 0.0038 (n=5) | 0.7658 | 0.7616 | -0.0020 | [-0.0070, +0.0036] | 0.781 | PARTIAL vs binary FCA |
| pubmed | interval_qbins4_i2_soft | 128 | target_entropy | soft | 0.7542 ± 0.0037 (n=5) | 0.7541 ± 0.0033 (n=5) | 0.7658 | 0.7616 | -0.0074 | [-0.0102, -0.0036] | 1.000 | NO vs SVD |

## Interpretation

- **citeseer**: best pattern run is `gaware_qbins4_i2_soft` / K=128 / `target_entropy` / `soft` with accuracy 0.6988; Δ vs SVD +0.0064, bootstrap CI [-0.0038, +0.0178], P(Δ≤0)=0.119. Verdict: SUCCESS vs K-matched SVD.
- **cora**: best pattern run is `gaware_qbins4_i2_soft` / K=128 / `target_entropy` / `soft` with accuracy 0.8006; Δ vs SVD -0.0110, bootstrap CI [-0.0214, +0.0004], P(Δ≤0)=0.968. Verdict: NO vs SVD.
- **pubmed**: best pattern run is `gaware_qbins4_i2_soft` / K=256 / `target_entropy` / `soft` with accuracy 0.7794; Δ vs SVD +0.0180, bootstrap CI [+0.0126, +0.0228], P(Δ≤0)=0.000. Verdict: SUCCESS vs K-matched SVD.

## Write-up guidance

Treat small positive deltas cautiously when the bootstrap CI crosses zero. For the coursework text, use `competitive with SVD` unless the CI is clearly positive. The figure `figures/pattern_delta_vs_svd.png` visualises K=128 pattern variants against the K-matched SVD control.

