# K08-K09 Checkpoint-Contract Sensitivity Appendix

K08 and K09 are shown side by side and are never pooled. The difference is descriptive, not a causal estimate of checkpoint selection alone: K09 is a separately preregistered campaign with a distinct source hash and added diagnostic instrumentation.

| Dataset | Condition | K08 MSE | K09 MSE | Mean delta K09-K08 | K09 lower seeds | Equal within 1e-6 |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | DLinear | 0.060650 +/- 0.000238 | 0.064489 +/- 0.001843 | 0.003839 | 0/5 | 0/5 |
| ETTh1 | KAN_WOA_RegBestClip | 0.164912 +/- 0.014927 | 0.164913 +/- 0.014927 | 0.000000 | 0/5 | 4/5 |
| ETTh1 | moderate_retain | 1.695296 +/- 2.716982 | 0.417187 +/- 0.159815 | -1.278109 | 3/5 | 0/5 |
| ETTh1 | none_retain | 0.398114 +/- 0.123440 | 0.277007 +/- 0.039970 | -0.121106 | 4/5 | 0/5 |
| ETTh2 | DLinear | 0.128562 +/- 0.000231 | 0.127925 +/- 0.000598 | -0.000637 | 4/5 | 1/5 |
| ETTh2 | KAN_WOA_RegBestClip | 0.195536 +/- 0.028677 | 0.195536 +/- 0.028679 | -0.000000 | 2/5 | 0/5 |
| ETTh2 | moderate_retain | 0.580647 +/- 0.243673 | 0.369897 +/- 0.081077 | -0.210750 | 4/5 | 0/5 |
| ETTh2 | none_retain | 0.286589 +/- 0.100378 | 0.329198 +/- 0.071826 | 0.042609 | 1/5 | 0/5 |

Interpretation: validation-best selection reduces late-epoch damage for several Grid Update conditions, but it does not reverse the DLinear-versus-KAN ranking. DLinear's K08 final-epoch and K09 validation-best results should not be treated as identical evaluation contracts.
