# Claim-Evidence Matrix

This matrix defines the maximum defensible wording for the manuscript. JSON
records are primary evidence; derived CSV, figures and prose cannot expand a
claim beyond the limits below.

| Claim | Primary evidence | Allowed wording | Prohibited extension |
|---|---|---|---|
| Corrected KAN training improves ETTh1/h96 mean MSE by 35.47% | Targeted KAN-WOA records and Stage 1 `KAN_WOA_RegBestClip` records | HPO-consistent regularization, validation checkpointing and clipping materially improved the repeated configuration under the pilot protocol | The correction improves every KAN, dataset or protocol |
| Corrected KAN is close to PatchTST but weaker than DLinear at ETTh1/h96 | `targeted_results_all/records` and `kan_improvement_results/records` | Report the three-seed means and SD under `pilot_60_20_20_v1` | KAN is SOTA or statistically equivalent to PatchTST |
| DLinear look-back 336 is the lowest observed pilot-protocol mean | Stage 1/2 internal DLinear records | It is the strongest observed same-protocol configuration | It is universally superior or a canonical ETT benchmark winner |
| KAN residual variants are rejected by validation | Stage 1/2 hybrid JSON, selected epoch 0 and residual gain 0 | Controlled negative result; final predictions equal the selected DLinear checkpoint | The hybrid improves DLinear |
| Moderate Adam reset is worse for every paired ETTh1 seed | K06 moderate retain/reset records | Exact paired-seed statement is allowed | Generalize to every schedule or dataset |
| Adam reset is not a sufficient Grid Update remedy | K06/K07 non-initial event logs and outcomes | Reset executed correctly and did not produce reliable stabilization; it can amplify failure | Stale Adam moments are proven not to contribute causally |
| Aggressive Grid Update is unstable | ETTh1 targeted ablation, K06 and K07 | Large mean/SD and catastrophic seed outcomes under tested configurations | Every aggressive schedule or KAN implementation is unstable |
| ETTh2 provides cross-dataset confirmation | K07 records | DLinear beats corrected KAN on all three ETTh2 seeds; reset has a worse mean-level direction | Independent external validation or reset is worse on every ETTh2 seed |
| Metaheuristic HPO explores a useful mixed KAN configuration space | Historical 18 HPO records | Descriptive landscape evidence with equal stated limitations | WOA/PSO significantly outperform other optimizers |
| KAN-metaheuristic novelty | Verified literature review, not experiment records | Novelty must be limited to the controlled training-consistency/Grid Update analysis if supported by the final review | First KAN-metaheuristic or first PSO-KAN study |
| Canonical DLinear versus corrected KAN | 40 K09 uniform-checkpoint JSON records | DLinear is stronger in all ten paired dataset-seed comparisons under canonical ETTh1/ETTh2 h96; K09 uses validation-best selection for every condition | Universal DLinear superiority beyond the tested task |
| Canonical moderate versus no Grid Update | K09 `none_retain` and `moderate_retain` records | Moderate is worse in 3/5 ETTh2 and 4/5 ETTh1 paired seeds, with worse means on both datasets | Every Grid Update schedule is harmful |
| K08-to-K09 checkpoint sensitivity | Separate K08/K09 JSON and `analysis/publication/CHECKPOINT_CONTRACT_APPENDIX.md` | Checkpoint contract materially changes several Grid Update outcomes; K08/K09 are reported side by side, never pooled | Validation-best selection alone causally explains every K08/K09 difference |
| Canonical multi-horizon model comparison | 120 K10 v2 JSON/NPZ records and `analysis/k10_multihorizon/` | DLinear has the lowest mean MSE in all eight dataset-horizon cells; it beats corrected KAN in 38/40 and PatchTST in 36/40 paired horizon-seed comparisons | Universal DLinear superiority, SOTA, or equivalence of models at ETTh2/h720 |
| K10 long-horizon behavior | K10 h720 prediction NPZ arrays and fixed-midpoint figure | All three direct models smooth or attenuate some abrupt excursions at the fixed origin; aggregate ranking comes from all-origin MSE | The fixed origin proves deployment behavior for every period or transformer |

## Statistical boundary

The frozen campaigns use three seeds. Report individual observations, mean,
sample SD, median and range. Do not add p-values, significance language or
equivalence claims. The canonical confirmation uses five seeds for robustness,
but remains descriptive unless a separately justified inferential design is
registered before observing its results.
