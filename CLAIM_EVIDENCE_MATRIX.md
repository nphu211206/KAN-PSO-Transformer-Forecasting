# Claim-Evidence Matrix

This matrix defines the maximum defensible wording for the manuscript. JSON
records are primary evidence for predecessor campaigns; the v9.1 CSV matrices
and `reproduce_statistical_tests.py` are the evidence base for the PSO-TCWavKAN
claims. Derived figures and prose cannot expand a claim beyond the limits below.

## PSO-TCWavKAN v9.1 claims (K11)

| Claim | Primary evidence | Allowed wording | Prohibited extension |
|---|---|---|---|
| ETTh2 multi-horizon performance | `TABLE1_BENCHMARK_FINAL.csv` (16 configs, 64 rows) | PSO-TCWavKAN has the lowest MSE in 7 of 8 ETTh2 configurations; at L=96 it reduces MSE vs DLinear by 6.3%, 6.2%, and 8.5% at H=96/192/336, and by 13.6% at H=720; at L=336 it wins 3 of 4 horizons | State-of-the-art, universal superiority, or a win at every horizon |
| ETTh1 and aggregate behavior | `TABLE1_BENCHMARK_FINAL.csv` | 10/16 wins vs DLinear, 12/16 vs NLinear, 10/16 vs PatchTST Lite; overall mean MSE 0.2565 vs DLinear 0.2529 (+1.4%); do not describe the model as the overall ETTh1 winner | "Outperforms DLinear overall" or "best model on ETT" |
| Nominal statistical improvement vs NLinear | 16 paired MSE means; `reproduce_statistical_tests.py` | Two-sided Wilcoxon W=28.0, uncorrected p=0.0386; Holm-corrected p=0.1159 across three comparisons; rounded-table variant W=29.0, p=0.0443, Holm p=0.1329; report both variants | "Statistically significant after correction", "p<0.05 confirms superiority", or significance for DLinear/PatchTST |
| Ablation ranking | `TABLE2_ABLATION_FINAL.csv`; A1-A4 three seeds (192 runs), A5 benchmark-referenced five-seed mean | A5 mean MSE 0.21265 for H<=336; A4 +1.63%, A3 +2.37%, A1 +2.44%, A2 +4.62%, descriptive only | Causal attribution under a controlled seed budget, or significance claims; A5 is not an independent three-seed run |
| Noise robustness | `TABLE5_NOISE_ROBUSTNESS_FINAL.csv` | Exploratory mixed-source single-checkpoint diagnostic; report raw MSE only; clean entries for three configs are five-seed benchmark means, not Seed 1000; no within-checkpoint degradation ratio | "Controlled multi-seed noise experiment", "within-checkpoint degradation", "robust across all horizons", or per-seed noise claims |
| Computational efficiency | Original Kaggle T4 observations; `reproduce_table7_params.py` for parameter counts | 44,841 / 796.9K parameters; 3.4 ms batch latency; 3.3x faster training than the standard B-spline KAN diagnostic (1.24 vs 4.15 s/epoch), not PatchTST Lite (2.85 s/epoch); all timing/memory values are observational | Independently reproduced profiling, hardware-independent latency, or energy claims |
| Proposition 1 | Analytical proof in the manuscript | The zero-gate limit contains the multi-kernel linear subspace: a representational-containment result | Optimization guarantee, test-error guarantee, or empirical proof of superiority |
| Figure provenance | Manuscript captions and figure archive | Fig. 3 is a B-spline diagnostic; Figs. 5, 6, 9 and 10 are predecessor-checkpoint diagnostics (KAN_WOA_RegBestClip) | Presenting these figures as PSO-TCWavKAN v9.1 empirical results |
| Release reproducibility | `pso_tcwavkan_v91.py`, CSVs, `reproduce_statistical_tests.py` | Aggregate matrices and statistical tests are reproducible from the release; per-seed raw logs, checkpoints, noise trajectories, and Table 7 profiling logs are not archived | Byte-for-byte replay of every Kaggle run or full independent replication |

## Statistical boundary for v9.1

The benchmark uses five seeds for PSO-TCWavKAN, DLinear and NLinear and three
seeds for PatchTST Lite. The Wilcoxon tests are descriptive paired comparisons
over 16 configurations, not a preregistered confirmatory analysis. The
uncorrected NLinear p-value may be reported as nominal evidence, but the
Holm-corrected result (p=0.1159) is not significant at alpha=0.05. Do not
describe any comparison as significant after correction, and do not pool the
v9.1 benchmark with K01-K10 predecessor results.

## Predecessor claims (K01-K10)

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

## Statistical boundary for predecessor campaigns

The frozen predecessor campaigns use three seeds. Report individual
observations, mean, sample SD, median and range. Do not add p-values,
significance language or equivalence claims. The canonical confirmation uses
five seeds for robustness, but remains descriptive unless a separately
justified inferential design is registered before observing its results.
