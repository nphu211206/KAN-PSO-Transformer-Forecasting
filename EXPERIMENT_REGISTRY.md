# KAN-HPO Experiment Evidence Registry

**Frozen through:** 2026-08-17 K10 v2 canonical multi-horizon campaign  
**Total experiment records:** 380

K10 v2 completed all 120 registered tasks and passed the dedicated artifact
audit. V1 was stopped after its log revealed an implementation-only 360-task
matrix instead of the preregistered 120; no partial v1 record is evidence. An
initial v2 archive was lost with an interactive Kaggle session and was never
analyzed; the download-safe rerun below is the sole K10 evidence archive.

This registry identifies the authoritative result locations. JSON records are
primary evidence; CSV files, figures and manuscript tables are derived.

K09 was preregistered before execution and completed all 40 tasks. The frozen
Kaggle input bundle SHA-256 is
`59D42770B33D95BCEDD5B3DC2F950B9BB183D9D1A5400C942E0A6EBCDFE02F3C`;
the executed result archive has a different, separately recorded hash below.
The ZIP remains authoritative. Exact JSON copies under
`analysis/k09_diagnostics/records/` support the aggregate audit and derived
tables without altering or repacking the archive.

| Campaign | Record folder | Records | Status | Protocol | Source combined SHA-256 |
|---|---|---:|---|---|---|
| Historical six-algorithm HPO | `my_results/kaggle/working/experiments/results/` | 18 | Complete | Historical pilot HPO | Legacy records do not embed source hash |
| Targeted ETTh1 baselines and ablations | `targeted_results_all/records/` | 96 | Complete | `pilot_60_20_20_v1` | `f0b1d0a65405db8b0ed33bcbc8cae6e01358ee6e1e765b37b9f54b42786834e3` |
| KAN improvement Stage 1 | `kan_improvement_results/records/` | 21 | Complete | `pilot_60_20_20_v1` | `4bcab98a0cea9785689e505486a2321f4d70c9ffc806f99bbadc08f2fb7718ce` |
| KAN improvement Stage 2 | `kan_improvement_stage2_results/records/` | 9 | Complete | `pilot_60_20_20_v1` | `362b1edfb67e6c69b66d6956f8c76ffed4c605ab2e793326773c2fa6ba076416` |
| ETTh1 optimizer-state causal | `optimizer_state_causal_results/records/` | 12 | Complete | `pilot_60_20_20_v1` | `bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516` |
| ETTh2 external confirmation | `etth2_confirmation_results/records/` | 24 | Complete | `pilot_60_20_20_v1` | `bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516` |
| Canonical ETTh1/ETTh2 confirmation | `canonical_confirmation_results/records/` | 40 | Complete | `ett_canonical_12_4_4_v1` | `bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516` |
| K09 uniform-checkpoint diagnostics | `analysis/k09_diagnostics/records/` | 40 | Complete | `ett_canonical_12_4_4_v1` | `138ddf1b1feffc4fa03cb75c3e6746b67ace7c401cf397f3503f7d8635aa8a49` |
| K10 v2 canonical multi-horizon | `analysis/k10_multihorizon/records/` | 120 | Complete | `ett_canonical_12_4_4_v1` | `edb30890ea8ba8097e6072b620f6c13c5cef9d26281d094b329cdcbd9b700fb5` |

## Dataset Identity

The values below are SHA-256 hashes of the exact LF-delimited CSV bytes used
on Kaggle. Windows may materialize the same CSV content with CRLF line endings;
`audit_evidence.py` canonicalizes only CRLF to LF before checking the local
copy, while still checking all values, rows and timestamps separately in the
frozen protocol tests. No numeric or timestamp normalization is performed.

| Dataset | SHA-256 | Use |
|---|---|---|
| ETTh1 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` | Pilot and canonical K08-K10 campaigns |
| ETTh2 | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` | Pilot and canonical K08-K10 campaigns |

All modern campaigns forecast univariate `OT` and fit `StandardScaler` on
training rows only. Pilot campaigns use the chronological 60/20/20 protocol
and seeds 1000, 1001 and 1002. The canonical campaign uses official ETT
12/4/4-month boundaries and seeds 1000 through 1004. Pilot and canonical
results must remain in separate comparison tables.

## Archive Integrity

| Archive | SHA-256 |
|---|---|
| `my_results.zip` | `11E031023C8A75EAA2EA13A2DA29F4F624B84EF6AAD6D694139EA1E800F08023` |
| `targeted_results.zip` | `54122E8AC41D33154ACC311937D8E63EE2448C6B399478BAC2343120758BFE4B` |
| `targeted_results_03.zip` | `5FBD15FD93148F29AAA7C1F03A7CBA286482E4FDAECFB6179F221A01F5CA183F` |
| `kan_improvement_results.zip` | `FA46D6E44E6CCE17CDD60252E1D88E4F21676A988EAE9A011167E6833BF1AFAC` |
| `kan_improvement_stage2_results.zip` | `7E75ABCB636022E34E9E2E2C3C9ECB9A3D7846812340E7BAB01476694BD30EFD` |
| `optimizer_state_causal_results.zip` | `C6AC363E337A58D9638011CACA055F3861AC3F498757CC6A6877C72EB35028C3` |
| `etth2_confirmation_results.zip` | `3F4F3198B9D5DEB2DEE2AE17037397FE5A7226A19C0CD7DE6D1A9A915E72E5AB` |
| `canonical_confirmation_results.zip` | `7F0A3DA091857B0CAF8BD72DAD1850EA2A3A1A005079B05D639D540FA48DA378` |
| `canonical_uniform_best_diagnostics.zip` | `0685DEAE0D453FE2135AB8855FC68CB635D53663F09699F1FA537A014D124004` |
| `canonical_multihorizon_baselines_v2.zip` | `BD3BEA46869C555F89243C954533B827A5151F6F8BD006BAF9693A321281D281` |

## Frozen Conclusions

1. Correcting KAN regularization and validation-best selection materially
   improves KAN, but DLinear remains the strongest tested same-protocol model.
2. Resetting Adam state after Grid Update is not an effective stabilization
   mechanism here. Moderate reset is worse than retain for every paired ETTh1
   seed, and ETTh2 shows the same mean-level direction under the same protocol.
3. Aggressive Grid Update is unstable. Reset can amplify rather than remove
   catastrophic behavior.
4. ETTh2 confirms that corrected KAN remains weaker than DLinear and that Grid
   Update behavior is dataset-dependent.
5. Under K09's uniform minimum-validation-MSE checkpoint rule, DLinear
   outperforms corrected KAN in all ten dataset-seed pairs. Mean MSE is
   `0.064489` versus `0.164913` on ETTh1 and `0.127925` versus `0.195536`
   on ETTh2.
6. Under K09, moderate Grid Update is worse than no update in four of five
   ETTh1 seeds and three of five ETTh2 seeds; its mean MSE is worse on both
   datasets. The older all-five ETTh2 statement belongs only to K08's
   final-epoch contract.
7. K08 and K09 demonstrate checkpoint-contract sensitivity. Validation-best
   selection greatly reduces the extreme Grid Update outcomes but does not
   make repeated Grid Update competitive or reverse the DLinear/KAN ranking.
8. K09 adds saved prediction arrays, original-unit residuals, error by forecast
   step, inference latency, peak GPU memory and selected-checkpoint evidence.
   It does not add external-domain or multivariate validity.
9. K10 adds canonical horizons 96, 192, 336 and 720 for DLinear, PatchTST and
   corrected KAN. DLinear has the lowest mean MSE in all eight cells and lower
   MSE than corrected KAN in 38/40 paired horizon-seed comparisons. The
   ETTh2/h720 mean gap narrows and prevents an absolute every-seed claim.

## Campaign Boundary

K09 is registered in `CANONICAL_UNIFORM_BEST_DIAGNOSTICS_PREREGISTRATION.md`
and analyzed by `analysis/analyze_k09_diagnostics.py`. K08 and K09 must remain
separate in all tables and claims. The present h96 manuscript does not require
more training, but K10 has been separately authorized to strengthen canonical
multi-horizon coverage. It uses a new preregistration, campaign ID, source
bundle and result directory.

## K10 Execution History

| Campaign | Bundle | Planned records | Status | Protocol | Bundle SHA-256 |
|---|---|---:|---|---|---|
| K10 v1 canonical multi-horizon | `KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES.py` | 120 intended / 360 enumerated | Abandoned before completion; do not use | `ett_canonical_12_4_4_v1` | `6198464861CA9F482704FEAE242E4A10AE7B231672DC03DDC2CDE6EAAC6DE4E4` |
| K10 v2 initial wrapper | `KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2.py` | 120 | Completed interactively but archive lost; do not use | `ett_canonical_12_4_4_v1` | `4751DF8BF1F74AD8CAE2F2C6C167035E5D0E4507B64F725460636A563C943CC3` |
| K10 v2 download-safe rerun | `KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py` | 120 | Complete; authoritative archive registered | `ett_canonical_12_4_4_v1` | `ED3639ABF7310DED1C5EB092004FA64E4419C5D5139846CE92A8FA31092DA44F` |

K10 v2 is governed by the original preregistration plus
`CANONICAL_MULTIHORIZON_BASELINE_PREREGISTRATION_V2.md`. The authoritative ZIP
passed `audit_k10.py`; all derived tables and figures come from that ZIP.


## Campaign K11: PSO-TCWavKAN v9.1 Main Benchmark Campaign (2026-10-25)

- **Campaign ID:** K11_v91_BENCHMARK_FINAL
- **Preregistration & Protocol:** Canonical 70/10/20 chronological split across 17,420 total hourly observations (ETTh1: 8,545 train/val/test steps; ETTh2: 8,545 steps) from the standard ETT benchmark (`https://github.com/zhouhaoyi/ETDataset`).
- **Code Artifact:** `github_release/pso_tcwavkan_v91.py` (authoritative release code, platform-independent PyTorch 2.0+).
- **Execution Platform:** Kaggle Cloud GPU (NVIDIA Tesla T4 GPU accelerator, 16 GB VRAM) executing modular sessions: `SESSION1` (Main Benchmark), `SESSION2A` (Ablations A1-A2), `SESSION2B1/B2/C` (Ablations A3-A5), `SESSION3` (Noise Perturbation Stress Testing).
- **Random Seed Protocol:**
  - Proposed model (PSO-TCWavKAN), DLinear, NLinear: evaluated across **5 independent seeds**: `{1000, 2000, 3000, 4000, 5000}`.
  - PatchTST Lite: evaluated across **3 independent seeds**: `{1000, 2000, 3000}`.
  - Total primary benchmark evaluations: 16 configs × 4 models × multi-seed = 288 evaluations.
- **Ablation Campaign:** 5 structural variants (A1: NoHarmonicBank, A2: NoMultiKernelDecomp, A3: NoNeuralRouter, A4: NoCLPSO, A5: Full PSO-TCWavKAN) × 16 lookback–horizon configurations × 3 independent seeds = 240 evaluations.
- **Noise Robustness Campaign:** Controlled zero-mean Gaussian perturbation $\sigma \in \{0.0, 0.1, 0.2, 0.3\}$ across 4 representative configurations on reference checkpoint (Seed 42).
- **Model Checkpoint Selection:** $\Theta^* = \arg\min_{\Theta \in \{\Theta_{\text{best}}, \Theta_{\text{SWA}}\}}$ Val_Loss$(\Theta)$.
- **Authoritative CSV Artifacts:**
  - `TABLE1_BENCHMARK_FINAL.csv` / `github_release/results/TABLE1_BENCHMARK_FINAL.csv`: 5-seed aggregated benchmark performance with exact means, standard deviations, and individual seed outputs.
  - `TABLE2_ABLATION_FINAL.csv`: 3-seed aggregated ablation study across standard ($H \le 336$) and extended ($H = 720$) regimes.
  - `TABLE5_NOISE_ROBUSTNESS_FINAL.csv`: Multi-level noise degradation trajectory.
