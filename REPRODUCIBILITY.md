# Reproducibility Guide - PSO-TCWavKAN v9.1

This package contains the code, benchmark CSV matrices, statistical reproduction script, and publication artifacts for:

**PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Structural Recovery for Resilient Power Transformer Thermal Dynamics Forecasting**

- **Author:** Nguyen Phu Toan (nphu211206@gmail.com)
- **Repository:** https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting
- **Release version:** v9.1
- **Base commit:** `c473e974b145853a779cf7f339dd7ebbda3e0a94` (release content commit; this documentation update follows it)
- **License:** MIT

---

## 1. Environment

- **OS:** platform independent (tested on Windows 11 and Ubuntu 22.04 LTS)
- **Python:** 3.10-3.13
- **Dependencies:**
  - torch >= 2.0.0
  - numpy >= 1.24
  - pandas >= 2.0
  - scipy >= 1.10
  - scikit-learn >= 1.3
  - matplotlib >= 3.7

```bash
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for the full sessions. The demo runs on CPU.

---

## 2. Experimental Data

- **Dataset:** Electricity Transformer Temperature (ETT), https://github.com/zhouhaoyi/ETDataset
- **Files:** `ETTh1.csv`, `ETTh2.csv` (17,420 hourly observations each)
- **Split:** chronological 70/10/20, giving 12,194 training, 1,742 validation, and 3,484 test steps
- **Target:** univariate oil temperature (OT), standardized on training rows only

---

## 3. Running the Code

The execution mode is selected at runtime. It is not hard-coded.

```bash
# Fast smoke test (CPU-friendly, ETTh1, L=96, H=96, 5 epochs)
python pso_tcwavkan_v91.py --mode demo --dataset ETTh1 --lookback 96 --horizon 96 --epochs 5

# Main benchmark: DLinear, NLinear, PSO-TCWavKAN (16 configs x 5 seeds)
python pso_tcwavkan_v91.py --run_mode SESSION1

# PatchTST Lite baseline (16 configs x 3 seeds)
python pso_tcwavkan_v91.py --run_mode SESSION2A

# Ablation sessions A1-A4 (three seeds per cell; A5 comes from SESSION1)
python pso_tcwavkan_v91.py --run_mode SESSION2B1
python pso_tcwavkan_v91.py --run_mode SESSION2B2
python pso_tcwavkan_v91.py --run_mode SESSION2C

# Aggregate session CSVs into TABLE1/TABLE2/TABLE5
python pso_tcwavkan_v91.py --run_mode MERGE

# Help
python pso_tcwavkan_v91.py --help
```

`--mode` is an alias for `--run_mode`. Mode names are case-insensitive.

### Session semantics

| Mode | What it runs | Seeds | Expected output |
|---|---|---|---|
| `DEMO` | Small CLPSO + short training smoke test | 1 | console MSE/MAE only |
| `SESSION1` | DLinear, NLinear, PSO-TCWavKAN on 16 configs + noise diagnostic + Wilcoxon + bootstrap CI | 5 | `pso_tcwavkan_v91_results.csv`, `_pso_log.csv`, `_noise.csv`, `_bootstrap_ci.csv` |
| `SESSION2A` | PatchTST Lite on 16 configs | 3 | `pso_tcwavkan_v91_patchtst.csv`, `_patchtst_ci.csv` |
| `SESSION2B1` | A1 NoHarmonicBank | 3 | `pso_tcwavkan_v91_ablation_s2b1.csv` |
| `SESSION2B2` | A2 NoMultiKernelDecomp | 3 | `pso_tcwavkan_v91_ablation_s2b2.csv` |
| `SESSION2C` | A3 NoNeuralRouter + A4 NoCLPSO | 3 | `pso_tcwavkan_v91_ablation_s2c.csv` |
| `MERGE` | Concatenate session outputs | - | `TABLE1_BENCHMARK_FINAL.csv`, `TABLE2_ABLATION_FINAL.csv`, `TABLE5_NOISE_ROBUSTNESS_FINAL.csv` |

The noise diagnostic runs inside `SESSION1`; there is no separate `noise-only` mode. The `_noise.csv` written by `SESSION1` is a fresh single-checkpoint diagnostic produced from the checkpoint trained in that session. It does not byte-for-byte reproduce the archived mixed-source Table 6 in the manuscript, whose first three clean references are five-seed benchmark means and whose noise columns are archived observations.

---

## 4. Reproducing the Statistical Tests

```bash
python reproduce_statistical_tests.py --csv results/TABLE1_BENCHMARK_FINAL.csv
```

Expected output:

- Rounded Table 4 values: NLinear W=29.0, p=0.0443, Holm p=0.1329; DLinear W=46.0, p=0.2744; PatchTST Lite W=35.0, p=0.0934.
- Full-precision CSV values: NLinear W=28.0, p=0.0386, Holm p=0.1159; DLinear W=46.0, p=0.2744; PatchTST Lite W=35.0, p=0.0934.

No comparison is significant at alpha=0.05 after Holm correction. The manuscript reports the nominal NLinear p-value together with the corrected value.

### 4.1 Parameter counts (Table 7)

```bash
python reproduce_table7_params.py
```

Expected output: DLinear 18,624 / 485,280; NLinear 9,312 / 242,640; PatchTST Lite 169,570 / 1,993,810; PSO-TCWavKAN 44,841 / 796,921 for (L,H) = (96,96) and (336,720). The Standard B-Spline KAN baseline is an external diagnostic and is not instantiated by this script. Timing, latency, and memory remain observational.

### 4.2 Figure 3

```bash
python plot_grid_dilemma_v2.py
```

The script regenerates `figures/grid_update_dilemma_v2.png` and `figures/dilemma_v2_data.json`. It requires the public `efficient-kan` package and the ETT CSV files under `ETDataset/ETT-small`. The archived JSON is included so the plotted arrays can be inspected without retraining.

---

## 5. Protocol Details That Are Easy to Get Wrong

- **CLPSO budget:** 5 particles x 5 generations = exactly 25 surrogate evaluations. Generation 1 evaluates the initial population; it is not an extra five evaluations on top of the loop. The code counts and asserts the total.
- **Scale parameter:** the search variable is `q = log_scale` on [-2.5, 0.0]. The model uses `s = softplus(q) + 1e-4`, giving effective s approximately in [0.079, 0.693]. It is not a log10(s) search.
- **PatchTST Lite:** patch length 16, stride 8 for both L=96 and L=336 (50% overlap). The code does not override these defaults.
- **Standard deviation:** `MSE_std` in the benchmark CSV is the sample standard deviation (ddof=1). The code uses `_sample_std`, which applies ddof=1 whenever more than one seed is present.
- **A5 ablation:** A5 is the five-seed benchmark mean, not a separate three-seed run. A1-A4 use three seeds (192 training runs). The A5 row in `TABLE2_ABLATION_FINAL.csv` is benchmark-referenced; its seed columns 1000-3000 are copied from the benchmark CSV, while its mean includes seeds 4000 and 5000.
- **Table 6 noise diagnostic:** exploratory and mixed-source. The first three clean entries are five-seed benchmark means; the ETTh2 (336,336) row uses the pre-SWA Seed 1000 checkpoint throughout. The noise columns are archived single-checkpoint observations; per-seed noise trajectories are not included. Because the clean and noise sources differ for the first three rows, Table 6 reports raw MSE only and makes no within-checkpoint degradation or multi-seed robustness claim. The `_noise.csv` produced by `SESSION1` is a fresh diagnostic from the trained checkpoint, not a byte-for-byte replay of the archived Table 6.
- **Figures 3, 5, 6, 9, 10:** Fig. 3 is a B-spline diagnostic, not a PSO-TCWavKAN comparison. Figs. 5, 6, 9, and 10 are predecessor-checkpoint diagnostics, not v9.1 outputs.

---

## 6. Artifact Manifest

| Artifact | Description |
|---|---|
| `pso_tcwavkan_v91.py` | Official PyTorch implementation: model, CLPSO, RevIN, training, evaluation, CLI |
| `reproduce_statistical_tests.py` | Two-sided Wilcoxon + Holm reproduction (rounded and full precision) |
| `reproduce_table7_params.py` | Table 7 parameter-count reproduction (analytical, no GPU) |
| `plot_grid_dilemma_v2.py` | Figure 3 B-spline diagnostic script |
| `figures/dilemma_v2_data.json` | Archived Figure 3 source arrays |
| `results/TABLE1_BENCHMARK_FINAL.csv` | 16-config benchmark, 64 model-config rows, per-seed values |
| `results/TABLE2_ABLATION_FINAL.csv` | 80-cell ablation matrix (A1-A4 three seeds; A5 benchmark-referenced) |
| `results/TABLE5_NOISE_ROBUSTNESS_FINAL.csv` | Mixed-source noise diagnostic |
| `paper_v91_FINAL.docx` / `.pdf` | Manuscript |
| `COVER_LETTER_ELECTRICAL_ENGINEERING.docx` / `.pdf` | Cover letter |
| `DECLARATIONS_ELECTRICAL_ENGINEERING.docx` / `.pdf` | Declarations |

Not included: per-seed raw training logs, model checkpoints, per-seed noise trajectories, predecessor-diagnostic prediction arrays for Figures 5, 6, 9, and 10, and Table 7 profiling logs. These were generated in Kaggle Cloud environments and are available from the author on reasonable request. The public release therefore reproduces the aggregate CSV matrices, parameter counts, and statistical tests, but not a byte-for-byte replay of every training run.
