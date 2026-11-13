# Reproducibility Guide — PSO-TCWavKAN (NCAA Submission Package)

This repository provides a complete, deterministic, leak-free reproduction package for:

> **PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting**  
> Phu Toan Nguyen (ORCID: [0009-0005-0529-2131](https://orcid.org/0009-0005-0529-2131)), *Neural Computing and Applications* (Submission)

---

## 1. Execution Environment

Verified experimental configuration (recorded in [`results/ENVIRONMENT_MANIFEST.json`](results/ENVIRONMENT_MANIFEST.json)):
- **Hardware:** 2 × NVIDIA Tesla T4 (Dual-GPU coordinator shard execution)
- **Python:** `3.13.16` (`PyTorch 2.11.0+cu130`)
- **Total Wall-Clock Runtime (`FULL_OPTION_A`):** `9,124.03 s` (Shard 0: `7,737.58 s`; Shard 1: `9,122.56 s`)

Install local verification and plotting dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Canonical Data Split & Leak-Free Protocol

- **Dataset:** Public Electricity Transformer Temperature (`ETTh1.csv`, `ETTh2.csv`) from [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), univariate Top-Oil Temperature (`OT`, $C=1$).
- **12/4/4-Month Canonical Window Split:**
  - `train`: `[0, 8640)`
  - `val`: `[8640 - L, 11520)`
  - `test`: `[11520 - L, 14400)`
- **Standardization:** `StandardScaler` fit strictly on `[0, 8640)` (`ETTh1`: $\mu = 17.1283^\circ\mathrm{C}, \sigma = 9.1765^\circ\mathrm{C}$; `ETTh2`: $\mu = 26.8720^\circ\mathrm{C}, \sigma = 11.5847^\circ\mathrm{C}$).
- **Leak-Free CLPSO Surrogate:** 15% subsample strictly inside `[0, 8640)`, partitioned 85%/15% into disjoint surrogate train/validation windows. Zero validation or test leakage occurs during swarm optimization.
- **Seeds & Variance Convention:** Three independent seeds `{2024, 2025, 2026}` with Bessel-corrected sample standard deviation (`ddof=1`).

---

## 3. Verification Without GPU Re-Training (Seconds)

All `144` NPZ arrays (`16` ground-truth `trues`, `48` Master 3-seed `preds`, `80` baseline Seed-2024 `preds`) and `8` CSV/JSON result files are included directly in this release.

```bash
# 1. Verify schema completeness across results/ (8 files) and predictions/ (144 NPZs)
python ingest_kaggle_results.py --verify_only

# 2. Verify LaTeX balance, 76/76 bijective BibTeX citations, 600 DPI figures, DOCX & stale-token sweep
python verify_latex_package.py

# 3. Verify Bessel ddof=1 statistics, all 144 NPZs, exact Wilcoxon & Holm tests, and TeX/MD parity
pytest tests/test_challenger_m6_benchmarks.py -v

# 4. Regenerate all 6 600-DPI figures and synchronize Word / Markdown manuscripts
python generate_ncaa_figures.py
python update_paper_markdown.py
python generate_paper_v91_docx.py
python build_submission_docs.py
```

---

## 4. Full From-Scratch Benchmark Execution (GPU)

The entire 4-suite empirical study lives inside a single self-contained runner: [`KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py`](KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py).

```bash
# Full dual-GPU (or single-GPU fallback) reproduction:
python KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py --mode FULL_OPTION_A --data_dir /path/to/ETT-small
```

| Suite | Scope | Key Output Files |
| :--- | :--- | :--- |
| **Suite 1** | 16 configs × 6 models × 3 seeds + CLPSO hyperparameters + 144 NPZs | `results/MASTER_BENCHMARK_RESULTS.csv`, `results/CLPSO_PER_CONFIG_LOG.csv`, `predictions/*.npz` |
| **Suite 2** | 5-variant ablation (`full`, `no_revin`, `no_tcd`, `spline_kan_block`, `no_clpso`) on `ETTh1 (336, 720)` × 3 seeds | `results/ABLATION_RESULTS.csv` |
| **Suite 3** | Convergence trace (`CLPSO seeds 42,43,44` vs `Std PSO` & `LDW-PSO`, $G=10$) & swarm sensitivity ($S \in \{5,10,20,40\}$) | `results/CLPSO_CONVERGENCE_LOG.csv`, `results/CLPSO_SENSITIVITY_LOG.csv` |
| **Suite 4** | Synchronized Tesla T4 VRAM/latency profiling ($B=1$), two-sided exact Wilcoxon + Holm tests, env manifest | `results/EDGE_PROFILE.csv`, `results/WILCOXON_STATS.csv`, `results/ENVIRONMENT_MANIFEST.json` |

