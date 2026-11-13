# PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11+cu130-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Target Journal: NCAA](https://img.shields.io/badge/Target%20Journal-Neural%20Computing%20%26%20Applications-003366?style=for-the-badge)](https://link.springer.com/journal/521)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Official PyTorch implementation, verified experimental archive, 600 DPI publication figures, and submission package**

**Author:** [Phu Toan Nguyen](mailto:nphu211206@gmail.com)  
*Faculty of Information Technology, Hanoi University of Business and Technology (HUBT), Hanoi 100000, Vietnam*  
ORCID iD: [0009-0005-0529-2131](https://orcid.org/0009-0005-0529-2131)

---

## Executive Summary

Accurately forecasting mineral-oil power transformer top-oil temperature (`OT`) across multi-day horizons ($H \in \{96, 192, 336, 720\}$ hours) underpins cellulose insulation aging monitoring under IEEE Std C57.91-2011 / IEC Std 60076-7:2018 and utility Dynamic Thermal Rating (DTR).

**PSO-TCWavKAN** replaces discrete B-spline grid updates with infinitely differentiable ($C^\infty$) multi-octave continuous Morlet wavelet KAN blocks, addressing the B-spline *Grid Update Dilemma* (optimizer momentum resets and compact-support boundary gradient vanishing). The framework couples:

1. **Invertible RevIN with Detached Statistics & Denominator Clamping** ($\epsilon_\gamma = 10^{-4}$).
2. **Multi-Kernel Symmetric Moving-Average Decomposition** ($k \in \{25, 49\}$) with linear macro-trend and seasonal anchors.
3. **Scale-Adaptive Patch Tokenization & Depthwise Inter-Patch Mixing** ($\mathcal{O}(B D N)$ backbone token-mixing complexity, **Theorem 1**).
4. **Float32 Adaptive Dual FFT Spectral Head & Instance-Adaptive Neural Router** satisfying exact non-degenerate asymptotic DLinear subspace recovery (**Proposition 1**).
5. **Leak-Free Clerc-Constrained CLPSO Pre-Optimization** ($\chi = 0.72984$, $c_1=c_2=2.05$, $S=20$, $G_{\max}=5$) evaluated strictly within the training partition ($t \le 8,640$, 85/15 surrogate split).

![Architectural Schematic](figures/fig1_architecture.png)

---

## Verified 16-Configuration Benchmark Summary

All six paradigms (`PSO-TCWavKAN Master`, `DLinear`, `NLinear`, `PatchTST Lite`, `B-Spline KAN`, and fixed-init `TCWavKAN (No PSO)`) were trained under identical canonical 12/4/4 ETT splits (`train` $[0, 8640)$, `val` $[8640-L, 11520)$, `test` $[11520-L, 14400)$) across three independent seeds $\{2024, 2025, 2026\}$ with Bessel-corrected sample standard deviation (`ddof=1`) on dual NVIDIA Tesla T4 GPUs (Python 3.13.16, PyTorch 2.11.0+cu130).

| Metric / Comparison | Result (16 Canonical ETTh1 & ETTh2 Configs) |
| :--- | :--- |
| **16-Config Mean MSE / MAE** | `NLinear` (0.1298 / 0.2748), `TCWavKAN (No PSO)` (0.1301 / 0.2766), **`PSO-TCWavKAN Master` (0.1312 / 0.2774)**, `B-Spline KAN` (0.1322 / 0.2803), `PatchTST Lite` (0.1386 / 0.2853), `DLinear` (0.1568 / 0.3025) |
| **vs. Co-Evaluated DLinear** | **+13.22%** mean MSE reduction (up to **+49.92%** at ETTh1 $L=96, H=720$); **14 / 0 / 2** W/D/L; $W^+=125.0, W^-=11.0, z=+2.947$; **$p_{\mathrm{exact}} = 0.0017$ ($1.68\times 10^{-3}$)**, **Holm $p_{\mathrm{adj}} = 0.0084$** |
| **vs. PatchTST Lite** | **+4.14%** mean MSE reduction; **12 / 0 / 4** W/D/L; $W^+=110.0, W^-=26.0, z=+2.172$; **$p_{\mathrm{exact}} = 0.0290$**, Holm $p_{\mathrm{adj}} = 0.1160$ |
| **vs. Non-Stationary ETTh2** | Outperforms `NLinear` on **5 / 8** `ETTh2` configurations and earns **4** first-place MSE / **5** first-place MAE wins overall |
| **Edge Footprint (Tesla T4)** | **49.8K–800.7K** params; **0.050–0.122 MB** inference VRAM ($B=1$); **16.041–34.763 MB** train VRAM; **3.61–3.88 ms** latency ($B=1$) |

---

## Repository Structure

```text
.
├── KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py   # Complete dual-GPU benchmark, ablation, CLPSO & edge runner
├── generate_ncaa_figures.py             # Generates all 6 publication figures at 600 DPI from CSV/NPZ
├── ingest_kaggle_results.py             # Zip-Slip-safe ingestor & LaTeX table synchronizer (--verify_only)
├── update_paper_markdown.py             # Synchronizes Markdown tables from results/*.csv
├── generate_paper_v91_docx.py           # Builds publication-grade paper_v91_FINAL.docx
├── build_submission_docs.py             # Builds COVER_LETTER_NCAA.docx and DECLARATIONS_NCAA.docx
├── verify_latex_package.py              # End-to-end forensic validator (LaTeX, BibTeX, DPI, DOCX, stale audit)
├── paper_ncaa.tex                       # Springer Nature (sn-jnl, sn-mathphys-num) manuscript source
├── sn-bibliography.bib                  # 76 verified BibTeX references (100% bijective with paper_ncaa.tex)
├── paper_v91_FINAL.md / .docx           # Synchronized Markdown & Word manuscript
├── COVER_LETTER_NCAA.md / .docx         # NCAA cover letter
├── DECLARATIONS_NCAA.md / .docx         # NCAA standalone declarations
├── figures/                             # 6 publication figures at 600 DPI (fig1..fig6.png)
├── results/                             # 8 CSV/JSON ground-truth tables & environment manifest
├── predictions/                         # 144 NPZ arrays (48 Master + 80 Seed-2024 Baselines + 16 Ground Truth)
├── tests/                               # Automated verification suite (pytest)
├── requirements.txt
├── pytest.ini
├── REPRODUCIBILITY.md                 # Complete environment, split & protocol guide
├── SUBMISSION_CHECKLIST.md            # NCAA Editorial Manager upload & QA checklist
├── ZENODO_RELEASE_GUIDE.md            # GitHub webhook & Zenodo DOI archival steps
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

---

## Quick Verification & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting.git
cd KAN-PSO-Transformer-Forecasting
pip install -r requirements.txt
```

### 2. Verify All Results, 144 NPZ Tensors, Statistical Tests, Figures & Manuscript Package
```bash
python ingest_kaggle_results.py --verify_only
python verify_latex_package.py
pytest tests/test_challenger_m6_benchmarks.py -v
```

### 3. Regenerate All 6 Figures & Synchronized Documents
```bash
python generate_ncaa_figures.py
python update_paper_markdown.py
python generate_paper_v91_docx.py
python build_submission_docs.py
```

### 4. Re-Run Full Experiments (Dual-GPU T4 / Single GPU)
```bash
# Quick smoke verification (~2 mins)
python KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py --mode SMOKE_TEST

# Full Option A 4-Suite Benchmark, Ablation, CLPSO Trace & Edge Profiling (~60–90 mins on T4x2)
python KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py --mode FULL_OPTION_A
```

---

## Data Availability & Citation

- **Benchmark Datasets:** Public ETT datasets (`ETTh1.csv`, `ETTh2.csv`) at [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset).
- **Archival DOI:** Pending Zenodo release webhook (`https://doi.org/10.5281/zenodo.24200012`). See [ZENODO_RELEASE_GUIDE.md](ZENODO_RELEASE_GUIDE.md).

```bibtex
@article{nguyen2026psotcwavkan,
  title={PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov--Arnold Networks for Power Transformer Thermal Dynamics Forecasting},
  author={Nguyen, Phu Toan},
  journal={Neural Computing and Applications (Submitted)},
  year={2026},
  note={Code and archive: https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting}
}
```
