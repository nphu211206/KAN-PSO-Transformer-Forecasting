# Reproducibility Guide — PSO-TCWavKAN v9.1

This package reproduces the experimental evaluation, statistical tests, and publication artifacts for:
**“PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Structural Recovery for Resilient Power Transformer Thermal Dynamics Forecasting”**

- **Author:** Nguyen Phu Toan (nphu211206@gmail.com)
- **Repository:** https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting
- **Release Version:** v9.1 (Official Submission Release)
- **License:** MIT License

---

## 1. Environment Requirements

- **Operating System:** Platform independent (tested on Windows 11 and Ubuntu 22.04 LTS)
- **Python:** 3.10 – 3.13
- **Primary Dependencies:**
  - `torch >= 2.0.0`
  - `numpy >= 1.24`
  - `pandas >= 2.0`
  - `scipy >= 1.10`
  - `scikit-learn >= 1.3`
  - `matplotlib >= 3.7`
  - `python-docx >= 1.1`
  - `PyMuPDF (fitz) >= 1.23`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Experimental Data

The benchmark utilizes the open-source Electricity Transformer Temperature (ETT) telemetry dataset:
- **Repository:** https://github.com/zhouhaoyi/ETDataset
- **Files used:** `ETTh1.csv`, `ETTh2.csv` (17,420 hourly observations each)
- **Partition:** 70% Training (12,194 steps), 10% Validation (1,742 steps), 20% Testing (3,484 steps)

---

## 3. Running Experiments via Authoritative Script

The end-to-end model code and evaluation routine is contained in `pso_tcwavkan_v91.py`. To execute on GPU hardware (e.g. NVIDIA Tesla T4 on Kaggle Cloud):

```bash
# Execute main benchmark (DLinear, NLinear, PSO-TCWavKAN across 5 seeds: 1000, 2000, 3000, 4000, 5000)
python pso_tcwavkan_v91.py --run_mode SESSION1

# Execute PatchTST Lite baseline (3 seeds: 1000, 2000, 3000)
python pso_tcwavkan_v91.py --run_mode SESSION2A

# Execute Ablations A1-A5 across 16 configurations (3 seeds each)
python pso_tcwavkan_v91.py --run_mode SESSION2B1
python pso_tcwavkan_v91.py --run_mode SESSION2B2
python pso_tcwavkan_v91.py --run_mode SESSION2C

# Execute Synthetic Noise Perturbation Stress Tests (sigma = 0.0, 0.1, 0.2, 0.3)
python pso_tcwavkan_v91.py --run_mode SESSION3
```

---

## 4. Reproducing Statistical Tests & Verification Suites

To independently verify the statistical comparisons and reproducibility constraints:

```bash
# Reproduce Wilcoxon signed-rank tests across the 16-condition benchmark matrix
python reproduce_statistical_tests.py

# Run comprehensive Round 4 verification audit (33 checks)
python audit_round4_complete.py

# Run supplemental Round 4 verification audit (19 checks)
python audit_round4_supplemental.py
```

---

## 5. Authoritative Artifact Manifest

| Artifact File | Description |
|---|---|
| `pso_tcwavkan_v91.py` | Official PyTorch implementation of PSO-TCWavKAN, CLPSO warm-start, RevIN, and training protocols |
| `TABLE1_BENCHMARK_FINAL.csv` | 16-configuration multi-seed benchmark performance (means, sample std, bootstrap 95% CI, individual seeds) |
| `TABLE2_ABLATION_FINAL.csv` | 80-cell ablation matrix (5 variants × 16 configurations × 3 seeds = 240 runs) |
| `TABLE5_NOISE_ROBUSTNESS_FINAL.csv` | Synthetic Gaussian noise perturbation trajectory across representative configurations |
| `paper_v91_FINAL.docx` / `.pdf` | Complete submission manuscript formatted for Electrical Engineering |
| `COVER_LETTER_ELECTRICAL_ENGINEERING.docx` / `.pdf` | Submission cover letter |
| `DECLARATIONS_ELECTRICAL_ENGINEERING.docx` / `.pdf` | Submission declarations (ethics, competing interests, data/code availability) |

Checkpoints and detailed per-seed raw training logs are archived in Kaggle Cloud GPU environments and are available from the author upon reasonable request.
