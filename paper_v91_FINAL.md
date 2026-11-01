# PSO-TCWavKAN v9.1 - Manuscript Companion

**Title:** PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Structural Recovery for Resilient Power Transformer Thermal Dynamics Forecasting

**Author:** Nguyen Phu Toan, Department of Information Technology, Hanoi University of Business and Technology, Hanoi, Vietnam. ORCID: 0009-0005-0529-2131.

**Status:** Submitted to *Electrical Engineering (Archiv fuer Elektrotechnik)*, Springer.

> **Authoritative files:** `paper_v91_FINAL.docx` and `paper_v91_FINAL.pdf`. This Markdown file is a companion summary, not a substitute for the manuscript. Equations, tables, figure captions, and the reference list are authoritative in the DOCX/PDF; a full-text Markdown rendering is not maintained.

---

## Abstract (corrected)

Forecasting power transformer thermal dynamics supports predictive maintenance, dynamic thermal rating, and asset protection. Kolmogorov-Arnold Networks (KANs) offer expressive function representation, but dynamic B-spline grid adaptation can destabilize training (the Grid Update Dilemma) and long-horizon accuracy can lag linear baselines. PSO-TCWavKAN integrates continuous dyadic Morlet wavelet filter banks, multi-kernel trend-seasonal decomposition, and an instance-adaptive neural routing gate. The gate provides asymptotic linear structural recovery (Proposition 1): as the gate approaches zero, the hypothesis-space closure contains the multi-kernel linear subspace. This is a representational-containment result, not an unconstrained optimization or test-error guarantee. A constriction-factor CLPSO warm-starts a bounded 4-D search using 5 particles for 5 generations (exactly 25 surrogate evaluations), followed by AdamW fine-tuning and Stochastic Weight Averaging.

Across 16 canonical ETT configurations, PSO-TCWavKAN has the lowest MSE in 7 of 8 ETTh2 configurations. At ETTh2 L=96 it reduces MSE versus DLinear by 6.3%, 6.2%, 8.5%, and 13.6% at H=96/192/336/720. Its overall mean MSE is 1.4% above DLinear because of long-horizon H=720 dispersion. The nominal two-sided Wilcoxon test versus NLinear gives W=28.0, p=0.0386; the Holm-corrected value is p=0.116. The rounded-table variant gives W=29.0, p=0.0443, Holm p=0.133. No comparison is significant after Holm correction at alpha=0.05. A1-A4 comprise 192 three-seed ablation runs; A5 is the five-seed benchmark reference, so the ablation is descriptive. The noise study is an exploratory mixed-source single-checkpoint diagnostic; the manuscript reports raw MSE only and does not use it for a multi-seed robustness claim. The model has 44.8K parameters at L=96/H=96 and 3.4 ms observational batch latency in the original T4 run.

**Keywords:** Kolmogorov-Arnold Networks; wavelet filter bank; particle swarm optimization; transformer temperature forecasting; time-series forecasting; multi-kernel decomposition.

---

## Corrected claims and boundaries

| Item | Corrected statement |
|---|---|
| Benchmark runs | 192 A1-A4 training runs plus the A5 benchmark reference; 288 benchmark evaluations (5 seeds for PSO/DLinear/NLinear, 3 for PatchTST Lite) |
| A5 provenance | Five-seed benchmark mean, not a separate three-seed ablation run; A5 seed columns 1000-3000 are copied from the benchmark CSV |
| NLinear test | Full precision W=28.0, p=0.0386, Holm p=0.116; rounded W=29.0, p=0.0443, Holm p=0.133; neither significant after correction |
| Scale parameter | Raw q = `log_scale` in [-2.5, 0.0]; effective s = softplus(q) + 1e-4 |
| CLPSO budget | 5 particles x 5 generations = 25 surrogate evaluations; generation 1 is the initial population |
| PatchTST Lite | Patch length 16, stride 8 for both L=96 and L=336 |
| Standard deviation | Sample SD (ddof=1), matching the benchmark CSV |
| Table 6 | Exploratory mixed-source diagnostic: three clean entries are benchmark means; the ETTh2 (336,336) row uses the pre-SWA Seed 1000 checkpoint. Noise columns are archived single-checkpoint observations; no within-checkpoint degradation ratio is claimed |
| Table 7 | Observational profiling from the original T4 run; raw profiling logs are not archived |
| Figures 3, 5, 6, 9, 10 | Fig. 3 is a B-spline diagnostic; Figs. 5, 6, 9, 10 are predecessor-checkpoint diagnostics, not v9.1 outputs |

---

## Availability

- Code and aggregate CSV matrices: https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting
- Statistical reproduction: `reproduce_statistical_tests.py`
- Parameter-count reproduction (four released models): `reproduce_table7_params.py`
- Figure 3 source data and script: `figures/dilemma_v2_data.json`, `plot_grid_dilemma_v2.py` (requires the public efficient-kan package and ETT CSV files)
- Data: https://github.com/zhouhaoyi/ETDataset
- Per-seed raw logs, checkpoints, per-seed noise trajectories, predecessor-diagnostic prediction arrays for Figures 5, 6, 9, and 10, and Table 7 profiling logs are available from the author on reasonable request.
