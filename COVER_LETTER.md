# Cover Letter – Electrical Engineering (Archiv für Elektrotechnik, Springer Nature)

**Date:** November 1, 2026  
**To:** Prof. Dr. Igor Kuzle, Editor-in-Chief, *Electrical Engineering* (*Archiv für Elektrotechnik*)  
Faculty of Electrical Engineering and Computing, University of Zagreb, Croatia  

**Subject:** Submission of Original Research Article (via Springer Nature Transfer Desk):  
**"PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Structural Recovery for Resilient Power Transformer Thermal Dynamics Forecasting"**

Dear Prof. Dr. Igor Kuzle and Esteemed Editorial Board,

I am pleased to submit my original research manuscript entitled **"PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Structural Recovery for Resilient Power Transformer Thermal Dynamics Forecasting"** for consideration as an **Original Research Paper** in *Electrical Engineering (Archiv für Elektrotechnik)*.

### Scientific Context and Alignment with *Electrical Engineering*

Accurate multi-step forecasting of winding hot-spot and top-oil temperatures supports dynamic thermal rating, condition-based asset management, and cooling control for oil-immersed transformers. IEEE Std C57.91 and IEC 60076-7 associate each sustained 6 °C rise above the 110 °C reference with an approximately doubled aging rate.

The work builds directly on the second-generation wavelet forecasting methodology published in this journal by **Bezerra et al. (2024)** (*Electrical Engineering*, Vol. 106, Issue 3, pp. 3017–3029, DOI: 10.1007/s00202-023-02134-5). It advances that line in three respects:

1. **Continuous dyadic Morlet wavelet KAN.** Learnable wavelet responses replace discrete B-spline knot adaptation, avoiding grid-reallocation discontinuities.
2. **Multi-kernel trend-seasonal decomposition and a routing gate.** A neural gate interpolates between a linear projection and the wavelet pathway. Proposition 1 establishes asymptotic linear structural recovery as representational containment: the zero-gate limit contains the multi-kernel linear subspace, not an optimization or test-error guarantee.
3. **Bounded CLPSO warm-start.** A constriction-factor CLPSO searches a four-dimensional space before AdamW fine-tuning and SWA, using 5 particles for 5 generations (exactly 25 evaluations; generation 1 is the initial population).

### Summary of Contributions and Evidence

- **Theoretical scope.** Proposition 1 provides a representational-containment result, stated and proved as such rather than as a generalization guarantee.
- **Benchmark evidence.** Across 16 canonical ETT configurations (lookbacks {96, 336}; horizons {96, 192, 336, 720}), PSO-TCWavKAN has the lowest MSE in 7 of 8 ETTh2 configurations. At ETTh2 L=96 it reduces MSE versus DLinear by 6.3%, 6.2%, 8.5%, and 13.6% at H=96/192/336/720. Its overall mean MSE is 1.4% above DLinear because of long-horizon H=720 dispersion, and it wins 10/16 conditions against DLinear and 12/16 against NLinear.
- **Statistical reporting.** The nominal two-sided Wilcoxon test versus NLinear gives W=28.0, p=0.0386; the Holm-corrected value is p=0.116. The rounded variant gives W=29.0, p=0.0443, Holm p=0.133. Neither is significant after correction at alpha=0.05.
- **Ablation evidence.** A1–A4 comprise 192 three-seed runs; A5 is the five-seed benchmark reference, not a separate three-seed run. The ablation is a descriptive comparison across different seed budgets.
- **Noise diagnostic.** The noise study is an exploratory mixed-source, single-checkpoint diagnostic; the manuscript reports raw MSE only and does not use it for a multi-seed robustness claim.
- **Computational profile.** The model has 44,841 parameters at L=96/H=96 and 796.9 K at L=336/H=720. The reported 3.4 ms batch latency and the 3.3x training-speed advantage over the standard B-spline KAN diagnostic (1.24 vs 4.15 s/epoch; PatchTST Lite is 2.85 s/epoch) are observational measurements from the original Kaggle T4 run; the profiling logs are not archived.

### Reproducibility and Data Availability

- **Open benchmark data:** the standardized ETT datasets (https://github.com/zhouhaoyi/ETDataset).
- **Public code and result archive:** https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting. The release includes the v9.1 implementation, the aggregate benchmark and ablation CSV matrices, a script that reproduces both the rounded and full-precision Wilcoxon/Holm results, and a parameter-count reproduction script for the four released models in Table 7. The Fig. 3 source-data JSON and plotting script are included; the diagnostic script additionally requires the public efficient-kan package and the ETT CSV files.
- **Release boundaries:** per-seed raw logs, model checkpoints, per-seed noise trajectories, predecessor-diagnostic prediction arrays, and Table 7 profiling logs are available from the author on reasonable request.

### Publication Model Declaration

In accordance with Springer Nature policies for hybrid journals, this manuscript is submitted under the **traditional subscription route (without open access fees / 0 APC)**.

### Ethical Declarations

- **Originality:** This manuscript represents original research that has not been published previously and is not under consideration elsewhere.
- **Competing Interests:** The author declares no competing financial or non-financial interests.
- **Funding:** This research received no external grant funding.
- **Transfer Notice:** This manuscript is transferred via the official Springer Nature Transfer Desk following editorial recommendation.

Thank you for your time and consideration.

Sincerely yours,

**Nguyen Phu Toan**  
Department of Information Technology  
Hanoi University of Business and Technology  
29A, Lane 124 Vinh Tuy Street, Vinh Tuy Ward, Hai Ba Trung District, Hanoi, Vietnam  
Email: nphu211206@gmail.com | ORCID: 0009-0005-0529-2131 (https://orcid.org/0009-0005-0529-2131)
