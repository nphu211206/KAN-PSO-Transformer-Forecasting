# Cover Letter for Manuscript Submission to *Neural Computing and Applications* (Springer Nature)

**Date:** November 2026

**To:**  
Prof. John MacIntyre  
Editor-in-Chief, *Neural Computing and Applications*  
Springer Nature London / University of Sunderland, United Kingdom  

**Subject:** Submission of Original Research Article:  
*"PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting"*  

**Dear Prof. John MacIntyre and Esteemed Members of the Editorial Board,**

I am pleased to submit my original research manuscript entitled **"PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting"** for peer review and publication consideration as an **Original Research Paper** in *Neural Computing and Applications* (NCAA).

---

### 1. Journal Scope Alignment & Motivation

*Neural Computing and Applications* is recognized internationally as a premier venue for original advancements bridging neural architectures, evolutionary swarm intelligence, and industrial engineering telemetry. Our manuscript directly engages with and advances methodological directions recently published in *Neural Computing and Applications*:

1. **Kolmogorov–Arnold Networks (KAN)**: Advancing beyond piecewise linear finite-element edge functions such as *P1-KAN* (Warin, *NCAA* 2026, 38:703, DOI: 10.1007/s00521-026-12354-y) into infinitely differentiable ($C^\infty$) continuous Morlet wavelet edge activations tailored to smooth oscillatory thermal inertia.
2. **End-to-End Continuous Wavelet Processing**: Unifying multi-octave wavelet filter banks and depthwise inter-patch convolutions within a single differentiable pipeline, resolving the offline decomposition separation of *Wavelet-LSTM* (Tamilselvi et al., *NCAA* 2025, 37(17):10521–10530, DOI: 10.1007/s00521-024-10561-z).
3. **Non-Stationary Distribution-Shift Neutralization**: Pairing detached-moment Reversible Instance Normalization (with clamped denominator safety) and a non-degenerate DLinear recovery router, avoiding the multi-subnet pre-training overhead of *GCINet* (Fan et al., *NCAA* 2025, 37(5):3227–3244, DOI: 10.1007/s00521-024-10692-3) and building upon frequency-enhanced decompositions such as *FEDAF* (Yang et al., *NCAA* 2024, 36(26):16271–16288, DOI: 10.1007/s00521-024-09937-y).
4. **Leak-Free Metaheuristic Swarm Pre-Optimization**: Coupling Clerc-constrained Comprehensive Learning Particle Swarm Optimization (CLPSO) with deep wavelet neural pre-conditioning strictly inside the training partition.

---

### 2. Core Scientific Innovations & Technical Contributions

Our manuscript addresses key theoretical and empirical challenges in Long-Term Time Series Forecasting (LTSF) for electric power transformer monitoring:

- **Continuous Morlet WavKAN with Smooth Analytic Gradients**: Standard B-spline KANs suffer from the *Grid Update Dilemma*—periodic knot re-gridding (requiring 18 grid updates in our three-seed ablation study) perturbs optimizer momentum buffers in Adam/AdamW, while compact spline support risks zero derivatives during extreme thermal excursions. Replacing discrete splines with continuous real Morlet wavelets equipped with softplus-bounded dilations ($\sigma_{d,k} > 0$) guarantees smooth $C^\infty$ analytic gradients over $\mathbb{R}$.
- **Invertible RevIN with Detached Statistics and Denominator Protection**: Detached batch-time moments paired with $\operatorname{safe\_}\boldsymbol{\gamma} = \operatorname{sign}(\boldsymbol{\gamma}) \odot \max(|\boldsymbol{\gamma}|, 10^{-4})$ prevent autograd leakage and inverse singularity explosions—critical under non-stationary regime shift, where removing RevIN induces a +59.40% MSE degradation in our ablation experiments.
- **Clerc-Constrained CLPSO Metaheuristic Pre-Optimization**: Using Maurice Clerc's constriction factor ($\chi = 0.72984, \phi = c_1 + c_2 = 4.1$) on a strictly isolated 15% training-only 3-epoch surrogate proxy ($\le 8,640$ timestamps), CLPSO tunes the 4D hyperparameter vector $(\omega_0, \log s, \log\alpha, \log_{10}\eta)$ without test-set leakage, establishing an $S=20$ accuracy–compute Pareto knee (84.4% of $S=40$ gain at 49.4% runtime).
- **Formal Recovery Guarantee (Proposition 1) & Linear Backbone Complexity (Theorem 1)**: Proposition 1 mathematically proves that PSO-TCWavKAN non-degenerately recovers the affine-RevIN multi-kernel DLinear structural subspace as routing gates vanish ($\Delta_{\max} = 0.0$ verified experimentally), while Theorem 1 establishes linear $\mathcal{O}(BDN)$ backbone token-mixing scaling versus $\mathcal{O}(BDN^2)$ self-attention.
- **Physical Grounding in IEEE Std C57.91-2011 / IEC Std 60076-7:2018**: Predictive horizons ($H \in \{96, 192, 336, 720\}$ hours) directly anchor top-oil temperature (`OT`) forecasting to Arrhenius cellulose insulation aging kinetics and utility Dynamic Thermal Rating (DTR).

---

### 3. Rigorous Empirical, Statistical, and Edge-Profiling Validation

All six model families (`PSO-TCWavKAN Master`, `DLinear`, `NLinear`, `PatchTST Lite`, `B-Spline KAN`, and fixed-init `TCWavKAN (No PSO)`) were re-trained from scratch under identical canonical 12/4/4 ETT splits and three independent seeds (2024, 2025, 2026) with Bessel-corrected standard deviation ($ddof=1$) on dual NVIDIA Tesla T4 GPUs (Python 3.13.16, PyTorch 2.11.0+cu130):

- **Multi-Horizon Benchmark Accuracy**: Achieves an average **13.22% MSE reduction** over co-evaluated `DLinear` across all 16 canonical configurations (improving 14/16 settings and reaching up to **+49.92%** at ETTh1 $L=96, H=720$ and **0.0548 ± 0.0045** at $L=336, H=96$; `ETTh1` $L=96, H=96$: **0.0587 ± 0.0010**), outperforming `DLinear` (0.1568 / 0.3025), `PatchTST Lite` (0.1386 / 0.2853), and `B-Spline KAN` (0.1322 / 0.2803) with a global 16-config mean MSE/MAE of **0.1312 / 0.2774**, and beating `NLinear` on 5 of 8 non-stationary `ETTh2` settings.
- **Wilcoxon–Holm Statistical Significance**: Exact two-sided paired Wilcoxon signed-rank tests confirm statistically significant superiority over `DLinear` ($W^+ = 125.0, W^- = 11.0, z = +2.947, p_{\mathrm{exact}} = 1.68 \times 10^{-3}, p_{\mathrm{asymp}} = 3.20 \times 10^{-3}$; Holm–Bonferroni adjusted $p_{\mathrm{adj}} = 0.0084 < 0.01$; $14/0/2$ wins) and over `PatchTST Lite` ($12/0/4$ wins, +4.14% mean gain, $p_{\mathrm{exact}} = 0.0290 < 0.05$, Holm $p_{\mathrm{adj}} = 0.1160$), alongside transparent comparison against `B-Spline KAN`, `TCWavKAN (No PSO)`, and `NLinear`.
- **Edge-Embedded SCADA Resource Profile**: Requires 49.8K–800.7K parameters (less than half of `PatchTST Lite` at $L=336, H=720$), consumes **0.050–0.122 MB** inference VRAM ($B=1$; **16.041–34.763 MB** training VRAM), and achieves **3.61–3.88 ms** synchronized T4 single-sample latency.

---

### 4. Mandatory Statements & Publication Route

- **Originality:** This manuscript is completely original, has not been published previously, and is not under consideration elsewhere.
- **Publication Route (0-APC Traditional Subscription Model):** In accordance with Springer Nature hybrid journal policies, this manuscript is submitted under the **Traditional Subscription Model (0-APC Free Publishing, zero author-facing processing charges)**.
- **Ethical Compliance & Full Reproducibility:** All mandatory Springer Nature declarations are included in both the manuscript and standalone Declarations file. The complete PyTorch training code, dual-GPU experiment coordinator, 600 DPI figure scripts, CSV logs, and 144 NPZ prediction/ground-truth arrays are openly hosted at `https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting` (and slated for Zenodo archival at `https://doi.org/10.5281/zenodo.xxxxxxx`).

Thank you very much for your time, consideration, and editorial stewardship of our submission.

Sincerely yours,

**Phu Toan Nguyen**  
Faculty of Information Technology, Hanoi University of Business and Technology (HUBT)  
No. 29A, Lane 124, Vinh Tuy Street, Vinh Hung Ward, Hai Ba Trung District, Hanoi 100000, Vietnam  
Email: `nphu211206@gmail.com`  
ORCID iD: [0009-0005-0529-2131](https://orcid.org/0009-0005-0529-2131)
