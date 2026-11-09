# Cover Letter for Manuscript Submission to *Neural Computing and Applications* (Springer Nature)

**Date:** November 2026

**To:**  
Prof. John MacIntyre  
Editor-in-Chief, *Neural Computing and Applications*  
Springer Nature London / University of Sunderland, United Kingdom  

**Subject:** Submission of Original Research Article:  
*"PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting"*  

**Dear Prof. John MacIntyre and Esteemed Members of the Editorial Board,**

I am honored to submit my original research manuscript entitled **"PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting"** for peer review and publication consideration as an **Original Research Paper** in *Neural Computing and Applications* (NCAA).

---

### 1. Journal Scope Alignment & Motivation

*Neural Computing and Applications* is globally recognized as the preeminent venue for pioneering theoretical advancements in neural network algorithms, swarm intelligence, and their robust deployment to demanding real-world engineering telemetry. Our research directly engages with the cutting-edge architectural directions and foundational methodologies recently ratified within *Neural Computing and Applications*:

1. **Kolmogorov–Arnold Networks (KAN)**: Advancing beyond the piecewise linear finite element formulation of *P1-KAN* (Warin, *NCAA* 2026, DOI: 10.1007/s00521-026-12354-y) into continuous, infinitely differentiable wavelet edge activations.
2. **Multiresolution Wavelet Processing**: Unifying multi-scale continuous wavelet transforms within an end-to-end differentiable neural backbone, surmounting the decoupled offline filter limitations of *Wavelet-LSTM* (Tamilselvi et al., *NCAA* 2025, DOI: 10.1007/s00521-024-10561-z).
3. **Distribution-Shift Neutralization**: Enhancing reversible normalization and depthwise temporal convolutions beyond the three-subnet complexity of *GCINet* (Fan et al., *NCAA* 2025, DOI: 10.1007/s00521-024-10692-3).
4. **Metaheuristic Swarm Pre-Optimization**: Synergizing Clerc-constrained Comprehensive Learning Particle Swarm Optimization (CLPSO) with deep neural parameter initialization.

---

### 2. Core Scientific Innovations & Technical Contributions

Our manuscript resolves critical theoretical and empirical bottlenecks in Long-Term Time Series Forecasting (LTSF) for high-stakes electric power infrastructure:

- **Continuous Morlet WavKAN with Smooth Analytical Gradients**: Standard B-spline KANs suffer from the "Grid Update Dilemma"—discrete knot adaptation disrupts optimizer momentum accumulators in algorithms like Adam, precipitating severe loss spikes and numerical instability. We replace discrete splines with continuous Morlet wavelets featuring softplus-bounded dilations ($\sigma > 0$), guaranteeing well-behaved analytical gradients everywhere on $\mathbb{R}$.
- **Invertible Reversible Instance Normalization (RevIN) with Detached Statistics**: We implement RevIN with detached mean and variance statistics and safe denominator clamping ($\epsilon_{\gamma} = 10^{-4}$), eliminating gradient interference and preventing scale explosion under severe distribution shift.
- **Clerc-Constrained CLPSO Metaheuristic Pre-Optimization**: To navigate the non-convex multi-scale wavelet parameter landscape, we deploy a Comprehensive Learning Particle Swarm Optimizer governed by Maurice Clerc's constriction factor ($\chi = 0.72984$), mathematically guaranteeing bounded swarm velocities and preventing premature convergence on an 85/15 leak-free surrogate proxy.
- **Strict Linear Asymptotic Complexity $\mathcal{O}(DN)$ & Proposition 1**: We provide formal mathematical proofs showing that PSO-TCWavKAN achieves strictly linear computational complexity $\mathcal{O}(DN)$, bypassing the quadratic bottleneck $\mathcal{O}(DN^2)$ of Transformers. Furthermore, Proposition 1 proves asymptotic linear structural recovery, demonstrating that our architecture gracefully contains DLinear as a non-degenerate subspace when the neural routing gate vanishes ($g(x) \to 0$).
- **Physical Domain Grounding in IEEE Std C57.91-2011**: We ground our predictive modeling in the thermodynamic and insulation degradation equations of IEEE Std C57.91-2011 and the Arrhenius loss-of-life model for mineral-oil-immersed power transformers, directly enabling proactive Dynamic Thermal Rating (DTR).

---

### 3. Rigorous Empirical & Statistical Validation

The empirical validity of PSO-TCWavKAN has been comprehensively evaluated across all 16 canonical configurations on the standard Electricity Transformer Temperature benchmarks ($\text{ETTh1}$ and $\text{ETTh2}$) across three independent random seeds (2024, 2025, 2026) using sample Bessel correction ($ddof=1$):

- **Benchmark Superiority**: Achieves an average **12.30% MSE reduction** over the co-evaluated DLinear baseline across the complete 16-configuration suite, with top performance at $H=96$ on $\text{ETTh1}$ ($\text{MSE} = 0.0591 \pm 0.0016$).
- **Statistical Significance**: Validated via the non-parametric Wilcoxon signed-rank test across all 16 configurations ($W^+ = 136.0, W^- = 0.0, z = -3.516$), yielding an exact permutation $p$-value of **$p = 3.05 \times 10^{-5}$** and asymptotic $p = 4.38 \times 10^{-4}$ ($p < 0.001$), establishing an unblemished **16 wins / 0 draws / 0 losses** record.
- **Edge-Ready Computational Profile**: Sub-12 ms observational inference latency and peak GPU memory footprint under 342 MB VRAM on an NVIDIA Tesla T4, confirming immediate viability for real-time SCADA edge deployment.

---

### 4. Mandatory Statements & Publication Route

- **Originality**: This manuscript represents 100% original research. It has not been published previously, is not currently under consideration for publication elsewhere, and will not be submitted to any other journal while under evaluation at *Neural Computing and Applications*.
- **Publication Model (Traditional Subscription Route / 0-APC Free Publishing)**: In accordance with Springer Nature hybrid publishing policies, this manuscript is submitted under the **Traditional Subscription Model (100% 0-APC Free Publishing, zero author-facing processing charges / $0.00 USD / 0 VNĐ)**.
- **Ethical Compliance & Reproducibility**: The paper contains all seven mandatory Springer Nature declarations. Complete PyTorch source code, CLPSO routines, configuration files, and prediction arrays are openly archived on GitHub and Zenodo.

Thank you very much for your time, consideration, and editorial stewardship of our manuscript.

Sincerely yours,

**Phu Toan Nguyen**  
Faculty of Information Technology, Hanoi University of Business and Technology (HUBT)  
No. 29A, Lane 124, Vinh Tuy Street, Vinh Hung Ward, Hai Ba Trung District, Hanoi 100000, Vietnam  
Primary Email: `nphu211206@gmail.com`  
ORCID iD: [0009-0005-0529-2131](https://orcid.org/0009-0005-0529-2131)  
