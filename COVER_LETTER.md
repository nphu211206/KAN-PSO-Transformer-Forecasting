# Cover Letter – Journal of Engineering and Applied Science (Springer Nature)

**Date:** October 19, 2026  
**To:** Editor-in-Chief, *Journal of Engineering and Applied Science* (JEAS)  

**Subject:** Submission of Original Research Article titled:  
**"PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Recovery for Resilient Power Transformer Thermal Dynamics Forecasting"**

Dear Editor-in-Chief,

I am pleased to submit our original research manuscript entitled **"PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Recovery for Resilient Power Transformer Thermal Dynamics Forecasting"** for consideration as a Research Article in the *Journal of Engineering and Applied Science*.

### Context and Suitability for JEAS
Accurate, multi-step-ahead thermal forecasting of high-voltage electrical power transformers is a cornerstone of smart grid condition-based maintenance, dynamic thermal rating (DTR), and catastrophic insulation failure prevention. While Kolmogorov-Arnold Networks (KANs) have recently captured significant attention as an expressive functional approximation paradigm, their direct application to volatile industrial time-series has been impeded by two systemic bottlenecks:
1. **The Grid Update Dilemma:** Training optimization instability and validation loss spikes caused by discrete B-spline knot reallocation during dynamic grid adaptation;
2. **The Linear Model Dilemma:** Empirical degradation across extended forecasting horizons (96 to 720 steps) when benchmarked against simple linear baselines (e.g., DLinear).

This investigation directly aligns with the stated scope of the *Journal of Engineering and Applied Science* in electrical engineering, computational intelligence, and industrial optimization, following the rigorous standards demonstrated by recent high-impact publications in JEAS on hybrid neural architectures and swarm metaheuristics (such as ACKLSTM-KAN, PSO-SVR, and PSO-GBM).

### Core Scientific and Practical Contributions
The manuscript delivers a mathematically rigorous, structurally protected, and empirically validated framework:
1. **Resolution of the Grid Update Dilemma by Construction:** We replace discrete B-splines with continuous, multi-scale dyadic Morlet wavelet filter banks across $K_w = 5$ octave scales. Being continuously differentiable, our parameterization entirely eliminates knot reallocation events and gradient discontinuities.
2. **Asymptotic Linear Recovery Guarantee (Proposition 1):** We introduce a parameterized neural routing gate blending linear and wavelet-KAN pathways, rigorously proving that in the topological closure of the hypothesis space as the routing gate tends to zero, the network identically recovers the generalized multi-kernel linear projection model, and recovers canonical DLinear when decomposition kernel lengths match ($K_1=K_2=25$).
3. **CLPSO Global Warm-Start:** To bypass non-convex wavelet ripple traps, a Comprehensive Learning Particle Swarm Optimizer with constriction factor 0.7298 performs an efficient global warm-start over a fast surrogate landscape prior to AdamW fine-tuning and Stochastic Weight Averaging (SWA).
4. **Dominant Benchmark Performance:** Evaluated across 16 canonical Electricity Transformer Temperature configurations (ETTh1 and ETTh2; lookbacks in {96, 336}; horizons in {96, 192, 336, 720}; 3 independent seeds), PSO-TCWavKAN achieves a 75.0% win rate against NLinear and 62.5% against DLinear and PatchTST Lite, conclusively dominating 7 out of 8 horizons on the non-stationary ETTh2 benchmark (achieving up to 13.6% MSE reduction over DLinear).
5. **Systematic Ablation and Operational Resilience:** An 80-cell ablation campaign proves that omitting multi-kernel trend decomposition causes a severe +4.62% error surge in standard operational forecasting ($H \le 336$). Under harsh 30% Gaussian noise perturbation (sigma = 0.3), degradation is tightly bounded within 1.4% on ETTh2 ($L=336, H=336$), demonstrating robust operational resilience against substation electromagnetic interference.
6. **Edge SCADA Feasibility:** With 44.8 K parameters in the base configuration (scaling to 796.9 K at L=336, H=720, maintaining a 2.5x to 4.0x parameter economy relative to PatchTST Lite) and a batch inference latency of 3.4 ms on an NVIDIA Tesla T4 GPU, the architecture is readily deployable on industrial edge computing devices.

### Reproducibility and Data Availability
In the spirit of complete scientific transparency and reproducible research:
- **Source Code and Artifacts:** All source code, pretrained models, configuration files, and benchmark scripts are publicly available in the open-source repository: https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting
- **Benchmark Data:** All evaluation datasets (ETTh1 and ETTh2) are public open-source benchmarks.

### Ethical and Policy Declarations
- **Originality:** This manuscript is an original work that has not been published previously and is not currently under consideration for publication elsewhere.
- **Competing Interests:** The author declares no competing financial or non-financial interests.
- **Funding:** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
- **Ethics and Consent:** The study uses publicly available telemetry data and involves no human participants or animal subjects.
- **Author Approval:** The author has read and approved the final version of the manuscript for submission.

Thank you very much for your time, consideration, and handling of our submission. I look forward to receiving the reviewers' constructive feedback.

Sincerely yours,

**Nguyen Phu Toan**  
Department of Information Technology  
Hanoi University of Business and Technology  
29A, Lane 124 Vinh Tuy Street, Vinh Tuy Ward, Hai Ba Trung District, Hanoi, Vietnam  
Email: nphu211206@gmail.com  
ORCID: 0009-0005-0529-2131 (https://orcid.org/0009-0005-0529-2131)  
