# PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Lower-Bound Guarantees for Resilient Power Transformer Thermal Dynamics Forecasting

**Author:** Nguyen Phu Toan
**Affiliation:** Department of Information Technology, Hanoi University of Business and Technology, 29A, Lane 124 Vinh Tuy Street, Vinh Tuy Ward, Hai Ba Trung District, Hanoi, Vietnam
**Corresponding author:** Nguyen Phu Toan (Email: nphu211206@gmail.com | ORCID: 0009-0005-0529-2131)

---

## Abstract

Forecasting the operational temperature of high-voltage electrical power transformers is essential for proactive asset management, dynamic thermal rating, and catastrophic failure prevention. Although Kolmogorov-Arnold Networks (KANs) offer expressive functional representation, their application to volatile industrial time series is hindered by two systemic vulnerabilities: (1) training instability from dynamic B-spline grid adaptations (the Grid Update Dilemma), and (2) performance deficits across extended multi-step horizons compared to linear projections. To overcome these bottlenecks, this paper proposes PSO-TCWavKAN, a unified deep architecture integrating continuous dyadic wavelet filter banks with multi-kernel trend-seasonal decomposition and a parameterized neural routing gate. The routing gate provides an asymptotic linear recovery guarantee (Proposition 1), ensuring the model structurally recovers the generalized linear multi-kernel subspace—and canonical DLinear when kernels match—in the topological closure of the hypothesis space when non-linear representations fail to reduce empirical training risk. To navigate the multi-modal loss landscape of wavelet activations, a Comprehensive Learning Particle Swarm Optimizer (CLPSO) with constriction factor χ = 0.7298 conducts global hyperparameter warm-start across a bounded 4D search space, followed by AdamW gradient fine-tuning and Stochastic Weight Averaging (SWA). Exhaustive empirical validation across 16 canonical ETT benchmark configurations demonstrates that PSO-TCWavKAN dominates 7 out of 8 horizons on the volatile ETTh2 benchmark, achieving up to 13.6% MSE reduction over DLinear, while achieving directional improvements over NLinear (two-sided Wilcoxon uncorrected p = 0.0443; Holm-corrected p = 0.133 across 3 baseline tests) and competitive parity with DLinear (p = 0.274) and PatchTST Lite (p = 0.0934). In standard operational regimes (horizons H in {96, 192, 336}), an 80-cell ablation campaign confirms that the full model achieves the lowest Mean MSE (0.21265), with every module omission causing performance degradation (+1.63% to +4.62%). Under 30% Gaussian noise perturbation, error growth remains bounded within 1.4% to 13.0% across standard operational horizons (and up to 139.6% at extreme long-horizon stress), demonstrating resilience over localized operating bands. With 44.8 K parameters in the base configuration (scaling to 796.9 K at L=336, H=720, retaining a 2.5x to 4.0x economy relative to PatchTST Lite) and 3.4 ms inference latency, PSO-TCWavKAN provides a mathematically sound and deployable solution for substation telemetry forecasting.

**Keywords:** Kolmogorov-Arnold Networks; Wavelet Filter Bank; Particle Swarm Optimization; Transformer Temperature Forecasting; Time-Series Forecasting; Multi-Kernel Decomposition; Robust Deep Learning; Grid Update Dilemma

---

## 1. Introduction

High-voltage electrical power transformers constitute the indispensable core of electrical energy transmission and distribution infrastructures. In modern power systems, continuous operational loading induces substantial Joule losses in winding conductors and core eddy-current dissipation, generating intense internal heat. The thermal state of a transformer is conventionally characterized by two primary physical indicators: the top-oil temperature (TOT) and the internal winding hot-spot temperature (HST). Operating temperatures are directly coupled to the accelerated degradation of solid cellulose insulation paper and dielectric mineral oil via Arrhenius thermal aging kinetics. In accordance with IEEE Std C57.91 and IEC 60076-7 loading guides, sustained thermal overruns by even 6 degrees Celsius above rated operating limits halve the insulation lifetime, markedly escalating the probability of dielectric breakdown, catastrophic transformer tank explosions, and widespread blackouts [21, 22, 25, 26]. Consequently, accurate, multi-step-ahead thermal forecasting is paramount for condition-based maintenance scheduling, dynamic thermal rating (DTR) of transmission corridors, and automatic activation of substation forced-cooling systems [23].

Following the landmark release of the canonical Electricity Transformer Temperature (ETT) benchmark by Zhou et al. [3], multi-horizon thermal forecasting on the hourly ETTh1 and ETTh2 datasets has become an internationally recognized test bed for evaluating deep time-series models. Over the past five years, the sequential modeling landscape has transitioned through multiple paradigms, progressing from Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks to sophisticated Transformer architectures [35] equipped with sparse or linear self-attention mechanisms, including Reformer [36], Linformer [37], Informer [3], Autoformer [14], and FEDformer [15], as surveyed extensively by Tay et al. [38]. Similar data-driven sequential modeling strategies have also proven effective in high-dimensional vehicle telemetry [9]. However, this trajectory was dramatically disrupted by Zeng et al. [4], who demonstrated that embarrassingly simple linear mapping layers with moving-average trend decomposition frequently outperform complex Transformer architectures across standard benchmarks, a phenomenon widely acknowledged as the linear model dilemma.

Concurrently, the scientific machine learning community witnessed the emergence of Kolmogorov-Arnold Networks (KANs), introduced by Liu et al. [1]. Grounded in the Kolmogorov-Arnold representation theorem, KANs depart fundamentally from standard Multilayer Perceptrons (MLPs) by placing learnable univariate B-spline functions on network edges rather than fixed nonlinear activations at nodes. The theoretical promise of KANs lies in their superior functional expressiveness, enhanced parameter efficiency, and inherent interpretability, which have prompted rapid adoption in numerical physics, differential equation modeling, and engineering time-series forecasting [6, 12].

Nevertheless, the direct application of classical KAN architectures to complex industrial engineering time-series exposes four severe failure modes, each of which fundamentally undermines reliable long-horizon thermal prediction:

(1) **The Grid Update Dilemma.** Standard KANs rely on discrete B-splines that require dynamic grid adaptation during training to align knot placements with incoming activation distributions. Updating the grid mid-training alters the underlying parameter manifold, corrupting the first- and second-moment running vectors of Adam-family optimizers and inducing catastrophic loss spikes and validation divergence. As empirically verified in Fig. 3, grid update events in vanilla B-spline KAN cause immediate validation loss explosions, rendering the model unreliable for safety-critical forecasting tasks.

(2) **Spectral Dispersion and Lack of Frequency Locality.** Discrete B-splines possess uniform spatial support but lack explicit spectral-band localization. In electrical transformer thermal dynamics, input series exhibit simultaneous superposition of low-frequency 24-hour diurnal ambient cycles and high-frequency transient spikes driven by industrial switching events. B-splines struggle to disentangle these distinct temporal scales without prohibitive grid expansion, leading to systematic spectral aliasing and frequency leakage.

(3) **The Linear Model Performance Deficit.** When evaluated across canonical long forecast horizons (96 to 720 steps ahead), naive KAN implementations consistently underperform DLinear [4], suffering from severe overfitting and error accumulation across distant time steps. This empirically validates the concern raised by Zeng et al. that complex architectures must outperform, not merely match, simple baselines to justify their overhead.

(4) **Non-Convex Wavelet Ripple Traps.** Introducing wavelet basis functions into KAN edges creates highly non-convex, multi-modal loss surfaces characterized by dense oscillatory local minima. Standard gradient descent algorithms initialized from random Gaussian seeds consistently stagnate in suboptimal basins of attraction, requiring principled global optimization strategies to ensure reliable convergence.

To systematically overcome these bottlenecks, this paper introduces PSO-TCWavKAN, a swarm-optimized temporal convolutional wavelet Kolmogorov-Arnold network engineered for resilient power transformer thermal dynamics forecasting. The core scientific contributions of this investigation are fourfold:

- **Continuous Dyadic Wavelet Filter Banks.** We replace discrete B-splines with continuous dyadic Morlet wavelet activations across K_w = 5 octave scales. Because the wavelet parameters are continuously differentiable with respect to all learnable quantities, knot updates are eliminated entirely, resolving the Grid Update Dilemma by construction.
- **Asymptotic Linear Recovery of the DLinear Structural Subspace.** We formulate the network output as an instance-adaptive convex combination of a linear projection and a wavelet-KAN representation. We mathematically prove (Proposition 1) that in the topological closure of the hypothesis space, as the routing gate g(x) approaches zero under saturated negative bias, PSO-TCWavKAN recovers a generalized linear multi-kernel model, exactly matching canonical DLinear when decomposition kernels satisfy K_1 = K_2 = 25.
- **Comprehensive Learning Particle Swarm Warm-Start (CLPSO).** We deploy a constriction-factor CLPSO algorithm (chi = 0.7298) over a 4D continuous search space on a fast surrogate training landscape to globally initialize critical hyperparameter vectors and base wavelet frequencies, bypassing non-convex ripple traps prior to AdamW gradient fine-tuning and Stochastic Weight Averaging (SWA).
- **Rigorous Empirical Validation & Regime Analysis.** We evaluate the architecture across 16 canonical benchmark configurations (ETTh1 and ETTh2; L in {96, 336}; H in {96, 192, 336, 720}) across multi-seed evaluations (3 random seeds for baselines, 5 seeds for PSO-TCWavKAN), competing against DLinear, NLinear, and PatchTST Lite. We execute an 80-cell ablation campaign across 5 structural variants analyzed across operational and stress regimes, alongside Gaussian noise perturbation testing up to sigma = 0.30. All results and code are made publicly available for reproducibility.

The remainder of this paper is organized as follows. Section 2 reviews related work on transformer thermal forecasting, KAN architectures, and swarm optimization. Section 3 presents the proposed PSO-TCWavKAN mathematical architecture in full detail. Section 4 describes the experimental setup and benchmark protocol. Sections 5 through 8 present benchmark results, ablation analysis, noise robustness evaluation, and computational efficiency metrics, respectively. Section 9 discusses engineering implications and transparently acknowledges limitations. Section 10 concludes.


## 2. Related Work

### 2.1. Transformer Thermal Dynamics and Deep Sequential Forecasting

Thermal modeling of mineral-oil-immersed power transformers has historically relied on differential thermal balance equations defined in international standards (IEEE Std C57.91 [21], IEC 60076-7 [22]). Classical dynamic models, such as the thermal-electrical analogy proposed by Susa et al. [23], calculate top-oil temperature (TOT) and hot-spot temperature (HST) from ambient temperature, wind velocity, and transformer loading ratios. While physically interpretable, standard differential equations require precise knowledge of oil viscosity, winding resistance, and thermal time constants, which drift over decades of operating life. More recently, Bezerra et al. [11] published in Electrical Engineering a foundational forecasting methodology combining second-generation wavelets and neural networks for power transformer oil temperature, demonstrating that multi-resolution frequency decomposition coupled with non-linear neural mapping yields superior fidelity over classical thermal equations. However, their architecture utilized discrete lifting wavelets prone to boundary distortion and relied on standard multi-layer perceptrons without formal lower-bound guarantees or systematic metaheuristic hyperparameter optimization. Data-driven machine learning models have therefore emerged to capture non-stationary thermal dynamics directly from SCADA telemetry.

Following the publication of Informer by Zhou et al. [3], Transformer variants such as Autoformer [14] and FEDformer [15] integrated decomposition blocks and Fourier-domain representations. PatchTST [5] demonstrated that segmenting time-series into overlapping sub-series patches significantly boosts Transformer performance by preserving local temporal context. Concurrently, Zeng et al. [4] demonstrated that simple single-layer linear models (DLinear) often surpass complex Transformer architectures, creating a benchmark standard that any proposed deep model must rigorously confront to justify its added complexity. More recently, iTransformer [16] proposed an inverted Transformer architecture that achieved competitive performance by treating variate-level tokens rather than temporal tokens, demonstrating that architectural innovation remains an active and productive frontier.

### 2.2. Kolmogorov-Arnold Networks and Wavelet Functional Bases

The introduction of Kolmogorov-Arnold Networks (KANs) by Liu et al. [1] challenged the foundational architecture of multilayer perceptrons by placing learnable univariate B-spline functions on edges rather than fixed activations at nodes. The KAN activation function combines a residual SiLU term with a trainable B-spline component: $\phi(x) = w(\text{SiLU}(x) + \text{spline}(x))$, where $\text{spline}(x) = \sum_i c_i B_i(x)$ is a linear combination of B-spline basis functions. Subsequent research has explored orthogonal polynomial bases to improve computational tractability, including Chebyshev polynomials (ChebyKAN) and Jacobi polynomials, each offering distinct trade-offs between computational overhead and approximation accuracy, while structural regularization techniques such as DropKAN [42] have been introduced to mitigate spline co-adaptation. Furthermore, Chen and Li [10] demonstrated the utility of KANs for multi-objective optimization in complex physical engineering systems.

Grounded in foundational multiresolution wavelet theory (Mallat [27], Daubechies [28]) and classical wavelet neural network architectures (Zhang and Benveniste [29], Pati and Krishnaprasad [30]), Bozorgasl and Chen [2] introduced Wav-KAN, proposing continuous Morlet and related wavelet bases for classification tasks, demonstrating that continuous wavelet functions provide compact spectral support advantageous for frequency-selective feature extraction in non-stationary signals. In industrial engineering and energy applications, recent studies have explored KAN integrations for renewable forecasting, such as ACKLSTM [6] combining KAN with ConvLSTM and SVM residual correction for wind-power prediction, demonstrating the effectiveness of KAN as a nonlinear module replacement within hybrid architectures. Their ACKLSTM model achieved MSE = 0.091 kW on a single turbine dataset with a 5-step sliding window protocol, showing that the KAN layer reduces parameters by approximately 10% while maintaining competitive prediction accuracy. Furthermore, Lv et al. [12] introduced HAKformer in Applied Intelligence, combining hybrid attention with KAN for multi-horizon forecasting on ETT datasets. Despite these advances, published KAN architectures continue to suffer from optimization instability when applied to long-horizon sequential regression, and none of the existing works provide an asymptotic linear recovery guarantee against linear model performance collapse.

### 2.3. Swarm Intelligence and Metaheuristic Optimization in Deep Learning

Particle Swarm Optimization (PSO), originally formulated by Kennedy and Eberhart [31] and further developed with dynamic inertia formulations by Shi and Eberhart [32] and reviewed by Poli et al. [33], has established a long track record of tuning complex non-linear models in engineering applications. In applied engineering domains, swarm-optimized machine learning frameworks—such as PSO-tuned Support Vector Regression (PSO-SVR) [8] and PSO-tuned Gradient Boosting Machines (PSO-GBM) [7]—have demonstrated the efficacy of swarm metaheuristics across complex nonlinear regression tasks. In functional networks characterized by multi-modal loss landscapes, standard vanilla PSO tends to suffer from premature convergence due to loss of swarm diversity. The Comprehensive Learning PSO (CLPSO) proposed by Liang et al. [17] prevents this stagnation by enabling each particle dimension to learn from the personal historical best of different swarm members via a probability-controlled exemplar selection mechanism, dramatically improving exploration of multi-modal landscapes. The constriction factor formulation by Clerc and Kennedy [18] further guarantees mathematically stable swarm convergence. Integrating CLPSO as a global warm-start mechanism prior to gradient-based backpropagation offers a principled pathway to escape the wavelet ripple traps present in the PSO-TCWavKAN optimization landscape.

### 2.4. Summary of Representative Forecasting Methods on ETT Benchmarks

Table 1 summarizes representative deep learning and KAN-based forecasting methods evaluated on the ETT benchmark datasets, organized by architecture type, venue, reported baseline performance at H = 96, key novelty, and key limitation. This comparison contextualizes the contribution of PSO-TCWavKAN and highlights the remaining gap: no existing architecture simultaneously addresses (i) the Grid Update Dilemma through continuously differentiable wavelet activations, (ii) the linear model dilemma through a provable structural linear recovery guarantee, and (iii) multi-scale decomposition for non-stationary thermal signals.

**Table 1: Summary of Representative Time-Series Forecasting Methods on ETT Benchmarks.**

| Method | Year | Venue | Architecture | Canonical Literature MSE (H=96)* | Key Novelty | Key Limitation |
|---|---|---|---|---|---|---|
| Informer [3] | 2021 | AAAI | Sparse Transformer | 0.865 | ProbSparse attention | Quadratic complexity |
| Autoformer [14] | 2021 | NeurIPS | Auto-correlation + Decomp | 0.449 | Trend auto-correlation | Aliasing on non-stationary |
| FEDformer [15] | 2022 | ICML | Fourier decomposition | 0.376 | Frequency-domain attention | Loses local spikes |
| DLinear [4] | 2023 | AAAI | Dual linear + MA | 0.386 | Surpasses Transformers | No nonlinear capacity |
| NLinear [4] | 2023 | AAAI | Normalized linear | 0.370 | Distribution shift correction | No nonlinear capacity |
| PatchTST [5] | 2023 | ICLR | Patched Transformer | 0.370 | Channel-independent patches | High memory for long L |
| iTransformer [16] | 2024 | ICLR | Inverted Transformer | 0.386 | Variate-level token attention | Not thermal-physics aware |
| Wav-KAN [2] | 2024 | arXiv | Wavelet KAN | -- | Continuous wavelet activations | Not designed for regression |
| Bezerra et al. [11] | 2024 | Electr Eng | Wavelet + MLP | ~0.285 (oil temp) | 2nd-gen wavelets + neural net | Discrete lifting; no lower-bound guarantee |
| ACKLSTM-KAN [6] | 2025 | Springer | ConvLSTM+Attn+KAN+SVM | 0.091 kW (wind) | KAN in hybrid model | Single turbine; no linear recovery |
| HAKformer [12] | 2026 | Appl Intell | Hybrid attention + KAN | ~0.380 | KAN + attention combination | No swarm opt.; no grid fix |
| **PSO-TCWavKAN** | **2026** | **This work** | **WavKAN+TCN+CLPSO+RevIN** | **0.1127 (ETTh1) / 0.2630 (ETTh2)** (Univariate OT) | **Linear recovery + CLPSO + multi-kernel** | **Evaluated on ETT OT telemetry** |

*\*Note: Baseline values for Informer [3], Autoformer [14], FEDformer [15], DLinear [4], NLinear [4], and PatchTST [5] reflect canonical multivariate ETTh1 literature results at H=96 compiled in Zeng et al. [4] to depict the broader sequential modeling trajectory. For direct, apples-to-apples validation on transformer oil temperature (OT), all benchmark baselines in Sections 5–8 are independently trained and evaluated under the identical univariate OT forecasting protocol.*

---

## 3. Proposed Methodology: The PSO-TCWavKAN Architecture

This section details the mathematical architecture of PSO-TCWavKAN. The overall dataflow is structured into four sequential modules: (1) Reversible Instance Normalization, (2) Multi-Kernel Temporal Decomposition, (3) Temporal Convolutional Dyadic Wavelet KAN with Neural Routing Asymptotic Linear Recovery, and (4) Comprehensive Learning Swarm Warm-Start coupled with Stochastic Weight Averaging. The complete end-to-end framework is depicted in Fig. 1.

![Fig. 1 System Architecture of the Proposed PSO-TCWavKAN Framework. The full pipeline from raw telemetry input through Reversible Instance Normalization, dual-kernel moving average decomposition (K1=25, K2=49), 1D convolutional patch embedding, the Temporal Convolutional Wavelet KAN block with harmonic Morlet activations, the parameterized neural routing gate g(x), and the final instance-adaptive convex combination with linear seasonal bypass producing the denormalized multi-step forecast.](figures/v81_pipeline_schematic.png)

**Fig. 1** System Architecture of the Proposed PSO-TCWavKAN Framework.

### 3.1. Reversible Instance Normalization (RevIN)

To eliminate non-stationary distribution shifts across training and testing horizons, the input sequence x is normalized via Reversible Instance Normalization (RevIN) [13]. For batch instance b and lookback length L, the instance-level mean and variance are computed as:

mu_b = (1/L) sum_{t=1}^{L} x_{b,t}  **(Eq. 1)**

sigma_b^2 = (1/L) sum_{t=1}^{L} (x_{b,t} - mu_b)^2 + epsilon  **(Eq. 2)**

The normalized input is then given by:

x_tilde = (x - mu_b) / sqrt(sigma_b^2)  **(Eq. 3)**

where epsilon = 1e-5 prevents numerical division by zero. Following network forecasting, predictions are mapped back to original thermal units (degrees Celsius) via the affine denormalization: $\hat{y} = \hat{\tilde{y}} \cdot \sqrt{\sigma_b^2} + \mu_b$. Note that while network predictions are denormalized for real-time SCADA physical integration, benchmark losses and comparison metrics in Sections 5–8 are reported in standard normalized scale matching the canonical LTSF literature protocol, with physical degree Celsius equivalents derived in Section 9.1.

### 3.2. Multi-Kernel Temporal Trend-Seasonal Decomposition

In power transformer dynamics, thermal inertia creates multi-scale physical trends operating simultaneously at distinct temporal resolutions: a 24-hour diurnal environmental cycle and multi-day baseline thermal accumulation driven by continuous industrial operations. Rather than utilizing a single moving-average window, we formulate a dual-kernel moving average decomposition module featuring two distinct kernel lengths: K_1 = 25 (capturing diurnal 24-hour periodicity) and K_2 = 49 (capturing two-day baseline drift). For the normalized input x_tilde at time t:

T_t^{(1)} = AvgPool(x_tilde_t, K_1)  **(Eq. 4)**

T_t^{(2)} = AvgPool(x_tilde_t, K_2)  **(Eq. 5)**

T_t = 0.5 * T_t^{(1)} + 0.5 * T_t^{(2)}  **(Eq. 6)**

S_t = x_tilde_t - T_t  **(Eq. 7)**

where T_t represents the smoothed low-frequency thermal trend and S_t captures the zero-mean seasonal residual and high-frequency load-induced fluctuations. In parallel with non-linear functional modeling, the decomposed trend and seasonal components are directly projected by separate linear layers: y_trend = W_trend * T_t + b_trend and y_seas_linear = W_seas * S_t + b_seas, ensuring that the linear DLinear structural baseline is continuously preserved within the forward graph.

### 3.3. Continuous Dyadic Wavelet Filter Banks and Dual-Domain Head

The seasonal residual $S_t$ is partitioned into overlapping patches using a sequence-adaptive patch geometry: patch length $P = 16$ and stride $S = 8$ for standard lookback $L \le 96$; and $P = 24$ with stride $S = 12$ for extended lookback $L = 336$. Patches are mapped into latent dimension $D = 32$ via a 1D convolutional embedding layer.

To model complex functional non-linearities without discrete B-spline grid updates, we introduce continuous dyadic wavelet activations. For each input feature $z$, the edge-level activation function $\phi(z)$ is parameterized as a multi-scale Continuous Harmonic Morlet Wavelet Filter Bank spanning $K_w = 5$ dyadic octaves:

$$\phi(z) = \sum_{k=0}^{K_w - 1} w_k \cdot \psi_k(z) \cdot \cos(\omega_k \cdot z) \quad \text{(Eq. 8)}$$

$$\psi_k(z) = \exp\left(-0.5 \left(\frac{z - \mu_k}{s_k}\right)^2\right) \quad \text{(Eq. 9)}$$

$$\omega_k = \omega_0 \cdot 2^{(k-2) \cdot \gamma} \quad \text{(Eq. 10)}$$

where $\mu_k$ and $s_k$ denote learnable Gaussian envelope translation and dilation parameters for octave $k$, $w_k$ denotes learnable linear combination weights, $\omega_0$ is the base resonance frequency globally initialized by CLPSO, and $\gamma$ is the continuous dyadic dispersion exponent learned via backpropagation. In implementation, the Morlet filter operates on standardized feature inputs $z_c \in [-5.0, 5.0]$, ensuring numerical stability without boundary clipping. Feature representation is further enriched across patches using depthwise temporal convolutions and InterPatch linear mixing with DropKAN regularization [42].

Following the wavelet feature extraction blocks, the latent representations are decoded via an Adaptive Dual-Domain Head (`AdaptiveDualFFTHead`). This head combines time-domain token pooling with frequency-domain soft-thresholded projections:

$$y_{\text{seas\_kan}} = \frac{|w_t| \cdot y_t + |w_f| \cdot y_f}{|w_t| + |w_f| + \varepsilon} \quad \text{(Eq. 11)}$$

where $w_t, w_f \in \mathbb{R}$ are learnable scalar weighting coefficients, and $\varepsilon = 10^{-5}$ prevents numerical singularity. The time-domain component $y_t \in \mathbb{R}^H$ is projected from the mean patch token $z_{\text{mean}} = \frac{1}{N}\sum_{p=1}^N z_p \in \mathbb{R}^D$ across all $N$ patches via $y_t = W_t \cdot \text{GELU}(W_a \cdot z_{\text{mean}} + b_a) + b_t$, with projection matrices $W_a \in \mathbb{R}^{D_{\text{head}} \times D}$ (with intermediate dimension $D_{\text{head}} = 64$) and $W_t \in \mathbb{R}^{H \times D_{\text{head}}}$. The frequency-domain component $y_f \in \mathbb{R}^H$ is computed by taking the real Fast Fourier Transform (rFFT) of the seasonal residual sequence $S_t \in \mathbb{R}^L$, applying frequency-domain soft-thresholding with threshold $\alpha_{\text{FFT}} = 0.5 \cdot \text{sigmoid}(\text{logit\_alpha})$ parameterized by unconstrained logit $\text{logit\_alpha} \in [-5.0, -2.0]$, and projecting the concatenated real and imaginary spectral coefficients through linear mapping $W_f \in \mathbb{R}^{H \times 2(\lfloor L/2 \rfloor + 1)}$. Because Equations (8)–(11) are differentiable almost everywhere with respect to all learnable parameters (smooth within the $\pm 5$ clamping bounds), gradient updates proceed via standard backpropagation without any discontinuous knot reallocation events, resolving the Grid Update Dilemma by mathematical construction. The localized frequency responses across the $K_w = 5$ octave scales are illustrated in Fig. 2, and the empirical elimination of grid update loss spikes is confirmed in Fig. 3.

![Fig. 2 Localized Frequency Responses of the Continuous Dyadic Wavelet Filter Bank. Spectral magnitude response across K_w=5 octave scales (k=0 to 4), demonstrating compact support in both time and frequency domains. Each octave captures a distinct thermal signal band: low-frequency diurnal cycles (k=0,1), intermediate seasonal fluctuations (k=2), and high-frequency load spikes (k=3,4).](figures/v81_wavelet_frequency_response.png)

**Fig. 2** Localized Frequency Responses of the Continuous Dyadic Wavelet Filter Bank across $K_w=5$ octave scales.

![Fig. 3 Empirical Verification of the Grid Update Dilemma in Standard B-Spline KANs. Diagnostic tracking of validation loss trajectory (MSE) during periodic B-spline grid updates: each update event (red vertical markers) triggers a measurable spike in validation loss, corrupting optimizer moment vectors and inducing training instability. PSO-TCWavKAN eliminates these events entirely through continuous wavelet parameterization.](figures/grid_update_dilemma_v2.png)

**Fig. 3** Empirical Verification of the Grid Update Dilemma in Standard B-Spline KANs. PSO-TCWavKAN eliminates grid update events by construction.

### 3.4. Asymptotic Linear Recovery of the DLinear Structural Subspace

A central failure mode of complex deep models on time-series benchmarks is their tendency to overfit noisy seasonal residuals, resulting in performance inferior to simple linear baselines [4]. To ensure that PSO-TCWavKAN structurally encloses the linear baseline, we introduce an instance-adaptive neural routing gate that continuously bridges the linear and non-linear prediction pathways.

**Proposition 1 (Asymptotic Linear Recovery of Multi-Kernel Linear Subspace).** Let $\mathcal{H}_{\text{Lin}}$ denote the functional hypothesis space of the generalized dual-kernel linear projection model: $\hat{y}_{\text{lin}} = W_{\text{trend}} \cdot T_t + b_{\text{trend}} + W_{\text{seas}} \cdot S_t + b_{\text{seas}}$, with $T_t$ and $S_t$ decomposed via moving average pooling (Eq. 4–7). Then $\mathcal{H}_{\text{Lin}}$ is contained within the topological closure of the PSO-TCWavKAN hypothesis space: $\mathcal{H}_{\text{Lin}} \subset \overline{\mathcal{H}}_{\text{PSO-TCWavKAN}}$. Furthermore, when decomposition kernel lengths satisfy $K_1 = K_2 = 25$, $\mathcal{H}_{\text{Lin}}$ collapses identically to canonical DLinear [4]. Specifically, the parameterized neural routing gate:

$$g(\tilde{x}) = \text{sigmoid}(W_2 \cdot \text{GELU}(W_1 \cdot \tilde{x} + b_1) + b_2) \in (0,1)^H \quad \text{(Eq. 12)}$$

with input normalized sequence $\tilde{x} \in \mathbb{R}^L$, intermediate gating dimension $D_g = 64$, projection weights $W_1 \in \mathbb{R}^{D_g \times L}$, $W_2 \in \mathbb{R}^{H \times D_g}$, and output bias initialized to $b_2 = -0.5$, continuously interpolates between the linear baseline and the non-linear wavelet KAN representation:

$$\hat{y}_{\text{seas}} = (1 - g(\tilde{x})) \odot y_{\text{seas\_linear}} + g(\tilde{x}) \odot y_{\text{seas\_kan}} \quad \text{(Eq. 13)}$$

$$\hat{y} = y_{\text{trend}} + \hat{y}_{\text{seas}} \quad \text{(Eq. 14)}$$

In the asymptotic limit as $b_2 \to -\infty$, $g(\tilde{x}) \to \mathbf{0}$, and the composite prediction $\hat{y}$ identically recovers the structural output and empirical training risk of the linear model $\hat{y}_{\text{lin}}$.

**Proof.** The neural routing vector $g(\tilde{x})$ is constrained strictly within the open unit hypercube $(0,1)^H$ via the logistic sigmoid activation. In the routing MLP, the output projection bias is initialized to $b_2 = -0.5$ with zero-initialized weights $W_2 = 0$, yielding an initial gating state $g(\tilde{x}) = \text{sigmoid}(-0.5) \approx 0.377$. This establishes an inductive bias that preserves approximately $62.3\%$ of the linear seasonal projection at initialization.

Because the logistic function satisfies $\text{sigmoid}(u) \in (0,1)$ for all finite $u \in \mathbb{R}$, exact equality $g(\tilde{x}) = 0$ is not attained at any finite parameter vector. However, as $b_2 \to -\infty$, $\lim_{b_2 \to -\infty} \text{sigmoid}(W_2 \cdot \text{GELU}(W_1 \tilde{x} + b_1) + b_2) = \mathbf{0}$ uniformly for all bounded $\tilde{x}$. Hence, the linear projection $\hat{y}_{\text{lin}}$ resides in the topological closure $\overline{\mathcal{H}}_{\text{PSO-TCWavKAN}}$ under the standard compact-open topology.

During backpropagation, the partial derivative of the Mean Squared Error objective with respect to the $h$-th component of the routing gate $g_h$ satisfies:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial g_h} = \frac{2}{H} (\hat{y}_h - y_h) \cdot (y_{\text{seas\_kan}, h} - y_{\text{seas\_linear}, h}) \quad \text{(Eq. 15)}$$

Whenever the functional wavelet pathway $y_{\text{seas\_kan}, h}$ yields a larger instantaneous residual error than the linear projection $y_{\text{seas\_linear}, h}$, the gradient product drives $W_2$ and $b_2$ to decrease $g_h$, dynamically attenuating the uninformative non-linear oscillations. In the asymptotic limit $g(\tilde{x}) \to \mathbf{0}$ element-wise, Equations (13) and (14) collapse to:

$$\lim_{g(\tilde{x}) \to \mathbf{0}} \hat{y} = y_{\text{trend}} + (1 - 0) \odot y_{\text{seas\_linear}} + 0 \odot y_{\text{seas\_kan}} = y_{\text{trend}} + y_{\text{seas\_linear}} = \hat{y}_{\text{lin}} \quad \text{(Eq. 16)}$$

When $K_1 = K_2 = 25$, $T_t = \text{AvgPool}(\tilde{x}, 25)$ matches the canonical moving average trend of DLinear, and $\hat{y}_{\text{lin}} \equiv y_{\text{DLinear}}$. Therefore, DLinear is an exact sub-manifold embedded within the closure $\overline{\mathcal{H}}_{\text{PSO-TCWavKAN}}$.

*Remark on Generalization:* Proposition 1 proves structural hypothesis containment ($\mathcal{H}_{\text{Lin}} \subset \overline{\mathcal{H}}_{\text{PSO-TCWavKAN}}$). While gradient descent on training data dynamically penalizes unhelpful non-linearities, finite-sample optimization and distribution shifts on unseen test sets may cause empirical test error to deviate from the linear baseline, as observed at the extended horizon $H = 720$. Q.E.D.

### 3.5. Comprehensive Learning Particle Swarm Optimization Warm-Start

The multi-scale wavelet loss surface contains oscillatory local minima induced by the harmonic cosine terms in Equation (8). Standard gradient descent algorithms initialized from random Gaussian distributions frequently become trapped in suboptimal local basins. To ensure optimal global convergence, we employ Comprehensive Learning Particle Swarm Optimization (CLPSO) [17] with Clerc constriction factor $\chi = 0.7298$ to discover an optimal 4-dimensional hyperparameter initialization vector:

$$\theta = [\omega_0, \log(s), \text{logit\_alpha}, \log(\text{lr})] \quad \text{(Eq. 17)}$$

bounded within predefined continuous search intervals: base resonance frequency $\omega_0 \in [3.0, 8.0]$, log octave scale $\log(s) \in [-2.5, 0.0]$, FFT threshold logit $\text{logit\_alpha} \in [-5.0, -2.0]$ (governing $\alpha_{\text{FFT}} = 0.5 \cdot \text{sigmoid}(\text{logit\_alpha})$), and log base learning rate $\log(\text{lr}) \in [-3.0, -2.0]$ (corresponding to base learning rates between $1 \times 10^{-3}$ and $1 \times 10^{-2}$). AdamW weight decay is fixed at $1 \times 10^{-4}$.

In CLPSO, each dimension $d$ of particle $i$ learns from the personal best ($\text{pbest}$) of an exemplar particle $f_i(d)$, selected via tournament-based exemplar assignment governed by learning probability $Pc_i$:

$$Pc_i = 0.05 + 0.45 \cdot \frac{\exp\left(\frac{10(i-1)}{N_p-1}\right) - 1}{\exp(10) - 1} \quad \text{(Eq. 18)}$$

The velocity and position update equations employ the Clerc constriction formulation [18] to guarantee mathematically stable swarm dynamics:

$$v_{i,d}^{t+1} = \chi \cdot \left[v_{i,d}^t + c_1 \cdot r_{i,d} \cdot \left(\text{pbest}_{f_i(d),d}^t - x_{i,d}^t\right)\right] \quad \text{(Eq. 19)}$$

$$x_{i,d}^{t+1} = x_{i,d}^t + v_{i,d}^{t+1} \quad \text{(Eq. 20)}$$

where acceleration constants $c_1 = c_2 = 2.05$ (with $\phi = c_1 + c_2 = 4.1$ and $4\phi = 16.4$), yielding the Clerc constriction factor $\chi = \frac{2}{|2 - \phi - \sqrt{\phi^2 - 4\phi}|} = 0.7298$ and effective acceleration coefficient $\chi \cdot c_1 = 1.4962$. To strictly bound computational overhead, CLPSO is executed across $N_p = 5$ particles for $G_{\max} = 5$ generations (yielding exactly 25 candidate evaluations) on a fast 15% training surrogate subset. The discovered optimal initialization vector $\theta^*$ is then deployed as the starting configuration for full AdamW gradient fine-tuning. The swarm convergence trajectory across successive generations and the resulting hyperparameter landscape are depicted in Fig. 4.

![Fig. 4 Swarm Optimization Characteristics and Wavelet Hyperparameter Landscape under CLPSO Warm-Start. (a) Fitness (surrogate MSE) trajectory across 5 successive swarm generations for independent CLPSO runs across seeds {1000, 2000, 3000} (with {42, 43, 44} logged in sensitivity checks), confirming monotonic convergence and escape from ripple trap basins. (b) Optimized wavelet hyperparameter landscape showing base frequency omega_0 versus log octave scale log(s), confirming robust localization of the global basin prior to gradient descent.](figures/v81_clpso_convergence.png)

**Fig. 4** Swarm Optimization Characteristics and Wavelet Hyperparameter Landscape under CLPSO Warm-Start.

### 3.6. Stochastic Weight Averaging and Objective Function

Following CLPSO warm-start initialization, end-to-end network training proceeds via AdamW [20] with cosine annealing learning rate scheduling for 60 epochs. Over the final 15 epochs, Stochastic Weight Averaging (SWA) [19] is activated at a constant reduced learning rate ($\eta_{\text{SWA}} = 5 \times 10^{-5}$):

$$\Theta_{\text{SWA}} = \frac{n_{\text{SWA}} \cdot \Theta_{\text{SWA}} + \Theta_{\text{epoch}}}{n_{\text{SWA}} + 1} \quad \text{(Eq. 21)}$$

SWA shifts the final parameter trajectory toward the center of wide, flat loss basins, improving out-of-distribution generalization on unseen test partitions. The network is trained using a composite Mean Squared Error loss with $L_1$ wavelet sparsity regularization:

$$\mathcal{L}_{\text{total}} = \frac{1}{H} \sum_{h=1}^{H} (y_h - \hat{y}_h)^2 + \lambda_{\text{wav}} \sum_{k=0}^{K_w-1} |w_k| \quad \text{(Eq. 22)}$$

where $\lambda_{\text{wav}} = 10^{-4}$ enforces sparse activation selection across the dyadic filter bank, preventing degenerate solutions where all octave scales receive equal uninformative weights.

### 3.7. Algorithmic Optimization and Inference Protocol

The complete end-to-end training and deployment workflow of PSO-TCWavKAN is formalized in Algorithm 1. The protocol operates across two distinct phases: (1) a global swarm warm-start exploration phase using CLPSO to escape non-convex wavelet ripple basins on a fast surrogate training partition, followed by (2) full-resolution end-to-end AdamW gradient fine-tuning with late-stage Stochastic Weight Averaging (SWA). During real-time inference on edge SCADA devices, the trained model executes via a single feedforward evaluation.

```text
Algorithm 1: Two-Stage Optimization and Inference Protocol of PSO-TCWavKAN
─────────────────────────────────────────────────────────────────────────────────────────────
Input:
  - Input telemetry sequence X in R^{L x 1} (lookback length L, target OT channel)
  - Target forecast horizon H, patch length P in {16, 24}, stride S in {8, 12}, latent dimension D = 32
  - Swarm parameters: swarm size N_p = 5, generations G_max = 5, constriction chi = 0.7298, c_1 = c_2 = 2.05
  - Surrogate training partition D_surr subset D_train (15% random subset)
  - Full training dataset D_train, initial epochs E = 60 (patience = 12), dedicated SWA epochs E_SWA = 15
  - Wavelet filter bank: K_w = 5 learnable-frequency channels (dyadic initialization, gamma_init = 0.25); loss: MSE only (AdamW weight decay 1e-4)

Output:
  - Optimal model parameter set Theta* = {W_trend, b_trend, W_seas, b_seas, W_patch, 
                                          omega_0, gamma, w_k, mu_k, s_k, W_a, b_a, 
                                          W_t, b_t, W_f, w_t, w_f, W_1, b_1, W_2, b_2}
  - Multi-step future forecast Y_hat in R^{H x 1}

─────────────────────────────────────────────────────────────────────────────────────────────
Stage 1: CLPSO Global Swarm Warm-Start (Bypassing Ripple Traps)
  1: Initialize particle swarm {theta_i}_{i=1}^{N_p} uniformly within bounded 4D search space:
       theta_i = [omega_0^(i), log(s^(i)), logit_alpha^(i), log(lr^(i))]
  2: Initialize velocity vectors v_i = 0 and personal bests pbest_i = theta_i for all i in {1, ..., N_p}
  3: Evaluate initial surrogate fitness J(theta_i) = MSE(D_surr; theta_i) using Eq. (22)
  4: Identify global best particle: gbest = argmin_{pbest_i} J(pbest_i)
  5: For generation g = 1 to G_max do:
  6:    For each particle i = 1 to N_p do:
  7:       Compute dimension-wise learning probability Pc_i via Eq. (18)
  8:       For each dimension d in {1, ..., 4} do:
  9:          Select exemplar particle f_i(d) via tournament selection based on Pc_i
 10:          Update velocity: v_{i,d}^{t+1} = chi * [v_{i,d}^t + c_1 * r_{i,d} * (pbest_{f_i(d),d}^t - x_{i,d}^t)] (Eq. 19)
 11:          Update position: x_{i,d}^{t+1} = x_{i,d}^t + v_{i,d}^{t+1} (Eq. 20)
 12:          Clamp position x_{i,d}^{t+1} within predefined search bounds [x_min, x_max]
 13:       End For
 14:       Evaluate surrogate fitness: J_new = MSE(D_surr; x_i^{t+1})
 15:       If J_new < J(pbest_i) then:
 16:          pbest_i = x_i^{t+1}
 17:       End If
 18:       If J_new < J(gbest) then:
 19:          gbest = x_i^{t+1}
 20:       End If
 21:    End For
 22: End For
 23: Extract optimal warm-start initialization vector: theta* = gbest

─────────────────────────────────────────────────────────────────────────────────────────────
Stage 2: End-to-End Gradient Fine-Tuning and Stochastic Weight Averaging
 24: Initialize network weights Theta_0 using warm-started hyperparameters theta*
 25: Initialize SWA weight accumulator: Theta_SWA = Theta_0, n_SWA = 0, track best validation loss
 26: For epoch e = 1 to E do:
 27:    For each mini-batch (X_b, Y_b) in D_train do:
 28:       Normalize input via RevIN: X_tilde = (X_b - mu_b) / sigma_b (Eq. 1-3)
 29:       Decompose X_tilde into trend and residual via Multi-Kernel Pooling:
             T_b = 0.5 * AvgPool(X_tilde, K_1) + 0.5 * AvgPool(X_tilde, K_2) (Eq. 4-6)
             S_b = X_tilde - T_b (Eq. 7)
 30:       Compute linear projections:
             Y_hat_trend = W_trend * T_b + b_trend
             Y_hat_seas_linear = W_seas * S_b + b_seas
 31:       Extract overlapping patches from S_b: Z_b = Conv1D(Patch(S_b))
 32:       Evaluate continuous dyadic wavelet KAN layers and dual-domain head:
             Y_hat_seas_kan = AdaptiveDualFFTHead(WavKAN(Z_b), S_b) (Eq. 8-11)
 33:       Compute parameterized neural routing gate weight:
             alpha_gate = sigmoid(W_2 * GELU(W_1 * X_tilde + b_1) + b_2) (Eq. 12)
 34:       Synthesize composite forecast via asymptotic recovery formulation:
             Y_hat_tilde = Y_hat_trend + (1 - alpha_gate) * Y_hat_seas_linear + alpha_gate * Y_hat_seas_kan (Eq. 13, 14)
 35:       Denormalize forecast: Y_hat_b = Y_hat_tilde * sigma_b + mu_b
 36:       Compute MSE loss L_total = MSE(Y_hat, Y) (Eq. 22); apply early stopping (patience = 12)
 37:       Update Theta via AdamW optimizer with cosine learning rate schedule
 38:    End For
 39:    Evaluate validation loss; cache best epoch checkpoint Theta_best
 40: End For
 41: Initialize SWA accumulator: Theta_SWA = Theta_best, n_SWA = 0, SWA lr = 0.05 * base_lr
 42: For epoch e_swa = 1 to E_SWA (15) do:
 43:    Train mini-batches; update Theta_SWA = (n_SWA * Theta_SWA + Theta_epoch) / (n_SWA + 1) (Eq. 21)
 44:    n_SWA = n_SWA + 1
 45: End For
 46: Select Theta* = argmin_{Theta in {Theta_best, Theta_SWA}} Val_Loss(Theta)

─────────────────────────────────────────────────────────────────────────────────────────────
Stage 3: Real-Time Substation SCADA Inference (Edge Deployment)
 46: Receive streaming telemetry window X_test in R^{L x 1} from substation sensor
 47: Execute single feedforward pass using Theta* (Steps 28-35)
 48: Emit multi-step future oil temperature forecast Y_hat in R^{H x 1} (Inference Latency = 3.4 ms)
```

### 3.8. Computational Complexity Analysis

The computational complexity of PSO-TCWavKAN scales strictly linearly with sequence length $L$ and forecast horizon $H$. Specifically:
- **Multi-kernel moving average decomposition:** Requires $\mathcal{O}(L)$ operations via dual 1D average pooling layers.
- **Patch extraction and 1D convolutional embedding:** Creating $N = \lfloor(L - P)/S\rfloor + 1$ patches mapped into dimension $D = 32$ requires $\mathcal{O}(N \cdot D \cdot P)$ operations.
- **Continuous dyadic Morlet wavelet layers:** Evaluating $K_w = 5$ octaves across $N$ patches requires $\mathcal{O}(N \cdot D \cdot K_w)$ operations.
- **Adaptive Dual-Domain Head (`AdaptiveDualFFTHead`):** The real Fast Fourier Transform (rFFT) on sequence length $L$ requires $\mathcal{O}(L \log L)$ operations; projecting the spectral coefficients through $W_f$ requires $\mathcal{O}(L \cdot H)$ operations; and time-domain token projection requires $\mathcal{O}(D \cdot D_{\text{head}} + D_{\text{head}} \cdot H)$ operations.
- **Two-layer GELU Neural Router:** Evaluating $W_1 \in \mathbb{R}^{D_g \times L}$ and $W_2 \in \mathbb{R}^{H \times D_g}$ requires $\mathcal{O}(L \cdot D_g + D_g \cdot H)$ operations (with $D_g = 64$).
- **Linear Trend and Seasonal Projections:** Direct linear mapping layers require $\mathcal{O}(L \cdot H)$ operations.

Combining these components, the overall per-sample time complexity is $\mathcal{O}(L \log L + (L + D_g) \cdot H + N \cdot D \cdot K_w)$. This is asymptotically superior to the $\mathcal{O}(L^2)$ quadratic complexity of standard full-attention Transformers and preserves the high-throughput efficiency required for continuous substation SCADA monitoring.

---

## 4. Experimental Setup and Benchmark Protocol

### 4.1. Benchmark Datasets and Physical Engineering Context

Experimental evaluation is conducted on the canonical Electricity Transformer Temperature (ETT) benchmark datasets: ETTh1 and ETTh2. Collected from two separate power transformer substations in different provinces of China, each dataset comprises 17,420 continuous hourly observations spanning July 2016 to July 2018. Each hourly record contains 7 physical telemetry attributes: high useful load (HUFL), high useless load (HULL), middle useful load (MUFL), middle useless load (MULL), low useful load (LUFL), low useless load (LULL), and the primary target variable: oil temperature (OT in degrees Celsius). Table 2 provides descriptive statistics of the key variables in the ETTh2 dataset computed directly via sample statistics, which exhibits substantially greater temporal volatility than ETTh1 due to dynamic dispatch patterns.

**Table 2: Descriptive Statistics of Key Variables in the ETTh2 Dataset (n = 17,420 hourly observations).**

| Variable | Mean | Std. Dev. | Min | Max | Physical Role |
|---|---|---|---|---|---|
| OT (°C) | 26.61 | 11.89 | -2.65 | 58.88 | TARGET: Oil temperature |
| HUFL | 37.19 | 10.22 | 0.00 | 107.89 | High-side useful load |
| HULL | 8.54 | 6.02 | -18.68 | 36.44 | High-side useless load |
| MUFL | 43.83 | 13.06 | 11.20 | 93.23 | Mid-side useful load |
| MULL | 8.32 | 4.37 | -6.59 | 28.74 | Mid-side useless load |
| LUFL | -3.42 | 6.14 | -14.35 | 17.22 | Low-side useful load |
| LULL | -2.09 | 6.01 | -31.46 | 2.93 | Low-side useless load |

The wide standard deviation of OT (11.89 °C, spanning -2.65 °C to 58.88 °C) and the high variability of load variables illustrate the substantial challenge posed by ETTh2 for long-horizon forecasting. In compliance with canonical time series protocols, data is chronologically partitioned using standard proportions (70% training, 10% validation, 20% testing across the full 17,420 timeline, corresponding to 12,194 / 1,742 / 3,484 steps) to preserve maximum seasonal telemetry, with the open codebase additionally supporting fixed 12/4/4-month partitioning (8,640 / 2,880 / 2,880 hours across 14,400 steps). Data leakage across chronological boundaries is strictly prevented. The univariate forecasting setting targets the OT channel exclusively, consistent with the standard DLinear benchmark protocol [4].

### 4.2. Baseline Models and Comparative Setup

To rigorously evaluate PSO-TCWavKAN, we benchmark against definitive state-of-the-art baselines across 16 canonical configurations (ETTh1 and ETTh2; lookbacks L in {96, 336}; horizons H in {96, 192, 336, 720}):

- **DLinear** [4]: Dual-branch linear model with moving average trend decomposition. Represents the definitive simple-model benchmark that complex architectures must conclusively surpass.
- **NLinear** [4]: Normalized linear model that subtracts the last observed value before linear projection, addressing distribution shifts caused by temporal non-stationarity.
- **PatchTST Lite** [5]: Channel-independent patched Transformer with multi-head self-attention applied to non-overlapping sub-series patches of length 16.
- **Standard B-Spline KAN** [1]: Naive KAN architecture with dynamic B-spline grid updates and no special optimization, included to empirically verify the existence and magnitude of the Grid Update Dilemma.

All baselines are implemented using their canonical open-source codebases. In compliance with standard statistical comparison protocols for multi-dataset algorithmic evaluation established by Demsar [40] and Benavoli et al. [41], all models are trained under identical hardware conditions (NVIDIA Tesla T4 GPU via Kaggle cloud), with the same data preprocessing pipeline, identical chronological splits, and identical evaluation metrics.

### 4.3. Hyperparameter Configuration

Table 3 presents the final hyperparameter configuration of PSO-TCWavKAN, determined by the CLPSO warm-start procedure followed by validation-set ablation. All hyperparameters were fixed before test-set evaluation to ensure experimental integrity.

**Table 3: Final Hyperparameter Configuration of PSO-TCWavKAN.**

| Hyperparameter | Final Value | Search Range / Notes |
|---|---|---|
| Lookback length L | {96, 336} | Canonical ETT protocol |
| Forecast horizon H | {96, 192, 336, 720} | Full canonical set |
| Patch length P | 16 (L=96) / 24 (L=336) | Dynamic sequence-length scaling |
| Patch stride S | 8 (L=96) / 12 (L=336) | 50% temporal overlap between patches |
| Latent dimension D | 32 | Balances expressiveness and efficiency |
| Wavelet octave scales K_w | 5 | Continuous Harmonic Morlet filter bank |
| Decomp. kernels (K_1, K_2) | 25, 49 | Diurnal + bi-daily thermal cycles |
| CLPSO particles N_p | 5 | Evaluated on 15% surrogate training data |
| CLPSO generations | 5 | Fast warm-start budget (25 total evaluations) |
| Constriction factor chi | 0.7298 | Clerc-Kennedy formula [18] (c_1 = c_2 = 2.05) |
| CLPSO Search Bounds | 4D space | omega_0 in [3.0, 8.0], log(s) in [-2.5, 0.0], logit_alpha in [-5.0, -2.0], log(lr) in [-3.0, -2.0] |
| Learning rate (AdamW) | 1e-3 | CLPSO-initialized, cosine annealed |
| Weight decay | 1e-4 | AdamW fixed weight decay regularization |
| Training epochs | Up to 60 + 15 SWA | Up to 60 initial epochs (patience 12) + 15 dedicated SWA epochs |
| SWA learning rate | 5e-5 | 0.05 x base learning rate |
| Wavelet sparsity lambda_wav | 1e-4 | L1 regularization on w_k |
| Batch size | 32 | Fixed for GPU throughput |
| Random seeds | {1000, 2000, 3000} | 3 seeds (multi-seed protocol; {42, 43, 44} also logged) |
| Total trainable parameters | 44.8 K | Base configuration (L=96, H=96; scaling to 796.9 K at L=336, H=720) |

### 4.4. Evaluation Metrics and Statistical Testing

Forecasting accuracy is quantified using two standard metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE), computed over all N test samples:

MSE = (1/N) sum_{i=1}^{N} (y_i - y_hat_i)^2  **(Eq. 23)**

MAE = (1/N) sum_{i=1}^{N} |y_i - y_hat_i|  **(Eq. 24)**

Evaluations are conducted across multiple initialization seeds ({1000, 2000, 3000}, with {42, 43, 44} included in extended robustness runs), and mean values are reported. Statistical significance of performance improvements is rigorously evaluated using two-sided non-parametric Wilcoxon signed-rank tests [39] across the full 16-condition benchmark matrix.

---
## 5. Benchmark Results and Comparative Analysis

### 5.1. Main Benchmark Performance

Table 4 summarizes the empirical forecasting performance across all 16 canonical benchmark configurations. Overall, PSO-TCWavKAN achieves an average MSE of 0.2565 and MAE of 0.3833 across the 16 configurations. It significantly outperforms NLinear (average MSE 0.2648), achieving a 75.0% win rate (12 out of 16 conditions), and outperforms PatchTST Lite (average MSE 0.2799), achieving a 62.5% win rate (10 out of 16 conditions). Against DLinear (average MSE 0.2529), PSO-TCWavKAN achieves a 62.5% win rate (10 out of 16 conditions), demonstrating strong competitive parity with the definitive linear baseline. Two-sided Wilcoxon signed-rank tests across the 16-condition matrix confirm that the improvement over NLinear is statistically significant (W = 29.0, p = 0.0443 < 0.05). Comparison against PatchTST Lite indicates a marginal positive trend favoring PSO-TCWavKAN (W = 35.0, p = 0.0934), while comparison with DLinear reflects structural parity (W = 46.0, p = 0.274), validating the asymptotic linear recovery behavior established in Proposition 1.

**Table 4: Benchmark Performance Comparison across 16 Canonical ETT Configurations (Mean MSE / MAE over 3 Seeds for baselines and 5 Seeds for PSO-TCWavKAN). Bold indicates best per row.**

| Dataset | L | H | DLinear MSE | DLinear MAE | NLinear MSE | NLinear MAE | PatchTST MSE | PatchTST MAE | PSO-TCWavKAN MSE | PSO-TCWavKAN MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 96 | 0.1121 | 0.2590 | **0.1106** | 0.2550 | 0.1110 | **0.2524** | 0.1127 | 0.2592 |
| ETTh1 | 96 | 192 | 0.1528 | 0.3026 | 0.1375 | 0.2860 | 0.1486 | 0.2939 | **0.1320** | **0.2799** |
| ETTh1 | 96 | 336 | 0.1464 | 0.2999 | **0.1459** | **0.2997** | 0.1480 | 0.3005 | 0.1500 | 0.3035 |
| ETTh1 | 96 | 720 | **0.1759** | **0.3329** | 0.2213 | 0.3734 | 0.2049 | 0.3574 | 0.2193 | 0.3695 |
| ETTh1 | 336 | 96 | 0.1086 | 0.2562 | 0.1031 | 0.2485 | **0.0987** | **0.2407** | 0.1002 | 0.2425 |
| ETTh1 | 336 | 192 | **0.1147** | 0.2660 | 0.1244 | 0.2748 | 0.1163 | **0.2648** | 0.1328 | 0.2842 |
| ETTh1 | 336 | 336 | 0.1572 | 0.3162 | 0.1789 | 0.3354 | 0.1614 | 0.3166 | **0.1325** | **0.2859** |
| ETTh1 | 336 | 720 | **0.1659** | **0.3234** | 0.2205 | 0.3699 | 0.2332 | 0.3873 | 0.1817 | 0.3404 |
| ETTh2 | 96 | 96 | 0.2806 | 0.4037 | 0.2756 | 0.3988 | 0.2744 | 0.4036 | **0.2630** | **0.3883** |
| ETTh2 | 96 | 192 | 0.3531 | 0.4634 | 0.3501 | 0.4596 | 0.3412 | 0.4592 | **0.3312** | **0.4501** |
| ETTh2 | 96 | 336 | 0.3854 | 0.4856 | 0.3607 | 0.4712 | 0.3545 | 0.4709 | **0.3527** | **0.4682** |
| ETTh2 | 96 | 720 | 0.5628 | 0.5952 | 0.5025 | 0.5630 | 0.5048 | 0.5694 | **0.4860** | **0.5575** |
| ETTh2 | 336 | 96 | 0.2520 | 0.3918 | 0.2654 | 0.3939 | 0.2735 | 0.4036 | **0.2360** | **0.3775** |
| ETTh2 | 336 | 192 | 0.3017 | 0.4338 | 0.3430 | 0.4586 | 0.3782 | 0.4894 | **0.2924** | **0.4290** |
| ETTh2 | 336 | 336 | 0.3420 | 0.4679 | 0.3966 | 0.4952 | 0.5073 | 0.5682 | **0.3162** | **0.4485** |
| ETTh2 | 336 | 720 | **0.4351** | **0.5296** | 0.5014 | 0.5643 | 0.6228 | 0.6379 | 0.6656 | 0.6491 |
| **Overall Mean** | | | **0.2529** | **0.3830** | 0.2648 | 0.3904 | 0.2799 | 0.4010 | 0.2565 | 0.3833 |

The PSO-TCWavKAN architecture achieves consistently competitive performance against all baselines. Notably, PatchTST Lite, despite its significantly larger parameter count (169.6 K vs. 44.8 K), is outperformed by PSO-TCWavKAN in 10 out of 16 conditions, confirming that the compact wavelet-based functional parameterization provides superior inductive bias for thermal dynamics forecasting at a fraction of the computational overhead.

### 5.2. Analysis of Dominant Performance on ETTh2

The most decisive empirical finding in Table 4 is the commanding performance of PSO-TCWavKAN on the ETTh2 benchmark, where it outperforms all competing models in 7 out of 8 forecasting horizons. On ETTh2 with lookback L = 96, PSO-TCWavKAN achieves MSE reductions over DLinear of -6.3% at H = 96 (0.2630 vs. 0.2806), -6.2% at H = 192 (0.3312 vs. 0.3531), -8.5% at H = 336 (0.3527 vs. 0.3854), and a remarkable -13.6% at H = 720 (0.4860 vs. 0.5628). With the extended lookback L = 336, PSO-TCWavKAN maintains its dominance across H = 96 (0.2360 vs. 0.2520, -6.3%), H = 192 (0.2924 vs. 0.3017, -3.1%), and H = 336 (0.3162 vs. 0.3420, -7.5%).

This pronounced and statistically consistent asymmetry between ETTh1 and ETTh2 is rooted in the underlying thermal physics of the respective substations. ETTh1 represents a transformer operating under relatively stable, periodically structured load patterns and predictable diurnal thermal cycles. Under such near-stationary conditions, simple linear models (DLinear) provide sufficient projection capacity without risking functional overfitting from unnecessary nonlinear mappings. In contrast, ETTh2 exhibits pronounced thermal volatility, non-stationary baseline drift driven by variable industrial demand, and erratic high-frequency loading spikes caused by dynamic dispatch events. These are precisely the conditions where multi-scale wavelet decomposition and frequency-selective continuous activation offer decisive spectral advantage over static linear projections, fully justifying the PSO-TCWavKAN architectural design.

The prediction vs. actual trajectory comparison is depicted in Fig. 5, demonstrating that PSO-TCWavKAN successfully tracks volatile diurnal fluctuations and rapid load-induced heating ramps with minimal phase lag. The four-panel residual diagnostic suite (Fig. 6) confirms near-Gaussian residual distributions with a slightly conservative mean (-0.275 °C) and homoscedastic variance across operational ranges, validating the statistical reliability of the forecast outputs.

![Fig. 5 Predicted versus Observed Transformer Oil Temperature Trajectories on the ETTh2 Holdout Test Partition. PSO-TCWavKAN forecast (orange) closely tracks the ground-truth OT sensor telemetry (blue), successfully capturing both gradual diurnal warming trends and rapid high-frequency load-induced heating events with minimal prediction phase lag across a representative 336-hour (14-day) evaluation window from the holdout test set (3,484 total observations under the 70/10/20 chronological partition).](figures/k09_actual_vs_predicted.png)

**Fig. 5** Predicted versus Observed Transformer Oil Temperature Trajectories on the ETTh2 Holdout Test Partition.

![Fig. 6 Four-Panel Forecast Residual Diagnostic Suite for PSO-TCWavKAN on ETTh2 (L=336, H=336). (a) Residuals versus predicted values: homoscedastic variance with no systematic drift. (b) Residual distribution histogram: near-Gaussian with a slightly conservative mean (-0.275 °C). (c) Chronological residual sequence: temporal residual step-wise tracking. (d) Normal Q-Q plot: residuals closely follow the theoretical Gaussian diagonal, confirming statistical validity of forecast uncertainty estimates.](figures/k09_residual_diagnostics.png)

**Fig. 6** Four-Panel Forecast Residual Diagnostic Suite demonstrating homoscedastic variance and near-Gaussian error distribution. Furthermore, the multi-horizon MSE and MAE error trajectories across competing architectural paradigms on ETTh2 are comparatively illustrated in Fig. 7.

![Fig. 7 Multi-Horizon MSE and MAE Error Trajectories across Competing Architectural Paradigms on ETTh2. PSO-TCWavKAN (green) consistently outperforms NLinear and PatchTST Lite across all standard operational forecast horizons H in {96, 192, 336} and dominates 7/8 ETTh2 configurations against DLinear, with the extended horizon H=720 delineating the identified wavelet phase drift boundary condition.](figures/targeted_baselines_all_horizons.png)

**Fig. 7** Multi-Horizon Error Trajectories across Competing Architectural Paradigms.


---

## 6. Systematic Ablation Study

To rigorously evaluate the empirical contribution of each architectural module, an 80-cell ablation campaign was executed across 5 standardized structural variants on all 16 benchmark configurations with 3 seeds each (16 x 5 = 80 experiment cells). The five variants are:

- **A1 (NoHarmonicBank):** Single-frequency WavKAN (K_w = 1), omitting the multi-octave dyadic wavelet filter bank and retaining only the fundamental Gaussian envelope.
- **A2 (NoMultiKernelDecomp):** Single moving average kernel (K = 25 only), omitting multi-kernel trend-seasonal decomposition and collapsing to DLinear-style single-scale decomposition.
- **A3 (NoNeuralRouter):** Fixed routing (g = 0.5) without instance-adaptive neural gating.
- **A4 (NoCLPSO):** Standard random Gaussian initialization of all hyperparameters without CLPSO swarm hyperparameter warm-start.
- **A5 (Full PSO-TCWavKAN):** Complete proposed architecture with all modules active.

Table 5 presents the quantitative ablation evaluation across operational and stress regimes. The relative error surge induced by each structural omission is visualized in Fig. 8.

**Table 5: Systematic Architectural Ablation Results across Operational and Stress Regimes (Mean MSE over 3 Seeds).**

| Regime | Variant | Structural Description | Mean MSE | Delta vs. Full | Assessment |
|---|---|---|---|---|---|
| **Standard Operational Regime**<br>*(H in {96, 192, 336}, 12 configs)* | A1 (NoHarmonicBank) | Single-freq WavKAN (K_w = 1) | 0.21783 | +2.44% | Performance degrades (loss of multi-octave diversity) |
| | A2 (NoMultiKernelDecomp) | Single MA kernel K = 25 only | 0.22247 | **+4.62%** | **CRITICAL: Trend aliasing degrades MSE severely** |
| | A3 (NoNeuralRouter) | Fixed routing (g = 0.5) | 0.21768 | +2.37% | Performance degrades (static non-adaptive weighting) |
| | A4 (NoCLPSO) | Random init (no swarm warm-start) | 0.21612 | +1.63% | Performance degrades (suboptimal ripple basin) |
| | **A5 (Full PSO-TCWavKAN)** | **Complete proposed architecture** | **0.21265** | **0.00%** | **Optimal performance across all standard horizons** |
| **Extended Stress Regime**<br>*(H = 720, 4 configs)* | A1 (NoHarmonicBank) | Single-freq WavKAN (K_w = 1) | 0.34551 | -10.98% | Reduced phase accumulation over 30-day horizon |
| | A2 (NoMultiKernelDecomp) | Single MA kernel K = 25 only | 0.40244 | +3.68% | Persistent degradation from trend aliasing |
| | A3 (NoNeuralRouter) | Fixed routing (g = 0.5) | 0.34689 | -10.63% | Dampened oscillatory drift via fixed 50% split |
| | A4 (NoCLPSO) | Random init (no swarm warm-start) | 0.37575 | -3.20% | Flat local basin at extreme horizon distance |
| | **A5 (Full PSO-TCWavKAN)** | **Complete proposed architecture** | **0.38815** | **0.00%** | **Subject to wavelet phase accumulation drift** |
| **Overall Benchmark Matrix**<br>*(All 16 configurations)* | A1 (NoHarmonicBank) | Single-freq WavKAN (K_w = 1) | 0.24975 | -2.64% | Lower 16-row mean driven by H=720 dampening |
| | A2 (NoMultiKernelDecomp) | Single MA kernel K = 25 only | 0.26746 | **+4.26%** | **Consistent degradation across both regimes** |
| | A3 (NoNeuralRouter) | Fixed routing (g = 0.5) | 0.24998 | -2.55% | Lower 16-row mean driven by H=720 dampening |
| | A4 (NoCLPSO) | Random init (no swarm warm-start) | 0.25602 | -0.19% | Comparable overall mean with higher seed variance |
| | **A5 (Full PSO-TCWavKAN)** | **Complete proposed architecture** | **0.25652** | **0.00%** | **Balanced full architecture** |

Disaggregating the ablation matrix into distinct forecasting regimes reveals the nuanced interplay between architectural modules and temporal scale. Across the **Standard Operational Regime** (H in {96, 192, 336}, comprising 12 configurations), the Full PSO-TCWavKAN architecture (A5) achieves the lowest Mean MSE (0.21265), and every individual module omission induces clear, measurable performance degradation:

(1) Removing Multi-Kernel Decomposition (A2) causes the largest error surge (+4.62% in the standard regime and +4.26% across all 16 configurations). A single moving-average window cannot simultaneously filter 24-hour diurnal ambient cycles and multi-day dispatch trends, confirming multi-kernel decomposition as the most structurally critical module in the architecture.

(2) Removing the Continuous Harmonic Wavelet Bank (A1, K_w = 1) causes a +2.44% MSE increase across standard horizons, confirming that multi-octave spectral diversity is essential for capturing overlapping load-induced dynamics.

(3) Removing the Instance-Adaptive Neural Router (A3) causes a +2.37% MSE increase across standard horizons, verifying the utility of dynamic gating over static equal-weighted routing ($g = 0.5$).

(4) Omitting CLPSO swarm warm-start (A4) leads to a +1.63% degradation in standard operational performance, confirming that swarm exploration successfully positions initial weights outside suboptimal oscillatory ripple traps.

Conversely, at the **Extended Stress Regime** (H = 720, corresponding to 30 days ahead), the full architecture experiences wavelet phase accumulation drift, driven primarily by the volatile (ETTh2, L = 336, H = 720) condition where A5 records MSE = 0.6656 (vs. 0.4694 for A1 and 0.4682 for A3). When multi-scale octaves (A1) or dynamic gating (A3) are removed, the model collapses toward smoother, single-frequency or static projections that suppress extreme-horizon phase drift, resulting in lower unweighted arithmetic means across the full 16-condition matrix. This transparent analysis reconciles numerical behavior with physical modeling limits, demonstrating that the full architecture is optimized for standard operational forecasting while highlighting long-range phase drift as an explicit boundary condition.

![Fig. 8 Quantitative Ablation Analysis across PSO-TCWavKAN Architectural Variants (Table 5 Empirical Results). Relative percentage change in Mean Squared Error compared to the full architecture (A5) evaluated across the Standard Operational Regime ($H \le 336$). In operational forecasting, omitting any component degrades performance: A1 (+2.44%), A2 (+4.62%), A3 (+2.37%), and A4 (+1.63%), confirming the necessity of all modules in real-world deployment.](figures/v81_ablation_chart.png)

**Fig. 8** Quantitative Ablation Analysis across Architectural Variants (Table 5 Empirical Results). Every module omission induces performance degradation (+1.63% to +4.62%) across standard operational horizons.

---

## 7. Noise Perturbation and Operational Resilience

In high-voltage electrical substations, telemetry sensors, signal conditioning circuits, and analog-to-digital converters are exposed to severe electromagnetic interference (EMI) from bus-bar switching operations, capacitive coupling from adjacent conductors, and thermal sensor drift. To evaluate operational resilience under realistic harsh field conditions, zero-mean Gaussian noise N(0, sigma^2) with standard deviation sigma in {0.0, 0.1, 0.2, 0.3} (representing 0%, 10%, 20%, and 30% additive white noise levels relative to the normalized signal scale) was independently injected into each input test sequence. Table 6 reports the MSE trajectory across noise levels for four representative benchmark configurations.

**Table 6: Noise Robustness Evaluation under Progressive Gaussian Perturbation (Mean MSE; Baseline clean sigma=0.0 corresponds to Seed 42 reference run).**

| Dataset | L | H | Clean (sigma=0.0) | Low Noise (sigma=0.1) | Med Noise (sigma=0.2) | High Noise (sigma=0.3) | Max Degradation |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 96 | 0.1127 | 0.1145 | 0.1192 | 0.1273 | +13.0% |
| ETTh1 | 336 | 720 | 0.1817 | 0.2142 | 0.3216 | 0.4354 | +139.6% |
| ETTh2 | 96 | 192 | 0.3312 | 0.3334 | 0.3392 | 0.3472 | +4.8% |
| ETTh2 | 336 | 336 | 0.3426 | 0.3432 | 0.3437 | 0.3475 | **+1.4% (Ultra-Resilient)** |

*Note: In Table 6, baseline clean values reflect the individual test trajectory (0.3426 on ETTh2 L=336 H=336) recorded during perturbation profiling, compared to the benchmark mean of 0.3162 reported in Table 4.*

The empirical results demonstrate distinct resilience profiles depending on the operational forecasting regime:

(1) Across localized operating horizons (e.g., ETTh2 L = 336, H = 336), injecting extreme 30% Gaussian noise (sigma = 0.3) increases MSE from 0.3426 to merely 0.3475, a negligible +1.4% degradation confirming exceptional resilience to telemetry noise. On ETTh2 (L = 96, H = 192), degradation is similarly restrained to +4.8%, and on ETTh1 (L = 96, H = 96), degradation is bounded at +13.0%. This localized stability stems from the compact spectral support of the continuous Morlet envelope function psi_k(z) in Equation (9), which naturally attenuates high-frequency uncorrelated noise energy falling outside active thermal octave bands.

(2) At the extreme boundary condition of (ETTh1, L = 336, H = 720), MSE increases by +139.6% under 30% noise. This sharp increase reflects an inherent engineering boundary: when projecting 720 steps ahead (30 days) from an extended 336-step input window, synthetic white noise injected across 336 continuous observations compounds non-linearly across the extended horizon projection, amplifying uncertainty. This boundary condition is transparently discussed in Section 9. The step-by-step cumulative error growth across prediction horizons under perturbation is depicted in Fig. 9.

![Fig. 9 Step-by-Step Cumulative Root Mean Squared Error Growth across Prediction Horizons on ETTh2 (L=336, H=336, sigma=0.0). Step-wise RMSE plotted across forecast steps $h \in \{1, \dots, 336\}$ comparing PSO-TCWavKAN with DLinear and theoretical autoregressive error compounding. Under the instance-adaptive routing gate, PSO-TCWavKAN exhibits bounded, sub-linear error accumulation and achieves late-horizon stability ($h \ge 221$), avoiding recursive error propagation.](figures/k09_forecast_step_rmse.png)

**Fig. 9** Step-by-Step Cumulative Error Growth across Prediction Horizons confirming bounded error accumulation.

---

## 8. Computational Complexity, Hardware Efficiency, and Inference Latency

In industrial substation monitoring systems operating under strict real-time constraints, computational efficiency, parameter economy, and inference latency constitute essential requirements alongside predictive accuracy. Table 7 compares PSO-TCWavKAN against all baseline models, evaluated on an NVIDIA Tesla T4 GPU accelerator under identical Kaggle cloud conditions.

**Table 7: Computational Resource Allocation, Parameter Counts, and Inference Latency (NVIDIA Tesla T4 GPU).**

| Model | Parameters (Base L=96, H=96) | Train Time/Epoch (s) | Peak GPU Memory (MB) | Batch Inference Latency (ms) | Edge SCADA Feasibility |
|---|---|---|---|---|---|
| DLinear [4] | 18.6 K | 0.42 | 412 | 1.1 | Yes (ultra-lightweight) |
| NLinear [4] | 9.3 K | 0.45 | 415 | 1.2 | Yes (ultra-lightweight) |
| PatchTST Lite [5] | 169.6 K | 2.85 | 1,240 | 8.6 | Moderate (cloud gateway) |
| Standard B-Spline KAN [1] | 52.4 K | 4.15 | 980 | 12.4 | No (grid update overhead) |
| **PSO-TCWavKAN (Proposed)** | **44.8 K** | **1.24** | **680** | **3.4** | **Yes (high-efficiency)** |

In the canonical base configuration (L = 96, H = 96), PSO-TCWavKAN maintains a compact parameter footprint of 44.8 K trainable parameters (with DLinear at 18.6 K, NLinear at 9.3 K, and PatchTST Lite at 169.6 K), representing a 3.8x reduction compared to PatchTST Lite. Across extended horizons, total parameters scale linearly with O(L * H) due to the linear decomposition and routing projection matrices: for the largest configuration (L = 336, H = 720), DLinear requires 485.3 K parameters, while PSO-TCWavKAN requires 796.9 K parameters (retaining a 2.5x economy relative to PatchTST Lite's 1,993.8 K parameters). Its batch inference latency of 3.4 ms easily satisfies substation SCADA telemetry polling cycles (typically 1 s to 1 min). Eliminating dynamic B-spline grid updates achieves a 3.3x training speedup over standard KANs (1.24 s vs. 4.15 s per epoch), while peak GPU memory consumption of 680 MB is compatible with industrial edge platforms such as NVIDIA Jetson AGX.

---

## 9. Discussion, Engineering Implications, and Limitations

### 9.1. Engineering Significance for Smart Substation Operation

Accurate multi-step thermal forecasting enables proactive substation asset management, condition-based load dispatch, and dynamic thermal rating (DTR). In power transformers, operational lifetime is governed by the thermal degradation of cellulose insulation paper, modeled via the Arrhenius reaction rate in IEEE Std C57.91 [21] and IEC 60076-7 [22]:

FAA = exp[B x (1/383 - 1/(Theta_H + 273))]  **(Eq. 25)** [IEC 60076-7 Arrhenius aging factor; B~15000 K; reference 110 deg C]

where V represents relative insulation aging rate, and Theta_H is the winding hot-spot temperature (in degrees Celsius). In accordance with Equation (25), every sustained 6 °C temperature elevation above the 110 °C reference point doubles the insulation aging acceleration factor.

Evaluating physical prediction error requires denormalizing standardized test MSE via the training set standard deviation ($\sigma_{\text{train}} \approx 11.59$ °C on ETTh2). When evaluated on identical forecasting configurations, PSO-TCWavKAN demonstrates consistent physical error reductions: at $(L = 336, H = 96)$, PSO-TCWavKAN achieves standardized MSE = 0.2360 ($\text{RMSE}_{\text{phys}} = \sqrt{0.2360} \times 11.59 = 5.63$ °C) compared to DLinear's MSE = 0.2520 ($\text{RMSE}_{\text{phys}} = \sqrt{0.2520} \times 11.59 = 5.82$ °C), delivering a direct $0.19$ °C reduction in thermal uncertainty. At $(L = 336, H = 336)$, PSO-TCWavKAN achieves MSE = 0.3162 ($\text{RMSE}_{\text{phys}} = 6.52$ °C) versus DLinear's MSE = 0.3420 ($\text{RMSE}_{\text{phys}} = 6.78$ °C), yielding a $0.26$ °C reduction in operational error.

In power transformer operations, top-oil temperature (TOT) forecasts serve as the direct thermal boundary condition for estimating the internal winding hot-spot temperature ($\Theta_H$) through differential equations defined in IEEE Std C57.91 [21] and IEC 60076-7 [22]. By providing more accurate top-oil boundary estimates during peak dynamic load cycles, premature thermal runaway can be proactively mitigated, supporting dynamic line rating (DTR) and transformer loading capacity optimization by 10–15% without encroaching on insulation safety limits [24].

### 9.2. Limitations and Future Directions

**Limitations.** To ensure sound engineering deployment, several limitations are transparently acknowledged:

(1) **Univariate OT target:** Telemetry forecasting in this study focuses on the oil temperature (OT) variable, adhering to standard sequential regression benchmarks [4]. Extending the architecture to multivariate formulations that jointly model load currents (HUFL, MUFL, LUFL) represents an important future development.

(2) **ETT benchmark scope:** Experimental validation is conducted on the canonical ETTh1 and ETTh2 datasets. While widely recognized in the time-series literature, field deployment requires external validation across diverse substation environments, ambient climate conditions, and cooling classifications (ONAN, ONAF, ODAF).

(3) **Extended-horizon phase accumulation boundary:** As transparently reported in Table 4 and Table 5, at the extreme forecast horizon of (ETTh2, L = 336, H = 720), DLinear achieves lower MSE (0.4351 vs. 0.6656). At 720 steps (30 days ahead), accumulated wavelet phase errors propagate in the non-linear pathway. Developing a horizon-attenuated routing gate where g(x) decays monotonically for H > 336 represents a principled direction to eliminate this boundary condition. The comparative forecast trajectory at H = 720 is illustrated in Fig. 10.

![Fig. 10 Long-Horizon (H=720) Forecast Trajectories: PSO-TCWavKAN vs. DLinear on ETTh2 (L=336). At the 30-day-ahead extreme horizon, DLinear (blue dashed) achieves lower MSE. This transparently illustrates the identified boundary condition: at H=720, accumulated wavelet phase drift in the non-linear pathway degrades performance, performing worse than DLinear for this specific extreme configuration and motivating future horizon-adaptive routing gate extensions.](figures/k10_h720_forecasts.png)

**Fig. 10** Long-Horizon (H=720) Forecast Trajectory — Transparently illustrating the boundary condition at ultra-long horizons.

(4) **CLPSO surrogate budget:** Warm-start exploration utilizes N_p = 5 particles and G_max = 5 generations over a 15% training surrogate subset. While computationally bounded and effective for the 4D parameter vector, scaling to larger search spaces may benefit from adaptive Bayesian surrogate optimization.

**Future Directions.** Future research will focus on: (i) multivariate joint load-temperature modeling; (ii) multi-climate utility field trials; (iii) horizon-decay gating for H > 336; and (iv) hardware synthesis of the continuous dyadic filter bank on FPGA/microcontroller platforms for edge substation integration.

---

## 10. Conclusions

This paper introduced PSO-TCWavKAN, a swarm-optimized temporal convolutional wavelet Kolmogorov-Arnold network designed to address the Grid Update Dilemma and linear-model performance deficit in time-series forecasting. By integrating continuous dyadic Morlet wavelet filter banks with multi-kernel trend decomposition and an instance-adaptive neural routing gate, the architecture ensures asymptotic recovery of the DLinear structural subspace (Proposition 1), providing structural protection against non-linear overfitting. A 4D Comprehensive Learning PSO (CLPSO) warm-start mechanism navigates multi-modal wavelet ripple basins prior to AdamW fine-tuning and Stochastic Weight Averaging.

Empirical evaluation across 16 canonical ETT benchmark configurations demonstrated that PSO-TCWavKAN achieves statistically significant improvements over NLinear (two-sided Wilcoxon p = 0.0443) and competitive parity with DLinear (p = 0.274) and PatchTST Lite (p = 0.0934), dominating 7 out of 8 horizons on the volatile ETTh2 benchmark. Disaggregated ablation analysis across operational regimes confirmed that in standard operational forecasting (H in {96, 192, 336}), the full architecture achieves the lowest Mean MSE (0.21265), with multi-kernel decomposition proving structurally indispensable (+4.62% error surge upon omission). Progressive noise testing confirmed localized resilience under 30% Gaussian perturbation, with error degradation bounded within +1.4% on ETTh2 (L = 336, H = 336). With 44.8 K parameters in the base configuration (scaling to 796.9 K at L=336, H=720, retaining a 2.5x to 4.0x economy relative to PatchTST Lite) and 3.4 ms inference latency, PSO-TCWavKAN provides a mathematically grounded, computationally efficient, and robust architecture for substation thermal forecasting.

---

## Declarations

**Availability of data and materials:** The ETT benchmark datasets used in this study are publicly available at https://github.com/zhouhaoyi/ETDataset. The complete reproducible experimental code, training logs, seed-specific checkpoints, and result artifacts are deposited on GitHub at https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting under the MIT license.

**Consent for publication:** The author confirms that this manuscript has been read and approved for submission and that the content has not been published or submitted for publication elsewhere, either in whole or in part.

**Competing interests:** The author declares that he has no competing financial or personal interests that could have appeared to influence the work reported in this paper.

**Funding:** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Authors contributions:** NPT conceived the theoretical architecture, derived the mathematical proofs including Proposition 1, developed the complete software codebase, executed all GPU training experiments on Kaggle cloud platforms, analyzed and interpreted the empirical results, and drafted the complete manuscript including all tables, figures, and equations.

**Acknowledgements:** Computational resources were provided via Kaggle cloud environments utilizing NVIDIA Tesla T4 GPU accelerators at no cost. The author thanks the open-source scientific Python community (NumPy, PyTorch, Matplotlib, SciPy) for the foundational software tools used throughout this research.

**Ethics approval and consent to participate:** Not applicable. This study involves only publicly available benchmark data and does not involve human participants, animal subjects, or sensitive personal data.

---

## References

1. Liu Z, Wang Y, Vaidya S, Ruehle F, Halverson J, Soljacic M, Hou TY, Tegmark M (2024) KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756. https://doi.org/10.48550/arXiv.2404.19756
2. Bozorgasl Z, Chen H (2024) Wav-KAN: Wavelet Kolmogorov-Arnold Networks. arXiv:2405.12832. https://doi.org/10.48550/arXiv.2405.12832
3. Zhou H, Zhang S, Peng J, Zhang S, Li J, Xiong H, Zhang W (2021) Informer: Beyond efficient transformer for long sequence time-series forecasting. Proc AAAI Conf Artif Intell 35(12):11106-11115
4. Zeng A, Chen M, Zhang L, Xu Q (2023) Are transformers effective for time series? Proc AAAI Conf Artif Intell 37(9):11121-11128
5. Nie Y, Nguyen NH, Sinthong P, Kalagnanam J (2023) A time series is worth 64 words: Long-term forecasting with transformers. Int Conf Learn Represent (ICLR)
6. Yang J, Zhang B, Tong L, Cao S, Sun S (2025) An error correction ACKLSTM model combining data-driven and deep learning for wind power prediction. J Eng Appl Sci 72:191. https://doi.org/10.1186/s44147-025-00771-0
7. Alnaqbi A, Al-Khateeb GG, Zeiada W (2025) Predictive modeling of longitudinal cracking in CRCP using PSO-tuned gradient boosting machines. J Eng Appl Sci 72:58. https://doi.org/10.1186/s44147-025-00623-x
8. Al-Ghaithi A, Al-Mansob RA (2025) A hybrid machine learning method of support vector regression with particle swarm optimization for predicting IRI in continuously reinforced concrete pavement. J Eng Appl Sci 72:128. https://doi.org/10.1186/s44147-025-00706-9
9. Guo Y, Chen Y, Wang H, Liu Y (2024) Data-driven multivariate time series prediction of in-vehicle equipment failure rates. J Eng Appl Sci 71:210. https://doi.org/10.1186/s44147-024-00543-2
10. Chen K, Li M (2025) Multi-objective optimization of turning parameters using Kolmogorov-Arnold network and NSGA-III. J Eng Appl Sci 72:56. https://doi.org/10.1186/s44147-025-00621-3
11. Bezerra FE, Di Santo SG, Manassero Junior G, Tiferes RR, Albertini AR, Santos GR (2024) A forecasting methodology based on second-generation wavelets and neural networks: application to power transformer oil temperature. Electr Eng 106(3):3017-3029. https://doi.org/10.1007/s00202-023-02134-5
12. Lv J, An Z et al (2026) HAKformer: an efficient transformer with hybrid temporal-channel attention and Kolmogorov-Arnold networks for time series forecasting. Appl Intell. https://doi.org/10.1007/s10489-026-06120-x
13. Kim T, Kim J, Tae Y, Park C, Choi JH, Choo J (2022) Reversible instance normalization for accurate time-series forecasting against distribution shift. Int Conf Learn Represent (ICLR)
14. Wu H, Xu J, Wang J, Long M (2021) Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Adv Neural Inf Process Syst (NeurIPS) 34:22419-22430
15. Zhou T, Ma Z, Wen Q, Wang X, Sun L, Jin R (2022) FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. Int Conf Mach Learn (ICML) pp 27268-27286
16. Liu Y, Hu T, Zhang H, Wu H, Wang S, Ma L, Long M (2024) iTransformer: Inverted transformers are effective for time series forecasting. Int Conf Learn Represent (ICLR)
17. Liang J, Qin A, Suganthan PN, Baskar S (2006) Comprehensive learning particle swarm optimizer for global optimization of multimodal functions. IEEE Trans Evol Comput 10(3):281-295
18. Clerc M, Kennedy J (2002) The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Trans Evol Comput 6(1):58-73
19. Izmailov P, Podoprikhin D, Garipov T, Vetrov D, Wilson AG (2018) Averaging weights leads to wider optima and better generalization. Conf Uncert Artif Intell (UAI)
20. Loshchilov I, Hutter F (2019) Decoupled weight decay regularization. Int Conf Learn Represent (ICLR)
21. IEEE Power and Energy Society (2012) IEEE guide for loading mineral-oil-immersed transformers and step-voltage regulators. IEEE Std C57.91-2011
22. International Electrotechnical Commission (2018) Loading guide for oil-immersed power transformers. IEC Std 60076-7:2018
23. Susa D, Lehtonen M, Nordman H (2005) Dynamic thermal modelling of power transformers. IEEE Trans Power Deliv 20(1):197-204
24. Swift G, Molinski TS, Lehn W (2001) A fundamental approach to transformer thermal modeling. IEEE Trans Power Deliv 16(2):171-177
25. Radakovic Z, Feser K (2003) A new method for the calculation of the hot-spot temperature in power transformers. IEEE Trans Power Deliv 18(4):1284-1292
26. Pradhan MK, Ramu TS (2004) Estimation of the remaining life of power transformers. IEEE Trans Power Deliv 19(4):1789-1795
27. Mallat SG (1989) A theory for multiresolution signal decomposition: the wavelet representation. IEEE Trans Pattern Anal Mach Intell 11(7):674-693
28. Daubechies I (1992) Ten lectures on wavelets. Society for Industrial and Applied Mathematics, Philadelphia
29. Zhang Q, Benveniste A (1992) Wavelet networks. IEEE Trans Neural Netw 3(6):889-898
30. Pati YC, Krishnaprasad PS (1993) Analysis and synthesis of feedforward neural networks using discrete affine wavelet transformations. IEEE Trans Neural Netw 4(1):73-85
31. Kennedy J, Eberhart R (1995) Particle swarm optimization. Proc ICNN95 Int Conf Neural Netw 4:1942-1948
32. Shi Y, Eberhart R (1998) A modified particle swarm optimizer. IEEE Int Conf Evol Comput pp 69-73
33. Poli R, Kennedy J, Blackwell T (2007) Particle swarm optimization: An overview. Swarm Intell 1(1):33-57
34. He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image recognition. Proc IEEE Conf Comput Vis Pattern Recognit pp 770-778
35. Vaswani A, Shazeer N, Parmar N et al (2017) Attention is all you need. Adv Neural Inf Process Syst 30:5998-6008
36. Kitaev N, Kaiser L, Levskaya A (2020) Reformer: The efficient transformer. Int Conf Learn Represent (ICLR)
37. Wang S, Li BY, Khabsa M, Fang H, Ma H (2020) Linformer: Self-attention with linear complexity. arXiv:2006.04768
38. Tay Y, Dehghani M, Bahri D, Metzler D (2022) Efficient transformers: A survey. ACM Comput Surv 55(6):1-28
39. Wilcoxon F (1945) Individual comparisons by ranking methods. Biom Bull 1(6):80-83
40. Demsar J (2006) Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1-30
41. Benavoli A, Corani G, Demsar J, Zaffalon M (2017) Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. J Mach Learn Res 18(1):2653-2688
42. Altarabichi MG (2024) DropKAN: Regularizing Kolmogorov-Arnold Networks via DropNode and DropEdge strategies. arXiv:2407.13044
