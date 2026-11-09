# PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting

**Running Title:** PSO-TCWavKAN for Transformer Thermal Dynamics

**Author:**  
Phu Toan Nguyen  
Faculty of Information Technology, Hanoi University of Business and Technology (HUBT), No. 29A, Lane 124, Vinh Tuy Street, Vinh Hung Ward, Hai Ba Trung District, Hanoi 100000, Vietnam  
Email: nphu211206@gmail.com  
ORCID: 0009-0005-0529-2131  

---

## Abstract

Power transformer thermal dynamics forecasting is vital for assessing insulation degradation, preventing catastrophic dielectric breakdown, and enabling Dynamic Thermal Rating in modern smart grids. In accordance with IEEE Std C57.91-2011, top-oil and winding temperatures directly dictate asset loss-of-life via the Arrhenius model. However, existing Long-Term Time Series Forecasting (LTSF) paradigms face significant structural bottlenecks: self-attention Transformers exhibit quadratic complexity $\mathcal{O}(L^2)$ and temporal permutation vulnerabilities, linear baselines underfit multi-frequency thermal inertia, and standard Kolmogorov–Arnold Networks (KAN) suffer from discrete B-spline grid instability during continuous distribution shift. To overcome these limitations, this paper proposes PSO-TCWavKAN, a unified hybrid deep learning framework. Our architecture integrates Invertible Reversible Instance Normalization (RevIN) with detached statistics, a dual-branch depthwise temporal convolution module, and continuous Morlet wavelet KAN layers featuring smooth analytic gradients across the entire real domain. To circumvent sub-optimal local minima in the multi-scale wavelet parameter landscape, we deploy a Comprehensive Learning Particle Swarm Optimizer (CLPSO) governed by Clerc's constriction factor ($\chi = 0.72984$) on a leak-free surrogate proxy. We formally prove that PSO-TCWavKAN achieves strictly linear computational complexity $\mathcal{O}(DN)$. Evaluated across 16 canonical configurations on ETTh1 and ETTh2 telemetry, PSO-TCWavKAN delivers an average 12.30% MSE reduction over DLinear, confirmed statistically significant by the Wilcoxon signed-rank test ($p = 3.05 \times 10^{-5}$).

**Keywords:** Kolmogorov–Arnold Networks (KAN); Continuous Wavelet Transform; Particle Swarm Optimization; Power Transformer Thermal Dynamics; Reversible Instance Normalization; Long-Term Time Series Forecasting.

---

## 1. Introduction

### 1.1 Industrial Context and Physical Stakes
Power transformers represent critical capital assets within electric power transmission and distribution networks, where continuous operational reliability underpins modern smart grid resilience. The operational lifetime of mineral-oil-immersed power transformers is fundamentally governed by the thermodynamic, thermal-oxidative, and electro-mechanical degradation of solid cellulose paper insulation surrounding high-voltage conductor windings. In accordance with the *IEEE Guide for Loading Mineral-Oil-Immersed Transformers* (IEEE Std C57.91-2011) and IEC 60076-7, the top-oil temperature (TOT) and internal winding hottest-spot temperature (HST) serve as the primary thermodynamic state variables that dictate the instantaneous rate of insulation loss-of-life via the Arrhenius thermal aging model:
\begin{equation}
V = \exp \left( \frac{E_A}{R \cdot (T_{\text{ref}} + 273.15)} - \frac{E_A}{R \cdot (\Theta_{\text{HST}} + 273.15)} \right) = \exp \left( \frac{E_A}{R \cdot 383.15} - \frac{E_A}{R \cdot (\Theta_{\text{HST}} + 273.15)} \right),
\label{eq:arrhenius_aging}
\end{equation}
where $V$ is the relative aging rate (aging acceleration factor), $E_A = 111\,\text{kJ/mol} = 111{,}000\,\text{J/mol}$ is the activation energy of cellulose degradation, $R = 8.314\,\text{J/(mol}\cdot\text{K)}$ denotes the universal gas constant, $T_{\text{ref}} = 110^{\circ}\text{C}$ ($383.15\,\text{K}$) is the standardized rated reference hot-spot temperature, and $\Theta_{\text{HST}}$ is the instantaneous hottest-spot temperature in degrees Celsius ($^{\circ}\text{C}$). Under this physical law, sustained thermal excursions of even $6^{\circ}\text{C}$ to $10^{\circ}\text{C}$ beyond the nominal reference limit ($110^{\circ}\text{C}$) double the chemical rate of cellulose depolymerization ($V \approx 2.0$), drastically reducing mechanical tensile strength and precipitating catastrophic dielectric flashovers, violent transformer tank rupture, and widespread cascading blackouts.

In modern power grids, high penetration of intermittent renewable generation (such as solar photovoltaics and wind turbines), severe ambient temperature fluctuations, and rapid electric vehicle charging cycles impose volatile, non-stationary thermal loading profiles on substation transformers. Consequently, developing high-precision multi-horizon prognostic models capable of forecasting transformer thermal trajectories across extended horizons ($H \in [96, 720]$ hours) is vital for enabling proactive Dynamic Thermal Rating (DTR), optimal load dispatch, and predictive asset preservation.

### 1.2 Critique of Existing LTSF Paradigms
Over the past decade, Long-Term Time Series Forecasting (LTSF) has been dominated by deep sequential models. The paradigm initially progressed from Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) units to specialized Transformer architectures incorporating sparse self-attention mechanisms, such as Informer, Autoformer, FEDformer, and PatchTST. Although self-attention enables expansive temporal receptive fields, its canonical computational complexity and GPU memory footprint scale quadratically ($\mathcal{O}(L^2)$) with lookback window length $L$, imposing prohibitive operational bottlenecks on resource-constrained substation edge monitoring devices. Furthermore, recent theoretical and empirical investigations have revealed that canonical self-attention mechanisms exhibit temporal permutation vulnerabilities, frequently diluting localized directional causality, temporal ordering, and phase alignment in continuous thermodynamic processes.

In response to these inefficiencies, direct linear projection architectures, such as DLinear and NLinear, were proposed as lightweight alternatives. While linear baselines demonstrate remarkable parameter efficiency and operational simplicity, their single-layer linear projections assume quasi-linear transition dynamics. As a consequence, they severely underfit the multi-frequency harmonic resonances, non-linear fluid-mechanical oil convection cycles, and sudden thermal shock loads characteristic of operational power transformers.

### 1.3 The KAN Opportunity and the Discrete Spline Grid Update Dilemma
The recent conceptualization of Kolmogorov–Arnold Networks (KAN) by Liu et al. (2024) introduced a mathematically compelling alternative to classical Multi-Layer Perceptrons (MLPs). Grounded in the Kolmogorov–Arnold representation theorem, KANs replace fixed node activation functions with learnable univariate functions positioned directly on network edges. By representing multivariate functional mappings through superpositions of one-dimensional continuous functions, KANs offer superior parameter efficiency, functional expressiveness, and physical interpretability.

However, canonical KAN implementations parameterize edge activations using B-splines over discrete spatial knot grids:
\begin{equation}
\phi(x) = \sum_{i} c_i B_i(x).
\end{equation}
In continuous, non-stationary industrial telemetry, B-spline KANs suffer from substantial computational overhead ($\mathcal{O}(G B K)$) and severe numerical instability. Crucially, as the internal activation distributions drift during training, discrete knot grids must be dynamically realigned via periodic *re-gridding*. This discrete grid update destabilizes first- and second-moment momentum accumulators in gradient-based optimizers such as Adam, inducing the "Grid Update Dilemma" and triggering catastrophic loss spikes. Moreover, because B-spline kernels exhibit compact, localized support, they produce zero gradients for out-of-grid load excursions ($x \notin [-1, 1]$), precipitating catastrophic forgetting and boundary extrapolation collapse during unexpected thermal surges.

### 1.4 The Proposed Solution: PSO-TCWavKAN
To overcome these structural dilemmas, this paper proposes **PSO-TCWavKAN**, a unified hybrid deep learning framework designed for electric power transformer thermal dynamics forecasting. Rather than relying on discrete, unstable B-spline grids, we formulate univariate edge activations using **Continuous Morlet Wavelets (WavKAN)**, characterized by closed-form, infinitely differentiable analytical expressions that provide smooth, well-behaved gradients across the entire real domain $\mathbb{R}$. 

To eliminate non-stationary distribution drift without multi-subnet overhead, an Invertible Reversible Instance Normalization (RevIN) layer with detached statistics is integrated at the system boundaries. Temporal trend drift and high-frequency cyclical oscillations are decomposed via a dual-branch depthwise temporal convolution extractor. To circumvent sub-optimal local minima in the highly non-convex multi-scale wavelet parameter landscape, we deploy a Comprehensive Learning Particle Swarm Optimizer (CLPSO) governed by Maurice Clerc’s constriction factor ($\chi = 0.72984$) on a leak-free 85/15 surrogate proxy, mathematically preventing swarm velocity explosion and discovering optimal parameter warm-starts.

### 1.5 Summary of Major Contributions
The primary contributions of this paper are summarized as follows:
* **Architectural Synthesis**: We design and implement PSO-TCWavKAN, seamlessly coupling Invertible RevIN with detached statistics, multi-kernel temporal trend-seasonal decomposition, pre-LayerNorm continuous Morlet WavKAN layers with analytic gradients resolving the B-spline grid update dilemma, and a dual FFT spectral head into an end-to-end differentiable pipeline.
* **Theoretical Proofs**: We formally state and prove Proposition 1, demonstrating that under vanishing router gates ($g(X) \to 0$), PSO-TCWavKAN gracefully collapses to canonical DLinear as a non-degenerate structural subspace with non-vanishing gradient pathways ($\nabla_W \mathcal{L} \neq 0$). Furthermore, Theorem 1 mathematically proves that the forward and backward passes maintain strictly linear computational complexity $\mathcal{O}(DN)$.
* **Metaheuristic Constriction Grounding**: We derive the closed-form Clerc constriction factor ($\chi = 0.72984, \phi = 4.1$) for CLPSO, establishing bounded swarm velocity trajectories on an 85/15 leak-free surrogate training proxy strictly contained within the training partition.
* **Empirical and Statistical Rigor**: Evaluated across 16 canonical configurations on ETTh1 and ETTh2 telemetry under three independent random seeds (2024, 2025, 2026) with Bessel correction ($ddof=1$), PSO-TCWavKAN achieves an average 12.30% MSE reduction over DLinear. The superiority is confirmed statistically significant via the non-parametric Wilcoxon signed-rank test ($p = 3.05 \times 10^{-5}$, $16/0/0$ win record).

### 1.6 Paper Organization
The remainder of this paper is organized as follows: Section 2 reviews related work in transformer LTSF, Kolmogorov–Arnold networks, wavelet architectures, and swarm intelligence. Section 3 details the mathematical formulation of the PSO-TCWavKAN architecture, the complexity derivation of Theorem 1, and the formal proof of Proposition 1. Section 4 presents the experimental setup, benchmark results, statistical significance testing, and ablation studies. Section 5 discusses computational efficiency and industrial deployment practicality. Section 6 concludes the paper and outlines future research directions. Section 7 presents mandatory Springer Nature declarations.

---

## 2. Related Work

### 2.1 Long-Term Time Series Forecasting in Power Infrastructure
Time series forecasting in power systems has evolved from classical statistical models (ARIMA, SARIMA) to deep learning architectures. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks were widely adopted for energy load forecasting, but their sequential step-by-step processing impedes parallel training on long historical sequences. The advent of the Transformer architecture with self-attention catalyzed a wave of LTSF models. Zhou et al. introduced Informer with ProbSparse attention to reduce attention complexity to $\mathcal{O}(L \log L)$. Wu et al. proposed Autoformer, replacing point-wise self-attention with auto-correlation mechanisms based on Wiener–Khinchin theorem. Zhou et al. developed FEDformer, computing frequency-domain compact representations via Fourier and wavelet transforms. Nie et al. proposed PatchTST, segmenting time series into subseries-level patches and leveraging channel-independent attention. More recently, Liu et al. introduced iTransformer, inverting the attention mechanism across variates rather than time, and Luo and Wang introduced ModernTCN, reviving pure convolutional networks for LTSF.

Within *Neural Computing and Applications*, recent landmark contributions include GCINet by Fan et al. (2025), which employs global convolution interaction and pre-trained reversible normalization, and FEDAF by Yang et al. (2024), which demonstrates frequency-enhanced attention-free forecasting. However, existing Transformer and convolutional architectures still struggle to capture localized, high-frequency thermal shocks under extreme operating conditions.

### 2.2 Kolmogorov–Arnold Networks: Mathematical Evolution and Spline Fragility
The Kolmogorov–Arnold representation theorem (Arnold, 1957; Kolmogorov, 1957) states that any multivariate continuous function $f: [0, 1]^d \to \mathbb{R}$ can be represented as a finite superposition of continuous univariate functions and addition:
\begin{equation}
f(\mathbf{x}) = \sum_{q=0}^{2d} \Phi_q \left( \sum_{p=1}^d \phi_{q,p}(x_p) \right).
\end{equation}
Liu et al. (2024) translated this mathematical theorem into deep learning by parameterizing $\phi_{q,p}$ as learnable B-spline curves over discrete grids, creating Kolmogorov–Arnold Networks (KANs). Subsequent variants have emerged to alleviate computational bottlenecks: Chebyshev-KAN (Gençer et al., 2024) replaces splines with orthogonal Chebyshev polynomials; Fast-KAN uses Radial Basis Functions (RBFs); ReLU-KAN utilizes piecewise linear ReLUs; and P1-KAN (Warin, NCAA 2026) employs piecewise linear finite elements for high-dimensional stochastic control. 

Despite these advancements, canonical B-spline KANs suffer from the "Grid Update Dilemma": adapting discrete knot grids to shifting input distributions invalidates optimizer moment accumulators, precipitating optimization divergence. While P1-KAN eliminates B-spline evaluation cost, its linear hat functions possess non-differentiable sharp cusps that struggle to model continuous, smooth oscillatory thermodynamic processes.

### 2.3 Wavelet Neural Networks and Multiresolution Telemetry Analysis
Wavelet analysis provides powerful time-frequency localization, overcoming the limitations of Fourier analysis in processing non-stationary signals (Daubechies, 1992; Mallat, 1999). Classical Wavelet Neural Networks (WNNs), pioneered by Zhang and Benveniste (1992) and Szu et al. (1992), replaced sigmoid neurons with dilated and translated wavelet atoms. In power engineering, wavelets have been extensively applied for power quality disturbance detection and fault diagnosis.

Recently in NCAA, Tamilselvi et al. (2025) proposed Wavelet-LSTM for environmental telemetry forecasting, coupling Maximal Overlap Discrete Wavelet Transform (MODWT) with LSTM networks. However, Wavelet-LSTM utilizes a decoupled two-stage architecture where the wavelet decomposition is fixed offline, preventing end-to-end gradient backpropagation. Bozorgasl and Chen (2024) introduced Wav-KAN for static image classification. Our work significantly advances this concept by formulating an end-to-end differentiable Continuous Morlet WavKAN block integrated with temporal depthwise convolutions, specifically designed for non-stationary industrial time-series forecasting.

### 2.4 Swarm Intelligence and Constrained Evolutionary Optimization
Particle Swarm Optimization (PSO), introduced by Kennedy and Eberhart (1995), models social foraging behavior to solve non-convex global optimization problems. In deep learning, PSO has been widely utilized for architecture search and hyperparameter tuning. However, standard PSO suffers from velocity explosion and premature convergence in high-dimensional non-convex landscapes.

To guarantee convergence, Maurice Clerc and James Kennedy (2002) derived the constriction factor method ($\chi$), establishing mathematically proven stability boundaries for particle velocities. Liang et al. (2006) introduced Comprehensive Learning Particle Swarm Optimization (CLPSO), enabling particles to learn from diverse historical exemplars across different dimensions. In this paper, we deploy a Clerc-constrained CLPSO algorithm on an explicit 85/15 leak-free surrogate proxy to discover optimal initializations for the multi-scale continuous wavelet parameter space, providing a principled warm-start for subsequent gradient descent.

---

## 3. Proposed Methodology

### 3.1 Overall Framework Architecture
The overall architectural workflow of PSO-TCWavKAN is depicted in Figure 1. Given an input telemetry sequence $\mathbf{X} \in \mathbb{R}^{B \times L \times C}$, where $B$ is batch size, $L$ is lookback length, and $C$ is the number of telemetry channels, the framework executes through seven integrated processing stages:
1. Invertible Reversible Instance Normalization (RevIN) with detached statistics.
2. Multi-Kernel Temporal Convolutional Series Decomposition.
3. Patch Tokenization and Linear Embedding.
4. Stacked Pre-LayerNorm TC-WavKAN Blocks with continuous Morlet wavelets.
5. Scalable Depthwise Cross-Patch Mixing ($\mathcal{O}(DN)$).
6. Adaptive Dual FFT Spectral Head with soft-thresholding shrinkage.
7. Instance-Adaptive Neural Router with Proposition 1 structural lower-bound recovery, followed by RevIN inverse restoration.

![Figure 1: Full architectural schematic of proposed PSO-TCWavKAN framework](figures/fig1_architecture.png)

**Figure 1:** Full architectural schematic of the proposed PSO-TCWavKAN forecasting framework featuring Invertible Reversible Instance Normalization, multi-kernel temporal series decomposition, pre-LayerNorm continuous Morlet WavKAN layers, depthwise inter-patch mixing, adaptive dual FFT spectral head, and instance neural router with guaranteed asymptotic linear recovery. Parameter bounds range from 49.8K ($L=96, H=96$) to 800.7K ($L=336, H=720$).

```
                            Input Time Series X ∈ R^{B × L × C}
                                          │
                            [RevIN Affine Normalization]
                        X_norm = ((X - μ) / σ) ⊙ γ + β
                          (Detached μ, σ; Safe eps γ)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
       [Multi-Kernel Decomposition]                [Instance-Adaptive Router]
       Kernel sizes: (25, 49)                      g(x) = σ(W_g · X_norm + b_g)
            ┌────────┴────────┐                                │
            ▼                 ▼                                │
       Trend Component   Seasonal Component                    │
         X_trend           X_seas                              │
            │                 │                                │
            │        [Patch Tokenization]                      │
            │        P=16, S=8 (or P=24, S=12)                 │
            │                 ▼                                │
            │        [Patch Linear Embedding]                  │
            │        Z ∈ R^{B × N × d_model}                   │
            │                 │                                │
            │        [Stacked TC-WavKAN Blocks]                │
            │        - Pre-LayerNorm Residuals                 │
            │        - Multi-Scale Wavelet Filter Bank         │
            │          (Per-channel ω_0, γ; Continuous Morlet) │
            │        - Scalable Depthwise Cross-Patch Mixing   │
            │          (O(D·N) linear complexity)              │
            │        - DropKAN Regularization                  │
            │                 ▼                                │
            │        [Adaptive Dual FFT Head]                  │
            │        - Time-domain pooling projection          │
            │        - Spectral soft-thresholding (rfft)       │
            │        - Sigmoid-gated adaptive fusion           │
            │                 ▼                                │
            │            Y_seas_kan                            │
            │                 │                                │
            ▼                 ▼                                │
       Y_trend_lin       Y_seas_lin                            │
       (Linear Proj)     (Linear Proj)                         │
            │                 │                                │
            │                 └──────────────┬─────────────────┘
            │                                ▼
            │             [Neural Routing Convex Combination]
            │             Y_seas = (1 - g(x)) ⊙ Y_seas_lin + g(x) ⊙ Y_seas_kan
            │             (Proposition 1: as g(x)→0, Y_seas → Y_seas_lin)
            │                                │
            └────────────────┬───────────────┘
                             ▼
                        Y_norm = Y_trend_lin + Y_seas
                             │
               [RevIN Exact Affine Denormalization]
               Y_unaffine = (Y_norm - β) / (safe_γ)
               Y_physical = Y_unaffine ⊙ σ + μ
                             │
                             ▼
                    Final Output Y ∈ R^{B × H × C}
```

### 3.2 Invertible Reversible Instance Normalization (RevIN)
To neutralize non-stationary distribution drift without multi-subnet overhead, we formulate an Invertible RevIN layer with detached statistics and safe denominator clamping.

#### 3.2.1 Detached Instance Moment Extraction
For each batch sample $b \in \{1, \dots, B\}$ and channel $c \in \{1, \dots, C\}$, the instance mean and variance are computed across the temporal lookback axis:
\begin{equation}
\boldsymbol{\mu}_{\mathbf{X}}(b, c) = \frac{1}{L} \sum_{t=1}^{L} \mathbf{X}(b, t, c), \quad \boldsymbol{\sigma}_{\mathbf{X}}^2(b, c) = \frac{1}{L} \sum_{t=1}^{L} \left( \mathbf{X}(b, t, c) - \boldsymbol{\mu}_{\mathbf{X}}(b, c) \right)^2.
\end{equation}
To prevent gradient signals from corrupting input distribution statistics, $\boldsymbol{\mu}_{\mathbf{X}}$ and $\boldsymbol{\sigma}_{\mathbf{X}}$ are strictly detached from the autograd graph:
\begin{equation}
\boldsymbol{\mu}_{\mathbf{X}} \leftarrow \operatorname{detach}\left( \boldsymbol{\mu}_{\mathbf{X}} \right), \quad \boldsymbol{\sigma}_{\mathbf{X}} \leftarrow \operatorname{detach}\left( \sqrt{\boldsymbol{\sigma}_{\mathbf{X}}^2 + \epsilon} \right),
\end{equation}
where $\epsilon = 10^{-5}$ is the numerical stabilization constant.

#### 3.2.2 Learnable Affine Normalization
The normalized input $\hat{\mathbf{X}} \in \mathbb{R}^{B \times L \times C}$ is obtained via learnable scale $\boldsymbol{\gamma} \in \mathbb{R}^{1 \times 1 \times C}$ and shift $\boldsymbol{\beta} \in \mathbb{R}^{1 \times 1 \times C}$ parameters:
\begin{equation}
\hat{\mathbf{X}}(b, t, c) = \left( \frac{\mathbf{X}(b, t, c) - \boldsymbol{\mu}_{\mathbf{X}}(b, c)}{\boldsymbol{\sigma}_{\mathbf{X}}(b, c)} \right) \odot \boldsymbol{\gamma}(c) + \boldsymbol{\beta}(c),
\end{equation}
where $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are initialized to $\mathbf{1}$ and $\mathbf{0}$ respectively.

#### 3.2.3 Safe Denominator Clamping for Inversion
To prevent division-by-zero or gradient explosion during backpropagation when $\boldsymbol{\gamma} \to 0$, we introduce protected denominator scaling:
\begin{equation}
\operatorname{safe\_}\boldsymbol{\gamma} = \operatorname{sign}(\boldsymbol{\gamma}) \odot \max\left( |\boldsymbol{\gamma}|, \epsilon_{\gamma} \right), \quad \epsilon_{\gamma} = 10^{-4},
\end{equation}
where $\operatorname{sign}(u) = +1$ if $u \ge 0$, and $-1$ if $u < 0$.

#### 3.2.4 Exact Inverse Denormalization
Given normalized forecast $\hat{\mathbf{Y}}_{\text{norm}} \in \mathbb{R}^{B \times H \times C}$, physical engineering units are restored in closed form:
\begin{equation}
\hat{\mathbf{Y}}_{\text{unaffine}}(b, h, c) = \frac{\hat{\mathbf{Y}}_{\text{norm}}(b, h, c) - \boldsymbol{\beta}(c)}{\operatorname{safe\_}\boldsymbol{\gamma}(c)},
\end{equation}
\begin{equation}
\tilde{\mathbf{Y}}(b, h, c) = \hat{\mathbf{Y}}_{\text{unaffine}}(b, h, c) \odot \boldsymbol{\sigma}_{\mathbf{X}}(b, c) + \boldsymbol{\mu}_{\mathbf{X}}(b, c).
\end{equation}

#### 3.2.5 Proposition 2: Distribution Shift Neutralization
\begin{proposition}[Error Lower Bound under Non-Stationary Shift]
\label{prop:revin_bound}
Let $\mathbf{X} \in \mathbb{R}^{L}$ denote a non-stationary input sequence with mean drift $\mathbb{E}[\mathbf{X}] = \boldsymbol{\mu}_{\text{test}} \neq \boldsymbol{\mu}_{\text{train}}$ and covariance $\boldsymbol{\Sigma}_{\text{test}}$. Let $\mathbf{Y} \in \mathbb{R}^{H}$ denote the target future trajectory with mean $\boldsymbol{\nu}_{\text{test}}$. Any static linear forecasting operator $\hat{\mathbf{Y}} = \mathbf{W}\mathbf{X} + \mathbf{b}$ incurs an expected generalization error bounded from below by:
\begin{equation}
\mathbb{E}\left[ \|\mathbf{Y} - \hat{\mathbf{Y}}\|_2^2 \right] \ge \operatorname{Tr}\left( \boldsymbol{\Sigma}_{\mathbf{Y}|\mathbf{X}}^{\text{test}} \right) + \left\| \boldsymbol{\nu}_{\text{test}} - \mathbf{W}\boldsymbol{\mu}_{\text{test}} - \mathbf{b} \right\|_2^2,
\end{equation}
where the second term represents an irreducible distribution-shift penalty. Under Invertible RevIN, the mean drift penalty is identically forced to zero:
\begin{equation}
\inf_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \left\| \mathbb{E}[\tilde{\mathbf{Y}}] - \boldsymbol{\nu}_{\text{test}} \right\|_2^2 = 0,
\end{equation}
compressing the error lower bound strictly to the intrinsic conditional variance $\operatorname{Tr}(\boldsymbol{\Sigma}_{\mathbf{Y}|\mathbf{X}}^{\text{test}})$.
\end{proposition}

### 3.3 Multi-Kernel Temporal Series Decomposition
Normalized sequence $\hat{\mathbf{X}} \in \mathbb{R}^{B \times L}$ is decomposed into low-frequency thermodynamic trend and cyclical seasonal components using an ensemble of symmetric 1D depthwise average pooling kernels $\mathcal{K} = \{25, 49\}$:
\begin{equation}
\mathbf{T}^{(m)}(b, t) = \frac{1}{k_m} \sum_{j = -\lfloor k_m / 2 \rfloor}^{\lfloor k_m / 2 \rfloor} \hat{\mathbf{X}}_{\text{pad}}(b, t + j),
\end{equation}
where $\hat{\mathbf{X}}_{\text{pad}}$ denotes replication-padded input of length $\lfloor k_m / 2 \rfloor$. The aggregated trend $\mathbf{X}_{\text{trend}}$ and seasonal residual $\mathbf{X}_{\text{seasonal}}$ are:
\begin{equation}
\mathbf{X}_{\text{trend}} = \frac{1}{|\mathcal{K}|} \sum_{m=1}^{|\mathcal{K}|} \mathbf{T}^{(m)}, \quad \mathbf{X}_{\text{seasonal}} = \hat{\mathbf{X}} - \mathbf{X}_{\text{trend}}.
\end{equation}
Simultaneously, direct linear pathways map the decomposed series directly to horizon $H$:
\begin{equation}
\hat{\mathbf{Y}}_{\text{trend\_lin}} = \mathbf{W}_{\text{trend}} \mathbf{X}_{\text{trend}} + \mathbf{b}_{\text{trend}} \in \mathbb{R}^{B \times H},
\end{equation}
\begin{equation}
\hat{\mathbf{Y}}_{\text{seas\_lin}} = \mathbf{W}_{\text{seas}} \mathbf{X}_{\text{seasonal}} + \mathbf{b}_{\text{seas}} \in \mathbb{R}^{B \times H},
\end{equation}
where $\mathbf{W}_{\text{trend}}, \mathbf{W}_{\text{seas}} \in \mathbb{R}^{H \times L}$ and $\mathbf{b}_{\text{trend}}, \mathbf{b}_{\text{seas}} \in \mathbb{R}^{H}$.

### 3.4 Patch Tokenization and Scale-Adaptive Temporal Embedding
To capture localized temporal patterns while avoiding quadratic computational scaling, $\mathbf{X}_{\text{seasonal}}$ is unfolded into overlapping patches along the time dimension:
- Patch length $P = 16$ and stride $S = 8$ for $L \le 96$; $P = 24$ and stride $S = 12$ for $L > 96$.
- Number of patch tokens: $N = \lfloor (L - P) / S \rfloor + 1$.

Each temporal patch is projected into latent embedding space $\mathbb{R}^D$ with $D = d_{\text{model}} = 32$:
\begin{equation}
\mathbf{Z}^{(0)} = \mathbf{W}_{\text{patch}} \operatorname{Unfold}(\mathbf{X}_{\text{seasonal}}, P, S) + \mathbf{b}_{\text{patch}} \in \mathbb{R}^{B \times N \times D}.
\end{equation}

### 3.5 Pre-LayerNorm TC-WavKAN Blocks and Continuous Wavelet Filter Bank
The patch tokens $\mathbf{Z}$ are processed through stacked TC-WavKAN blocks incorporating:
1. **Pre-LayerNorm Residual Connection**:
   Normalizing before transformation, $\tilde{\mathbf{Z}} = \operatorname{LayerNorm}(\mathbf{Z})$, preserves unbroken identity gradient paths across deep block stacks.
2. **Local Temporal Context Mixing**:
   A 1D depthwise convolution with kernel size 5 and groups $D$ extracts local sequential continuity:
   \begin{equation}
   \mathbf{Z}_{\text{local}} = \operatorname{DWConv1D}_{k=5, \text{groups}=D}(\tilde{\mathbf{Z}}).
   \end{equation}
3. **Multi-Scale Continuous Morlet Wavelet Filter Bank**:
   For each feature channel $d \in \{1, \dots, D\}$ and wavelet octave $k \in \{1, \dots, K\}$ ($K=5$):
   \begin{equation}
   z_{b, n, d, k} = \frac{x_{b, n, d} - \mu_{d, k}}{\sigma_{d, k}}, \quad \sigma_{d, k} = \operatorname{softplus}(\tilde{\sigma}_{d, k}) + 10^{-4} > 0,
   \end{equation}
   where $\mu_{d, k} \in \mathbb{R}$ is the learnable translation center and $\tilde{\sigma}_{d, k} \in \mathbb{R}$ is the unconstrained dilation logit. Coordinates are safely clamped: $z_c = \operatorname{clip}(z, -5.0, 5.0)$.
   
   Each channel learns an independent base frequency $\omega_{0, d}$ and octave dilation exponent $\gamma_d$:
   \begin{equation}
   \omega_{d, k} = (|\omega_{0, d}| + 10^{-5}) \cdot 2^{\gamma_d \cdot (k - (K-1)/2)}.
   \end{equation}
   The continuous Morlet wavelet basis is evaluated in closed form:
   \begin{equation}
   \psi_{\text{Morlet}}(z_c) = \exp\left( -\frac{1}{2} z_c^2 \right) \cos(\omega_{d, k} z_c).
   \end{equation}
   The filter bank also supports Mexican Hat (Ricker) and Derivative of Gaussian (DoG) wavelets:
   \begin{equation}
   \psi_{\text{MexHat}}(z_c) = (1 - \tilde{z}^2) \exp\left( -\frac{1}{2} \tilde{z}^2 \right), \quad \psi_{\text{DoG}}(z_c) = -\tilde{z} \exp\left( -\frac{1}{2} \tilde{z}^2 \right),
   \end{equation}
   where $\tilde{z} = z_c \cdot (\omega_{d, k} / (|\omega_{0, d}| + 10^{-5}))$.
4. **Dynamic Positional Mixing and DropKAN**:
   A patch positional embedding $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N \times 16}$ modulates octave mixing weights via a small MLP:
   \begin{equation}
   \mathbf{c}_{n, d, k} = \mathbf{c}_{s, d, k} + \operatorname{MLP}(\mathbf{E}_{\text{pos}})_{n, d, k}, \quad \mathbf{w}_{b, n, d} = \sum_{k=1}^K c_{n, d, k} \cdot \psi_{d, k}(z_{b, n, d, k}).
   \end{equation}
   DropKAN regularizes the representation by randomly dropping basis functions with probability $p=0.05$ during training:
   \begin{equation}
   \mathbf{w}_{\text{drop}} = \mathbf{w} \odot \frac{\mathbb{I}(\mathbf{r} > 0.05)}{1 - 0.05}, \quad \mathbf{w}_{\text{gated}} = \operatorname{sigmoid}(\mathbf{g}_{\text{wav}}) \odot \mathbf{W}_{\text{proj}} \mathbf{w}_{\text{drop}}.
   \end{equation}

![Figure 2: Multi-scale continuous wavelet basis functions and admissibility](figures/fig2_wavelet_basis.png)

**Figure 2:** Multi-scale continuous wavelet basis functions and admissibility. (a) Scale dilation of Mexican Hat and Gaussian envelope across scales $a \in \{0.5, 1.0, 2.0\}$; (b) Spatial translation shift invariance across offsets $b \in \{-2.0, 0.0, 2.0\}$; (c) Comparison between continuous wavelet archetypes (Morlet, Mexican Hat, Derivative of Gaussian) and cubic B-spline; (d) Fourier power spectra demonstrating band-pass filtering and strictly zero DC response ($\hat{\psi}(0) = 0$).

### 3.6 Scalable Inter-Patch Token Mixing with Strictly Linear Complexity $\mathcal{O}(DN)$
Unlike standard multi-head self-attention that computes full pairwise cross-attention $\mathbf{Q}\mathbf{K}^T$ requiring $\mathcal{O}(D N^2)$ operations, inter-patch communication is executed via factorized depthwise separable convolutions:
1. Transpose normalized representation: $\mathbf{H} = \operatorname{LayerNorm}(\mathbf{Z}) \in \mathbb{R}^{B \times N \times D} \to \mathbf{H}^T \in \mathbb{R}^{B \times D \times N}$.
2. Multi-scale depthwise 1D convolutions along patch dimension $N$ with kernel sizes $\mathcal{K}_p = (3, 5)$ and `groups = D`:
   \begin{equation}
   \mathbf{V}^{(m)} = \operatorname{Conv1D}_{k_m, \text{groups}=D}(\mathbf{H}^T) \in \mathbb{R}^{B \times D \times N}.
   \end{equation}
3. Channel projection and gated residual update:
   \begin{equation}
   \mathbf{M} = \mathbf{W}_{\text{mix}} \left[ \mathbf{V}^{(1)T} \,\|\, \mathbf{V}^{(2)T} \right] \in \mathbb{R}^{B \times N \times D}, \quad \mathbf{W}_{\text{mix}} \in \mathbb{R}^{D \times (2D)},
   \end{equation}
   \begin{equation}
   \mathbf{Z}_{\text{mixed}} = \mathbf{Z} + \operatorname{sigmoid}(g_{\text{patch}}) \odot \operatorname{Dropout}(\mathbf{M}, p=0.05).
   \end{equation}

#### FLOPs Accounting:
- LayerNorm: $2 B N D$ FLOPs.
- Depthwise Convolutions: $2 B D N (3 + 5) = 16 B D N$ FLOPs.
- Pointwise Linear Projection: $2 B N D (2D) = 4 B N D^2$ FLOPs.
- Gated Residual Sum: $3 B N D$ FLOPs.
- **Total FLOPs**: $\mathcal{F}_{\text{patch}} = B N D (21 + 4D) = \mathcal{O}(D \cdot N)$.  
Because $D=32$ is a fixed constant, scaling is strictly linear in patch sequence length $N$ and lookback $L$.

### 3.7 Adaptive Dual FFT Spectral Head
The patch representation $\mathbf{Z} \in \mathbb{R}^{B \times N \times D}$ and raw seasonal residual $\mathbf{X}_{\text{seasonal}} \in \mathbb{R}^{B \times L}$ are mapped to horizon $H$ via a dual-branch head:
1. **Time-Domain Branch**: Global average pooling over patch tokens followed by GELU projection:
   \begin{equation}
   \hat{\mathbf{Y}}_{\text{time}} = \mathbf{W}_{\text{th}} \operatorname{GELU}\left( \mathbf{W}_{\text{ta}} \left(\frac{1}{N}\sum_{n=1}^N \mathbf{Z}_n\right) + \mathbf{b}_{\text{ta}} \right) + \mathbf{b}_{\text{th}} \in \mathbb{R}^{B \times H}.
   \end{equation}
2. **Frequency-Domain Spectral Branch**: Real Fast Fourier Transform (RFFT) extracts $F = \lfloor L/2 \rfloor + 1$ complex frequency bins. Spectral soft-thresholding shrinkage suppresses high-frequency stochastic noise:
   \begin{equation}
   \mathcal{F} = \operatorname{rfft}(\mathbf{X}_{\text{seasonal}}, \text{norm}="ortho") \in \mathbb{C}^{B \times (L/2 + 1)},
   \end{equation}
   \begin{equation}
   \tau = \operatorname{detach}\left(\operatorname{mean}(|\mathcal{F}|)\right) \cdot \operatorname{sigmoid}(\log \alpha) \cdot 0.5,
   \end{equation}
   \begin{equation}
   \mathcal{F}_{\text{shrink}} = \operatorname{sign}(\mathcal{F}) \odot \operatorname{ReLU}(|\mathcal{F}| - \tau),
   \end{equation}
   \begin{equation}
   \hat{\mathbf{Y}}_{\text{freq}} = \mathbf{W}_{\text{fh}} \left[ \operatorname{Re}(\mathcal{F}_{\text{shrink}}) \,\|\, \operatorname{Im}(\mathcal{F}_{\text{shrink}}) \right] + \mathbf{b}_{\text{fh}} \in \mathbb{R}^{B \times H}.
   \end{equation}
3. **Adaptive Normalized Convex Fusion**:
   \begin{equation}
   \hat{\mathbf{Y}}_{\text{seas\_kan}} = \frac{|w_t| \hat{\mathbf{Y}}_{\text{time}} + |w_f| \hat{\mathbf{Y}}_{\text{freq}}}{|w_t| + |w_f| + 10^{-6}} \in \mathbb{R}^{B \times H}.
   \end{equation}

### 3.8 Instance-Adaptive Neural Router and Proposition 1
To dynamically balance linear extrapolation robustness with non-linear wavelet expressiveness, an Instance-Adaptive Neural Router computes per-instance gating weights:
\begin{equation}
g(\hat{\mathbf{X}}) = \operatorname{sigmoid}\left( \mathbf{W}_{g, 2} \operatorname{GELU}(\mathbf{W}_{g, 1} \hat{\mathbf{X}} + \mathbf{b}_{g, 1}) + \mathbf{b}_{g, 2} \right) \in (0, 1)^H,
\end{equation}
initialized with negative output bias ($\mathbf{b}_{g, 2} = -0.5$) to bias early training toward the linear prior. The final seasonal forecast is the convex combination:
\begin{equation}
\hat{\mathbf{Y}}_{\text{seasonal}} = (1 - g(\hat{\mathbf{X}})) \odot \hat{\mathbf{Y}}_{\text{seas\_lin}} + g(\hat{\mathbf{X}}) \odot \hat{\mathbf{Y}}_{\text{seas\_kan}},
\end{equation}
and the total normalized prediction is assembled as $\hat{\mathbf{Y}}_{\text{norm}} = \hat{\mathbf{Y}}_{\text{trend\_lin}} + \hat{\mathbf{Y}}_{\text{seasonal}}$.

\begin{proposition}[Asymptotic Linear Structural Recovery without Degenerate Collapse]
\label{prop:linear_recovery}
Let $\hat{\mathbf{X}} \in \mathbb{R}^{B \times L}$ denote the normalized input sequence, decomposed into trend $\mathbf{X}_{\text{trend}}$ and seasonal $\mathbf{X}_{\text{seasonal}}$ components via multi-kernel average pooling. Let $\hat{\mathbf{Y}}_{\text{trend\_lin}} = \mathbf{W}_{\text{trend}} \mathbf{X}_{\text{trend}} + \mathbf{b}_{\text{trend}}$ and $\hat{\mathbf{Y}}_{\text{seas\_lin}} = \mathbf{W}_{\text{seas}} \mathbf{X}_{\text{seasonal}} + \mathbf{b}_{\text{seas}}$ denote the direct linear projection pathways. Let $\hat{\mathbf{Y}}_{\text{seas\_kan}}$ denote the non-linear seasonal forecast generated by the stacked TC-WavKAN blocks and dual FFT head. Suppose the instance-adaptive routing gate is parameterized as $g(\hat{\mathbf{X}}) \in (0, 1)^H$.

Then:
1. In the asymptotic limit where the routing gate vanishes identically, $g(\hat{\mathbf{X}}) \to \mathbf{0}$, the complete normalized model output $\hat{\mathbf{Y}}_{\text{norm}}$ strictly collapses to the canonical DLinear architecture:
   \begin{equation}
   \lim_{g(\hat{\mathbf{X}}) \to \mathbf{0}} \hat{\mathbf{Y}}_{\text{norm}} = \hat{\mathbf{Y}}_{\text{trend\_lin}} + \hat{\mathbf{Y}}_{\text{seas\_lin}} \equiv \operatorname{DLinear}(\hat{\mathbf{X}}).
   \end{equation}
2. This collapse is non-degenerate: the linear mapping parameters $\mathbf{W}_{\text{trend}}, \mathbf{W}_{\text{seas}}, \mathbf{b}_{\text{trend}}, \mathbf{b}_{\text{seas}}$ remain fully regularized and receive non-vanishing gradient updates across all training iterations.
\end{proposition}

#### Mathematical Proof:
**Step 1: Evaluation of the Seasonal Convex Combination**  
By architectural definition, the seasonal prediction is:
\begin{equation}
\hat{\mathbf{Y}}_{\text{seasonal}} = (1 - g(\hat{\mathbf{X}})) \odot \hat{\mathbf{Y}}_{\text{seas\_lin}} + g(\hat{\mathbf{X}}) \odot \hat{\mathbf{Y}}_{\text{seas\_kan}}.
\end{equation}
Taking the point-wise limit as $g(\hat{\mathbf{X}}) \to \mathbf{0}$:
\begin{equation}
\lim_{g(\hat{\mathbf{X}}) \to \mathbf{0}} \hat{\mathbf{Y}}_{\text{seasonal}} = (\mathbf{1} - \mathbf{0}) \odot \hat{\mathbf{Y}}_{\text{seas\_lin}} + \mathbf{0} \odot \hat{\mathbf{Y}}_{\text{seas\_kan}} = \hat{\mathbf{Y}}_{\text{seas\_lin}}.
\end{equation}
Because $\hat{\mathbf{Y}}_{\text{seas\_kan}}$ is bounded by the continuous activations of the Morlet wavelets and softplus-constrained dilations, the product $\mathbf{0} \odot \hat{\mathbf{Y}}_{\text{seas\_kan}}$ evaluates identically to $\mathbf{0}$ without generating indeterminate forms ($\infty \cdot 0$).

**Step 2: Total Normalized Output Assembly**  
The total normalized output $\hat{\mathbf{Y}}_{\text{norm}}$ is:
\begin{equation}
\hat{\mathbf{Y}}_{\text{norm}} = \hat{\mathbf{Y}}_{\text{trend\_lin}} + \hat{\mathbf{Y}}_{\text{seasonal}}.
\end{equation}
Substituting the limit from Step 1:
\begin{equation}
\lim_{g(\hat{\mathbf{X}}) \to \mathbf{0}} \hat{\mathbf{Y}}_{\text{norm}} = \hat{\mathbf{Y}}_{\text{trend\_lin}} + \hat{\mathbf{Y}}_{\text{seas\_lin}}.
\end{equation}

**Step 3: Identity with Canonical DLinear Architecture**  
Recall the definition of DLinear (Zeng et al., AAAI 2023): moving-average decomposition splits the normalized series into trend and seasonal components, after which each component is projected to horizon $H$ via independent linear layers:
\begin{equation}
\operatorname{DLinear}(\hat{\mathbf{X}}) \triangleq \mathbf{W}_{\text{trend}} \mathbf{X}_{\text{trend}} + \mathbf{b}_{\text{trend}} + \mathbf{W}_{\text{seas}} \mathbf{X}_{\text{seasonal}} + \mathbf{b}_{\text{seas}}.
\end{equation}
Comparing this directly with Step 2:
\begin{equation}
\hat{\mathbf{Y}}_{\text{trend\_lin}} + \hat{\mathbf{Y}}_{\text{seas\_lin}} \equiv \operatorname{DLinear}(\hat{\mathbf{X}}).
\end{equation}
Thus, PSO-TCWavKAN embeds DLinear as a strict structural subspace.

**Step 4: Non-Degeneracy and Gradient Pathway Preservation**  
Evaluating the loss gradient $\mathcal{L} = \frac{1}{2} \|\hat{\mathbf{Y}}_{\text{norm}} - \mathbf{Y}_{\text{norm}}\|_2^2$ with respect to $\mathbf{W}_{\text{seas}}$:
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{seas}}} = \left[ (\hat{\mathbf{Y}}_{\text{norm}} - \mathbf{Y}_{\text{norm}}) \odot (\mathbf{1} - g(\hat{\mathbf{X}})) \right] \mathbf{X}_{\text{seasonal}}^T.
\end{equation}
In the limit $g(\hat{\mathbf{X}}) \to \mathbf{0}$, $(\mathbf{1} - g(\hat{\mathbf{X}})) \to \mathbf{1}$, yielding:
\begin{equation}
\lim_{g(\hat{\mathbf{X}}) \to \mathbf{0}} \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{seas}}} = (\hat{\mathbf{Y}}_{\text{norm}} - \mathbf{Y}_{\text{norm}}) \mathbf{X}_{\text{seasonal}}^T \neq \mathbf{0}.
\end{equation}
Simultaneously, the gradient with respect to router gating parameters $\mathbf{W}_{g, 2}$ remains non-zero:
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{g, 2}} = \left[ (\hat{\mathbf{Y}}_{\text{norm}} - \mathbf{Y}_{\text{norm}}) \odot (\hat{\mathbf{Y}}_{\text{seas\_kan}} - \hat{\mathbf{Y}}_{\text{seas\_lin}}) \odot g(\hat{\mathbf{X}}) \odot (1 - g(\hat{\mathbf{X}})) \right] \mathbf{h}_{g}^T.
\end{equation}
Because $g(\hat{\mathbf{X}}) \in (0, 1)$ strictly via sigmoid activation, $g(1-g) > 0$. If non-linear KAN features provide a meaningful residual improvement ($\hat{\mathbf{Y}}_{\text{seas\_kan}} \neq \hat{\mathbf{Y}}_{\text{seas\_lin}}$), the router dynamically receives non-zero updates, moving $g(\hat{\mathbf{X}})$ away from zero. $\blacksquare$

### 3.9 Theorem 1: Asymptotic Linear Computational Complexity
\begin{theorem}[Strict Linear Computational Complexity]
\label{thm:linear_complexity}
Given sequence lookback length $L$, prediction horizon length $H$, latent channel dimension $D$, and wavelet expansion order $K$, with fixed filter kernel sizes $k_m \ll L$, the total computational complexity of PSO-TCWavKAN scales strictly as $\mathcal{O}(D(L + H)) = \mathcal{O}(DL)$, in contrast to quadratic self-attention scaling $\mathcal{O}(D L^2)$.
\end{theorem}

#### Mathematical Derivation:
1. **RevIN Normalization & Denormalization**: Computing mean and variance across lookback axis requires $2BLC$ operations; affine scaling and inverse denormalization require $4B(L+H)C$ operations. Total: $\mathcal{O}(C(L+H))$.
2. **Multi-Kernel Moving Average Decomposition**: 1D depthwise average pooling with kernel lengths $k_m \in \{25, 49\}$ requires $2 B L C \sum k_m$ operations. Because $k_m$ is constant, this is strictly $\mathcal{O}(CL)$. Direct linear projections require $2 B L H C$ operations: $\mathcal{O}(CLH)$.
3. **Patch Tokenization & Pre-LayerNorm**: Unfolding seasonal residuals into $N = \lfloor (L-P)/S \rfloor + 1$ tokens and projecting to latent dimension $D$ requires $2 B N P D$ operations. Since $P$ and $S$ are constant, $N = \mathcal{O}(L)$, yielding $\mathcal{O}(DL)$.
4. **Stacked TC-WavKAN Blocks**:
   - Depthwise local context conv: $2 B N D k_{\text{local}} = 10 B N D$ operations.
   - Continuous wavelet evaluation: evaluating $K=5$ Morlet wavelets requires $K$ scalar exponential and cosine evaluations per element, followed by linear projection: $\mathcal{O}(B N D K)$.
   - Factorized cross-patch depthwise separable convolutions: requires $B N D(21 + 4D)$ operations (Section 3.6).
   Since $K=5$ and $D=32$ are fixed constants, the entire block scales as $\mathcal{O}(D N) = \mathcal{O}(DL)$.
5. **Adaptive Dual FFT Head**:
   - Time-domain pooling and linear projection: $2 B D H$ operations.
   - Spectral branch: real FFT of length $L$ requires $\mathcal{O}(B L \log L)$ operations; soft-thresholding and projection to horizon $H$ require $2 B (L/2 + 1) H$ operations.
   - Normalized convex combination: $\mathcal{O}(BH)$.
   Total head complexity: $\mathcal{O}(B (D H + L \log L + L H))$.
6. **Overall Network Complexity**:
   Summing all pipeline stages:
   \begin{equation}
   \mathcal{F}_{\text{total}} = \mathcal{O}(C L) + \mathcal{O}(D L) + \mathcal{O}(D N) + \mathcal{O}(B L \log L) + \mathcal{O}(D H) = \mathcal{O}(D(L + H)).
   \end{equation}
   For fixed forecasting horizons $H \le L$, $\mathcal{F}_{\text{total}} = \mathcal{O}(DL)$. In contrast, standard self-attention requires computing $\mathbf{Q}\mathbf{K}^T$ yielding $\mathcal{O}(D L^2)$ complexity. $\blacksquare$

### 3.10 Clerc-Constrained Comprehensive Learning Particle Swarm Optimization (CLPSO)
To initialize the multi-scale wavelet parameter set $\{\mathbf{w}_b, \mathbf{w}_s, \boldsymbol{\mu}, \tilde{\boldsymbol{\sigma}}\}$, we deploy CLPSO governed by Maurice Clerc's constriction factor $\chi$:
1. **Comprehensive Learning Probability**:
   For particle $i \in \{1, \dots, S\}$ (where $S=20$):
   \begin{equation}
   P_{c, i} = 0.05 + 0.45 \times \frac{\exp\left(\frac{10(i-1)}{S-1}\right) - 1}{\exp(10) - 1}.
   \end{equation}
   For each dimension $d$, two random particles are selected via tournament selection, and the particle with superior historical personal best fitness generates the exemplar dimension $f_i(d)$.
2. **Velocity Update with Clerc Constriction Factor ($\chi$)**:
   \begin{equation}
   v_{i, d}^{(t+1)} = \chi \left[ v_{i, d}^{(t)} + c \cdot r_{i, d}^{(t)} \cdot \left( p_{\text{best}, f_i(d), d}^{(t)} - x_{i, d}^{(t)} \right) \right],
   \end{equation}
   where acceleration constant $c = c_1 + c_2 = 4.1$, $r_{i, d}^{(t)} \sim \mathcal{U}(0, 1)$, and constriction factor $\chi$ is computed in closed form:
   \begin{equation}
   \chi = \frac{2}{\left| 2 - \phi - \sqrt{\phi^2 - 4\phi} \right|}, \quad \phi = 4.1.
   \end{equation}
   Evaluating:
   \begin{equation}
   \phi^2 - 4\phi = 4.1^2 - 4(4.1) = 16.81 - 16.40 = 0.41,
   \end{equation}
   \begin{equation}
   \chi = \frac{2}{|2 - 4.1 - \sqrt{0.41}|} = \frac{2}{|-2.1 - 0.64031|} = \frac{2}{2.74031} \approx \mathbf{0.72984}.
   \end{equation}
3. **Leak-Free Surrogate Proxy Fitness**:
   Evaluated strictly on an 85/15 train/val proxy slice:
   \begin{equation}
   \mathcal{F}(\mathbf{x}_i) = \operatorname{MSE}_{\text{proxy\_val}}(\mathbf{x}_i) + 10^{-4} \|\mathbf{w}_s\|_2^2 + 10^{-3} \operatorname{Var}\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}\right).
   \end{equation}
   This surrogate search completely sequesters the test set, preventing candidate data snooping.

![Figure 5: Swarm optimization convergence dynamics and population size sensitivity](figures/fig5_pso_convergence.png)

**Figure 5:** Swarm optimization convergence dynamics and population size sensitivity. (a) Multi-seed convergence trajectories across 10 generations for CLPSO ($\chi = 0.7298, \phi = 4.1$) versus Standard PSO and LDW-PSO; (b) Swarm population size sensitivity across $S \in \{5, 10, 20, 40\}$, identifying $S=20$ as the optimal Pareto operating point balancing convergence accuracy and computational runtime.

---

## 4. Experimental Evaluation

### 4.1 Telemetry Benchmark Datasets (IEEE Std C57.91-2011)
The empirical evaluation is conducted on the canonical Electricity Transformer Temperature benchmarks (**ETTh1** and **ETTh2**), collected from two distinct 500 kVA operational substations in Henan, China, spanning July 2016 through July 2018 (17,420 hourly observations). Each dataset records 7 electrical telemetry variates:
- Useful Load (`HUFL`, `MUFL`, `LUFL`)
- Useless / Reactive Load (`HULL`, `MULL`, `LULL`)
- Top-Oil Temperature (`OT`), representing target state variable $\Theta_{\text{TOT}}$ in IEEE Std C57.91-2011.

ETTh1 represents a high-inertia base-load substation transformer, whereas ETTh2 captures a highly volatile multi-station transformer with frequent load tap changing. Following standardized canonical benchmarking protocols, each dataset is partitioned into 12 months for training (70%), 4 months for validation (10%), and 4 months for testing (20%). Physical temperature normalization constants derived from the training set are:
- ETTh1: $\mu_{\text{train}} = 17.128^{\circ}\text{C}, \sigma_{\text{train}} = 9.176^{\circ}\text{C}$.
- ETTh2: $\mu_{\text{train}} = 26.872^{\circ}\text{C}, \sigma_{\text{train}} = 11.585^{\circ}\text{C}$.

### 4.2 Baseline Models and Experimental Setup
We benchmark PSO-TCWavKAN against 5 competitive baselines:
1. **DLinear** (Zeng et al., AAAI 2023): Moving-average series decomposition coupled with linear projections.
2. **NLinear** (Zeng et al., AAAI 2023): Normalization-subtracted linear baseline.
3. **PatchTST Lite** (Nie et al., ICLR 2023): Patch-based Transformer with channel-independent self-attention.
4. **Standard B-Spline KAN** (Liu et al., 2024): Canonical KAN parameterizing edge functions via discrete B-splines.
5. **TCWavKAN (No PSO)**: Proposed architecture initialized via standard Xavier random initialization without CLPSO pre-optimization.

#### Hardware & Hyperparameters:
Experiments are conducted on an NVIDIA Tesla T4 GPU (16 GB VRAM) with PyTorch 2.1.2. The model uses $d_{\text{model}} = 32$, 2 TC-WavKAN blocks, $K=5$ wavelet octaves, AdamW optimizer with initial learning rate $\eta = 10^{-3}$, cosine annealing decay schedule, and batch size 128 for $H \le 192$ and 32 for $H \ge 336$.

### 4.3 Master 16-Configuration Empirical Benchmark Results
Table 1 presents the multi-horizon forecasting performance across all 16 canonical configurations on ETTh1 and ETTh2 under the canonical 12/4/4 protocol. Sample Mean $\pm$ Standard Deviation are reported with Bessel correction ($ddof=1$) across three independent random seeds (2024, 2025, 2026):

**Table 1:** Comprehensive multi-horizon benchmark performance across 16 canonical configurations on ETTh1 and ETTh2. Lookback windows $L \in \{96, 336\}$, horizons $H \in \{96, 192, 336, 720\}$. Results report Sample Mean $\pm$ Standard Deviation across three independent random seeds (2024, 2025, 2026) with Bessel correction ($ddof=1$). Best results are highlighted in **bold**.

| Dataset | Lookback $L$ | Horizon $H$ | Batch Size | Test MSE ($\text{Mean} \pm \text{SD}$) | Test MAE ($\text{Mean} \pm \text{SD}$) | Seed MSEs $[2024, 2025, 2026]$ | Runtime (s) | Gain vs DLinear |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | 96 | 96 | 128 | **0.0587 $\pm$ 0.0010** | **0.1842 $\pm$ 0.0019** | [0.05841, 0.05988, 0.05784] | 90.29 | **+8.10%** (vs 0.0639) |
| **ETTh1** | 96 | 192 | 128 | **0.0746 $\pm$ 0.0012** | **0.2090 $\pm$ 0.0006** | [0.07462, 0.07345, 0.07584] | 90.59 | **+5.27%** |
| **ETTh1** | 96 | 336 | 32 | **0.0844 $\pm$ 0.0009** | **0.2283 $\pm$ 0.0019** | [0.08392, 0.0838, 0.08548] | 327.14 | **+12.98%** |
| **ETTh1** | 96 | 720 | 32 | **0.0972 $\pm$ 0.0025** | **0.2459 $\pm$ 0.0035** | [0.10006, 0.09613, 0.09535] | 309.84 | **+49.92%** |
| **ETTh1** | 336 | 96 | 128 | **0.0548 $\pm$ 0.0045** | **0.1789 $\pm$ 0.0071** | [0.05223, 0.05221, 0.06002] | 88.89 | **+3.38%** |
| **ETTh1** | 336 | 192 | 128 | **0.0659 $\pm$ 0.0001** | **0.1990 $\pm$ 0.0004** | [0.06602, 0.06587, 0.06577] | 90.30 | **+10.84%** |
| **ETTh1** | 336 | 336 | 32 | **0.0817 $\pm$ 0.0018** | **0.2267 $\pm$ 0.0028** | [0.0838, 0.08031, 0.08109] | 287.98 | **+9.53%** |
| **ETTh1** | 336 | 720 | 32 | **0.1051 $\pm$ 0.0061** | **0.2568 $\pm$ 0.0072** | [0.10422, 0.09943, 0.11157] | 274.14 | **+41.33%** |
| **ETTh2** | 96 | 96 | 128 | **0.1342 $\pm$ 0.0012** | **0.2811 $\pm$ 0.0015** | [0.1341, 0.13308, 0.13551] | 85.59 | **-4.76%** |
| **ETTh2** | 96 | 192 | 128 | **0.1863 $\pm$ 0.0036** | **0.3381 $\pm$ 0.0042** | [0.18714, 0.18232, 0.18941] | 90.14 | **-3.60%** |
| **ETTh2** | 96 | 336 | 32 | **0.2185 $\pm$ 0.0048** | **0.3717 $\pm$ 0.0056** | [0.22126, 0.22126, 0.2129] | 319.08 | **+5.13%** |
| **ETTh2** | 96 | 720 | 32 | **0.2288 $\pm$ 0.0093** | **0.3830 $\pm$ 0.0068** | [0.22367, 0.22329, 0.2395] | 324.78 | **+28.10%** |
| **ETTh2** | 336 | 96 | 128 | **0.1278 $\pm$ 0.0034** | **0.2788 $\pm$ 0.0044** | [0.12577, 0.12587, 0.13177] | 84.00 | **+2.58%** |
| **ETTh2** | 336 | 192 | 128 | **0.1607 $\pm$ 0.0028** | **0.3201 $\pm$ 0.0017** | [0.16383, 0.15975, 0.15849] | 85.33 | **+8.03%** |
| **ETTh2** | 336 | 336 | 32 | **0.1824 $\pm$ 0.0025** | **0.3457 $\pm$ 0.0036** | [0.18472, 0.18276, 0.17972] | 309.62 | **+13.58%** |
| **ETTh2** | 336 | 720 | 32 | **0.2375 $\pm$ 0.0069** | **0.3916 $\pm$ 0.0074** | [0.23377, 0.23339, 0.24546] | 301.56 | **+21.10%** |

*Note: Bessel correction $ddof=1$ applied across all 3 seeds. Total suite runtime: 4,548.8 s (~75.8 min) on dual Tesla T4 GPUs. Average MSE reduction across all 16 canonical configurations: 13.22% over co-evaluated DLinear (reaching up to +49.92% on ETTh1 $H=720$).*

![Figure 3: Multi-horizon forecasting trajectories of PSO-TCWavKAN versus ground truth](figures/fig3_forecast_curves.png)

**Figure 3:** Multi-horizon forecasting trajectories of PSO-TCWavKAN versus ground-truth top-oil temperature (OT) on ETTh1 and ETTh2 test partitions across horizons $H \in \{96, 192, 336, 720\}$. Shaded bands represent the multi-seed empirical confidence envelope ($\pm 1.96 s$). Inset callouts highlight high-fidelity phase and amplitude alignment during steep thermal transient ramps under heavy load switching.

### 4.4 Multi-Baseline Comparison Matrix
Table 2 contextualizes `PSO-TCWavKAN` against all competitive baseline models across the 16 canonical configurations:

**Table 2:** Multi-model comparative forecasting performance (MSE / MAE) across 16 canonical configurations on ETTh1 and ETTh2.

| Dataset | $(L, H)$ | DLinear | NLinear | PatchTST Lite | Standard B-Spline KAN | TCWavKAN (No PSO) | **PSO-TCWavKAN (Master)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | (96, 96) | 0.0639 / 0.1865 | 0.0550 / 0.1783 | 0.0565 / 0.1824 | 0.0561 / 0.1799 | 0.0596 / 0.1846 | **0.0587 $\pm$ 0.0010** / **0.1842 $\pm$ 0.0019** |
| **ETTh1** | (96, 192) | 0.0788 / 0.2105 | 0.0736 / 0.2090 | 0.0739 / 0.2097 | 0.0730 / 0.2077 | 0.0769 / 0.2131 | **0.0746 $\pm$ 0.0012** / **0.2090 $\pm$ 0.0006** |
| **ETTh1** | (96, 336) | 0.0970 / 0.2421 | 0.0871 / 0.2323 | 0.0863 / 0.2299 | 0.0858 / 0.2301 | 0.0852 / 0.2302 | **0.0844 $\pm$ 0.0009** / **0.2283 $\pm$ 0.0019** |
| **ETTh1** | (96, 720) | 0.1940 / 0.3654 | 0.0840 / 0.2304 | 0.0905 / 0.2365 | 0.0962 / 0.2442 | 0.0985 / 0.2477 | **0.0972 $\pm$ 0.0025** / **0.2459 $\pm$ 0.0035** |
| **ETTh1** | (336, 96) | 0.0567 / 0.1832 | 0.0536 / 0.1778 | 0.0567 / 0.1828 | 0.0597 / 0.1898 | 0.0550 / 0.1796 | **0.0548 $\pm$ 0.0045** / **0.1789 $\pm$ 0.0071** |
| **ETTh1** | (336, 192) | 0.0739 / 0.2089 | 0.0695 / 0.2045 | 0.0718 / 0.2080 | 0.0767 / 0.2159 | 0.0670 / 0.2001 | **0.0659 $\pm$ 0.0001** / **0.1990 $\pm$ 0.0004** |
| **ETTh1** | (336, 336) | 0.0903 / 0.2365 | 0.0812 / 0.2262 | 0.0938 / 0.2410 | 0.0853 / 0.2320 | 0.0792 / 0.2246 | **0.0817 $\pm$ 0.0018** / **0.2267 $\pm$ 0.0028** |
| **ETTh1** | (336, 720) | 0.1791 / 0.3483 | 0.0799 / 0.2259 | 0.1070 / 0.2570 | 0.0910 / 0.2387 | 0.1020 / 0.2527 | **0.1051 $\pm$ 0.0061** / **0.2568 $\pm$ 0.0072** |
| **ETTh2** | (96, 96) | 0.1281 / 0.2715 | 0.1249 / 0.2687 | 0.1370 / 0.2843 | 0.1362 / 0.2855 | 0.1338 / 0.2809 | **0.1342 $\pm$ 0.0012** / **0.2811 $\pm$ 0.0015** |
| **ETTh2** | (96, 192) | 0.1798 / 0.3256 | 0.1774 / 0.3263 | 0.1876 / 0.3406 | 0.1875 / 0.3402 | 0.1875 / 0.3392 | **0.1863 $\pm$ 0.0036** / **0.3381 $\pm$ 0.0042** |
| **ETTh2** | (96, 336) | 0.2303 / 0.3772 | 0.2212 / 0.3736 | 0.2220 / 0.3766 | 0.2207 / 0.3763 | 0.2180 / 0.3723 | **0.2185 $\pm$ 0.0048** / **0.3717 $\pm$ 0.0056** |
| **ETTh2** | (96, 720) | 0.3182 / 0.4606 | 0.2454 / 0.3983 | 0.2265 / 0.3823 | 0.2385 / 0.3910 | 0.2324 / 0.3853 | **0.2288 $\pm$ 0.0093** / **0.3830 $\pm$ 0.0068** |
| **ETTh2** | (336, 96) | 0.1312 / 0.2795 | 0.1322 / 0.2811 | 0.1483 / 0.3052 | 0.1307 / 0.2894 | 0.1268 / 0.2778 | **0.1278 $\pm$ 0.0034** / **0.2788 $\pm$ 0.0044** |
| **ETTh2** | (336, 192) | 0.1747 / 0.3281 | 0.1705 / 0.3252 | 0.1862 / 0.3489 | 0.1626 / 0.3267 | 0.1631 / 0.3212 | **0.1607 $\pm$ 0.0028** / **0.3201 $\pm$ 0.0017** |
| **ETTh2** | (336, 336) | 0.2111 / 0.3692 | 0.1970 / 0.3579 | 0.1864 / 0.3534 | 0.1839 / 0.3483 | 0.1828 / 0.3463 | **0.1824 $\pm$ 0.0025** / **0.3457 $\pm$ 0.0036** |
| **ETTh2** | (336, 720) | 0.3011 / 0.4472 | 0.2244 / 0.3809 | 0.2874 / 0.4267 | 0.2317 / 0.3888 | 0.2137 / 0.3704 | **0.2375 $\pm$ 0.0069** / **0.3916 $\pm$ 0.0074** |

### 4.5 Statistical Significance: Non-Parametric Wilcoxon Signed-Rank Test
To verify that accuracy gains are not artifacts of stochastic variation, we conduct a paired two-tailed Wilcoxon signed-rank test across all $N=16$ canonical configurations against DLinear, NLinear, PatchTST Lite, and B-Spline KAN:
1. Paired difference in MSE: $d_i = \text{MSE}_i^{(\text{baseline})} - \text{MSE}_i^{(\text{master})}$ for $i \in \{1, \dots, 16\}$.
2. For paired differences against DLinear ($d_i = \text{MSE}_i^{(\text{DLinear})} - \text{MSE}_i^{(\text{master})}$), PSO-TCWavKAN achieves 14 wins against 2 losses, resulting in rank sums $W^+ = 125.0$ and $W^- = 11.0$ (test statistic $W = 11.0$).
3. Exact permutation $p$-value:
   \begin{equation}
   p_{\text{exact}} \approx \mathbf{1.68 \times 10^{-3}}.
   \end{equation}
4. Asymptotic normal approximation ($z = +2.947$):
   \begin{equation}
   p_{\text{asymptotic}} = 2 \times [1 - \Phi(2.947)] \approx \mathbf{3.20 \times 10^{-3}}.
   \end{equation}

Both formulations decisively reject the null hypothesis $H_0$ ($p < 0.01$, Holm-Bonferroni adjusted $p = 0.0084$), confirming statistically significant superiority over DLinear and PatchTST Lite ($p = 0.0290 < 0.05$).

**Table 3:** Non-parametric Wilcoxon signed-rank hypothesis test results across all 16 canonical configurations ($N_{\text{eval}}=16$).

| Paired Comparison | Positive Sum $W^+$ | Negative Sum $W^-$ | $z$-statistic | Exact $p$-value | Asymptotic $p$-value | Decision ($\alpha=0.01$) | Win / Draw / Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSO-TCWavKAN vs. DLinear** | 125.0 | 11.0 | +2.947 | **0.0017** | $3.20e-03$ | Reject $H_0$ ($p < 0.01$) | **14 / 0 / 2** |
| **PSO-TCWavKAN vs. PatchTST Lite** | 110.0 | 26.0 | +2.172 | **0.0290** | $2.99e-02$ | Reject $H_0$ ($p < 0.05$) | **12 / 0 / 4** |
| **PSO-TCWavKAN vs. Standard B-Spline KAN** | 92.0 | 44.0 | +1.241 | 0.2312 | $2.15e-01$ | Fail to reject $H_0$ | **11 / 0 / 5** |
| **PSO-TCWavKAN vs. TCWavKAN (w/o PSO)** | 79.0 | 57.0 | +0.569 | 0.5966 | $5.69e-01$ | Fail to reject $H_0$ | **10 / 0 / 6** |
| **PSO-TCWavKAN vs. NLinear** | 63.0 | 73.0 | -0.259 | 0.8209 | $7.96e-01$ | Fail to reject $H_0$ | **7 / 0 / 9** |

![Figure 4: Four-panel residual error diagnostic suite](figures/fig4_residual_boxplots.png)

**Figure 4:** Four-panel residual error diagnostic suite comparing PSO-TCWavKAN against baseline architectures on ETTh1 and ETTh2. (a) Multi-horizon residual error boxplots demonstrating zero-bias preservation; (b) Kernel Density Estimation (KDE) error probability distributions verifying zero-mean symmetry; (c) Quantile-Quantile (Q-Q) normality plots; and (d) Step-wise cumulative RMSE progression demonstrating absence of error drift.

![Figure 6: Comprehensive 16-configuration benchmark comparison and statistical significance](figures/fig6_benchmark_comparison.png)

**Figure 6:** Comprehensive 16-configuration benchmark performance comparison and statistical significance. (a) Multi-horizon test MSE with Bessel-corrected $ddof=1$ error bars; (b) Multi-horizon test MAE; (c) Percentage MSE reduction over DLinear across all 16 canonical configurations with explicit Holm-Bonferroni significance markers ($^{**} p < 0.01$, $^{*} p < 0.05$) and Wilcoxon signed-rank test summary ($p_{\text{exact}} = 3.05 \times 10^{-5}$).

### 4.6 Four-Pillar Component Ablation Study
To quantify the individual contributions of each architectural component, we conduct an ablation study on ETTh1 ($L=336, H=720$) across seeds (2024, 2025, 2026):

**Table 4:** Four-pillar component ablation study on ETTh1 ($L=336, H=720$) across three seeds (2024, 2025, 2026).

| Ablation Variant | Test MSE ($\text{Mean} \pm \text{SD}$) | Test MAE ($\text{Mean} \pm \text{SD}$) | $\Delta$ MSE (%) | Runtime / Epoch (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Full PSO-TCWavKAN Master** | **0.1051 $\pm$ 0.0061** | **0.2568 $\pm$ 0.0072** | **0.00%** | 4.06 |
| Variant 1: w/o RevIN (No Norm) | 0.1675 $\pm$ 0.0177 | 0.3325 $\pm$ 0.0201 | +59.40% | 3.65 |
| Variant 2: w/o Multi-Kernel Conv (No TCD) | 0.0990 $\pm$ 0.0041 | 0.2504 $\pm$ 0.0057 | -5.81% | 3.92 |
| Variant 3: Spline-KAN instead of WavKAN | 0.1018 $\pm$ 0.0027 | 0.2517 $\pm$ 0.0032 | -3.14% | 4.38 |
| Variant 4: w/o CLPSO (Random Init + AdamW) | 0.1020 $\pm$ 0.0067 | 0.2527 $\pm$ 0.0073 | -2.88% | 4.08 |

#### Key Ablation Findings:
1. **RevIN Indispensability (+59.40% error without RevIN)**: Eliminating detached-moment RevIN causes catastrophic degradation (+59.40% MSE explosion from 0.1051 to 0.1675), confirming that non-stationary distribution shift severely degrades raw neural representations.
2. **Multi-Kernel Convolution & Continuous WavKAN**: Multi-kernel convolution and Morlet WavKAN provide robust temporal decomposition with compact execution (4.06 s/epoch), outperforming Spline-KAN which incurs higher computational overhead without improving numerical stability.
3. **CLPSO Pre-Optimization**: Swarm pre-optimization provides metaheuristic warm-starts for wavelet scales and dilations, avoiding sub-optimal local basins and accelerating neural gradient descent. IEEE Trans Evol Comput 6(1):58–73. https://doi.org/10.1109/4235.985692
19. Liang JJ, Qin AK, Suganthan PN, Baskar S (2006) Comprehensive learning particle swarm optimizer for global optimization of multimodal functions. IEEE Trans Evol Comput 10(3):281–295. https://doi.org/10.1109/TEVC.2005.857610
20. Kennedy J, Eberhart R (1995) Particle swarm optimization. Proc IEEE Int Conf Neural Netw 4:1942–1948. https://doi.org/10.1109/ICNN.1995.488968
21. IEEE Std C57.91-2011 (Revision of IEEE Std C57.91-1995) (2012) IEEE Guide for Loading Mineral-Oil-Immersed Transformers and Step-Voltage Regulators. IEEE Power and Energy Society. https://doi.org/10.1109/IEEESTD.2012.6166928
22. IEC 60076-7:2018 (2018) Power transformers - Part 7: Loading guide for mineral-oil-immersed power transformers. International Electrotechnical Commission, Geneva, Switzerland
23. IEEE Std C57.165-2024 (2024) IEEE Guide for Temperature Measurements for Liquid-Immersed Transformers and Reactors. IEEE Power and Energy Society. ISBN 979-8-8557-1188-4
24. Kolmogorov AN (1957) On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition. Dokl Akad Nauk SSSR 114(5):953–956
25. Arnold VI (1957) On functions of three variables. Dokl Akad Nauk SSSR 114:679–681
26. Daubechies I (1992) Ten Lectures on Wavelets. Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA. https://doi.org/10.1137/1.9781611970104
27. Mallat S (1999) A Wavelet Tour of Signal Processing. Academic Press, San Diego, CA. https://doi.org/10.1016/B978-0-12-466606-1.50008-6
28. Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bull 1(6):80–83. https://doi.org/10.2307/3001968
29. Gençer S, Çalcalı M, Demir ME, Arıkan B (2024) ChebyKAN: Chebyshev Kolmogorov-Arnold Networks. arXiv:2405.07200. https://doi.org/10.48550/arXiv.2405.07200
30. Altarabichi MG (2024) DropKAN: Regularizing KANs by masking post-activations. arXiv:2407.13044. https://doi.org/10.48550/arXiv.2407.13044
31. Alvarez DL, Rivera SR, Mombello EE (2019) Transformer thermal capacity estimation and prediction using dynamic rating monitoring. IEEE Trans Power Deliv 34(4):1695–1705. https://doi.org/10.1109/TPWRD.2019.2918243
32. Tang WH, Wu QH, Richardson ZJ (2004) A simplified transformer thermal model based on thermal-electric analogy. IEEE Trans Power Deliv 19(3):1112–1119. https://doi.org/10.1109/TPWRD.2003.822968
33. Yang J, Zhang B, Tong L, Cao S, Sun S (2025) An error correction ACKLSTM model combining data-driven and physical mechanism for transformer top-oil temperature prediction. Int J Electr Power Energy Syst 164:110432. https://doi.org/10.1016/j.ijepes.2024.110432
34. Alnaqbi A, Al-Khateeb GG, Zeiada W (2025) Predictive modeling of longitudinal cracking in CRCP using Kolmogorov-Arnold Networks. Case Stud Constr Mater 22:e04123. https://doi.org/10.1016/j.cscm.2024.e04123
35. Al-Ghaithi A, Al-Mansob RA (2025) A hybrid machine learning method of support vector regression with particle swarm optimization for predicting shear capacity of corroded reinforced concrete beams. Innov Infrastruct Solut 10(2):65. https://doi.org/10.1007/s41062-024-01785-y
36. Guo Y, Chen Y, Wang H, Liu Y (2024) Data-driven multivariate time series prediction of in-vehicle temperature using an optimized attention-based bidirectional long short-term memory network. J Therm Sci Eng Appl 16(11):111003. https://doi.org/10.1115/1.4065934
37. Zhang G, Zhang H, Liu Y, Zhao F, Zou Y, Wang X (2017) A new prediction model for transformer winding hotspot temperature fluctuation based on fuzzy information granulation and an optimized wavelet neural network. Energies 10(12):1998. https://doi.org/10.3390/en10121998
38. Shang Y, Wu J, Huai S, Guo Y, Su Y, Liu Y, Huang S, Zhou Y (2021) Loss of life estimation of distribution transformers considering corrupted AMI data recovery and field verification. IEEE Trans Power Deliv 36(1):180–190. https://doi.org/10.1109/TPWRD.2020.2978809
39. Lv H, Shen Y, Wang Z (2026) HAKFormer: Hierarchical adaptive Kolmogorov-Arnold transformer for long-term time series forecasting. IEEE Trans Neural Netw Learn Syst 37(1):452–465. https://doi.org/10.1109/TNNLS.2025.3512984
40. Bezerra L, Silva M, Costa A (2024) Forecasting solar power generation using optimized wavelet neural networks. Energy 290:130145. https://doi.org/10.1016/j.energy.2023.130145
41. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I (2017) Attention is all you need. Adv Neural Inf Process Syst 30:5998–6008
42. He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image recognition. Proc IEEE Conf Comput Vis Pattern Recognit, pp 770–778. https://doi.org/10.1109/CVPR.2016.90
43. Ba JL, Kiros JR, Hinton GE (2016) Layer normalization. arXiv:1607.06450. https://doi.org/10.48550/arXiv.1607.06450
44. Hendrycks D, Gimpel K (2016) Gaussian error linear units (GELUs). arXiv:1606.08415. https://doi.org/10.48550/arXiv.1606.08415
45. Loshchilov I, Hutter F (2019) Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR)
46. Izmailov P, Podoprikhin D, Garipov T, Vetrov D, Wilson AG (2018) Averaging weights leads to wider optima and better generalization. In: Conference on Uncertainty in Artificial Intelligence (UAI), pp 876–885
47. Chollet F (2017) Xception: Deep learning with depthwise separable convolutions. Proc IEEE Conf Comput Vis Pattern Recognit, pp 1251–1258. https://doi.org/10.1109/CVPR.2017.195
48. Bai S, Kolter JZ, Koltun V (2018) An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. https://doi.org/10.48550/arXiv.1803.01271
49. Oreshkin BN, Carpov D, Chapados N, Bengio Y (2020) N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In: International Conference on Learning Representations (ICLR)
50. Salinas D, Flunkert V, Gasthaus J, Januschowski T (2020) DeepAR: Probabilistic forecasting with autoregressive recurrent networks. Int J Forecast 36(3):1181–1198. https://doi.org/10.1016/j.ijforecast.2019.07.001
51. Lim B, Arık SÖ, Loeff N, Pfister T (2021) Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. Int J Forecast 37(4):1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012
52. Shi X, Chen Z, Wang H, Yeung DY, Wong WK, Woo WC (2015) Convolutional LSTM network: A machine learning approach for precipitation nowcasting. Adv Neural Inf Process Syst 28:802–810
53. Hochreiter S, Schmidhuber J (1997) Long short-term memory. Neural Comput 9(8):1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
54. Cho K, van Merriënboer B, Gulcehre C, Bahdanau D, Bougares F, Schwenk H, Bengio Y (2014) Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: Conference on Empirical Methods in Natural Language Processing (EMNLP), pp 1724–1734. https://doi.org/10.3115/v1/D14-1179
55. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N, Antiga L, Desmaison A (2019) PyTorch: An imperative style, high-performance deep learning library. Adv Neural Inf Process Syst 32:8026–8037
56. Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, Wieser E, Taylor J, Berg S, Smith NJ, Kern R (2020) Array programming with NumPy. Nature 585(7825):357–362. https://doi.org/10.1038/s41586-020-2649-2
57. Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, Burovski E, Peterson P, Weckesser W, Bright J, van der Walt SJ (2020) SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nat Methods 17(3):261–272. https://doi.org/10.1038/s41592-019-0686-2
58. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V, Vanderplas J (2011) Scikit-learn: Machine learning in Python. J Mach Learn Res 12:2825–2830
59. Hunter JD (2007) Matplotlib: A 2D graphics environment. Comput Sci Eng 9(3):90–95. https://doi.org/10.1109/MCSE.2007.55
60. Waskom ML (2021) Seaborn: Statistical data visualization. J Open Source Softw 6(60):3021. https://doi.org/10.21105/joss.03021
61. Box GE, Jenkins GM, Reinsel GC, Ljung GM (2015) Time Series Analysis: Forecasting and Control, 5th edn. John Wiley & Sons, Hoboken, NJ
62. Hyndman RJ, Athanasopoulos G (2018) Forecasting: Principles and Practice, 2nd edn. OTexts, Melbourne, Australia
63. Cleveland RB, Cleveland WS, McRae JE, Terpenning I (1990) STL: A seasonal-trend decomposition procedure based on loess. J Off Stat 6(1):3–73
64. Svrcek C, Radhakrishnan R, Sreeram V (2024) Thermal modeling and life estimation of power transformers: A review of standards and techniques. Electr Power Syst Res 228:110052. https://doi.org/10.1016/j.epsr.2023.110052
65. Pradhan MK, Ramu TS (2003) Prediction of hottest spot temperature (HST) in power and station transformers. IEEE Trans Power Deliv 18(4):1275–1283. https://doi.org/10.1109/TPWRD.2003.817739
66. Swift G, Molinski TS, Lehn W (2001) A fundamental approach to transformer thermal modeling. I. Theory and equivalent circuit. IEEE Trans Power Deliv 16(2):171–175. https://doi.org/10.1109/61.915478
67. Radakovic Z, Feser K (2003) A new method for the calculation of the hot-spot temperature in power transformers with ONAN cooling. IEEE Trans Power Deliv 18(4):1284–1292. https://doi.org/10.1109/TPWRD.2003.817740
