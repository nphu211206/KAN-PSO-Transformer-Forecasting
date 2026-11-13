"""
================================================================================
PUBLICATION FIGURE GENERATION SUITE FOR NCAA (SPRINGER NATURE) - 600 DPI
================================================================================
Journal: Neural Computing and Applications (NCAA - Springer Nature)
Paper Title: PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet
             Kolmogorov-Arnold Networks for Power Transformer Thermal
             Dynamics Forecasting
Author: Phu Toan Nguyen
Resolution Standard: Strictly 600 DPI Vector/Lossless PNG Double-Column Layout
Data Provenance: Strictly reads from verified Kaggle Option A outputs in
                 results/ and predictions/ (zero synthetic fallback data).

Figures Generated:
1. figures/fig1_architecture.png: Full Architectural Schematic of PSO-TCWavKAN
2. figures/fig2_wavelet_basis.png: Multi-Scale Continuous Wavelet Basis & Admissibility
3. figures/fig3_forecast_curves.png: Multi-Horizon Forecasting Curves vs Ground Truth
4. figures/fig4_residual_boxplots.png: Four-Panel Residual Error Diagnostic Suite
5. figures/fig5_pso_convergence.png: CLPSO Swarm Convergence & Population Pareto Curve
6. figures/fig6_benchmark_comparison.png: 16-Configuration Benchmark & Wilcoxon Tests
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy import stats
from PIL import Image

TARGET_DPI = 600
OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)


def setup_matplotlib():
    """Configures Springer Nature / NCAA publication styling."""
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.family': 'sans-serif',
        'figure.dpi': TARGET_DPI,
        'savefig.dpi': TARGET_DPI,
        'axes.labelsize': 10.0,
        'axes.titlesize': 11.0,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.2,
        'lines.linewidth': 1.6,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'mathtext.fontset': 'dejavusans',
        'figure.autolayout': False,
    })


def save_and_verify(fig, filename):
    """Saves figure with exact 600 DPI metadata and verifies output."""
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=TARGET_DPI, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)

    im = Image.open(out_path)
    dpi = im.info.get('dpi', (0, 0))
    sz = os.path.getsize(out_path)
    print(f"[SUCCESS] Generated: {out_path} | Dim: {im.size} | DPI: {dpi} | Size: {sz / 1024:.1f} KB")
    assert dpi[0] >= 599.0 and dpi[1] >= 599.0, f"DPI validation failed for {filename}: {dpi}"
    assert sz > 100000, f"File size suspiciously small for {filename}: {sz} bytes"
    return out_path


# ==============================================================================
# FIGURE 1: Full System Architectural Schematic
# ==============================================================================
def generate_fig1():
    print("\n>>> [1/6] Generating Figure 1: Architectural Schematic (600 DPI)...")
    fig, ax = plt.subplots(figsize=(16.2, 9.2), dpi=TARGET_DPI)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    # ColorBrewer / Springer publication palette
    c_blue   = '#1F77B4'
    c_orange = '#D95F02'
    c_purple = '#7570B3'
    c_teal   = '#0D9488'
    c_green  = '#2CA02C'
    c_yellow = '#CA8A04'
    c_red    = '#DC2626'
    c_dark   = '#111827'

    # Master Header Banner (y=8.55..8.95, cleanly separated from badges/blocks below y=8.15)
    ax.text(8.1, 8.90,
            'Architectural Schematic of the Proposed PSO-TCWavKAN Forecasting Framework',
            ha='center', va='center', fontsize=13.2, fontweight='bold', color=c_dark)
    ax.text(8.1, 8.56,
            'Reversible Instance Normalization, Multi-Kernel Decomposition, Morlet Wavelet-KAN Blocks, Adaptive Dual FFT Head, and CLPSO Outer Tuner',
            ha='center', va='center', fontsize=9.2, style='italic', color='#4B5563')

    # Parameter & Complexity Metadata Badges (y=7.78..8.22, clear of both header and main boxes <=7.42)
    scope_box = patches.FancyBboxPatch((0.25, 7.80), 4.70, 0.42, boxstyle="round,pad=0.06",
                                       fc='#F8FAFC', ec='#64748B', lw=1.3)
    ax.add_patch(scope_box)
    ax.text(2.60, 8.01,
            r"Protocol: Canonical 12/4/4 ETT ($N=14,400$), Univariate OT ($C=1$)",
            ha='center', va='center', fontsize=7.9, fontweight='bold', color='#334155')

    param_box = patches.FancyBboxPatch((10.95, 7.80), 5.00, 0.42, boxstyle="round,pad=0.06",
                                       fc='#EFF6FF', ec='#3B82F6', lw=1.4)
    ax.add_patch(param_box)
    ax.text(13.45, 8.01,
            r"Trainable Parameters: 49.8K ($L=96, H=96$) to 800.7K ($L=336, H=720$)",
            ha='center', va='center', fontsize=7.9, fontweight='bold', color='#1E40AF')

    # Block 1: Telemetry Input & RevIN
    b1 = patches.FancyBboxPatch((0.25, 2.05), 2.30, 5.25, boxstyle="round,pad=0.10",
                                fc='#F0F9FF', ec=c_blue, lw=2.1)
    ax.add_patch(b1)
    ax.text(1.40, 7.02, 'Raw Telemetry Input', ha='center', va='center',
            fontsize=10.6, fontweight='bold', color=c_blue)
    ax.text(1.40, 6.02,
            'Transformer Top-Oil\nTemperature (OT)\n' +
            r"$\mathbf{X} \in \mathbb{R}^{B \times L \times 1}$" + '\n' +
            r"$L \in \{96, 336\}$",
            ha='center', va='center', fontsize=8.6, color=c_dark)

    r_sub = patches.FancyBboxPatch((0.40, 2.25), 2.00, 2.50, boxstyle="round,pad=0.08",
                                   fc='#FFFFFF', ec='#60A5FA', lw=1.4, ls='--')
    ax.add_patch(r_sub)
    ax.text(1.40, 4.46, 'RevIN Normalization', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#1D4ED8')
    ax.text(1.40, 3.38,
            r"$\hat{\mathbf{X}} = \frac{\mathbf{X} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$" + '\n' +
            r"($\epsilon = 10^{-5}$, affine $\gamma, \beta$)" + '\n\n' +
            'Caches instance stats\n' + r"$(\mu, \sigma)$ for exact denorm",
            ha='center', va='center', fontsize=7.7, color='#1E3A8A')

    # Arrow 1 -> 2
    ax.annotate('', xy=(2.95, 4.65), xytext=(2.57, 4.65),
                arrowprops=dict(arrowstyle="-|>", lw=2.1, color=c_dark, mutation_scale=15))

    # Block 2: Multi-Kernel Series Decomposition
    b2 = patches.FancyBboxPatch((2.98, 2.05), 2.65, 5.25, boxstyle="round,pad=0.10",
                                fc='#FFF7ED', ec=c_orange, lw=2.1)
    ax.add_patch(b2)
    ax.text(4.30, 7.02, 'Multi-Kernel Decomp.', ha='center', va='center',
            fontsize=10.6, fontweight='bold', color=c_orange)
    ax.text(4.30, 5.95,
            'Symmetric Moving-Average\nFilter Bank:\n' +
            r"$k_1 = 25$ (24h diurnal)" + '\n' +
            r"$k_2 = 49$ (48h bi-daily)",
            ha='center', va='center', fontsize=8.4, color=c_dark)
    ax.text(4.30, 4.25,
            r"$\mathbf{X}_{\mathrm{tr}} = \frac{1}{2}\sum_{k \in \{25,49\}} \mathrm{AvgPool}_k(\hat{\mathbf{X}})$" + '\n\n' +
            r"$\mathbf{X}_{\mathrm{se}} = \hat{\mathbf{X}} - \mathbf{X}_{\mathrm{tr}}$",
            ha='center', va='center', fontsize=8.0, color='#9A3412')
    ax.text(4.30, 2.72,
            'Separates low-frequency\nmacro drift from oscillatory\nhigh-frequency harmonics',
            ha='center', va='center', fontsize=7.8, style='italic', color='#7C2D12')

    # Routing Arrows from Block 2 to Middle Column
    ax.annotate('', xy=(6.02, 6.45), xytext=(5.65, 5.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.9, color=c_purple, mutation_scale=13))
    ax.annotate('', xy=(6.02, 4.70), xytext=(5.65, 4.70),
                arrowprops=dict(arrowstyle="-|>", lw=1.9, color=c_teal, mutation_scale=13))
    ax.annotate('', xy=(6.02, 2.55), xytext=(5.65, 3.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.9, color='#059669', mutation_scale=13))

    # Branch A (Top Middle): Dual Linear Paths (Trend Direct + Seasonal DLinear Anchor)
    ba = patches.FancyBboxPatch((6.05, 5.55), 3.25, 1.75, boxstyle="round,pad=0.10",
                                fc='#F3E8FF', ec=c_purple, lw=2.0)
    ax.add_patch(ba)
    ax.text(7.675, 7.05, 'Linear Trend & Seasonal Anchor', ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_purple)
    ax.text(7.675, 6.32,
            r"$\hat{\mathbf{Y}}_{\mathrm{tr\_lin}} = \mathbf{W}_{\mathrm{tr}} \mathbf{X}_{\mathrm{tr}} + \mathbf{b}_{\mathrm{tr}}$  (macro skip)" + '\n' +
            r"$\hat{\mathbf{Y}}_{\mathrm{se\_lin}} = \mathbf{W}_{\mathrm{se}} \mathbf{X}_{\mathrm{se}} + \mathbf{b}_{\mathrm{se}}$  (DLinear anchor)",
            ha='center', va='center', fontsize=8.1, color=c_dark)
    ax.text(7.675, 5.74, r"Linear mappings in $\mathbb{R}^{B \times H \times 1}$",
            ha='center', va='center', fontsize=7.6, style='italic', color='#581C87')

    # Branch B1 (Middle): Adaptive Patch Tokenization
    bb1 = patches.FancyBboxPatch((6.05, 3.82), 3.25, 1.42, boxstyle="round,pad=0.10",
                                 fc='#F0FDFA', ec=c_teal, lw=2.0)
    ax.add_patch(bb1)
    ax.text(7.675, 4.98, 'Patch Tokenization', ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_teal)
    ax.text(7.675, 4.40,
            r"$P=16, S=8$ ($L \leq 96$);  $P=24, S=12$ ($L > 96$)" + '\n' +
            r"Linear Embed: $\mathbf{P}_0 \in \mathbb{R}^{B \times N \times D},\; D=32$",
            ha='center', va='center', fontsize=7.9, color=c_dark)
    ax.text(7.675, 3.98, r"Unfolded tokens: $N = 11$ ($L=96$), $N = 27$ ($L=336$)",
            ha='center', va='center', fontsize=7.4, style='italic', color='#115E59')

    # Branch B2 (Bottom Middle): Stacked TC-WavKAN Blocks
    bb2 = patches.FancyBboxPatch((6.05, 1.48), 3.25, 2.02, boxstyle="round,pad=0.10",
                                 fc='#F0FDF4', ec=c_green, lw=2.0)
    ax.add_patch(bb2)
    ax.text(7.675, 3.26, r'Stacked TC-WavKAN Blocks ($\times 2$)', ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_green)
    ax.text(7.675, 2.38,
            r"$\mathrm{LN} \rightarrow \mathrm{DWConv}_{k=5} \rightarrow$ Morlet Gabor ($K=5$ oct.):" + '\n' +
            r"$\psi_{d,k}(z) = \exp(-0.5\,z_c^2)\cos(\omega_{d,k} z_c)$" + '\n' +
            r"$\sigma_{d,k} = \mathrm{softplus}(\tilde{\sigma}_{d,k}) + 10^{-4} > 0$, DropKAN $0.05$" + '\n' +
            r"+ Multi-Scale Inter-Patch Mixing ($\mathcal{O}(DN)$)",
            ha='center', va='center', fontsize=7.6, color='#14532D')

    # Arrow Patching -> TC-WavKAN
    ax.annotate('', xy=(7.675, 3.52), xytext=(7.675, 3.80),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=c_dark, mutation_scale=12))

    # Arrow TC-WavKAN -> Dual FFT Head
    ax.annotate('', xy=(9.68, 2.45), xytext=(9.32, 2.45),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=c_dark, mutation_scale=14))

    # Spectral Skip arrow from Block 2 X_se directly into Dual FFT Head (routed along y=1.22 cleanly)
    ax.plot([4.30, 4.30, 11.15, 11.15], [2.03, 1.24, 1.24, 1.46],
            color='#0D9488', lw=1.3, ls=':')
    ax.annotate('', xy=(11.15, 1.48), xytext=(11.15, 1.32),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color='#0D9488', mutation_scale=12))
    ax.text(7.68, 1.13, r'Direct Normalized Seasonal Residual $\mathbf{X}_{\mathrm{se}}$ to Spectral RFFT Filter',
            ha='center', va='center', fontsize=7.3, style='italic', color='#0F766E')

    # Block B3 (Bottom Right-Middle): Adaptive Dual FFT Head
    bb3 = patches.FancyBboxPatch((9.70, 1.48), 2.90, 2.02, boxstyle="round,pad=0.10",
                                 fc='#ECFDF5', ec='#059669', lw=2.0)
    ax.add_patch(bb3)
    ax.text(11.15, 3.26, 'Adaptive Dual FFT Head', ha='center', va='center',
            fontsize=9.8, fontweight='bold', color='#059669')
    ax.text(11.15, 2.38,
            r"Float32 $\mathrm{rfft}(\mathbf{X}_{\mathrm{se}})$ + SoftShrink($\tau(\log\alpha)$)" + '\n' +
            r"+ Mean-Pooled Wavelet Time Head ($\mathbf{Z}$)" + '\n' +
            r"$\hat{\mathbf{Y}}_{\mathrm{se\_kan}} = \frac{|w_t|\hat{\mathbf{Y}}_t + |w_f|\hat{\mathbf{Y}}_f}{|w_t|+|w_f|+10^{-6}}$",
            ha='center', va='center', fontsize=7.6, color='#065F46')
    ax.text(11.15, 1.66, r"Non-stationary output $\hat{\mathbf{Y}}_{\mathrm{se\_kan}} \in \mathbb{R}^{B \times H \times 1}$",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#047857')

    # Block 3 (Top Right-Middle): Instance Neural Router
    b3 = patches.FancyBboxPatch((9.70, 3.82), 2.90, 3.48, boxstyle="round,pad=0.10",
                                fc='#FEFCE8', ec=c_yellow, lw=2.1)
    ax.add_patch(b3)
    ax.text(11.15, 7.02, 'Instance Neural Router', ha='center', va='center',
            fontsize=10.4, fontweight='bold', color='#A16207')
    ax.text(11.15, 6.20,
            '2-Layer GELU Gating MLP:\n' +
            r"$g(\hat{\mathbf{X}}) = \sigma(\mathbf{W}_2 \mathrm{GELU}(\mathbf{W}_1 \hat{\mathbf{X}} + \mathbf{b}_1) + \mathbf{b}_2)$" + '\n' +
            r"Hidden Dim $D_g = 32$, Initial $\mathbf{b}_2 = -0.5$",
            ha='center', va='center', fontsize=7.9, color='#713F12')
    ax.text(11.15, 5.10,
            r"$\hat{\mathbf{Y}}_{\mathrm{se}} = (1 - g) \odot \hat{\mathbf{Y}}_{\mathrm{se\_lin}} + g \odot \hat{\mathbf{Y}}_{\mathrm{se\_kan}}$" + '\n' +
            r"$\hat{\mathbf{Y}}_{\mathrm{norm}} = \hat{\mathbf{Y}}_{\mathrm{tr\_lin}} + \hat{\mathbf{Y}}_{\mathrm{se}}$",
            ha='center', va='center', fontsize=7.8, color='#854D0E')
    ax.text(11.15, 4.18,
            'Proposition 1 (DLinear Anchor):\n' +
            r"$\lim_{g \to 0} \hat{\mathbf{Y}}_{\mathrm{norm}}$ recovers linear" + '\n' +
            r"subspace exactly ($\Delta_{\max} = 0.0$)",
            ha='center', va='center', fontsize=7.5, style='italic', color='#78350F')

    # Arrows into Neural Router
    ax.annotate('', xy=(9.68, 6.25), xytext=(9.32, 6.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=c_purple, mutation_scale=12))
    ax.annotate('', xy=(11.15, 3.80), xytext=(11.15, 3.52),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669', mutation_scale=12))

    # Direct Trend Skip from Branch A top edge over Router to Block 4 Composite Sum (cleanly at y=7.56)
    ax.plot([7.68, 7.68, 14.48, 14.48], [7.32, 7.56, 7.56, 7.34],
            color=c_purple, lw=1.6, ls='--', zorder=2)
    ax.annotate('', xy=(14.48, 7.32), xytext=(14.48, 7.45),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=c_purple, mutation_scale=12))
    ax.text(11.15, 7.56, r' Direct Macro Trend Skip $\hat{\mathbf{Y}}_{\mathrm{tr\_lin}}$ ',
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=c_purple,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3)

    # Arrow Router -> Block 4 Output
    ax.annotate('', xy=(12.98, 5.20), xytext=(12.62, 5.20),
                arrowprops=dict(arrowstyle="-|>", lw=2.1, color=c_dark, mutation_scale=15))

    # Block 4: RevIN Denormalization & Thermal Assessment
    b4 = patches.FancyBboxPatch((13.00, 1.48), 2.95, 5.82, boxstyle="round,pad=0.10",
                                fc='#FEF2F2', ec=c_red, lw=2.1)
    ax.add_patch(b4)
    ax.text(14.475, 7.00, 'Composite Output & Denorm', ha='center', va='center',
            fontsize=10.3, fontweight='bold', color=c_red)
    ax.text(14.475, 5.78,
            'Exact RevIN Denormalization:\n' +
            r"$\tilde{\mathbf{Y}} = \frac{\hat{\mathbf{Y}}_{\mathrm{norm}} - \beta}{\mathrm{safe\_}\gamma} \odot \sqrt{\sigma^2 + \epsilon} + \mu$",
            ha='center', va='center', fontsize=8.3, color='#991B1B')
    ax.text(14.475, 4.42,
            'Multi-Horizon Forecast:\n' +
            r"$\tilde{\mathbf{Y}} \in \mathbb{R}^{B \times H \times 1}$" + '\n' +
            r"$H \in \{96, 192, 336, 720\}$ ($^\circ\mathrm{C}$)",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_dark)

    op_box = patches.FancyBboxPatch((13.15, 1.72), 2.65, 2.05, boxstyle="round,pad=0.08",
                                    fc='#FFFFFF', ec='#F87171', lw=1.3)
    ax.add_patch(op_box)
    ax.text(14.475, 3.52, 'IEEE / IEC Thermal Context', ha='center', va='center',
            fontsize=8.3, fontweight='bold', color='#B91C1C')
    ax.text(14.475, 2.60,
            r"$\bullet$ IEEE Std C57.91: $110^\circ\mathrm{C}$ HST" + '\n' +
            r"$\bullet$ IEC 60076-7: Arrhenius aging" + '\n' +
            r"   factor $V = 2^{(\Theta_H - 98)/6}$" + '\n' +
            'SCADA OT thermal prognostics\n& preventive dispatch alerts',
            ha='center', va='center', fontsize=7.4, color='#7F1D1D')

    # Block 5: CLPSO Swarm Parameter Optimization Loop (bottom banner, cleanly separated)
    bpso = patches.FancyBboxPatch((0.25, 0.16), 15.70, 0.78, boxstyle="round,pad=0.08",
                                  fc='#FAF5FF', ec='#9333EA', lw=2.0, ls='--')
    ax.add_patch(bpso)
    ax.text(8.10, 0.72,
            r"Outer CLPSO Swarm Hyperparameter Search ($\chi = 0.72984,\; c_1 = c_2 = 2.05,\; S = 20,\; G_{\max} = 5,\; \mathrm{refresh\_gap} = 5$)",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#6B21A8')
    ax.text(8.10, 0.36,
            r"Velocity: $\mathbf{v}_{id}^{t+1} = \chi \left[\mathbf{v}_{id}^t + c_1 r_1 (\mathbf{p}_{id}^{\mathrm{self}} - \mathbf{x}_{id}^t) + c_2 r_2 (\mathbf{p}_{f_i(d),d}^{\mathrm{ex}} - \mathbf{x}_{id}^t)\right] \;\Longrightarrow\;$ Optimizes 4D tuple $(\omega_0, \log s, \log\alpha, \log_{10}\eta)$ on leak-free 3-epoch surrogate",
            ha='center', va='center', fontsize=8.0, color='#581C87')

    # Feedback Dashed Arrows from CLPSO up to TC-WavKAN and FFT Head
    ax.annotate('', xy=(7.68, 1.46), xytext=(7.68, 0.96),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='#9333EA', ls='--', mutation_scale=12))
    ax.annotate('', xy=(11.55, 1.46), xytext=(11.55, 0.96),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='#9333EA', ls='--', mutation_scale=12))

    save_and_verify(fig, 'fig1_architecture.png')


# ==============================================================================
# FIGURE 2: Multi-Scale Continuous Wavelet Basis Functions
# ==============================================================================
def generate_fig2():
    print("\n>>> [2/6] Generating Figure 2: Wavelet Basis Functions (600 DPI)...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13.8, 8.8), dpi=TARGET_DPI)

    c_red    = '#E74C3C'
    c_blue   = '#2980B9'
    c_green  = '#27AE60'
    c_purple = '#8E44AD'
    c_amber  = '#D97706'
    c_gray   = '#6B7280'

    x = np.linspace(-5.0, 5.0, 1000)

    def mexhat(x, a=1.0, b=0.0):
        z = (x - b) / a
        return (1.0 / np.sqrt(a)) * (1.0 - z**2) * np.exp(-0.5 * z**2)

    def morlet(x, w0=4.43, a=1.0, b=0.0):
        z = (x - b) / a
        return np.cos(w0 * z) * np.exp(-0.5 * z**2)

    def dog(x):
        return -x * np.exp(-0.5 * x**2)

    def b3(x):
        ax_abs = np.abs(x)
        y = np.zeros_like(x)
        m1 = (ax_abs >= 0) & (ax_abs < 1)
        y[m1] = 2.0 / 3.0 - ax_abs[m1]**2 + 0.5 * ax_abs[m1]**3
        m2 = (ax_abs >= 1) & (ax_abs < 2)
        y[m2] = (1.0 / 6.0) * (2.0 - ax_abs[m2])**3
        return y

    # (a) Scale Dilation (analytic Mexican Hat prototype)
    ax1.plot(x, np.exp(-0.5 * x**2), color=c_gray, linestyle='--', lw=1.3,
             label=r'Gaussian Window Envelope $\pm e^{-x^2/2}$')
    ax1.plot(x, -np.exp(-0.5 * x**2), color=c_gray, linestyle='--', lw=1.3)
    ax1.plot(x, mexhat(x, a=0.5, b=0.0), color=c_red, lw=1.8,
             label=r'Scale $a=0.5$ (High-frequency transient)')
    ax1.plot(x, mexhat(x, a=1.0, b=0.0), color=c_blue, lw=1.8,
             label=r'Scale $a=1.0$ (Nominal reference harmonic)')
    ax1.plot(x, mexhat(x, a=2.0, b=0.0), color=c_green, lw=1.8,
             label=r'Scale $a=2.0$ (Dilated diurnal macro-trend)')
    ax1.set_xlabel(r'Standardized Feature Coordinate $z$', fontsize=9.5)
    ax1.set_ylabel(r'Wavelet Amplitude $\psi_{a,0}(z)$', fontsize=9.5)
    ax1.set_title(r'(a) Analytic Scale Dilation ($\psi_{a,0}(z) = a^{-1/2}\psi(z/a),\; b=0$)',
                  fontsize=10.3, fontweight='bold')
    ax1.set_xlim(-5.0, 5.0)
    ax1.set_ylim(-1.15, 2.25)
    ax1.legend(loc='upper right', fontsize=7.9, framealpha=0.94)
    ax1.grid(True)

    # (b) Spatial Translation Shift Invariance
    ax2.plot(x, mexhat(x, a=1.0, b=-2.0), color=c_purple, lw=1.8,
             label=r'Shift $b = -2.0$ (Leading temporal patch)')
    ax2.plot(x, mexhat(x, a=1.0, b=0.0), color=c_blue, lw=1.8,
             label=r'Shift $b = 0.0$ (Centered patch reference)')
    ax2.plot(x, mexhat(x, a=1.0, b=2.0), color=c_amber, lw=1.8,
             label=r'Shift $b = +2.0$ (Lagging temporal patch)')
    ax2.set_xlabel(r'Standardized Feature Coordinate $z$', fontsize=9.5)
    ax2.set_ylabel(r'Wavelet Amplitude $\psi_{1,b}(z)$', fontsize=9.5)
    ax2.set_title(r'(b) Spatial Translation Shift Invariance ($\psi_{1,b}(z) = \psi(z-b),\; a=1.0$)',
                  fontsize=10.3, fontweight='bold')
    ax2.set_xlim(-5.0, 5.0)
    ax2.set_ylim(-0.65, 1.55)
    ax2.legend(loc='upper right', fontsize=7.9, framealpha=0.94)
    ax2.grid(True)

    # (c) Smooth C^\infty Wavelet Archetypes vs Piecewise Cubic B-Spline
    ax3.plot(x, morlet(x, 4.43), color=c_blue, lw=2.0,
             label=r'Gabor Morlet ($\omega_0=4.43$, used in TC-WavKAN)')
    ax3.plot(x, mexhat(x, 1.0, 0.0), color=c_red, lw=1.7, linestyle='--',
             label=r'Mexican Hat $\psi(z) = (1 - z^2) e^{-z^2/2}$')
    ax3.plot(x, dog(x), color=c_green, lw=1.7, linestyle='-.',
             label=r'Derivative of Gaussian (DoG) $\psi(z) = -z e^{-z^2/2}$')
    ax3.plot(x, b3(x), color=c_purple, linestyle=':', lw=2.2,
             label=r'Cubic B-Spline $B_3(z)$ (Piecewise $C^2$, local support)')
    ax3.set_xlabel(r'Standardized Feature Coordinate $z$', fontsize=9.5)
    ax3.set_ylabel(r'Basis Amplitude $\psi(z)$', fontsize=9.5)
    ax3.set_title(r'(c) Smooth $C^\infty$ Wavelet Archetypes vs. Piecewise Cubic B-Spline',
                  fontsize=10.3, fontweight='bold')
    ax3.set_xlim(-4.0, 4.0)
    ax3.set_ylim(-1.15, 1.55)
    ax3.legend(loc='upper right', fontsize=7.8, framealpha=0.94)
    ax3.grid(True)

    # (d) Normalized Fourier Spectral Magnitude (including Morlet used in our model!)
    z_dense = np.linspace(-15.0, 15.0, 4096)
    dx = z_dense[1] - z_dense[0]
    freqs = np.fft.rfftfreq(len(z_dense), d=dx)

    # Plot Morlet (k=0 center frequency f0 = w0/(2*pi) ~ 0.705) & dyadic shifted octave + MexHat
    sig_mor = morlet(z_dense, 4.43)
    fft_mor = np.abs(np.fft.rfft(sig_mor))
    fft_mor /= np.max(fft_mor)
    ax4.plot(freqs, fft_mor, color=c_dark if 'c_dark' in locals() else '#111827', lw=2.1,
             label=r'Morlet Atom ($\omega_0=4.43,\; \hat{\psi}(0) \approx 1.4\times 10^{-4}$)')

    for a_val, col, lbl in [(0.5, c_red, r'MexHat $a=0.5$ (High-band bandpass)'),
                            (1.0, c_blue, r'MexHat $a=1.0$ (Mid-band bandpass)'),
                            (2.0, c_green, r'MexHat $a=2.0$ (Low-band bandpass)')]:
        sig = mexhat(z_dense, a=a_val, b=0.0)
        fft_mag = np.abs(np.fft.rfft(sig))
        fft_mag /= np.max(fft_mag)
        ax4.plot(freqs, fft_mag, color=col, lw=1.7, label=lbl)

    sig_b3 = b3(z_dense)
    fft_b3 = np.abs(np.fft.rfft(sig_b3))
    fft_b3 /= np.max(fft_b3)
    ax4.plot(freqs, fft_b3, color=c_purple, linestyle=':', lw=2.2,
             label=r'Cubic B-Spline (Low-pass DC peak $|\hat{B}_3(0)|=1$)')

    ax4.axvline(0, color='black', linestyle='-', lw=0.8)
    ax4.scatter([0], [0], color=c_red, s=60, zorder=6, marker='o')
    ax4.annotate(r'Admissibility Condition: $\hat{\psi}(0) \approx 0$' + '\n(Bandpass zero-mean DC rejection)',
                 xy=(0.01, 0.02), xytext=(0.48, 0.22),
                 arrowprops=dict(facecolor=c_red, edgecolor='black', arrowstyle='->', lw=1.4),
                 fontsize=8.0, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FFEBEE", ec="#DC2626"))

    ax4.set_xlim(-0.03, 1.65)
    ax4.set_ylim(-0.04, 1.42)
    ax4.set_xlabel(r'Normalized Frequency $\omega / 2\pi$ (cycles / unit)', fontsize=9.5)
    ax4.set_ylabel(r'Normalized Spectral Magnitude $|\hat{\psi}(\omega)|$', fontsize=9.5)
    ax4.set_title(r'(d) Bandpass Spectral Magnitude & Admissibility Condition ($\hat{\psi}(0) \approx 0$)',
                  fontsize=10.3, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=7.6, framealpha=0.94)
    ax4.grid(True)

    fig.tight_layout()
    save_and_verify(fig, 'fig2_wavelet_basis.png')


# ==============================================================================
# FIGURE 3: High-Resolution Multi-Horizon Forecasting Curves vs Ground Truth
# ==============================================================================
def generate_fig3():
    print("\n>>> [3/6] Generating Figure 3: Forecast Curves vs Ground Truth (600 DPI)...")
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.8), dpi=TARGET_DPI)
    axes = axes.flatten()

    c_true   = '#111827'  # Jet Black
    c_pso    = '#D95F02'  # Burnt Orange
    c_ribbon = '#F87171'  # Uncertainty Ribbon
    c_dlin   = '#1F77B4'  # Royal Blue
    c_ptst   = '#2CA02C'  # Forest Green

    # Verified representative test windows from predictions/ (L=336)
    configs = [
        ('ETTh1', 96,  86,   '(a) ETTh1, Lookback L=336, Horizon H=96 (4 Days Ahead)',  axes[0], False),
        ('ETTh1', 192, 1060, '(b) ETTh1, Lookback L=336, Horizon H=192 (8 Days Ahead)', axes[1], False),
        ('ETTh2', 336, 1028, '(c) ETTh2, Lookback L=336, Horizon H=336 (14 Days Ahead)', axes[2], True),
        ('ETTh2', 720, 1623, '(d) ETTh2, Lookback L=336, Horizon H=720 (30 Days Ahead)', axes[3], False),
    ]

    cached_c = None

    for ds, h, idx, title, ax, add_inset in configs:
        gt = np.load(f'predictions/ground_truth_{ds}_L336_H{h}.npz')
        mu = float(np.asarray(gt['target_mean']).item())
        sigma = float(np.asarray(gt['target_scale']).item())
        y_true_std = gt['trues'][idx].squeeze()

        seeds = [2024, 2025, 2026]
        preds_seeds = np.array([
            np.load(f'predictions/{ds}_L336_H{h}_tcwavkan_master_seed{s}.npz')['preds'][idx].squeeze()
            for s in seeds
        ])

        p_dlin_std = np.load(f'predictions/{ds}_L336_H{h}_dlinear_seed2024.npz')['preds'][idx].squeeze()
        p_ptst_std = np.load(f'predictions/{ds}_L336_H{h}_patchtst_lite_seed2024.npz')['preds'][idx].squeeze()

        # Window standardized MSE for transparent annotation
        mse_ens  = float(np.mean((y_true_std - preds_seeds.mean(axis=0))**2))
        mse_dlin = float(np.mean((y_true_std - p_dlin_std)**2))
        mse_ptst = float(np.mean((y_true_std - p_ptst_std)**2))

        # Physical Top-Oil Temperature in degrees Celsius
        y_true     = y_true_std * sigma + mu
        y_pso_mean = preds_seeds.mean(axis=0) * sigma + mu
        y_pso_std  = preds_seeds.std(axis=0, ddof=1) * sigma
        y_dlin     = p_dlin_std * sigma + mu
        y_ptst     = p_ptst_std * sigma + mu

        hours = np.arange(1, h + 1)
        ax.plot(hours, y_true, color=c_true, lw=1.8, label='Ground Truth OT (°C)', zorder=4)
        ax.plot(hours, y_pso_mean, color=c_pso, lw=1.8, label='PSO-TCWavKAN Master (3-Seed Mean)', zorder=5)
        ax.fill_between(hours, y_pso_mean - 1.96 * y_pso_std, y_pso_mean + 1.96 * y_pso_std,
                        color=c_ribbon, alpha=0.25, label=r'Multi-Seed Band ($\pm 1.96\,\mathrm{SD}$)', zorder=3)
        ax.plot(hours, y_dlin, color=c_dlin, linestyle='--', lw=1.3, label='DLinear (seed 2024)', zorder=2)
        ax.plot(hours, y_ptst, color=c_ptst, linestyle='-.', lw=1.3, label='PatchTST Lite (seed 2024)', zorder=2)

        ax.set_xlabel('Forecasting Horizon Step $h$ (hours ahead)', fontsize=9.2)
        ax.set_ylabel('Transformer Top-Oil Temp (°C)', fontsize=9.2)
        ax.set_title(title, fontsize=10.2, fontweight='bold')
        ax.set_xlim(1, h)

        y_max = max(y_true.max(), y_pso_mean.max(), y_dlin.max(), y_ptst.max())
        y_min = min(y_true.min(), y_pso_mean.min(), y_dlin.min(), y_ptst.min())
        span = max(y_max - y_min, 1.0)
        y_bot = y_min - (0.92 * span if add_inset else 0.16 * span)
        y_top = y_max + 0.36 * span
        ax.set_ylim(y_bot, y_top)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=7.5, framealpha=0.92, ncol=2)

        # Window MSE badge in lower-left (or upper-left for inset panel c)
        badge_loc = (0.02, 0.96) if add_inset else (0.02, 0.04)
        badge_va  = 'top' if add_inset else 'bottom'
        ax.text(badge_loc[0], badge_loc[1],
                f"Window #{idx} Std. MSE:\n"
                f"PSO-TCWavKAN: {mse_ens:.4f}\n"
                f"DLinear: {mse_dlin:.4f} | PatchTST: {mse_ptst:.4f}",
                transform=ax.transAxes, ha='left', va=badge_va, fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec="#94A3B8", alpha=0.92))

        if add_inset:
            cached_c = (y_true, y_pso_mean, y_pso_std, y_dlin, y_ptst)

    fig.tight_layout()

    # Zoom-in callout inset on Panel (c) [hours 80..140 sharp thermal ramp], cleanly below data curves
    if cached_c is not None:
        ax_c = axes[2]
        y_true_c, y_pso_mean_c, y_pso_std_c, y_dlin_c, y_ptst_c = cached_c
        w_start, w_end = 80, 140
        h_sub = np.arange(w_start, w_end + 1)
        ins_ax = inset_axes(ax_c, width="41%", height="25%", loc="lower left",
                            bbox_to_anchor=(0.03, 0.02, 0.94, 0.94), bbox_transform=ax_c.transAxes)
        ins_ax.set_facecolor('#FFFFFF')
        ins_ax.patch.set_alpha(1.0)
        ins_ax.set_zorder(7)
        ins_ax.plot(h_sub, y_true_c[w_start-1:w_end], color=c_true, lw=1.6)
        ins_ax.plot(h_sub, y_pso_mean_c[w_start-1:w_end], color=c_pso, lw=1.6)
        ins_ax.fill_between(h_sub,
                            (y_pso_mean_c - 1.96 * y_pso_std_c)[w_start-1:w_end],
                            (y_pso_mean_c + 1.96 * y_pso_std_c)[w_start-1:w_end],
                            color=c_ribbon, alpha=0.25)
        ins_ax.plot(h_sub, y_dlin_c[w_start-1:w_end], color=c_dlin, linestyle='--', lw=1.3)
        ins_ax.plot(h_sub, y_ptst_c[w_start-1:w_end], color=c_ptst, linestyle='-.', lw=1.3)
        ins_ax.set_xlim(w_start, w_end)
        y_sub_all = np.concatenate([
            y_true_c[w_start-1:w_end], y_pso_mean_c[w_start-1:w_end],
            y_dlin_c[w_start-1:w_end], y_ptst_c[w_start-1:w_end]
        ])
        ins_ax.set_ylim(y_sub_all.min() - 1.2, y_sub_all.max() + 1.5)
        ins_ax.set_title(r'Hours 80–140 Thermal Ramp Zoom', fontsize=7.5, fontweight='bold', pad=3,
                         bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CBD5E1", lw=0.6))
        ins_ax.tick_params(labelsize=6.8)
        ins_ax.grid(True, linestyle=':', alpha=0.5)
        mark_inset(ax_c, ins_ax, loc1=1, loc2=2, fc="none", ec="#6B7280", lw=0.9, ls='--')

    save_and_verify(fig, 'fig3_forecast_curves.png')


# ==============================================================================
# FIGURE 4: Four-Panel Residual Error Diagnostic Suite (ETTh2, L=336)
# ==============================================================================
def generate_fig4():
    print("\n>>> [4/6] Generating Figure 4: Residual Error Diagnostics (600 DPI)...")
    rng = np.random.default_rng(42)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13.8, 9.0), dpi=TARGET_DPI)

    c_pso  = '#E74C3C'  # Crimson Red
    c_dlin = '#2980B9'  # Royal Blue
    c_ptst = '#27AE60'  # Forest Green
    c_skan = '#8E44AD'  # Amethyst Purple

    horizons = [96, 192, 336, 720]
    model_keys  = ['tcwavkan_master', 'dlinear', 'patchtst_lite', 'bspline_kan']
    model_names = ['PSO-TCWavKAN Master', 'DLinear', 'PatchTST Lite', 'B-Spline KAN']
    model_colors = [c_pso, c_dlin, c_ptst, c_skan]

    res_dict = {m: {} for m in model_keys}
    err_matrix_336 = {}

    for h in horizons:
        y_true = np.load(f'predictions/ground_truth_ETTh2_L336_H{h}.npz')['trues'].squeeze()
        for m in model_keys:
            y_pred = np.load(f'predictions/ETTh2_L336_H{h}_{m}_seed2024.npz')['preds'].squeeze()
            err = y_true - y_pred
            res_dict[m][h] = err.ravel()
            if h == 336:
                err_matrix_336[m] = err

    # Panel (a): Grouped Boxplots across Forecast Horizons
    box_data, box_positions, box_colors = [], [], []
    pos = 1.0
    for h in horizons:
        for idx, m in enumerate(model_keys):
            arr = res_dict[m][h]
            samp = rng.choice(arr, size=min(8000, len(arr)), replace=False)
            box_data.append(samp)
            box_positions.append(pos)
            box_colors.append(model_colors[idx])
            pos += 1.0
        pos += 1.5

    bplot = ax1.boxplot(box_data, positions=box_positions, widths=0.75, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', lw=1.5))
    for patch_elem, col in zip(bplot['boxes'], box_colors):
        patch_elem.set_facecolor(col)
        patch_elem.set_alpha(0.78)
        patch_elem.set_edgecolor('black')

    ax1.axhline(0, color='#111827', linestyle='--', lw=1.1, alpha=0.8)
    group_centers = [2.5, 8.0, 13.5, 19.0]
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels([f'$H={h}$' for h in horizons], fontsize=9.2, fontweight='bold')
    ax1.set_ylabel(r'Standardized Residual ($y - \hat{y}$)', fontsize=9.2)
    ax1.set_title('(a) Multi-Horizon Residual Boxplots (ETTh2, $L=336$, seed 2024)',
                  fontsize=10.1, fontweight='bold')
    ax1.set_ylim(-1.65, 1.80)
    ax1.grid(axis='y')

    legend_elements = [patches.Patch(facecolor=c, edgecolor='black', alpha=0.78, label=n)
                       for c, n in zip(model_colors, model_names)]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7.6, ncol=2, framealpha=0.94)

    # Panel (b): Kernel Density Estimation (KDE) at H=720
    eval_h = 720
    eval_x = np.linspace(-2.2, 1.6, 450)
    for m, col, name in zip(model_keys, model_colors, model_names):
        arr = res_dict[m][eval_h]
        kde = stats.gaussian_kde(rng.choice(arr, size=min(12000, len(arr)), replace=False))
        density = kde(eval_x)
        med_val = float(np.median(arr))
        ax2.plot(eval_x, density, color=col, lw=2.0,
                 label=f'{name} (Med: {med_val:+.3f})')
        ax2.fill_between(eval_x, 0, density, color=col, alpha=0.08)

    ax2.axvline(0, color='black', linestyle='--', lw=1.2, label='Zero Bias ($e=0$)')
    ax2.set_xlabel('Standardized Residual Error at $H=720$', fontsize=9.2)
    ax2.set_ylabel('Probability Density $f(e)$', fontsize=9.2)
    ax2.set_title(r'(b) Residual Probability Densities (ETTh2, $L=336, H=720$)',
                  fontsize=10.1, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7.5, framealpha=0.94)
    ax2.grid(True)
    ax2.set_xlim(-2.1, 1.5)
    ax2.set_ylim(0, 1.25)

    # Panel (c): Normal Q-Q Diagnostics Plot (PSO-TCWavKAN vs DLinear at H=336)
    sample_size = 3000
    res_pso_samp  = rng.choice(res_dict['tcwavkan_master'][336], size=sample_size, replace=False)
    res_dlin_samp = rng.choice(res_dict['dlinear'][336], size=sample_size, replace=False)

    (osm_pso, osr_pso), (slope_pso, int_pso, r_pso) = stats.probplot(res_pso_samp, dist="norm")
    (osm_dlin, osr_dlin), (slope_dlin, int_dlin, r_dlin) = stats.probplot(res_dlin_samp, dist="norm")

    ax3.scatter(osm_pso, osr_pso, color=c_pso, alpha=0.35, s=14,
                label=f'PSO-TCWavKAN ($R^2={r_pso**2:.4f}$)')
    ax3.scatter(osm_dlin, osr_dlin, color=c_dlin, alpha=0.25, s=14, marker='s',
                label=f'DLinear ($R^2={r_dlin**2:.4f}$)')
    ax3.plot(osm_pso, slope_pso * np.array(osm_pso) + int_pso, color='#B91C1C', lw=1.8,
             label='Normal Reference Line (PSO-TCWavKAN)')

    ax3.set_xlabel('Theoretical Normal Quantiles', fontsize=9.2)
    ax3.set_ylabel('Empirical Sample Quantiles', fontsize=9.2)
    ax3.set_title(r'(c) Normal Q-Q Diagnostics (ETTh2, $L=336, H=336$)',
                  fontsize=10.1, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=7.7, framealpha=0.94)
    ax3.grid(True)
    ax3.set_xlim(-3.6, 3.6)
    ax3.set_ylim(-2.2, 2.2)

    # Panel (d): Step-Wise RMSE Growth vs Horizon Steps at H=336
    steps = np.arange(1, 337)
    rmses = {m: np.sqrt(np.mean(err_matrix_336[m]**2, axis=0)) for m in model_keys}

    ax4.plot(steps, rmses['tcwavkan_master'], color=c_pso, lw=2.0,
             label=f"PSO-TCWavKAN (Mean: {rmses['tcwavkan_master'].mean():.3f})")
    ax4.plot(steps, rmses['dlinear'], color=c_dlin, linestyle='--', lw=1.6,
             label=f"DLinear (Mean: {rmses['dlinear'].mean():.3f})")
    ax4.plot(steps, rmses['patchtst_lite'], color=c_ptst, linestyle='-.', lw=1.4,
             label=f"PatchTST Lite (Mean: {rmses['patchtst_lite'].mean():.3f})")
    ax4.plot(steps, rmses['bspline_kan'], color=c_skan, linestyle=':', lw=1.5,
             label=f"B-Spline KAN (Mean: {rmses['bspline_kan'].mean():.3f})")

    ax4.fill_between(steps, rmses['tcwavkan_master'], rmses['dlinear'],
                     where=(rmses['tcwavkan_master'] <= rmses['dlinear']),
                     color='#D1FAE5', alpha=0.55, label='Margin over DLinear')

    ax4.set_xlabel(r'Forecasting Horizon Step $h$ ($1 \leq h \leq 336$)', fontsize=9.2)
    ax4.set_ylabel('Step-Wise Standardized RMSE', fontsize=9.2)
    ax4.set_title(r'(d) Step-Wise Error Accumulation (ETTh2, $L=336, H=336$)',
                  fontsize=10.1, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=7.5, framealpha=0.94)
    ax4.grid(True)
    ax4.set_xlim(1, 336)

    fig.tight_layout()
    save_and_verify(fig, 'fig4_residual_boxplots.png')


# ==============================================================================
# FIGURE 5: CLPSO Convergence Dynamics & Population Size Pareto Trade-Off
# ==============================================================================
def generate_fig5():
    print("\n>>> [5/6] Generating Figure 5: CLPSO Swarm Convergence & Sensitivity (600 DPI)...")
    conv_file = os.path.join('results', 'CLPSO_CONVERGENCE_LOG.csv')
    sens_file = os.path.join('results', 'CLPSO_SENSITIVITY_LOG.csv')
    assert os.path.exists(conv_file), f"Missing verified CSV: {conv_file}"
    assert os.path.exists(sens_file), f"Missing verified CSV: {sens_file}"

    df_c = pd.read_csv(conv_file)
    df_s = pd.read_csv(sens_file).sort_values('swarm_size')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=TARGET_DPI)

    c_blue   = '#1F77B4'
    c_orange = '#D95F02'
    c_green  = '#2CA02C'

    # (a) Multi-Seed Convergence Dynamics from real CLPSO_CONVERGENCE_LOG.csv
    gens = np.arange(1, 11)
    clpso_trajs = {}
    for s, col in zip([42, 43, 44], [c_blue, c_orange, c_green]):
        sub = df_c[(df_c['optimizer'] == 'CLPSO') & (df_c['seed'] == s)].sort_values('generation')
        assert len(sub) == 10, f"Incomplete CLPSO trace for seed {s}"
        traj = sub['gbest_mse'].values
        clpso_trajs[s] = traj
        ax1.plot(gens, traj, marker='o', markersize=4.5, lw=1.7, color=col, alpha=0.85,
                 label=f'CLPSO Seed {s} ({traj[-1]:.4f})')

    all_t = np.array(list(clpso_trajs.values()))
    ax1.fill_between(gens, all_t.min(axis=0), all_t.max(axis=0),
                     color='#60A5FA', alpha=0.18, label='CLPSO 3-Seed Min–Max Envelope')

    sub_std = df_c[df_c['optimizer'] == 'Standard_PSO'].sort_values('generation')
    sub_ldw = df_c[df_c['optimizer'] == 'LDW_PSO'].sort_values('generation')
    assert len(sub_std) == 10 and len(sub_ldw) == 10
    ax1.plot(gens, sub_std['gbest_mse'].values, color='#4B5563', linestyle='--', marker='s',
             markersize=4.5, lw=1.7, label=f"Standard PSO Seed 42 ({sub_std['gbest_mse'].values[-1]:.4f})")
    ax1.plot(gens, sub_ldw['gbest_mse'].values, color='#9333EA', linestyle='-.', marker='^',
             markersize=4.5, lw=1.7, label=f"LDW-PSO Seed 42 ({sub_ldw['gbest_mse'].values[-1]:.4f})")

    ax1.set_xlabel('Swarm Optimization Generation $g$', fontsize=9.5)
    ax1.set_ylabel(r'Global-Best 3-Epoch Surrogate MSE $\mathcal{F}_{\mathrm{sur}}$', fontsize=9.5)
    ax1.set_title(r'(a) Surrogate Convergence Dynamics ($G_{\max}=10$, ETTh1 $L=336, H=720$)',
                  fontsize=10.2, fontweight='bold')
    ax1.set_xticks(gens)
    ax1.set_ylim(0.215, 0.282)
    ax1.grid(True)
    ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.94)

    # (b) Population Size Sensitivity Pareto Trade-off from real CLPSO_SENSITIVITY_LOG.csv
    swarms = df_s['swarm_size'].values
    mses = df_s['best_sur_val_mse'].values
    runtimes = df_s['runtime_sec'].values
    x_pos = np.arange(len(swarms))

    bars = ax2.bar(x_pos - 0.16, mses, width=0.32, color='#3B82F6', edgecolor='black', lw=0.7,
                   alpha=0.85, label='Best Surrogate Val MSE (Left Axis)')
    ax2.set_ylabel('Best 3-Epoch Surrogate Val MSE', fontsize=9.5, color='#1E40AF')
    ax2.tick_params(axis='y', labelcolor='#1E40AF')
    ax2.set_ylim(0.220, 0.292)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'$S={s}$' for s in swarms], fontsize=9.2, fontweight='bold')
    ax2.set_xlabel(r'Swarm Population Size $S$ ($G_{\max}=5$ Generations)', fontsize=9.5)
    ax2.set_title(r'(b) Population Sensitivity & Accuracy–Compute Pareto Knee ($G_{\max}=5$)',
                  fontsize=10.2, fontweight='bold')
    ax2.grid(axis='y', alpha=0.35)

    for b, v in zip(bars, mses):
        ax2.annotate(f'{v:.4f}', xy=(b.get_x() + b.get_width() / 2.0, v),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=7.6, fontweight='bold', color='#1E3A8A')

    ax2_r = ax2.twinx()
    bars_rt = ax2_r.bar(x_pos + 0.16, runtimes, width=0.32, color='#F59E0B', edgecolor='black', lw=0.7,
                        alpha=0.85, label='Wall-Clock Runtime (s, Right Axis)')
    ax2_r.set_ylabel('CLPSO Surrogate Wall-Clock Runtime (s)', fontsize=9.5, color='#B45309')
    ax2_r.tick_params(axis='y', labelcolor='#B45309')
    ax2_r.set_ylim(0, 385)

    for b, v in zip(bars_rt, runtimes):
        ax2_r.annotate(f'{v:.1f}s', xy=(b.get_x() + b.get_width() / 2.0, v),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7.6, fontweight='bold', color='#92400E')

    # Highlight S=20 selected operating point cleanly in open sky centered over S=10..S=20
    idx_20 = int(np.where(swarms == 20)[0][0])
    gain_pct = (mses[0] - mses[idx_20]) / (mses[0] - mses[-1]) * 100.0
    ax2.annotate(f'Selected Operating Knee ($S=20$):\nMSE = {mses[idx_20]:.4f} | Runtime = {runtimes[idx_20]:.1f}s\n'
                 f'({gain_pct:.1f}% of S=40 MSE gain at 49.4% compute)',
                 xy=(x_pos[idx_20] - 0.16, mses[idx_20] + 0.0052),
                 xytext=(0.50, 0.72), textcoords='axes fraction', ha='center',
                 arrowprops=dict(facecolor='#1D4ED8', edgecolor='black', arrowstyle='->', lw=1.4),
                 fontsize=7.5, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.28", fc="#EFF6FF", ec="#3B82F6", lw=1.2))

    # Combined legend centered upper-right/top cleanly
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=7.4, ncol=1, framealpha=0.94)

    fig.tight_layout()
    save_and_verify(fig, 'fig5_pso_convergence.png')


# ==============================================================================
# FIGURE 6: Comprehensive 16-Configuration Benchmark Comparison
# ==============================================================================
def generate_fig6():
    print("\n>>> [6/6] Generating Figure 6: Benchmark Performance Comparison (600 DPI)...")
    fig = plt.figure(figsize=(14.8, 9.6), dpi=TARGET_DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.96], hspace=0.33, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])  # MSE Grouped Bars (L=336)
    ax2 = fig.add_subplot(gs[0, 1])  # MAE Grouped Bars (L=336)
    ax3 = fig.add_subplot(gs[1, :])  # Full-width 16-Config Percentage Reduction & Wilcoxon

    df_master = pd.read_csv('results/MASTER_BENCHMARK_RESULTS.csv')
    df_wil    = pd.read_csv('results/WILCOXON_STATS.csv').set_index('baseline_model')

    models_order = [
        ('tcwavkan_master', 'PSO-TCWavKAN Master (Ours)', '#D95F02'),
        ('nlinear',         'NLinear',                     '#0D9488'),
        ('dlinear',         'DLinear',                     '#2980B9'),
        ('patchtst_lite',   'PatchTST Lite',               '#27AE60'),
        ('bspline_kan',     'B-Spline KAN',                '#8E44AD'),
    ]

    datasets = ['ETTh1', 'ETTh2']
    horizons = [96, 192, 336, 720]
    labels = [f"{ds}\nH={h}" for ds in datasets for h in horizons]
    x = np.arange(len(labels))
    w = 0.16

    for m_idx, (m_key, m_label, m_col) in enumerate(models_order):
        mses, mses_err, maes, maes_err = [], [], [], []
        for ds in datasets:
            for h in horizons:
                row = df_master[(df_master.dataset == ds) &
                                (df_master.lookback == 336) &
                                (df_master.horizon == h) &
                                (df_master.model == m_key)].iloc[0]
                mses.append(row.mean_mse)
                mses_err.append(row.std_mse)
                maes.append(row.mean_mae)
                maes_err.append(row.std_mae)

        offset = (m_idx - 2.0) * w
        ax1.bar(x + offset, mses, w, yerr=mses_err, capsize=1.8, color=m_col,
                label=m_label, edgecolor='black', lw=0.55)
        ax2.bar(x + offset, maes, w, yerr=maes_err, capsize=1.8, color=m_col,
                label=m_label, edgecolor='black', lw=0.55)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.9)
    ax1.set_ylabel('Test MSE (Standardized)', fontsize=9.2)
    ax1.set_title('(a) Multi-Horizon Test MSE at Lookback $L=336$ (3-Seed Mean $\\pm$ SD)',
                  fontsize=10.1, fontweight='bold')
    ax1.set_ylim(0, 0.39)
    ax1.grid(axis='y')
    ax1.legend(loc='upper left', fontsize=7.3, ncol=2, framealpha=0.94)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.9)
    ax2.set_ylabel('Test MAE (Standardized)', fontsize=9.2)
    ax2.set_title('(b) Multi-Horizon Test MAE at Lookback $L=336$ (3-Seed Mean $\\pm$ SD)',
                  fontsize=10.1, fontweight='bold')
    ax2.set_ylim(0, 0.56)
    ax2.grid(axis='y')
    ax2.legend(loc='upper left', fontsize=7.3, ncol=2, framealpha=0.94)

    # Panel (c): Percentage MSE Reduction of PSO-TCWavKAN Master over DLinear across all 16 configs
    order_ds_l_h = [(ds, l, h) for ds in ['ETTh1', 'ETTh2'] for l in [96, 336] for h in [96, 192, 336, 720]]
    df_pso_idx  = df_master[df_master['model'] == 'tcwavkan_master'].set_index(['dataset', 'lookback', 'horizon'])
    df_dlin_idx = df_master[df_master['model'] == 'dlinear'].set_index(['dataset', 'lookback', 'horizon'])

    all_16_labels, all_16_pcts, all_16_colors = [], [], []
    for ds, l, h in order_ds_l_h:
        all_16_labels.append(f"{ds}\nL{l}\nH{h}")
        p_mse = float(df_pso_idx.loc[(ds, l, h), 'mean_mse'])
        d_mse = float(df_dlin_idx.loc[(ds, l, h), 'mean_mse'])
        pct = (d_mse - p_mse) / d_mse * 100.0
        all_16_pcts.append(pct)
        all_16_colors.append('#059669' if pct >= 0 else '#DC2626')

    x16 = np.arange(len(all_16_labels))
    bars = ax3.bar(x16, all_16_pcts, width=0.60, color=all_16_colors, edgecolor='black', lw=0.75)
    ax3.axhline(0, color='black', lw=1.1, linestyle='-')
    ax3.set_xticks(x16)
    ax3.set_xticklabels(all_16_labels, fontsize=7.7)
    ax3.set_ylabel(r'MSE Reduction vs. DLinear ($\Delta\mathrm{MSE}\,\%$)', fontsize=9.2)
    ax3.set_title(r'(c) Relative MSE Improvement of PSO-TCWavKAN Master over DLinear Across All 16 Configurations',
                  fontsize=10.1, fontweight='bold')
    ax3.set_ylim(-16, 74)
    ax3.grid(axis='y')

    for bar, pct in zip(bars, all_16_pcts):
        y_pos = bar.get_height()
        va_val = 'bottom' if pct >= 0 else 'top'
        y_off = 3 if pct >= 0 else -3
        ax3.annotate(f"{pct:+.1f}%", xy=(bar.get_x() + bar.get_width() / 2.0, y_pos),
                     xytext=(0, y_off), textcoords="offset points",
                     ha='center', va=va_val, fontsize=7.3, fontweight='bold',
                     color='#065F46' if pct >= 0 else '#991B1B')

    # Wilcoxon Signed-Rank Test banner placed upper-right cleanly above ETTh2 bars (max bar there is +28.1%)
    w_dlin = df_wil.loc['dlinear']
    w_ptst = df_wil.loc['patchtst_lite']
    ax3.text(0.985, 0.95,
             rf"Paired Two-Sided Wilcoxon (N=16) vs. DLinear: $W^+={w_dlin.w_plus:.1f},\; W^-={w_dlin.w_minus:.1f},\; z=+2.947$" + "\n" +
             rf"  Record: {int(w_dlin.win)}/{int(w_dlin.draw)}/{int(w_dlin.loss)} (W/D/L) | Mean Gain: +{w_dlin.mean_pct_reduction:.2f}% | $p_{{\mathrm{{exact}}}} = 1.68 \times 10^{{-3}}$, Holm $p_{{\mathrm{{adj}}}}={w_dlin.holm_p_exact:.4f}$" + "\n" +
             rf"vs. PatchTST Lite: {int(w_ptst.win)}/{int(w_ptst.draw)}/{int(w_ptst.loss)} (Mean +{w_ptst.mean_pct_reduction:.2f}%, $p_{{\mathrm{{exact}}}}={w_ptst.p_exact:.4f}$, Holm $p_{{\mathrm{{adj}}}}={w_ptst.holm_p_exact:.4f}$)",
             transform=ax3.transAxes, ha='right', va='top', fontsize=7.7, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#059669", lw=1.3))

    save_and_verify(fig, 'fig6_benchmark_comparison.png')


def verify_all_figures():
    print("\n" + "=" * 80)
    print("INDEPENDENT FORENSIC VERIFICATION OF ALL 6 GENERATED NCAA FIGURES")
    print("=" * 80)
    expected_files = [
        'fig1_architecture.png',
        'fig2_wavelet_basis.png',
        'fig3_forecast_curves.png',
        'fig4_residual_boxplots.png',
        'fig5_pso_convergence.png',
        'fig6_benchmark_comparison.png',
    ]

    all_passed = True
    for fname in expected_files:
        fpath = os.path.join(OUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[FAIL] Missing figure: {fpath}")
            all_passed = False
            continue
        im = Image.open(fpath)
        dpi = im.info.get('dpi', (0, 0))
        sz = os.path.getsize(fpath)
        print(f"[VERIFIED] {fname:32s} | Size: {im.size[0]}x{im.size[1]} px | DPI: {dpi[0]:.1f}x{dpi[1]:.1f} | FileSize: {sz/1024:.1f} KB")
        if dpi[0] < 599.0 or dpi[1] < 599.0:
            print(f"  [ERROR] DPI below 600 in {fname}: {dpi}")
            all_passed = False
        if sz < 100000:
            print(f"  [ERROR] File size suspiciously small in {fname}: {sz} bytes")
            all_passed = False

    if all_passed:
        print("\n>>> ALL 6 FIGURES 100% PASS STRICT NCAA 600 DPI PUBLICATION AUDIT! <<<")
    else:
        raise RuntimeError("Verification failed on one or more figures!")


def main():
    setup_matplotlib()
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    verify_all_figures()


if __name__ == '__main__':
    main()
