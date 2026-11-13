#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
KAGGLE_NCAA_OPTION_A_SUPPLEMENT.py
100% Self-Contained Leak-Free Benchmark, Ablation, CLPSO & Edge Profiling Suite
================================================================================
Target Journal : Neural Computing and Applications (NCAA - Springer Nature)
Manuscript     : PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet
                 Kolmogorov-Arnold Networks for Power Transformer Thermal
                 Dynamics Forecasting
Author         : Nguyen Phu Toan (Hanoi University of Business and Technology)

What this single self-contained script executes (zero external dependencies
beyond PyTorch, NumPy, Pandas, SciPy, Scikit-Learn):

  SUITE 1 — Unified Leak-Free 16-Configuration Benchmark (Tables 1, 2, 3 & Figs 3, 4, 6)
    - Strict univariate Top-Oil Temperature (OT, C=1) under canonical 12/4/4
      protocol ([0:8640] train / [8640:11520] val / [11520:14400] test).
    - Leak-free CLPSO surrogate split STRICTLY confined inside train [0:8640]
      (disjoint 85/15 sur-train / sur-val windows; zero overlap with val/test).
    - All 6 models evaluated identically across 16 configs x 3 seeds [2024, 2025, 2026]:
        1. tcwavkan_master  (CLPSO warm-start + Decomp_TCWavKAN_Master)
        2. tcwavkan_no_pso  (Xavier/default init + Decomp_TCWavKAN_Master)
        3. dlinear          (Zeng et al., AAAI 2023)
        4. nlinear          (Zeng et al., AAAI 2023)
        5. patchtst_lite    (Nie et al., ICLR 2023)
        6. bspline_kan      (Liu et al., 2024; RevIN + Tikhonov B-spline KAN + safe_update_grid)
    - Saves full CSV summary (results/MASTER_BENCHMARK_RESULTS.csv), per-config CLPSO
      discovered hyperparameters (results/CLPSO_PER_CONFIG_LOG.csv), and prediction
      trajectories (.npz) for Figures 3 & 4.

  SUITE 2 — Four-Pillar Component Ablation Study (Table 4)
    - Evaluated on ETTh1 (L=336, H=720) across seeds [2024, 2025, 2026]:
        * full            (linked directly from Suite 1 tcwavkan_master)
        * no_revin        (w/o RevIN normalization/denormalization)
        * no_tcd          (w/o Multi-Kernel Trend-Seasonal Decomposition)
        * spline_kan_block(Discrete B-Spline KAN block w/ safe_update_grid vs Morlet WavKAN)
        * no_clpso        (linked directly from Suite 1 tcwavkan_no_pso)
    - Saves results/ABLATION_RESULTS.csv with mean epoch timing and grid update stats.

  SUITE 3 — Swarm Convergence & Population Sensitivity Analysis (Section 4.7 & Fig 5)
    - Leak-free surrogate on ETTh1 (L=336, H=720) strictly inside [0:8640]:
        * Convergence curves (G=10 generations, S=20): CLPSO (chi=0.72984, seeds 42,43,44)
          vs Standard PSO (w=0.8) vs LDW-PSO (w: 0.9->0.4) -> results/CLPSO_CONVERGENCE_LOG.csv
        * Population sensitivity: S in {5, 10, 20, 40} at G=5 -> results/CLPSO_SENSITIVITY_LOG.csv

  SUITE 4 — CUDA/CPU Edge Footprint Profiling & Non-Parametric Wilcoxon Tests (Sec 4.5, 5.1)
    - Exact parameter counts, isolated train/inference peak VRAM (MB), and synchronized
      single-stream bs=1 inference latency (ms) -> results/EDGE_PROFILE.csv
    - Two-sided paired Wilcoxon signed-rank + Holm-Bonferroni tests -> results/WILCOXON_STATS.csv
    - Environment manifest -> results/ENVIRONMENT_MANIFEST.json
    - Auto-resumes from existing CSV rows & auto-updates ncaa_option_a_results.zip!

HOW TO RUN ON KAGGLE:
  1. Create New Notebook -> Accelerator: GPU T4 x2 (recommended) or P100 / T4 -> Internet: ON
  2. RECOMMENDED FOR 2x SPEED ON GPU T4 x2:
     Click `File -> Editor Type -> Script` (or upload .py file & run `!python ...`).
     In Script mode, when 2 T4 GPUs are detected, this runner AUTOMATICALLY shards
     the workload across BOTH GPUs (`cuda:0` + `cuda:1` subprocesses) for ~2x speedup!
     (Notebook-cell paste also works safely via single-GPU fallback.)
  3. Keep EXECUTION_MODE = "FULL_OPTION_A" below and click "Save Version" -> "Save & Run All (Commit)"
  4. Expected wall-clock time:
       - GPU T4 x2 (Script / Dual-GPU auto-parallel): ~35-65 mins
       - Single GPU T4 / Colab T4                    : ~60-120 mins
     Download `/kaggle/working/ncaa_option_a_results.zip` when complete.
================================================================================
"""

from __future__ import annotations

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
EXECUTION_MODE = "FULL_OPTION_A"  # Options: "FULL_OPTION_A", "SMOKE_TEST", "CLI"

import argparse
import contextlib
import copy
import gc
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel


# ==============================================================================
# GLOBAL PROTOCOL CONSTANTS
# ==============================================================================
DATA_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
}

ALL_16_CONFIGS: List[Tuple[str, int, int]] = [
    ("ETTh1", 96, 96),
    ("ETTh1", 96, 192),
    ("ETTh1", 96, 336),
    ("ETTh1", 96, 720),
    ("ETTh1", 336, 96),
    ("ETTh1", 336, 192),
    ("ETTh1", 336, 336),
    ("ETTh1", 336, 720),
    ("ETTh2", 96, 96),
    ("ETTh2", 96, 192),
    ("ETTh2", 96, 336),
    ("ETTh2", 96, 720),
    ("ETTh2", 336, 96),
    ("ETTh2", 336, 192),
    ("ETTh2", 336, 336),
    ("ETTh2", 336, 720),
]

DEFAULT_SEEDS = [2024, 2025, 2026]

# Unified baseline list (6 models total across 16 configs = 96 config evaluations)
SUITE1_MODELS = [
    "tcwavkan_master",
    "tcwavkan_no_pso",
    "dlinear",
    "nlinear",
    "patchtst_lite",
    "bspline_kan",
]

# Standard published/verified learning rates per model (no baseline handicapped)
DEFAULT_MODEL_LRS: Dict[str, float] = {
    "tcwavkan_master": 0.005,   # Overridden per config by CLPSO search in [1e-3, 1e-2]
    "tcwavkan_no_pso": 0.005,   # Identical default LR without CLPSO warm-start
    "dlinear": 0.001,           # Canonical LTSF-Linear AdamW LR
    "nlinear": 0.001,           # Canonical LTSF-Linear AdamW LR
    "patchtst_lite": 0.001,     # Verified optimal Transformer LR (avoids 5e-3 instability)
    "bspline_kan": 0.001,       # Verified B-spline KAN LR (grid_range=[-4,4], safe_update_grid)
}

DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 12
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_SWA_EPOCHS = 10

CLPSO_SWARM_SIZE = 20
CLPSO_GENS = 5
CLPSO_EVAL_EPOCHS = 3
GRID_UPDATE_EPOCHS = (0, 5, 10, 15)
KAN_REG_LAMBDA = 1e-4

_RAW_SERIES_CACHE: Dict[str, np.ndarray] = {}


# ==============================================================================
# UTILITY HELPERS & REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_dynamic_batch_size(horizon: int) -> int:
    return 128 if int(horizon) <= 192 else 32


def compute_sample_metrics_ddof1(errors: Sequence[float]) -> Tuple[float, float]:
    n = len(errors)
    if n == 0:
        return float("nan"), float("nan")
    mean_val = float(np.mean(errors))
    if n == 1:
        return mean_val, float("nan")
    return mean_val, float(np.std(errors, ddof=1))


def get_cosine_warmup_lr(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 5,
    lr_min: float = 1e-5,
    lr_max: float = 0.005,
) -> float:
    if epoch < warmup_epochs:
        return lr_min + (lr_max - lr_min) * (epoch / max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine_decay


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


class _NoOpGradScaler:
    """Drop-in GradScaler fallback when running on CPU or without AMP."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _make_grad_scaler(enabled: bool):
    """Cross-version PyTorch AMP GradScaler helper (PyTorch 1.10 through 2.13+)."""
    if not enabled or not torch.cuda.is_available():
        return _NoOpGradScaler()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=True)
        except TypeError:
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)
    return _NoOpGradScaler()


def _cuda_autocast(enabled: bool, dtype: torch.dtype = torch.float16):
    """Cross-version PyTorch CUDA autocast context (no-op on CPU)."""
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            if enabled:
                return amp_mod.autocast("cuda", enabled=True, dtype=dtype)
            return amp_mod.autocast("cuda", enabled=False)
        except TypeError:
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        if enabled:
            try:
                return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
            except TypeError:
                return torch.cuda.amp.autocast(enabled=True)
        return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def refresh_zip_archive(out_root: str, zip_path: str) -> None:
    """Atomically updates downloadable zip bundle after each checkpointed milestone."""
    out_p = Path(out_root)
    tmp_zip = Path(f"{zip_path}.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(out_p.rglob("*")):
            if not p.is_file() or str(p) == str(zip_path) or p.name.endswith(".tmp"):
                continue
            rel = p.relative_to(out_p)
            if rel.parts and rel.parts[0].startswith("shard_"):
                continue
            zf.write(p, arcname=rel)
    os.replace(tmp_zip, zip_path)


def upsert_csv_row(csv_path: str, row: Dict[str, Any], key_cols: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df_old = pd.read_csv(csv_path)
        mask = pd.Series([True] * len(df_old))
        for col in key_cols:
            mask = mask & (df_old[col].astype(str) == str(row[col]))
        df_old = df_old[~mask]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    tmp_csv = f"{csv_path}.tmp"
    df_out.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)


def load_completed_keys(csv_path: str, key_cols: Sequence[str]) -> set:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    keys = set()
    for _, r in df.iterrows():
        keys.add(tuple(str(r[c]) for c in key_cols))
    return keys


# ==============================================================================
# 1. REVERSIBLE INSTANCE NORMALIZATION (RevIN)
# ==============================================================================

class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
        safe_gamma_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.safe_gamma_eps = safe_gamma_eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self.mean: Optional[torch.Tensor] = None
        self.stdev: Optional[torch.Tensor] = None

    def _get_safe_gamma(self) -> torch.Tensor:
        safe_sign = torch.where(self.gamma >= 0.0, torch.ones_like(self.gamma), -torch.ones_like(self.gamma))
        safe_abs = torch.clamp(self.gamma.abs(), min=self.safe_gamma_eps)
        return safe_sign * safe_abs

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(-1)
        mean = (x[:, -1:, :] if self.subtract_last else x.mean(dim=1, keepdim=True)).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False)
        stdev = torch.sqrt(var + self.eps).detach()
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        self.mean, self.stdev = mean, stdev
        if is_2d:
            x_norm = x_norm.squeeze(-1)
        return x_norm, (mean, stdev)

    def denormalize(
        self,
        x: torch.Tensor,
        stats_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(-1)
        if stats_tuple is not None:
            mean, stdev = stats_tuple
        else:
            if self.mean is None or self.stdev is None:
                raise RuntimeError("RevIN denormalize called without cached or explicit stats.")
            mean, stdev = self.mean, self.stdev
        if self.affine:
            safe_gamma = self._get_safe_gamma()
            x = (x - self.beta) / safe_gamma
        x = x * stdev + mean
        if is_2d:
            x = x.squeeze(-1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "norm",
        stats_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if mode == "norm":
            x_norm, st = self.normalize(x)
            return (x_norm, st) if return_stats else x_norm
        elif mode == "denorm":
            return self.denormalize(x, stats_tuple=stats_tuple)
        raise NotImplementedError(f"Unsupported RevIN mode '{mode}'")


# ==============================================================================
# 2. TIKHONOV B-SPLINE KAN (FLOAT32 + AMP-SAFE + SAFE_UPDATE_GRID)
# ==============================================================================

class KANLinear(nn.Module):
    """
    Efficient B-Spline KANLinear layer with:
      - enable_standalone_scale_spline = False by default (avoids double scaling)
      - Tikhonov ridge regularization (_lambda = 1e-2) in curve2coeff
      - Float32 forced execution inside CUDA AMP float16 contexts (never fails Half/Float or LinAlg)
      - Entropy safe guard (eps = 1e-10, torch.where(p > 0))
      - safe_update_grid(x) with snapshot & automatic rollback on non-finite outputs
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = False,
        base_activation: Callable[[], nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-4.0, 4.0),
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.enable_standalone_scale_spline = bool(enable_standalone_scale_spline)
        self.base_activation = base_activation()
        self.grid_eps = float(grid_eps)

        h = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = (
            (torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=torch.float32) * h + grid_range[0])
            .expand(self.in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features, self.grid_size + self.spline_order))
        if self.enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        else:
            self.register_parameter("spline_scaler", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order],
                noise,
            )
            scale = self.scale_spline if not self.enable_standalone_scale_spline else 1.0
            self.spline_weight.data.copy_(scale * coeff)
            if self.enable_standalone_scale_spline and self.spline_scaler is not None:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        with _cuda_autocast(False):
            xf = x.float()
            grid = self.grid.float()
            xf_u = xf.unsqueeze(-1)
            bases = ((xf_u >= grid[:, :-1]) & (xf_u < grid[:, 1:])).float()
            for k in range(1, self.spline_order + 1):
                denom1 = (grid[:, k:-1] - grid[:, : -(k + 1)]).clamp(min=1e-6)
                denom2 = (grid[:, k + 1 :] - grid[:, 1:(-k)]).clamp(min=1e-6)
                bases = ((xf_u - grid[:, : -(k + 1)]) / denom1 * bases[:, :, :-1]) + (
                    (grid[:, k + 1 :] - xf_u) / denom2 * bases[:, :, 1:]
                )
            return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with _cuda_autocast(False):
            xf = x.float()
            yf = y.float()
            A = self.b_splines(xf).transpose(0, 1)  # (in, batch, coeff)
            B = yf.transpose(0, 1)                  # (in, batch, out)
            _lambda = 1e-2
            At = A.transpose(1, 2)
            AtA = torch.bmm(At, A)
            lam_I = _lambda * torch.eye(AtA.shape[-1], device=A.device, dtype=torch.float32).unsqueeze(0)
            solution = torch.linalg.solve(AtA + lam_I, torch.bmm(At, B))
            return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline and self.spline_scaler is not None:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        orig_shape = x.shape
        with _cuda_autocast(False):
            xf = x.reshape(-1, self.in_features).float()
            base_out = F.linear(self.base_activation(xf), self.base_weight.float())
            spl = self.b_splines(xf).view(xf.size(0), -1)
            w_spl = self.scaled_spline_weight.float().view(self.out_features, -1)
            spline_out = F.linear(spl, w_spl)
            out = (base_out + spline_out).reshape(*orig_shape[:-1], self.out_features)
        return out

    @torch.no_grad()
    def safe_update_grid(self, x: torch.Tensor, margin: float = 0.01) -> bool:
        """Adaptive knot refinement with atomic snapshot & rollback protection."""
        x_flat = x.reshape(-1, self.in_features).float()
        if x_flat.size(0) < max(4, self.grid_size + self.spline_order):
            return False

        grid_backup = self.grid.data.clone()
        spline_backup = self.spline_weight.data.clone()
        try:
            with _cuda_autocast(False):
                splines = self.b_splines(x_flat).permute(1, 0, 2)
                orig_coeff = self.scaled_spline_weight.float().permute(1, 2, 0)
                unreduced_out = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

                x_sorted = torch.sort(x_flat, dim=0)[0]
                batch = x_flat.size(0)
                idx = torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x_flat.device)
                grid_adaptive = x_sorted[idx]

                span = (x_sorted[-1] - x_sorted[0]).clamp(min=1e-4)
                uniform_step = (span + 2.0 * margin) / self.grid_size
                grid_uniform = (
                    torch.arange(self.grid_size + 1, dtype=torch.float32, device=x_flat.device).unsqueeze(1)
                    * uniform_step
                    + x_sorted[0]
                    - margin
                )
                grid = self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive
                grid = torch.concatenate(
                    [
                        grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x_flat.device).unsqueeze(1),
                        grid,
                        grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x_flat.device).unsqueeze(1),
                    ],
                    dim=0,
                )
                self.grid.data.copy_(grid.T)
                new_coeff = self.curve2coeff(x_flat, unreduced_out)
                if not torch.isfinite(self.grid).all() or not torch.isfinite(new_coeff).all():
                    raise RuntimeError("Non-finite knots or coefficients after update")
                self.spline_weight.data.copy_(new_coeff)
                probe = self.forward(x_flat[:min(8, batch)])
                if not torch.isfinite(probe).all():
                    raise RuntimeError("Non-finite forward probe after update")
            return True
        except Exception:
            self.grid.data.copy_(grid_backup)
            self.spline_weight.data.copy_(spline_backup)
            return False

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> torch.Tensor:
        l1_fake = self.spline_weight.float().abs().mean(-1)
        reg_act = l1_fake.sum()
        p = l1_fake / (reg_act + 1e-10)
        reg_ent = -torch.sum(torch.where(p > 0, p * p.log(), torch.zeros_like(p)))
        return regularize_activation * reg_act + regularize_entropy * reg_ent


# ==============================================================================
# 3. CONTINUOUS WAVELET FILTER BANK, SCALABLE INTER-PATCH MIXING & MASTER MODEL
# ==============================================================================

class MultiWaveletFilterBank(nn.Module):
    def __init__(
        self,
        d_model: int,
        K: int = 5,
        omega0_init: float = 5.0,
        gamma_init: float = 0.25,
        wavelet_type: str = "morlet",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.wavelet_type = wavelet_type.lower()
        self.omega0 = nn.Parameter(torch.full((d_model, 1), float(omega0_init)))
        self.gamma = nn.Parameter(torch.full((d_model, 1), float(gamma_init)))
        k_idx = torch.arange(K, dtype=torch.float32) - (K - 1) / 2.0 if K > 1 else torch.zeros(1, dtype=torch.float32)
        self.register_buffer("k_idx", k_idx.unsqueeze(0))

    def get_frequencies(self) -> torch.Tensor:
        base_freq = self.omega0.abs() + 1e-5
        octaves = torch.pow(2.0, self.gamma * self.k_idx)
        return base_freq * octaves

    def forward(self, z: torch.Tensor, wavelet_type: Optional[str] = None) -> torch.Tensor:
        wtype = (wavelet_type or self.wavelet_type).lower()
        z_c = z.clamp(-5.0, 5.0)
        omegas = self.get_frequencies().unsqueeze(0).unsqueeze(0)
        if wtype == "morlet":
            return torch.exp(-0.5 * z_c**2) * torch.cos(omegas * z_c)
        elif wtype == "mexhat":
            z_scaled = z_c * (omegas / (self.omega0.abs().unsqueeze(0).unsqueeze(0) + 1e-5))
            return (1.0 - z_scaled**2) * torch.exp(-0.5 * z_scaled**2)
        elif wtype == "dog":
            z_scaled = z_c * (omegas / (self.omega0.abs().unsqueeze(0).unsqueeze(0) + 1e-5))
            return -z_scaled * torch.exp(-0.5 * z_scaled**2)
        raise ValueError(f"Unknown wavelet_type: {wtype}")


class WaveletPositionalMixing(nn.Module):
    def __init__(self, max_patches: int = 64, d_model: int = 32, K: int = 5, embed_dim: int = 16) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(max_patches, embed_dim)
        self.coeff_mlp = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, K))
        nn.init.uniform_(self.coeff_mlp[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.coeff_mlp[-1].bias)
        self.c_s = nn.Parameter(torch.zeros(d_model, K))

    def forward(self, n_patches: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(n_patches, device=device)
        delta_c = self.coeff_mlp(self.pos_embed(pos))
        return self.c_s.unsqueeze(0) + delta_c.unsqueeze(1)


class ScalableInterPatchMixing(nn.Module):
    def __init__(
        self,
        n_patches: int,
        d_model: int,
        kernel_sizes: Sequence[int] = (3, 5),
        drop: float = 0.05,
        init_gate: float = -2.0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.ln = nn.LayerNorm(d_model)
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model, bias=False)
            for k in self.kernel_sizes
        ])
        n_branches = len(self.kernel_sizes)
        if n_branches > 1:
            self.proj = nn.Linear(d_model * n_branches, d_model, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            self.proj = nn.Identity()
        self.drop = nn.Dropout(drop)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h_trans = self.ln(x).transpose(1, 2)
        branch_outs = [conv(h_trans) for conv in self.dw_convs]
        if len(branch_outs) > 1:
            mixed = self.proj(torch.cat(branch_outs, dim=1).transpose(1, 2))
        else:
            mixed = branch_outs[0].transpose(1, 2)
        return residual + torch.sigmoid(self.gate) * self.drop(mixed)


class MultiKernelSeriesDecomposition(nn.Module):
    def __init__(self, kernel_sizes: Sequence[int] = (25, 49)) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.pads = [k // 2 for k in self.kernel_sizes]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_3d = x.unsqueeze(1) if x.dim() == 2 else x
        trends = []
        for k, p in zip(self.kernel_sizes, self.pads):
            xp = F.pad(x_3d, (p, p), mode="replicate")
            trends.append(F.avg_pool1d(xp, kernel_size=k, stride=1).squeeze(1))
        trend = torch.stack(trends, dim=0).mean(dim=0)
        seasonal = x_3d.squeeze(1) - trend
        return trend, seasonal


class DropKAN(nn.Module):
    def __init__(self, p: float = 0.05) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return x * mask / (1.0 - self.p)


class AdaptiveDualFFTHead(nn.Module):
    def __init__(self, d_model: int, sl: int, pl: int, log_alpha: float = -3.0) -> None:
        super().__init__()
        self.ta = nn.Linear(d_model, d_model)
        self.th = nn.Linear(d_model, pl)
        self.fd = sl // 2 + 1
        self.fh = nn.Linear(self.fd * 2, pl)
        self.wt = nn.Parameter(torch.tensor(1.0))
        self.wf = nn.Parameter(torch.tensor(0.5))
        self.log_alpha = nn.Parameter(torch.tensor(float(log_alpha)))

    def forward(self, z: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        yt = self.th(F.gelu(self.ta(z.mean(dim=1))))
        # Execute RFFT strictly in float32 for CUDA FP16 safety on odd/non-pow2 lengths
        with _cuda_autocast(False):
            xf = torch.fft.rfft(xs.float(), dim=-1, norm="ortho")
            mag = torch.sqrt(xf.real**2 + xf.imag**2 + 1e-8)
            alpha = torch.sigmoid(self.log_alpha.float()) * 0.5
            thr = mag.mean(dim=-1, keepdim=True).detach() * alpha
            xf_r = torch.sign(xf.real) * F.relu(xf.real.abs() - thr)
            xf_i = torch.sign(xf.imag) * F.relu(xf.imag.abs() - thr)
            yf = self.fh.float()(torch.cat([xf_r, xf_i], dim=-1))
        w_t = self.wt.float().abs()
        w_f = self.wf.float().abs()
        return (w_t * yt.float() + w_f * yf) / (w_t + w_f + 1e-6)


class InstanceAdaptiveRouter(nn.Module):
    def __init__(self, sl: int, pl: int, hidden_dim: int = 32, init_bias: float = -0.5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(sl, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, pl))
        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.mlp[-1].bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))


class TCWavKANBlock_Master(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_patches: int,
        K: int = 5,
        drop: float = 0.05,
        omega0: float = 5.0,
        gamma_init: float = 0.25,
        log_scale: float = -1.0,
        wavelet_type: str = "morlet",
        use_harmonic_bank: bool = True,
    ) -> None:
        super().__init__()
        self.K = K if use_harmonic_bank else 1
        self.wavelet_type = wavelet_type
        self.norm_in = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=False)

        eff_K = self.K
        mu_init = (
            torch.linspace(-2.0, 2.0, eff_K).unsqueeze(0).expand(d_model, -1).clone()
            if eff_K > 1
            else torch.zeros(d_model, 1)
        )
        self.mu = nn.Parameter(mu_init)
        self.ls = nn.Parameter(torch.full((d_model, eff_K), float(log_scale)))
        self.wavelet_bank = MultiWaveletFilterBank(d_model, eff_K, omega0, gamma_init, wavelet_type)
        self.pos_mixing = WaveletPositionalMixing(n_patches + 16, d_model, eff_K, embed_dim=16)
        self.gate_wav = nn.Parameter(torch.full((1, 1, d_model), -3.0))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        self.dropkan = DropKAN(drop)
        self.inter_patch = ScalableInterPatchMixing(n_patches, d_model, (3, 5), drop, init_gate=-2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm_in(x)
        xr = xn + self.dwconv(xn.transpose(1, 2)).transpose(1, 2)
        c = self.pos_mixing(xr.size(1), xr.device)
        sigma = F.softplus(self.ls) + 1e-4
        z = (xr.unsqueeze(-1) - self.mu) / sigma
        psi = self.wavelet_bank(z, self.wavelet_type)
        wav = self.dropkan((c.unsqueeze(0) * psi).sum(dim=-1))
        wav_gated = torch.sigmoid(self.gate_wav) * self.proj(wav)
        delta_patch = self.inter_patch(xr) - xr
        return x + wav_gated + delta_patch


class SplineKANBlock_Master(nn.Module):
    """
    Ablation counterpart to TCWavKANBlock_Master:
    Preserves Pre-LayerNorm, dwconv local mixing, gate, DropKAN, proj, and
    ScalableInterPatchMixing O(DN), replacing ONLY the continuous Morlet wavelet
    filter bank + positional modulation with discrete cubic B-spline KANLinear.
    """

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        drop: float = 0.05,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=False)
        self.kan_layer = KANLinear(
            d_model, d_model,
            grid_size=grid_size, spline_order=spline_order,
            grid_range=(-4.0, 4.0), enable_standalone_scale_spline=False,
        )
        self.gate_wav = nn.Parameter(torch.full((1, 1, d_model), -3.0))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        self.dropkan = DropKAN(drop)
        self.inter_patch = ScalableInterPatchMixing(n_patches, d_model, (3, 5), drop, init_gate=-2.0)
        self._last_xr: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm_in(x)
        xr = xn + self.dwconv(xn.transpose(1, 2)).transpose(1, 2)
        if self.training:
            self._last_xr = xr.detach()
        spl = self.dropkan(self.kan_layer(xr))
        spl_gated = torch.sigmoid(self.gate_wav) * self.proj(spl.to(x.dtype))
        delta_patch = self.inter_patch(xr) - xr
        return x + spl_gated + delta_patch

    @torch.no_grad()
    def update_block_grid(self) -> bool:
        if self._last_xr is None:
            return False
        return self.kan_layer.safe_update_grid(self._last_xr.reshape(-1, self._last_xr.size(-1)))


class Decomp_TCWavKAN_Master(nn.Module):
    def __init__(
        self,
        sl: int = 96,
        pl: int = 96,
        d_model: int = 32,
        n_layers: int = 2,
        K: int = 5,
        drop: float = 0.05,
        omega0: float = 5.0,
        gamma_init: float = 0.25,
        log_scale: float = -1.0,
        log_alpha: float = -3.0,
        wavelet_type: str = "morlet",
        use_harmonic_bank: bool = True,
        kernel_sizes: Sequence[int] = (25, 49),
        use_router: bool = True,
        use_revin: bool = True,
        use_decomp: bool = True,
        block_type: str = "wavelet",
        patch_size: Optional[int] = None,
        stride: Optional[int] = None,
        affine_revin: bool = True,
        num_features: int = 1,
    ) -> None:
        super().__init__()
        self.sl = sl
        self.pl = pl
        self.d_model = d_model
        self.use_router = use_router
        self.use_revin = use_revin
        self.use_decomp = use_decomp
        self.block_type = block_type
        self.uses_kan_grid = (self.block_type == "spline_kan")

        self.revin = RevIN(num_features=num_features, affine=affine_revin) if use_revin else None
        self.decomp = MultiKernelSeriesDecomposition(kernel_sizes=kernel_sizes) if use_decomp else None
        self.trend_linear = nn.Linear(sl, pl)
        self.seas_linear = nn.Linear(sl, pl)

        self.patch_size = patch_size or (16 if sl <= 96 else 24)
        self.stride = stride or (8 if sl <= 96 else 12)
        n_patches = (sl - self.patch_size) // self.stride + 1
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(self.patch_size, d_model)

        if block_type == "wavelet":
            self.blocks = nn.ModuleList([
                TCWavKANBlock_Master(
                    d_model=d_model, n_patches=n_patches, K=K, drop=drop,
                    omega0=omega0, gamma_init=gamma_init, log_scale=log_scale,
                    wavelet_type=wavelet_type, use_harmonic_bank=use_harmonic_bank,
                )
                for _ in range(n_layers)
            ])
        elif block_type == "spline_kan":
            self.blocks = nn.ModuleList([
                SplineKANBlock_Master(d_model=d_model, n_patches=n_patches, drop=drop)
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        self.head = AdaptiveDualFFTHead(d_model=d_model, sl=sl, pl=pl, log_alpha=log_alpha)
        self.router = InstanceAdaptiveRouter(sl, pl) if use_router else None
        self.fixed_g = 0.5
        self._forced_gate: Optional[float] = None

    def set_forced_gate(self, gate_val: Optional[float]) -> None:
        """Force router gate g(x) to a fixed scalar (e.g. 0.0 for Proposition 1)."""
        self._forced_gate = gate_val

    def regularization_loss(self) -> torch.Tensor:
        if self.block_type != "spline_kan":
            return torch.tensor(0.0, device=self.patch_embed.weight.device)
        return sum(blk.kan_layer.regularization_loss() for blk in self.blocks)

    @torch.no_grad()
    def safe_update_grids(self, x_sample: torch.Tensor) -> Tuple[int, int]:
        if self.block_type != "spline_kan":
            return 0, 0
        was_training = self.training
        self.train()
        _ = self.forward(x_sample)
        succ = 0
        att = 0
        for blk in self.blocks:
            att += 1
            if blk.update_block_grid():
                succ += 1
        if not was_training:
            self.eval()
        return att, succ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(-1)
        B, L, C = x.shape

        if self.use_revin and self.revin is not None:
            x_norm, st = self.revin(x, mode="norm", return_stats=True)
        else:
            x_norm, st = x, None

        x_norm_1d = x_norm.squeeze(-1) if C == 1 else x_norm.permute(0, 2, 1).reshape(B * C, L)

        if self.use_decomp and self.decomp is not None:
            x_trend, x_seas = self.decomp(x_norm_1d)
            y_trend_lin = self.trend_linear(x_trend)
        else:
            x_seas = x_norm_1d
            y_trend_lin = torch.zeros(x_norm_1d.shape[0], self.pl, device=x.device, dtype=x.dtype)

        y_seas_lin = self.seas_linear(x_seas)
        patches = x_seas.unfold(-1, self.patch_size, self.stride)
        z = self.patch_embed(patches)
        for blk in self.blocks:
            z = blk(z)

        y_seas_kan = self.head(z, x_seas).to(y_seas_lin.dtype)

        if self._forced_gate is not None:
            g = torch.full((x_norm_1d.shape[0], self.pl), float(self._forced_gate), device=x.device, dtype=y_seas_lin.dtype)
        elif self.use_router and self.router is not None:
            g = self.router(x_norm_1d).to(y_seas_lin.dtype)
        else:
            g = torch.full((x_norm_1d.shape[0], self.pl), self.fixed_g, device=x.device, dtype=y_seas_lin.dtype)

        y_seas = (1.0 - g) * y_seas_lin + g * y_seas_kan
        y_norm_1d = y_trend_lin + y_seas
        y_norm = y_norm_1d.unsqueeze(-1) if C == 1 else y_norm_1d.view(B, C, self.pl).permute(0, 2, 1)

        if self.use_revin and self.revin is not None:
            y_phys = self.revin(y_norm, mode="denorm", stats_tuple=st)
        else:
            y_phys = y_norm
        return y_phys.squeeze(-1) if is_2d else y_phys


# ==============================================================================
# 4. COMPETITIVE BASELINES (DLinear, NLinear, PatchTST_Lite, BSplineKANForecast)
# ==============================================================================

class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pad = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dim = x.ndim
        x_in = x.unsqueeze(-1) if orig_dim == 2 else x
        x_t = x_in.transpose(1, 2)
        xp = F.pad(x_t, (self.pad, self.pad), mode="replicate")
        out = F.avg_pool1d(xp, self.kernel_size, stride=1).transpose(1, 2)
        return out.squeeze(-1) if orig_dim == 2 else out


class DLinearForecast(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            trend = self.moving_average(x)
            return self.seasonal(x - trend) + self.trend(trend)
        trend = self.moving_average(x)
        seas = x - trend
        return self.trend(trend.transpose(1, 2)).transpose(1, 2) + self.seasonal(seas.transpose(1, 2)).transpose(1, 2)


class NLinearForecast(nn.Module):
    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            last = x[:, -1:].detach()
            return self.linear(x - last) + last
        last = x[:, -1:, :].detach()
        return self.linear((x - last).transpose(1, 2)).transpose(1, 2) + last


class PatchTST_Lite(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model * self.n_patches, pred_len)
        self.drop = nn.Dropout(dropout)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dim = x.ndim
        x_in = x.unsqueeze(-1) if orig_dim == 2 else x
        B, L, C = x_in.shape
        mean = x_in.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x_norm = ((x_in - mean) / stdev) * self.affine_weight + self.affine_bias
        patches = x_norm.transpose(1, 2).reshape(B * C, L).unfold(-1, self.patch_len, self.stride)
        z = self.encoder(self.patch_embed(patches) + self.pos_embed)
        out = self.head(self.drop(z.flatten(start_dim=1))).view(B, C, self.pred_len).transpose(1, 2)
        safe_gamma = torch.where(
            self.affine_weight.abs() < 1e-4,
            torch.sign(self.affine_weight + 1e-8) * 1e-4,
            self.affine_weight,
        )
        out_phys = ((out - self.affine_bias) / safe_gamma) * stdev + mean
        return out_phys.squeeze(-1) if orig_dim == 2 else out_phys


class BSplineKANForecast(nn.Module):
    """
    Canonical Standard B-Spline KAN [L -> 32 -> H] baseline (Liu et al., 2024)
    wrapped with RevIN for fair non-stationary comparison, Tikhonov ridge lstsq,
    standalone_scale=False, grid_range=[-4, 4], and periodic safe_update_grid.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 32,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.uses_kan_grid = True
        self.revin = RevIN(num_features=1, affine=True)
        self.layers = nn.ModuleList([
            KANLinear(seq_len, hidden_dim, grid_size=grid_size, spline_order=spline_order, grid_range=(-4.0, 4.0)),
            KANLinear(hidden_dim, pred_len, grid_size=grid_size, spline_order=spline_order, grid_range=(-4.0, 4.0)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_2d = x.dim() == 2
        x_in = x.unsqueeze(-1) if is_2d else x
        B, L, C = x_in.shape
        x_norm, st = self.revin(x_in, mode="norm", return_stats=True)
        h = x_norm.transpose(1, 2).reshape(B * C, L)
        for layer in self.layers:
            h = layer(h)
        y_norm = h.view(B, C, self.pred_len).transpose(1, 2)
        y_phys = self.revin(y_norm.to(x.dtype), mode="denorm", stats_tuple=st)
        return y_phys.squeeze(-1) if is_2d else y_phys

    @torch.no_grad()
    def safe_update_grids(self, x_sample: torch.Tensor) -> Tuple[int, int]:
        x_in = x_sample.unsqueeze(-1) if x_sample.dim() == 2 else x_sample
        B, L, C = x_in.shape
        x_norm = self.revin(x_in, mode="norm")
        h = x_norm.transpose(1, 2).reshape(B * C, L)
        att, succ = 0, 0
        for layer in self.layers:
            att += 1
            if layer.safe_update_grid(h):
                succ += 1
            h = layer(h)
        return att, succ

    def regularization_loss(self) -> torch.Tensor:
        return sum(l.regularization_loss() for l in self.layers)


# ==============================================================================
# 5. FAST TENSOR BATCH LOADER & LEAK-FREE CANONICAL / SURROGATE DATA PIPELINE
# ==============================================================================

def load_raw_ot_series(dataset_name: str, data_dir: Optional[str] = None) -> np.ndarray:
    clean_name = dataset_name.replace(".csv", "")
    if clean_name in _RAW_SERIES_CACHE:
        return _RAW_SERIES_CACHE[clean_name]

    csv_file = f"{clean_name}.csv"
    search_dirs = ([data_dir] if data_dir else []) + [
        "./data", "data", "./ETDataset/ETT-small", "ETDataset/ETT-small",
        "/kaggle/input/ett-small/ETT-small", "/kaggle/input/etdataset/ETT-small",
        "/kaggle/input/ett-small", "/kaggle/input/etdataset",
        "/kaggle/working/ETDataset/ETT-small", "../input/ett-small/ETT-small", ".",
    ]
    filepath = None
    for d in search_dirs:
        cand = os.path.join(d, csv_file)
        if os.path.exists(cand) and os.path.isfile(cand):
            filepath = cand
            break

    if filepath is None and clean_name in DATA_URLS:
        dl_dir = data_dir or "./data"
        os.makedirs(dl_dir, exist_ok=True)
        dest = os.path.join(dl_dir, csv_file)
        print(f"Downloading {clean_name}.csv from official ETDataset GitHub to {dest}...")
        urllib.request.urlretrieve(DATA_URLS[clean_name], dest)
        filepath = dest

    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot locate dataset CSV for {clean_name}")

    df = pd.read_csv(filepath)
    if "OT" not in df.columns:
        raise ValueError(f"Expected column 'OT' in {filepath}, found {list(df.columns)}")
    raw_ot = df[["OT"]].values.astype(np.float32)  # (T, 1) strict univariate OT
    _RAW_SERIES_CACHE[clean_name] = raw_ot
    return raw_ot


class TensorBatchLoader:
    """
    Zero-overhead tensor-resident batch loader (pre-materializes sliding windows once on
    the target device so inner epochs execute with zero Python __getitem__ or PCIe transfers).
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 2024,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.indices = indices
        self.gen = torch.Generator(device=x.device)
        self.gen.manual_seed(int(seed))

    def __len__(self) -> int:
        n = self.x.size(0)
        if self.drop_last and n >= self.batch_size:
            return n // self.batch_size
        return max(1, math.ceil(n / self.batch_size)) if n > 0 else 0

    def __iter__(self):
        n = self.x.size(0)
        if n == 0:
            return
        if self.shuffle:
            perm = torch.randperm(n, generator=self.gen, device=self.x.device)
            eff_n = (n // self.batch_size) * self.batch_size if (self.drop_last and n >= self.batch_size) else n
            for start in range(0, eff_n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                yield self.x[idx], self.y[idx]
        else:
            eff_n = (n // self.batch_size) * self.batch_size if (self.drop_last and n >= self.batch_size) else n
            for start in range(0, eff_n, self.batch_size):
                end = min(start + self.batch_size, eff_n)
                yield self.x[start:end], self.y[start:end]

    def sample_batch(self, max_samples: int = 256) -> torch.Tensor:
        return self.x[:min(max_samples, self.x.size(0))]


def _make_windows(scaled_1d: np.ndarray, sl: int, pl: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.tensor(scaled_1d, dtype=torch.float32, device=device)  # (T, 1)
    w = t.unfold(0, sl + pl, 1).permute(0, 2, 1).contiguous()      # (N_win, sl+pl, 1)
    return w[:, :sl, :], w[:, sl:, :]


def build_dataloaders(
    dataset_name: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
    seed: int = 2024,
    data_dir: Optional[str] = None,
) -> Tuple[TensorBatchLoader, TensorBatchLoader, TensorBatchLoader, StandardScaler]:
    raw = load_raw_ot_series(dataset_name, data_dir=data_dir)
    total_len = len(raw)
    assert total_len >= 14400, f"Expected >= 14400 rows for canonical 12/4/4 protocol, got {total_len}"

    train_end, val_end, test_end = 8640, 11520, 14400
    train_slice = raw[0:train_end]
    val_slice = raw[train_end - seq_len : val_end]
    test_slice = raw[val_end - seq_len : test_end]

    scaler = StandardScaler()
    tr_sc = scaler.fit_transform(train_slice)
    va_sc = scaler.transform(val_slice)
    te_sc = scaler.transform(test_slice)

    x_tr, y_tr = _make_windows(tr_sc, seq_len, pred_len, device)
    x_va, y_va = _make_windows(va_sc, seq_len, pred_len, device)
    x_te, y_te = _make_windows(te_sc, seq_len, pred_len, device)

    eval_bs = max(batch_size, 256 if pred_len <= 192 else 128)
    tr_loader = TensorBatchLoader(x_tr, y_tr, batch_size=batch_size, shuffle=True, drop_last=True, seed=seed)
    va_loader = TensorBatchLoader(x_va, y_va, batch_size=eval_bs, shuffle=False, drop_last=False, seed=seed)
    te_loader = TensorBatchLoader(x_te, y_te, batch_size=eval_bs, shuffle=False, drop_last=False, seed=seed)
    return tr_loader, va_loader, te_loader, scaler


def make_surrogate_loaders(
    dataset_name: str,
    sl: int,
    pl: int,
    device: torch.device,
    bs: int = 32,
    pool_fraction: float = 0.15,
    split_ratio: float = 0.85,
    seed: int = 42,
    data_dir: Optional[str] = None,
) -> Tuple[TensorBatchLoader, TensorBatchLoader, int, int]:
    """
    Strictly leak-free surrogate partitioning inside canonical training horizon [0:8640].
    Hard-enforces:
      - n_train_raw == 8640 (zero overlap with val [8640:11520] or test [11520:14400])
      - sur_train_idx ∩ sur_val_idx == ∅
      - max window end index <= 8640
    """
    raw = load_raw_ot_series(dataset_name, data_dir=data_dir)
    assert len(raw) >= 14400
    n_train_raw = 8640  # STRICT canonical 12-month boundary (zero leakage!)
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(raw[:n_train_raw])
    x_all, y_all = _make_windows(tr_scaled, sl, pl, device)
    total_windows = x_all.size(0)
    assert total_windows > 64, f"Insufficient surrogate windows ({total_windows}) for L={sl}, H={pl}"

    n_pool = min(total_windows, max(bs * 2, int(total_windows * pool_fraction)))
    rng = random.Random(seed)
    sampled = rng.sample(range(total_windows), n_pool)
    n_sur_tr = max(1, min(int(n_pool * split_ratio), n_pool - 1))
    tr_idx = sampled[:n_sur_tr]
    va_idx = sampled[n_sur_tr:]

    # Mandatory leakage invariants
    assert len(set(tr_idx).intersection(va_idx)) == 0, "LEAKAGE ERROR: sur_train and sur_val overlap!"
    assert max(sampled) + sl + pl <= n_train_raw, "LEAKAGE ERROR: surrogate window exceeds n_train_raw=8640!"

    idx_tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    idx_va_t = torch.tensor(va_idx, dtype=torch.long, device=device)
    sur_tr = TensorBatchLoader(x_all[idx_tr_t], y_all[idx_tr_t], batch_size=bs, shuffle=True, drop_last=False, seed=seed, indices=tr_idx)
    sur_va = TensorBatchLoader(x_all[idx_va_t], y_all[idx_va_t], batch_size=max(bs, 128), shuffle=False, drop_last=False, seed=seed, indices=va_idx)
    return sur_tr, sur_va, len(tr_idx), len(va_idx)


# ==============================================================================
# 6. SWARM OPTIMIZERS (CLPSO chi=0.72984, STANDARD PSO w=0.8, LDW-PSO)
# ==============================================================================

def compute_clerc_constriction(c1: float = 2.05, c2: float = 2.05) -> float:
    phi = c1 + c2
    if phi <= 4.0:
        raise ValueError(f"phi must be > 4.0, got {phi}")
    denom = abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))
    return 2.0 / denom


def compute_comprehensive_learning_probabilities(n_particles: int) -> np.ndarray:
    if n_particles <= 1:
        return np.array([0.05], dtype=np.float64)
    exp10_minus_1 = math.exp(10.0) - 1.0
    return np.array(
        [0.05 + 0.45 * ((math.exp(10.0 * i / (n_particles - 1)) - 1.0) / exp10_minus_1) for i in range(n_particles)],
        dtype=np.float64,
    )


def eval_particle_fitness(
    particle: Sequence[float],
    sur_train_loader: TensorBatchLoader,
    sur_val_loader: TensorBatchLoader,
    dev: torch.device,
    n_epochs: int = 3,
    is_amp: bool = True,
    eval_seed: int = 42,
) -> float:
    set_seed(eval_seed)
    omega0, log_scale, log_alpha, log_lr = [float(v) for v in particle]
    lr = float(10.0 ** log_lr)
    sl = sur_train_loader.x.size(1)
    pl = sur_train_loader.y.size(1)

    model = Decomp_TCWavKAN_Master(
        sl=sl, pl=pl, d_model=32, n_layers=2, K=5, drop=0.05,
        omega0=omega0, gamma_init=0.25, log_scale=log_scale, log_alpha=log_alpha,
        wavelet_type="morlet", kernel_sizes=(25, 49), use_router=True, num_features=1,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_cuda_amp = bool(is_amp and dev.type == "cuda" and torch.cuda.is_available())
    scaler = _make_grad_scaler(use_cuda_amp)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in sur_train_loader:
            optimizer.zero_grad(set_to_none=True)
            if use_cuda_amp:
                with _cuda_autocast(True, dtype=torch.float16):
                    loss = F.mse_loss(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    model.eval()
    se, ne = 0.0, 0
    with torch.no_grad():
        for xb, yb in sur_val_loader:
            pred = model(xb)
            se += F.mse_loss(pred, yb, reduction="sum").item()
            ne += yb.numel()
    fitness = se / ne if ne > 0 else 999.0
    if not math.isfinite(fitness):
        fitness = 999.0

    del model, optimizer, scaler
    return float(fitness)


class CLPSOSwarm:
    """Clerc-constrained Comprehensive Learning PSO (chi ~ 0.72984, c1=c2=2.05)."""

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 5,
        lb: Optional[Sequence[float]] = None,
        ub: Optional[Sequence[float]] = None,
        c1: float = 2.05,
        c2: float = 2.05,
        refresh_gap: int = 5,
        seed: int = 42,
    ) -> None:
        self.n_particles = int(n_particles)
        self.n_iterations = int(n_iterations)
        self.lb = np.array(lb if lb is not None else [3.0, -2.5, -5.0, -3.0], dtype=np.float64)
        self.ub = np.array(ub if ub is not None else [8.0, 0.0, -2.0, -2.0], dtype=np.float64)
        self.dim = len(self.lb)
        self.c1, self.c2 = float(c1), float(c2)
        self.chi = compute_clerc_constriction(self.c1, self.c2)
        self.refresh_gap = int(refresh_gap)
        self.rng = np.random.RandomState(int(seed))
        self.pc = compute_comprehensive_learning_probabilities(self.n_particles)
        self.v_max = 0.20 * (self.ub - self.lb)

        self.positions = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.velocities = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.pbest_positions = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.pbest_fitness = np.full(self.n_particles, np.inf, dtype=np.float64)
        self.gbest_position = np.zeros(self.dim, dtype=np.float64)
        self.gbest_fitness = float("inf")
        self.exemplars = np.zeros((self.n_particles, self.dim), dtype=np.int32)
        self.stagnation_counts = np.zeros(self.n_particles, dtype=np.int32)
        self.history: List[Dict[str, Any]] = []
        self._init_done = False

    def init_swarm(self) -> None:
        for d in range(self.dim):
            self.positions[:, d] = self.rng.uniform(self.lb[d], self.ub[d], size=self.n_particles)
        self.velocities = self.rng.uniform(-self.v_max, self.v_max, size=(self.n_particles, self.dim))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness.fill(np.inf)
        self.gbest_fitness = float("inf")
        self.stagnation_counts.fill(0)
        self._select_all_exemplars()
        self._init_done = True

    def _select_exemplar(self, i: int) -> np.ndarray:
        ex = np.full(self.dim, i, dtype=np.int32)
        for d in range(self.dim):
            if self.rng.rand() < self.pc[i]:
                cands = self.rng.choice(self.n_particles, size=2, replace=(self.n_particles == 1))
                ex[d] = cands[0] if self.pbest_fitness[cands[0]] <= self.pbest_fitness[cands[1]] else cands[1]
        if self.n_particles > 1 and np.all(ex == i):
            rd = self.rng.randint(0, self.dim)
            others = [p for p in range(self.n_particles) if p != i]
            ex[rd] = self.rng.choice(others)
        return ex

    def _select_all_exemplars(self) -> None:
        for i in range(self.n_particles):
            self.exemplars[i] = self._select_exemplar(i)

    def optimize(self, fitness_fn: Callable[[np.ndarray], float], verbose: bool = True) -> Tuple[np.ndarray, float]:
        if not self._init_done:
            self.init_swarm()
        t_start = time.time()
        for it in range(self.n_iterations):
            fits = []
            for i in range(self.n_particles):
                pos = self.positions[i]
                fit = float(fitness_fn(pos))
                fits.append(fit)
                if fit < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fit
                    self.pbest_positions[i] = np.copy(pos)
                    self.stagnation_counts[i] = 0
                else:
                    self.stagnation_counts[i] += 1
                if fit < self.gbest_fitness:
                    self.gbest_fitness = fit
                    self.gbest_position = np.copy(pos)

            for i in range(self.n_particles):
                if self.stagnation_counts[i] >= self.refresh_gap:
                    self.exemplars[i] = self._select_exemplar(i)
                    self.stagnation_counts[i] = 0

            self.history.append({
                "generation": it + 1,
                "gbest_mse": float(self.gbest_fitness),
                "mean_mse": float(np.mean(fits)),
                "min_mse": float(np.min(fits)),
                "gbest_position": self.gbest_position.copy().tolist(),
                "cumulative_evals": (it + 1) * self.n_particles,
                "elapsed_sec": time.time() - t_start,
            })
            if verbose:
                print(
                    f"  [CLPSO Gen {it+1}/{self.n_iterations}] gbest={self.gbest_fitness:.6f} | "
                    f"params={np.round(self.gbest_position, 4).tolist()}",
                    flush=True,
                )

            for i in range(self.n_particles):
                for d in range(self.dim):
                    ex_pos = self.pbest_positions[self.exemplars[i, d], d]
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    v_new = self.chi * (
                        self.velocities[i, d]
                        + self.c1 * r1 * (self.pbest_positions[i, d] - self.positions[i, d])
                        + self.c2 * r2 * (ex_pos - self.positions[i, d])
                    )
                    v_new = np.clip(v_new, -self.v_max[d], self.v_max[d])
                    self.velocities[i, d] = v_new
                    self.positions[i, d] = np.clip(self.positions[i, d] + v_new, self.lb[d], self.ub[d])

        return self.gbest_position.copy(), float(self.gbest_fitness)


class InertiaPSOSwarm:
    """Comparison swarm optimizer for Suite 3: Standard PSO (w=0.8) & LDW-PSO (0.9->0.4)."""

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 10,
        mode: str = "standard",  # "standard" or "ldw"
        seed: int = 42,
    ) -> None:
        self.n_particles = int(n_particles)
        self.n_iterations = int(n_iterations)
        self.mode = mode
        self.lb = np.array([3.0, -2.5, -5.0, -3.0], dtype=np.float64)
        self.ub = np.array([8.0, 0.0, -2.0, -2.0], dtype=np.float64)
        self.dim = len(self.lb)
        self.c1, self.c2 = 1.49, 1.49
        self.rng = np.random.RandomState(int(seed))
        self.v_max = 0.20 * (self.ub - self.lb)
        self.history: List[Dict[str, Any]] = []

    def optimize(self, fitness_fn: Callable[[np.ndarray], float], verbose: bool = True) -> Tuple[np.ndarray, float]:
        pos = np.column_stack([self.rng.uniform(self.lb[d], self.ub[d], self.n_particles) for d in range(self.dim)])
        vel = self.rng.uniform(-self.v_max, self.v_max, size=(self.n_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_fit = np.full(self.n_particles, np.inf, dtype=np.float64)
        gbest_pos = np.zeros(self.dim, dtype=np.float64)
        gbest_fit = float("inf")
        t_start = time.time()

        for it in range(self.n_iterations):
            fits = []
            for i in range(self.n_particles):
                fit = float(fitness_fn(pos[i]))
                fits.append(fit)
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = np.copy(pos[i])
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = np.copy(pos[i])

            self.history.append({
                "generation": it + 1,
                "gbest_mse": float(gbest_fit),
                "mean_mse": float(np.mean(fits)),
                "min_mse": float(np.min(fits)),
                "gbest_position": gbest_pos.copy().tolist(),
                "cumulative_evals": (it + 1) * self.n_particles,
                "elapsed_sec": time.time() - t_start,
            })
            if verbose:
                print(f"  [{self.mode.upper()}_PSO Gen {it+1}/{self.n_iterations}] gbest={gbest_fit:.6f}", flush=True)

            w = 0.8 if self.mode == "standard" else (0.9 - 0.5 * (it / max(1, self.n_iterations - 1)))
            for i in range(self.n_particles):
                r1 = self.rng.rand(self.dim)
                r2 = self.rng.rand(self.dim)
                vel[i] = np.clip(
                    w * vel[i] + self.c1 * r1 * (pbest_pos[i] - pos[i]) + self.c2 * r2 * (gbest_pos - pos[i]),
                    -self.v_max, self.v_max,
                )
                pos[i] = np.clip(pos[i] + vel[i], self.lb, self.ub)

        return gbest_pos.copy(), float(gbest_fit)


# ==============================================================================
# 7. UNIFIED TRAINING ENGINE (AMP + COSINE WARMUP + FLOAT32 SWA + ARBITRATION)
# ==============================================================================

def evaluate_loader(
    model: nn.Module,
    loader: TensorBatchLoader,
    return_preds: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray, np.ndarray]]:
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    preds_list, trues_list = ([], []) if return_preds else (None, None)
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            se += F.mse_loss(out, yb, reduction="sum").item()
            ae += F.l1_loss(out, yb, reduction="sum").item()
            n += yb.numel()
            if return_preds:
                preds_list.append(out.detach().cpu().numpy())
                trues_list.append(yb.detach().cpu().numpy())
    mse = se / n if n > 0 else float("inf")
    mae = ae / n if n > 0 else float("inf")
    if return_preds:
        return mse, mae, np.concatenate(preds_list, axis=0), np.concatenate(trues_list, axis=0)
    return mse, mae


def train_eval_master(
    model: nn.Module,
    train_loader: TensorBatchLoader,
    val_loader: TensorBatchLoader,
    test_loader: TensorBatchLoader,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    swa_epochs: int = DEFAULT_SWA_EPOCHS,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    kan_reg_lambda: float = KAN_REG_LAMBDA,
    grid_update_epochs: Sequence[int] = GRID_UPDATE_EPOCHS,
) -> Tuple[float, float, np.ndarray, np.ndarray, Dict[str, Any]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda" and torch.cuda.is_available())
    scaler = _make_grad_scaler(amp_enabled)

    best_val = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    stale = 0
    has_kan_grid = bool(getattr(model, "uses_kan_grid", False)) and hasattr(model, "safe_update_grids") and hasattr(model, "regularization_loss")
    grid_attempts, grid_successes, grid_failures = 0, 0, 0
    phase1_epoch_times: List[float] = []
    t_total = time.time()
    epochs_run = 0

    for epoch in range(epochs):
        t_ep = time.time()
        epochs_run = epoch + 1

        # Optional periodic B-spline grid refinement on non-wavelet spline KAN variants
        if has_kan_grid and epoch in grid_update_epochs:
            x_probe = train_loader.sample_batch(256)
            att, succ = model.safe_update_grids(x_probe)
            grid_attempts += att
            grid_successes += succ
            grid_failures += (att - succ)

        cur_lr = get_cosine_warmup_lr(epoch, epochs, warmup_epochs=warmup_epochs, lr_min=1e-5, lr_max=lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with _cuda_autocast(True, dtype=torch.float16):
                    out = model(xb)
                    loss = F.mse_loss(out, yb)
                    if has_kan_grid and kan_reg_lambda > 0.0:
                        loss = loss + kan_reg_lambda * model.regularization_loss()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(xb)
                loss = F.mse_loss(out, yb)
                if has_kan_grid and kan_reg_lambda > 0.0:
                    loss = loss + kan_reg_lambda * model.regularization_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        phase1_epoch_times.append(time.time() - t_ep)
        val_mse, _ = evaluate_loader(model, val_loader)
        if math.isfinite(val_mse) and val_mse < best_val:
            best_val = val_mse
            best_weights = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    # Phase 2: Float32 SWA fine-tuning
    model.load_state_dict(best_weights)
    swa_model = AveragedModel(model)
    swa_lr = max(1e-5, lr * 0.05)
    swa_opt = torch.optim.AdamW(model.parameters(), lr=swa_lr, weight_decay=weight_decay)

    if swa_epochs > 0:
        for _ in range(swa_epochs):
            model.train()
            for xb, yb in train_loader:
                swa_opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(xb), yb)
                if has_kan_grid and kan_reg_lambda > 0.0:
                    loss = loss + kan_reg_lambda * model.regularization_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                swa_opt.step()
            swa_model.update_parameters(model)

    # Phase 3: Checkpoint Arbitration
    eval_model: nn.Module = model
    selected_type = "base"
    if swa_epochs > 0:
        swa_val_mse, _ = evaluate_loader(swa_model, val_loader)
        if math.isfinite(swa_val_mse) and swa_val_mse <= best_val:
            eval_model = swa_model
            selected_type = "swa"
        else:
            model.load_state_dict(best_weights)
            eval_model = model
    else:
        model.load_state_dict(best_weights)

    test_mse, test_mae, preds, trues = evaluate_loader(eval_model, test_loader, return_preds=True)
    total_time = time.time() - t_total

    info = {
        "best_val_mse": float(best_val),
        "selected_type": selected_type,
        "training_time_sec": float(total_time),
        "phase1_mean_epoch_sec": float(np.mean(phase1_epoch_times)) if phase1_epoch_times else 0.0,
        "epochs_run": int(epochs_run),
        "grid_attempts": int(grid_attempts),
        "grid_successes": int(grid_successes),
        "grid_failures": int(grid_failures),
    }
    return float(test_mse), float(test_mae), preds, trues, info


# ==============================================================================
# 8. MODEL FACTORY
# ==============================================================================

def create_model(
    model_key: str,
    seq_len: int,
    pred_len: int,
    hparams: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    hp = hparams or {}
    m = model_key.lower().strip()

    if m in ("tcwavkan_master", "tcwavkan_no_pso", "full", "no_clpso"):
        return Decomp_TCWavKAN_Master(
            sl=seq_len, pl=pred_len, d_model=32, n_layers=2, K=5, drop=0.05,
            omega0=float(hp.get("omega0", 5.0)),
            gamma_init=float(hp.get("gamma_init", 0.25)),
            log_scale=float(hp.get("log_scale", -1.0)),
            log_alpha=float(hp.get("log_alpha", -3.0)),
            wavelet_type="morlet", kernel_sizes=(25, 49),
            use_router=True, use_revin=True, use_decomp=True, block_type="wavelet", num_features=1,
        )
    elif m == "no_revin":
        return Decomp_TCWavKAN_Master(
            sl=seq_len, pl=pred_len, d_model=32, n_layers=2, K=5, drop=0.05,
            omega0=float(hp.get("omega0", 5.0)),
            log_scale=float(hp.get("log_scale", -1.0)),
            log_alpha=float(hp.get("log_alpha", -3.0)),
            use_revin=False, use_decomp=True, block_type="wavelet", num_features=1,
        )
    elif m == "no_tcd":
        return Decomp_TCWavKAN_Master(
            sl=seq_len, pl=pred_len, d_model=32, n_layers=2, K=5, drop=0.05,
            omega0=float(hp.get("omega0", 5.0)),
            log_scale=float(hp.get("log_scale", -1.0)),
            log_alpha=float(hp.get("log_alpha", -3.0)),
            use_revin=True, use_decomp=False, block_type="wavelet", num_features=1,
        )
    elif m == "spline_kan_block":
        return Decomp_TCWavKAN_Master(
            sl=seq_len, pl=pred_len, d_model=32, n_layers=2, K=5, drop=0.05,
            log_alpha=float(hp.get("log_alpha", -3.0)),
            use_revin=True, use_decomp=True, block_type="spline_kan", num_features=1,
        )
    elif m == "dlinear":
        return DLinearForecast(seq_len=seq_len, pred_len=pred_len, kernel_size=25)
    elif m == "nlinear":
        return NLinearForecast(seq_len=seq_len, pred_len=pred_len)
    elif m == "patchtst_lite":
        return PatchTST_Lite(seq_len=seq_len, pred_len=pred_len, patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2, dropout=0.05)
    elif m == "bspline_kan":
        return BSplineKANForecast(seq_len=seq_len, pred_len=pred_len, hidden_dim=32, grid_size=5, spline_order=3)
    raise ValueError(f"Unknown model key: {model_key}")


# ==============================================================================
# 9. SUITES 1, 2, 3, 4 ORCHESTRATOR
# ==============================================================================

def run_suite1_benchmark(
    configs: List[Tuple[str, int, int]],
    models: List[str],
    seeds: List[int],
    epochs: int,
    patience: int,
    swa_epochs: int,
    warmup_epochs: int,
    swarm_size: int,
    swarm_gens: int,
    pso_eval_epochs: int,
    device: torch.device,
    out_root: str,
    zip_path: str,
    data_dir: Optional[str] = None,
    refresh_zip: bool = True,
) -> None:
    print("\n" + "#" * 80)
    print("  SUITE 1: UNIFIED LEAK-FREE 16-CONFIG BENCHMARK (6 MODELS x 3 SEEDS)")
    print("#" * 80, flush=True)

    res_dir = os.path.join(out_root, "results")
    pred_dir = os.path.join(out_root, "predictions")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    bench_csv = os.path.join(res_dir, "MASTER_BENCHMARK_RESULTS.csv")
    pso_csv = os.path.join(res_dir, "CLPSO_PER_CONFIG_LOG.csv")

    done_bench = load_completed_keys(bench_csv, ["dataset", "lookback", "horizon", "model"])
    clpso_cache: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    if os.path.exists(pso_csv) and os.path.getsize(pso_csv) > 0:
        for _, r in pd.read_csv(pso_csv).iterrows():
            clpso_cache[(str(r["dataset"]), int(r["lookback"]), int(r["horizon"]))] = {
                "omega0": float(r["omega0"]),
                "log_scale": float(r["log_scale"]),
                "log_alpha": float(r["log_alpha"]),
                "lr": float(r["lr"]),
            }

    total_jobs = len(configs) * len(models)
    job_idx = 0

    for ds, sl, pl in configs:
        bs = get_dynamic_batch_size(pl)

        # Resolve CLPSO hyperparameters once per (dataset, sl, pl) if tcwavkan_master requested
        if "tcwavkan_master" in models and (ds, sl, pl) not in clpso_cache:
            print(f"\n>>> [CLPSO Surrogate] {ds} L={sl} H={pl} | S={swarm_size}, G={swarm_gens}", flush=True)
            t_pso = time.time()
            sur_tr, sur_va, n_str, n_sva = make_surrogate_loaders(ds, sl, pl, device=device, bs=32, seed=42, data_dir=data_dir)
            swarm = CLPSOSwarm(n_particles=swarm_size, n_iterations=swarm_gens, seed=42)
            best_pos, best_fit = swarm.optimize(
                lambda pos: eval_particle_fitness(pos, sur_tr, sur_va, device, n_epochs=pso_eval_epochs, is_amp=(device.type == "cuda")),
                verbose=True,
            )
            pso_sec = time.time() - t_pso
            hp = {
                "omega0": float(best_pos[0]),
                "log_scale": float(best_pos[1]),
                "log_alpha": float(best_pos[2]),
                "lr": float(10.0 ** float(best_pos[3])),
            }
            clpso_cache[(ds, sl, pl)] = hp
            upsert_csv_row(
                pso_csv,
                {
                    "dataset": ds, "lookback": sl, "horizon": pl,
                    "swarm_size": swarm_size, "n_iterations": swarm_gens,
                    "omega0": hp["omega0"], "log_scale": hp["log_scale"],
                    "log_alpha": hp["log_alpha"], "lr": hp["lr"],
                    "sur_val_mse": best_fit,
                    "sur_train_windows": n_str, "sur_val_windows": n_sva,
                    "pso_time_sec": round(pso_sec, 2),
                },
                ["dataset", "lookback", "horizon"],
            )
            del sur_tr, sur_va
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for model_key in models:
            job_idx += 1
            if (ds, str(sl), str(pl), model_key) in done_bench:
                print(f"[{job_idx}/{total_jobs}] SKIP (cached): {ds} L={sl} H={pl} {model_key}", flush=True)
                continue

            if model_key == "tcwavkan_master":
                hp = clpso_cache[(ds, sl, pl)]
                lr_used = hp["lr"]
            else:
                hp = {}
                lr_used = DEFAULT_MODEL_LRS[model_key]

            mses, maes, run_times, ep_times, val_mses = [], [], [], [], []
            n_params = 0

            for s_idx, seed in enumerate(seeds):
                set_seed(seed)
                tr_l, va_l, te_l, scaler = build_dataloaders(ds, sl, pl, bs, device=device, seed=seed, data_dir=data_dir)
                model = create_model(model_key, sl, pl, hparams=hp)
                _, n_params = count_parameters(model)

                test_mse, test_mae, preds, trues, info = train_eval_master(
                    model, tr_l, va_l, te_l, device=device,
                    epochs=epochs, patience=patience, lr=lr_used,
                    swa_epochs=swa_epochs, warmup_epochs=warmup_epochs,
                    use_amp=(device.type == "cuda"),
                )
                mses.append(test_mse)
                maes.append(test_mae)
                run_times.append(info["training_time_sec"])
                ep_times.append(info["phase1_mean_epoch_sec"])
                val_mses.append(info["best_val_mse"])

                # Store ground truth once per (ds, sl, pl) and compact float32 predictions
                gt_file = os.path.join(pred_dir, f"ground_truth_{ds}_L{sl}_H{pl}.npz")
                if not os.path.exists(gt_file):
                    np.savez_compressed(
                        gt_file,
                        trues=trues.astype(np.float32),
                        target_mean=np.array([float(scaler.mean_[0])], dtype=np.float32),
                        target_scale=np.array([float(scaler.scale_[0])], dtype=np.float32),
                    )
                if model_key == "tcwavkan_master" or s_idx == 0:
                    p_file = os.path.join(pred_dir, f"{ds}_L{sl}_H{pl}_{model_key}_seed{seed}.npz")
                    np.savez_compressed(p_file, preds=preds.astype(np.float32))

                del model, tr_l, va_l, te_l
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            mean_mse, std_mse = compute_sample_metrics_ddof1(mses)
            mean_mae, std_mae = compute_sample_metrics_ddof1(maes)

            row = {
                "dataset": ds, "lookback": sl, "horizon": pl, "model": model_key,
                "protocol": "canonical_12_4_4", "batch_size": bs,
                "lr_used": round(lr_used, 6),
                "omega0": round(hp.get("omega0", 5.0), 4) if "tcwavkan" in model_key else float("nan"),
                "log_scale": round(hp.get("log_scale", -1.0), 4) if "tcwavkan" in model_key else float("nan"),
                "log_alpha": round(hp.get("log_alpha", -3.0), 4) if "tcwavkan" in model_key else float("nan"),
                "seeds": str(seeds),
                "mean_mse": mean_mse, "std_mse": std_mse,
                "formatted_mse": f"{mean_mse:.4f} +/- {std_mse:.4f}",
                "mean_mae": mean_mae, "std_mae": std_mae,
                "formatted_mae": f"{mean_mae:.4f} +/- {std_mae:.4f}",
                "seed_mses": str([round(x, 5) for x in mses]),
                "seed_maes": str([round(x, 5) for x in maes]),
                "phase1_mean_epoch_sec": round(float(np.mean(ep_times)), 4),
                "total_runtime_sec": round(float(np.sum(run_times)), 2),
                "params": n_params,
                "best_val_mse": round(float(np.mean(val_mses)), 5),
            }
            upsert_csv_row(bench_csv, row, ["dataset", "lookback", "horizon", "model"])
            print(
                f"[{job_idx}/{total_jobs}] {ds} L={sl:3d} H={pl:3d} | {model_key:16s} | "
                f"MSE={row['formatted_mse']} | MAE={row['formatted_mae']} | {row['total_runtime_sec']:.1f}s",
                flush=True,
            )

        if refresh_zip:
            refresh_zip_archive(out_root, zip_path)


def run_suite2_ablation(
    ds: str,
    sl: int,
    pl: int,
    seeds: List[int],
    epochs: int,
    patience: int,
    swa_epochs: int,
    warmup_epochs: int,
    device: torch.device,
    out_root: str,
    zip_path: str,
    data_dir: Optional[str] = None,
    refresh_zip: bool = True,
) -> None:
    print("\n" + "#" * 80)
    print(f"  SUITE 2: FOUR-PILLAR ABLATION STUDY ({ds} L={sl}, H={pl})")
    print("#" * 80, flush=True)

    res_dir = os.path.join(out_root, "results")
    bench_csv = os.path.join(res_dir, "MASTER_BENCHMARK_RESULTS.csv")
    pso_csv = os.path.join(res_dir, "CLPSO_PER_CONFIG_LOG.csv")
    ablation_csv = os.path.join(res_dir, "ABLATION_RESULTS.csv")

    df_bench = pd.read_csv(bench_csv)
    df_pso = pd.read_csv(pso_csv) if os.path.exists(pso_csv) else pd.DataFrame()
    done_ab = load_completed_keys(ablation_csv, ["variant"])

    sub_full = df_bench[
        (df_bench["dataset"] == ds) & (df_bench["lookback"] == sl)
        & (df_bench["horizon"] == pl) & (df_bench["model"] == "tcwavkan_master")
    ]
    assert not sub_full.empty, "Suite 1 tcwavkan_master must complete before Suite 2 ablation!"
    full_row = sub_full.iloc[0]
    full_mse = float(full_row["mean_mse"])

    hp = {"omega0": 5.0, "log_scale": -1.0, "log_alpha": -3.0, "lr": 0.005}
    if not df_pso.empty:
        sub_p = df_pso[(df_pso["dataset"] == ds) & (df_pso["lookback"] == sl) & (df_pso["horizon"] == pl)]
        if not sub_p.empty:
            r_p = sub_p.iloc[0]
            hp = {"omega0": float(r_p["omega0"]), "log_scale": float(r_p["log_scale"]), "log_alpha": float(r_p["log_alpha"]), "lr": float(r_p["lr"])}

    # Record Variant 0 (full) and Variant 4 (no_clpso) directly from Suite 1 for exact consistency
    for v_key, v_label, src_model in [
        ("full", "Full PSO-TCWavKAN Master", "tcwavkan_master"),
        ("no_clpso", "Variant 4: w/o CLPSO (Random Init + AdamW)", "tcwavkan_no_pso"),
    ]:
        src_r = df_bench[
            (df_bench["dataset"] == ds) & (df_bench["lookback"] == sl)
            & (df_bench["horizon"] == pl) & (df_bench["model"] == src_model)
        ].iloc[0]
        delta_pct = (float(src_r["mean_mse"]) - full_mse) / full_mse * 100.0
        upsert_csv_row(
            ablation_csv,
            {
                "variant": v_key, "variant_label": v_label, "dataset": ds, "lookback": sl, "horizon": pl,
                "batch_size": int(src_r["batch_size"]), "lr_used": float(src_r["lr_used"]), "seeds": src_r["seeds"],
                "mean_mse": float(src_r["mean_mse"]), "std_mse": float(src_r["std_mse"]),
                "formatted_mse": src_r["formatted_mse"],
                "mean_mae": float(src_r["mean_mae"]), "std_mae": float(src_r["std_mae"]),
                "formatted_mae": src_r["formatted_mae"],
                "delta_mse_pct": round(delta_pct, 2),
                "seed_mses": src_r["seed_mses"], "seed_maes": src_r["seed_maes"],
                "phase1_mean_epoch_sec": float(src_r["phase1_mean_epoch_sec"]),
                "total_runtime_sec": float(src_r["total_runtime_sec"]),
                "params": int(src_r["params"]),
                "grid_attempts": 0, "grid_successes": 0, "grid_failures": 0,
            },
            ["variant"],
        )

    bs = get_dynamic_batch_size(pl)
    variants = [
        ("no_revin", "Variant 1: w/o RevIN (No Norm)", hp["lr"]),
        ("no_tcd", "Variant 2: w/o Multi-Kernel Conv (No TCD)", hp["lr"]),
        ("spline_kan_block", "Variant 3: Spline-KAN instead of WavKAN", 0.001),
    ]

    for v_key, v_label, v_lr in variants:
        if v_key in done_ab:
            print(f"[ABLATION] SKIP (cached): {v_key}", flush=True)
            continue

        mses, maes, ep_times, run_times = [], [], [], []
        g_att, g_succ, g_fail = 0, 0, 0
        n_params = 0

        for seed in seeds:
            set_seed(seed)
            tr_l, va_l, te_l, _ = build_dataloaders(ds, sl, pl, bs, device=device, seed=seed, data_dir=data_dir)
            model = create_model(v_key, sl, pl, hparams=hp)
            _, n_params = count_parameters(model)
            test_mse, test_mae, _, _, info = train_eval_master(
                model, tr_l, va_l, te_l, device=device,
                epochs=epochs, patience=patience, lr=v_lr,
                swa_epochs=swa_epochs, warmup_epochs=warmup_epochs,
                use_amp=(device.type == "cuda"),
            )
            mses.append(test_mse)
            maes.append(test_mae)
            ep_times.append(info["phase1_mean_epoch_sec"])
            run_times.append(info["training_time_sec"])
            g_att += info["grid_attempts"]
            g_succ += info["grid_successes"]
            g_fail += info["grid_failures"]
            del model, tr_l, va_l, te_l
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_mse, std_mse = compute_sample_metrics_ddof1(mses)
        mean_mae, std_mae = compute_sample_metrics_ddof1(maes)
        delta_pct = (mean_mse - full_mse) / full_mse * 100.0

        row = {
            "variant": v_key, "variant_label": v_label, "dataset": ds, "lookback": sl, "horizon": pl,
            "batch_size": bs, "lr_used": round(v_lr, 6), "seeds": str(seeds),
            "mean_mse": mean_mse, "std_mse": std_mse, "formatted_mse": f"{mean_mse:.4f} +/- {std_mse:.4f}",
            "mean_mae": mean_mae, "std_mae": std_mae, "formatted_mae": f"{mean_mae:.4f} +/- {std_mae:.4f}",
            "delta_mse_pct": round(delta_pct, 2),
            "seed_mses": str([round(x, 5) for x in mses]), "seed_maes": str([round(x, 5) for x in maes]),
            "phase1_mean_epoch_sec": round(float(np.mean(ep_times)), 4),
            "total_runtime_sec": round(float(np.sum(run_times)), 2),
            "params": n_params,
            "grid_attempts": g_att, "grid_successes": g_succ, "grid_failures": g_fail,
        }
        upsert_csv_row(ablation_csv, row, ["variant"])
        print(
            f"[ABLATION] {v_label:42s} | MSE={row['formatted_mse']} ({delta_pct:+.2f}%) | "
            f"Epoch={row['phase1_mean_epoch_sec']:.3f}s",
            flush=True,
        )

    if refresh_zip:
        refresh_zip_archive(out_root, zip_path)


def run_suite3_swarm_dynamics(
    ds: str,
    sl: int,
    pl: int,
    conv_gens: int,
    sens_sizes: List[int],
    sens_gens: int,
    pso_eval_epochs: int,
    device: torch.device,
    out_root: str,
    zip_path: str,
    data_dir: Optional[str] = None,
    conv_runs: Optional[List[Tuple[str, int, int]]] = None,
    refresh_zip: bool = True,
) -> None:
    print("\n" + "#" * 80)
    print(f"  SUITE 3: SWARM CONVERGENCE & POPULATION SENSITIVITY ({ds} L={sl}, H={pl})")
    print("#" * 80, flush=True)

    res_dir = os.path.join(out_root, "results")
    conv_csv = os.path.join(res_dir, "CLPSO_CONVERGENCE_LOG.csv")
    sens_csv = os.path.join(res_dir, "CLPSO_SENSITIVITY_LOG.csv")

    done_conv = load_completed_keys(conv_csv, ["optimizer", "seed", "generation"])
    done_sens = load_completed_keys(sens_csv, ["swarm_size"])

    sur_tr, sur_va, _, _ = make_surrogate_loaders(ds, sl, pl, device=device, bs=32, seed=42, data_dir=data_dir)
    fit_fn = lambda pos: eval_particle_fitness(pos, sur_tr, sur_va, device, n_epochs=pso_eval_epochs, is_amp=(device.type == "cuda"))

    # 3A: Convergence Curves (CLPSO x3 seeds vs Standard PSO vs LDW-PSO)
    runs = conv_runs if conv_runs is not None else [
        ("CLPSO", 42, 20), ("CLPSO", 43, 20), ("CLPSO", 44, 20),
        ("Standard_PSO", 42, 20), ("LDW_PSO", 42, 20),
    ]
    for opt_name, s_seed, s_size in runs:
        expected_last = (opt_name, str(s_seed), str(conv_gens))
        if expected_last in done_conv:
            print(f"[SWARM CONV] SKIP (cached): {opt_name} seed={s_seed}", flush=True)
            continue
        print(f">>> Running convergence trace: {opt_name} (seed={s_seed}, S={s_size}, G={conv_gens})", flush=True)
        if opt_name == "CLPSO":
            opt = CLPSOSwarm(n_particles=s_size, n_iterations=conv_gens, seed=s_seed)
        elif opt_name == "Standard_PSO":
            opt = InertiaPSOSwarm(n_particles=s_size, n_iterations=conv_gens, mode="standard", seed=s_seed)
        else:
            opt = InertiaPSOSwarm(n_particles=s_size, n_iterations=conv_gens, mode="ldw", seed=s_seed)
        opt.optimize(fit_fn, verbose=True)
        for rec in opt.history:
            pos = rec["gbest_position"]
            upsert_csv_row(
                conv_csv,
                {
                    "optimizer": opt_name, "seed": s_seed, "swarm_size": s_size,
                    "generation": rec["generation"], "gbest_mse": rec["gbest_mse"],
                    "mean_mse": rec["mean_mse"], "min_mse": rec["min_mse"],
                    "best_omega0": pos[0], "best_log_scale": pos[1],
                    "best_log_alpha": pos[2], "best_lr": 10.0 ** pos[3],
                    "cumulative_evals": rec["cumulative_evals"],
                    "elapsed_sec": round(rec["elapsed_sec"], 2),
                },
                ["optimizer", "seed", "generation"],
            )

    # 3B: Swarm Size Sensitivity (S in {5, 10, 20, 40})
    for pop in sens_sizes:
        if (str(pop),) in done_sens:
            print(f"[SWARM SENS] SKIP (cached): S={pop}", flush=True)
            continue
        print(f">>> Running sensitivity sweep: S={pop}, G={sens_gens}", flush=True)
        t0 = time.time()
        opt = CLPSOSwarm(n_particles=pop, n_iterations=sens_gens, seed=42)
        best_pos, best_fit = opt.optimize(fit_fn, verbose=True)
        dt = time.time() - t0
        upsert_csv_row(
            sens_csv,
            {
                "dataset": ds, "lookback": sl, "horizon": pl,
                "swarm_size": pop, "n_iterations": sens_gens,
                "omega0": best_pos[0], "log_scale": best_pos[1],
                "log_alpha": best_pos[2], "lr": 10.0 ** best_pos[3],
                "best_sur_val_mse": best_fit, "runtime_sec": round(dt, 2),
            },
            ["swarm_size"],
        )

    del sur_tr, sur_va
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if refresh_zip:
        refresh_zip_archive(out_root, zip_path)


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Cross-version SciPy two-sided paired Wilcoxon helper (handles exact/asymptotic/approx & zeros)."""
    def _call(method_name: Optional[str]):
        if method_name is None:
            return stats.wilcoxon(x, y, alternative="two-sided")
        try:
            return stats.wilcoxon(x, y, alternative="two-sided", method=method_name)
        except TypeError:
            return stats.wilcoxon(x, y, alternative="two-sided", mode=method_name)

    res_ex = None
    for m in ("exact", "asymptotic", "approx", None):
        try:
            res_ex = _call(m)
            break
        except Exception:
            continue

    res_as = None
    for m in ("asymptotic", "approx", None):
        try:
            res_as = _call(m)
            break
        except Exception:
            continue

    if res_ex is None and res_as is None:
        return 0.0, 1.0, 1.0
    if res_ex is None:
        res_ex = res_as
    if res_as is None:
        res_as = res_ex
    return float(res_ex.statistic), float(res_ex.pvalue), float(res_as.pvalue)


def run_suite4_profile_and_stats(
    models: List[str],
    device: torch.device,
    out_root: str,
    zip_path: str,
    total_elapsed_sec: float,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> None:
    print("\n" + "#" * 80)
    print("  SUITE 4: EDGE HARDWARE PROFILING & NON-PARAMETRIC WILCOXON TESTS")
    print("#" * 80, flush=True)

    res_dir = os.path.join(out_root, "results")
    edge_csv = os.path.join(res_dir, "EDGE_PROFILE.csv")
    bench_csv = os.path.join(res_dir, "MASTER_BENCHMARK_RESULTS.csv")
    wilcoxon_csv = os.path.join(res_dir, "WILCOXON_STATS.csv")
    manifest_json = os.path.join(res_dir, "ENVIRONMENT_MANIFEST.json")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU"
    profile_cfgs = [(96, 96, 128), (336, 720, 32)]

    for m_key in models:
        for sl, pl, bs_tr in profile_cfgs:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                base_mem = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)

            set_seed(42)
            model = create_model(m_key, sl, pl).to(device)
            tot_p, tr_p = count_parameters(model)

            # Train peak VRAM (single forward+backward+step)
            xb = torch.randn(bs_tr, sl, 1, device=device)
            yb = torch.randn(bs_tr, pl, 1, device=device)
            model.train()
            if tr_p > 0:
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    with _cuda_autocast(True, dtype=torch.float16):
                        loss = F.mse_loss(model(xb), yb)
                else:
                    loss = F.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()
                del opt, loss

            if device.type == "cuda":
                torch.cuda.synchronize()
                train_vram_mb = (torch.cuda.max_memory_allocated(device) - base_mem) / (1024.0 ** 2)
            else:
                train_vram_mb = 0.0
            del xb, yb

            # Inference peak VRAM & bs=1 synchronized latency (10 warmup + 100 timed)
            model.eval()
            xb_inf = torch.randn(1, sl, 1, device=device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                base_inf_mem = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(xb_inf)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    infer_vram_mb = (torch.cuda.max_memory_allocated(device) - base_inf_mem) / (1024.0 ** 2)
                else:
                    infer_vram_mb = 0.0
                latencies_ms = []
                for _ in range(100):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model(xb_inf)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            upsert_csv_row(
                edge_csv,
                {
                    "model": m_key, "lookback": sl, "horizon": pl, "batch_size_train": bs_tr,
                    "total_params": tot_p, "trainable_params": tr_p,
                    "peak_train_vram_mb": round(max(0.0, train_vram_mb), 3),
                    "peak_infer_vram_b1_mb": round(max(0.0, infer_vram_mb), 3),
                    "infer_latency_b1_ms_mean": round(float(np.mean(latencies_ms)), 4),
                    "infer_latency_b1_ms_std": round(float(np.std(latencies_ms, ddof=1)), 4),
                    "device_name": dev_name, "torch_version": torch.__version__,
                },
                ["model", "lookback", "horizon"],
            )
            print(
                f"[EDGE] {m_key:16s} ({sl},{pl}) | Params={tr_p:,} | "
                f"TrainVRAM={train_vram_mb:.1f} MB | InfVRAM={infer_vram_mb:.2f} MB | "
                f"Lat={np.mean(latencies_ms):.3f} ms",
                flush=True,
            )
            del model, xb_inf
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Exact paired Wilcoxon signed-rank tests against all baselines across all 16 configs
    if os.path.exists(bench_csv):
        df = pd.read_csv(bench_csv)
        master_df = df[df["model"] == "tcwavkan_master"].set_index(["dataset", "lookback", "horizon"]).sort_index()
        if len(master_df) >= 5:
            raw_rows = []
            for base_m in [m for m in models if m != "tcwavkan_master"]:
                base_df = df[df["model"] == base_m].set_index(["dataset", "lookback", "horizon"]).reindex(master_df.index)
                if base_df["mean_mse"].isna().any():
                    continue
                k_mse = master_df["mean_mse"].values.astype(np.float64)
                b_mse = base_df["mean_mse"].values.astype(np.float64)
                diff = b_mse - k_mse  # positive => PSO-TCWavKAN lower MSE (better)
                win = int(np.sum(diff > 0))
                draw = int(np.sum(diff == 0))
                loss_c = int(np.sum(diff < 0))
                mean_pct = float(np.mean(diff / b_mse * 100.0))

                nz = diff[diff != 0]
                if len(nz) > 0:
                    ranks = stats.rankdata(np.abs(nz))
                    w_plus = float(np.sum(ranks[nz > 0]))
                    w_minus = float(np.sum(ranks[nz < 0]))
                    w_stat, p_ex, p_as = _safe_wilcoxon(k_mse, b_mse)
                else:
                    w_plus = w_minus = w_stat = 0.0
                    p_ex = p_as = 1.0

                raw_rows.append({
                    "baseline_model": base_m, "n_configs": len(k_mse),
                    "win": win, "draw": draw, "loss": loss_c,
                    "mean_pct_reduction": round(mean_pct, 3),
                    "w_plus": w_plus, "w_minus": w_minus, "w_stat": w_stat,
                    "p_exact": p_ex, "p_asymptotic": p_as,
                })

            # Holm-Bonferroni correction across paired comparisons
            raw_rows.sort(key=lambda r: r["p_exact"])
            m_cmp = len(raw_rows)
            running_max = 0.0
            for rank_idx, r in enumerate(raw_rows):
                holm_p = min(1.0, max(running_max, r["p_exact"] * (m_cmp - rank_idx)))
                running_max = holm_p
                r["holm_p_exact"] = holm_p
                r["significant_005"] = bool(r["p_exact"] < 0.05)
                r["significant_001"] = bool(r["p_exact"] < 0.01)
                upsert_csv_row(wilcoxon_csv, r, ["baseline_model"])
                print(
                    f"[WILCOXON] vs {r['baseline_model']:16s} | W/D/L={r['win']}/{r['draw']}/{r['loss']} | "
                    f"MeanGain={r['mean_pct_reduction']:+.2f}% | W+={r['w_plus']:.1f} | p_exact={r['p_exact']:.3e} | holm={holm_p:.3e}",
                    flush=True,
                )

    # Exact Proposition 1 (g -> 0 structural recovery) verification
    prop1_diffs: Dict[str, float] = {}
    for p_sl, p_pl in [(96, 96), (336, 720)]:
        set_seed(42)
        m_prop = Decomp_TCWavKAN_Master(
            sl=p_sl, pl=p_pl, d_model=32, n_layers=2, K=5, drop=0.0,
            use_revin=False, use_decomp=True, block_type="wavelet", num_features=1,
        ).to(device)
        m_prop.eval()
        m_prop.set_forced_gate(0.0)
        x_p = torch.randn(4, p_sl, 1, device=device)
        with torch.no_grad():
            y_gate0 = m_prop(x_p)
            x_p_1d = x_p.squeeze(-1)
            tr_p_x, se_p_x = m_prop.decomp(x_p_1d)
            y_lin = (m_prop.trend_linear(tr_p_x) + m_prop.seas_linear(se_p_x)).unsqueeze(-1)
            diff_val = float((y_gate0 - y_lin).abs().max().item())
        prop1_diffs[f"L{p_sl}_H{p_pl}"] = diff_val
        print(f"[PROP1 CHECK] ({p_sl},{p_pl}) g=0 max|y - (W_t x_t + W_s x_s)| = {diff_val:.3e}", flush=True)
        del m_prop, x_p

    manifest = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": dev_name,
        "protocol": "canonical_12_4_4",
        "channels": "univariate_OT_C1",
        "seeds": DEFAULT_SEEDS,
        "clpso_leakage_free_train_cap": 8640,
        "prop1_max_abs_diff": prop1_diffs,
        "total_elapsed_sec": round(total_elapsed_sec, 2),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    refresh_zip_archive(out_root, zip_path)


# ==============================================================================
# 10. DUAL-GPU SHARD MERGER & ENTRYPOINT
# ==============================================================================

def _merge_csv_shards(
    shard_roots: List[str],
    rel_csv: str,
    out_csv: str,
    key_cols: Sequence[str],
    sort_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> None:
    dfs = []
    for sr in shard_roots:
        p = os.path.join(sr, rel_csv)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            dfs.append(pd.read_csv(p))
    if not dfs:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=list(key_cols), keep="last").reset_index(drop=True)
    if sort_fn is not None:
        merged = sort_fn(merged).reset_index(drop=True)
    tmp_csv = f"{out_csv}.tmp"
    merged.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, out_csv)


def _merge_all_shards(out_root: str, n_shards: int = 2) -> None:
    shard_roots = [os.path.join(out_root, f"shard_{i}") for i in range(n_shards)]
    res_dir = os.path.join(out_root, "results")
    pred_dir = os.path.join(out_root, "predictions")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    model_rank = {m: idx for idx, m in enumerate(SUITE1_MODELS)}
    variant_rank = {v: idx for idx, v in enumerate(["full", "no_revin", "no_tcd", "spline_kan_block", "no_clpso"])}

    def _sort_bench(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_mr"] = df["model"].map(lambda x: model_rank.get(str(x), 999))
        df = df.sort_values(["dataset", "lookback", "horizon", "_mr"]).drop(columns=["_mr"])
        return df

    def _sort_ablation(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_vr"] = df["variant"].map(lambda x: variant_rank.get(str(x), 999))
        df = df.sort_values(["_vr"]).drop(columns=["_vr"])
        return df

    _merge_csv_shards(
        shard_roots, os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv"),
        os.path.join(res_dir, "MASTER_BENCHMARK_RESULTS.csv"),
        ["dataset", "lookback", "horizon", "model"], _sort_bench,
    )
    _merge_csv_shards(
        shard_roots, os.path.join("results", "CLPSO_PER_CONFIG_LOG.csv"),
        os.path.join(res_dir, "CLPSO_PER_CONFIG_LOG.csv"),
        ["dataset", "lookback", "horizon"],
        lambda df: df.sort_values(["dataset", "lookback", "horizon"]),
    )
    _merge_csv_shards(
        shard_roots, os.path.join("results", "ABLATION_RESULTS.csv"),
        os.path.join(res_dir, "ABLATION_RESULTS.csv"),
        ["variant"], _sort_ablation,
    )
    _merge_csv_shards(
        shard_roots, os.path.join("results", "CLPSO_CONVERGENCE_LOG.csv"),
        os.path.join(res_dir, "CLPSO_CONVERGENCE_LOG.csv"),
        ["optimizer", "seed", "generation"],
        lambda df: df.sort_values(["optimizer", "seed", "generation"]),
    )
    _merge_csv_shards(
        shard_roots, os.path.join("results", "CLPSO_SENSITIVITY_LOG.csv"),
        os.path.join(res_dir, "CLPSO_SENSITIVITY_LOG.csv"),
        ["swarm_size"],
        lambda df: df.sort_values(["swarm_size"]),
    )

    for sr in shard_roots:
        sp = os.path.join(sr, "predictions")
        if os.path.isdir(sp):
            for fname in sorted(os.listdir(sp)):
                if fname.endswith(".npz"):
                    shutil.copy2(os.path.join(sp, fname), os.path.join(pred_dir, fname))


def _run_dual_gpu_coordinator(
    script_path: str,
    mode: str,
    out_root: str,
    zip_path: str,
    data_dir: Optional[str],
    t0: float,
) -> None:
    print(">>> Initializing Dual-GPU Coordinator: pre-caching ETTh1 & ETTh2 before worker fork...", flush=True)
    _ = load_raw_ot_series("ETTh1", data_dir=data_dir)
    _ = load_raw_ot_series("ETTh2", data_dir=data_dir)

    has_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    tag_prefix = "GPU" if has_multi_gpu else "SHARD"
    print_lock = threading.Lock()
    procs: List[Tuple[int, subprocess.Popen, float]] = []
    readers: List[threading.Thread] = []
    shard_elapsed: Dict[str, float] = {}

    for sid in (0, 1):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if has_multi_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(sid)
        cmd = [
            sys.executable, "-u", os.path.abspath(script_path),
            "--mode", mode,
            "--out_dir", out_root,
            "--role", "worker",
            "--shard_id", str(sid),
        ]
        if data_dir:
            cmd.extend(["--data_dir", data_dir])
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        t_start = time.time()
        procs.append((sid, proc, t_start))

        def _pump(p=proc, s_id=sid):
            assert p.stdout is not None
            for line in iter(p.stdout.readline, ""):
                with print_lock:
                    print(f"[{tag_prefix}{s_id}] {line}", end="", flush=True)
            p.stdout.close()

        th = threading.Thread(target=_pump, daemon=True)
        th.start()
        readers.append(th)

    rcs: Dict[int, int] = {}
    for sid, proc, t_start in procs:
        rcs[sid] = proc.wait()
        shard_elapsed[f"shard_{sid}"] = round(time.time() - t_start, 2)
    for th in readers:
        th.join(timeout=10.0)

    _merge_all_shards(out_root, n_shards=2)

    if rcs[0] != 0 or rcs[1] != 0:
        raise RuntimeError(f"Dual-GPU worker(s) failed with exit codes: {rcs}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_suite4_profile_and_stats(
        models=SUITE1_MODELS,
        device=device,
        out_root=out_root,
        zip_path=zip_path,
        total_elapsed_sec=time.time() - t0,
        extra_manifest={
            "dual_gpu_parallel": bool(has_multi_gpu),
            "sharded_execution": True,
            "per_shard_elapsed_sec": shard_elapsed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NCAA Option A Complete Verification & Benchmark Runner")
    parser.add_argument("--mode", type=str, default=EXECUTION_MODE, choices=["FULL_OPTION_A", "SMOKE_TEST"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--role", type=str, default="auto", choices=["auto", "coordinator", "worker"])
    parser.add_argument("--shard_id", type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument("--force_single_gpu", action="store_true")
    args, _ = parser.parse_known_args()

    mode = args.mode
    is_kaggle = os.path.exists("/kaggle")
    default_out = "/kaggle/working/ncaa_option_a" if is_kaggle else (
        "./ncaa_option_a_smoke" if mode == "SMOKE_TEST" else "./ncaa_option_a"
    )
    out_root = args.out_dir or default_out
    zip_path = (
        "/kaggle/working/ncaa_option_a_results.zip" if (is_kaggle and args.out_dir is None)
        else os.path.join(os.path.dirname(os.path.abspath(out_root)), f"{os.path.basename(out_root)}_results.zip")
    )
    os.makedirs(out_root, exist_ok=True)

    script_path = globals().get("__file__", "")
    has_script_file = bool(script_path) and os.path.isfile(script_path)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    t0 = time.time()

    # Auto-enable Dual-GPU Parallel Sharding on T4 x2 (or explicit --role coordinator)
    if args.role == "coordinator" or (
        args.role == "auto" and not args.force_single_gpu and mode == "FULL_OPTION_A" and n_gpus >= 2 and has_script_file
    ):
        print("=" * 80)
        print(f"  NCAA OPTION A DUAL-GPU COORDINATOR | Mode={mode} | GPUs={n_gpus}")
        print(f"  PyTorch={torch.__version__} | OutDir={out_root} | ZipBundle={zip_path}")
        print("=" * 80, flush=True)
        _run_dual_gpu_coordinator(script_path, mode, out_root, zip_path, args.data_dir, t0)
        print(f"\n>>> ALL SUITES COMPLETE IN {time.time() - t0:.2f}s! ARCHIVE READY: {zip_path}", flush=True)
        return

    if args.role == "auto" and mode == "FULL_OPTION_A" and n_gpus >= 2 and not has_script_file:
        print(
            "[INFO] Detected >=2 GPUs, but running inside a pasted notebook cell (__file__ unavailable).\n"
            "       Proceeding safely on GPU 0 (use `File -> Editor Type -> Script` to unlock 2x T4 speedup!).",
            flush=True,
        )

    if args.role == "worker":
        if args.shard_id not in (0, 1):
            raise ValueError("--shard_id must be 0 or 1 when --role worker")
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        shard_root = os.path.join(out_root, f"shard_{args.shard_id}")
        os.makedirs(shard_root, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("=" * 80)
        print(f"  NCAA OPTION A WORKER SHARD {args.shard_id} | Mode={mode} | Device={device}")
        print(f"  ShardOut={shard_root}")
        print("=" * 80, flush=True)

        if mode == "SMOKE_TEST":
            smoke_cfgs = [("ETTh1", 336, 720), ("ETTh1", 96, 96)]
            worker_cfgs = smoke_cfgs[args.shard_id :: 2]
            run_suite1_benchmark(
                configs=worker_cfgs, models=SUITE1_MODELS, seeds=[2024],
                epochs=1, patience=1, swa_epochs=1, warmup_epochs=1,
                swarm_size=2, swarm_gens=2, pso_eval_epochs=1,
                device=device, out_root=shard_root, zip_path=zip_path, data_dir=args.data_dir, refresh_zip=False,
            )
            if args.shard_id == 0:
                run_suite2_ablation(
                    ds="ETTh1", sl=336, pl=720, seeds=[2024],
                    epochs=1, patience=1, swa_epochs=1, warmup_epochs=1,
                    device=device, out_root=shard_root, zip_path=zip_path, data_dir=args.data_dir, refresh_zip=False,
                )
                run_suite3_swarm_dynamics(
                    ds="ETTh1", sl=336, pl=720, conv_gens=2, sens_sizes=[2], sens_gens=2,
                    pso_eval_epochs=1, device=device, out_root=shard_root, zip_path=zip_path,
                    data_dir=args.data_dir, conv_runs=[("CLPSO", 42, 4), ("Standard_PSO", 42, 4)], refresh_zip=False,
                )
            else:
                run_suite3_swarm_dynamics(
                    ds="ETTh1", sl=336, pl=720, conv_gens=2, sens_sizes=[3], sens_gens=2,
                    pso_eval_epochs=1, device=device, out_root=shard_root, zip_path=zip_path,
                    data_dir=args.data_dir, conv_runs=[("CLPSO", 43, 4), ("LDW_PSO", 42, 4)], refresh_zip=False,
                )
        else:
            target_ds = "ETTh1" if args.shard_id == 0 else "ETTh2"
            worker_cfgs = [c for c in ALL_16_CONFIGS if c[0] == target_ds]
            run_suite1_benchmark(
                configs=worker_cfgs, models=SUITE1_MODELS, seeds=DEFAULT_SEEDS,
                epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE,
                swa_epochs=DEFAULT_SWA_EPOCHS, warmup_epochs=DEFAULT_WARMUP_EPOCHS,
                swarm_size=CLPSO_SWARM_SIZE, swarm_gens=CLPSO_GENS, pso_eval_epochs=CLPSO_EVAL_EPOCHS,
                device=device, out_root=shard_root, zip_path=zip_path, data_dir=args.data_dir, refresh_zip=False,
            )
            if args.shard_id == 0:
                run_suite2_ablation(
                    ds="ETTh1", sl=336, pl=720, seeds=DEFAULT_SEEDS,
                    epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE,
                    swa_epochs=DEFAULT_SWA_EPOCHS, warmup_epochs=DEFAULT_WARMUP_EPOCHS,
                    device=device, out_root=shard_root, zip_path=zip_path, data_dir=args.data_dir, refresh_zip=False,
                )
            else:
                run_suite3_swarm_dynamics(
                    ds="ETTh1", sl=336, pl=720, conv_gens=10, sens_sizes=[5, 10, 20, 40], sens_gens=5,
                    pso_eval_epochs=CLPSO_EVAL_EPOCHS, device=device, out_root=shard_root, zip_path=zip_path,
                    data_dir=args.data_dir, refresh_zip=False,
                )
        return

    # Single-GPU / Single-process execution path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  NCAA OPTION A UNIFIED SUITE | Mode={mode} | Device={device}")
    print(f"  PyTorch={torch.__version__} | OutDir={out_root}")
    print(f"  ZipBundle={zip_path}")
    print("=" * 80, flush=True)

    if mode == "SMOKE_TEST":
        smoke_cfgs = [("ETTh1", 336, 720), ("ETTh1", 96, 96)]
        run_suite1_benchmark(
            configs=smoke_cfgs, models=SUITE1_MODELS, seeds=[2024, 2025],
            epochs=2, patience=2, swa_epochs=1, warmup_epochs=1,
            swarm_size=3, swarm_gens=2, pso_eval_epochs=1,
            device=device, out_root=out_root, zip_path=zip_path, data_dir=args.data_dir,
        )
        run_suite2_ablation(
            ds="ETTh1", sl=336, pl=720, seeds=[2024, 2025],
            epochs=2, patience=2, swa_epochs=1, warmup_epochs=1,
            device=device, out_root=out_root, zip_path=zip_path, data_dir=args.data_dir,
        )
        run_suite3_swarm_dynamics(
            ds="ETTh1", sl=336, pl=720, conv_gens=2, sens_sizes=[3, 5], sens_gens=2,
            pso_eval_epochs=1, device=device, out_root=out_root, zip_path=zip_path, data_dir=args.data_dir,
        )
        run_suite4_profile_and_stats(
            models=SUITE1_MODELS, device=device, out_root=out_root, zip_path=zip_path,
            total_elapsed_sec=time.time() - t0,
        )
    else:
        run_suite1_benchmark(
            configs=ALL_16_CONFIGS, models=SUITE1_MODELS, seeds=DEFAULT_SEEDS,
            epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE,
            swa_epochs=DEFAULT_SWA_EPOCHS, warmup_epochs=DEFAULT_WARMUP_EPOCHS,
            swarm_size=CLPSO_SWARM_SIZE, swarm_gens=CLPSO_GENS, pso_eval_epochs=CLPSO_EVAL_EPOCHS,
            device=device, out_root=out_root, zip_path=zip_path, data_dir=args.data_dir,
        )
        run_suite2_ablation(
            ds="ETTh1", sl=336, pl=720, seeds=DEFAULT_SEEDS,
            epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE,
            swa_epochs=DEFAULT_SWA_EPOCHS, warmup_epochs=DEFAULT_WARMUP_EPOCHS,
            device=device, out_root=out_root, zip_path=zip_path, data_dir=args.data_dir,
        )
        run_suite3_swarm_dynamics(
            ds="ETTh1", sl=336, pl=720, conv_gens=10, sens_sizes=[5, 10, 20, 40], sens_gens=5,
            pso_eval_epochs=CLPSO_EVAL_EPOCHS, device=device, out_root=out_root, zip_path=zip_path, data_dir=args.data_dir,
        )
        run_suite4_profile_and_stats(
            models=SUITE1_MODELS, device=device, out_root=out_root, zip_path=zip_path,
            total_elapsed_sec=time.time() - t0,
        )

    print(f"\n>>> ALL SUITES COMPLETE IN {time.time() - t0:.2f}s! ARCHIVE READY: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
