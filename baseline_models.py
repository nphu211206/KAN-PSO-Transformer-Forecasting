"""Forecasting baselines used by the targeted follow-up experiments.

Linear, NLinear, and DLinear follow the model definitions from the official
LTSF-Linear implementation.  PatchTST is loaded from Hugging Face Transformers
instead of maintaining an unverified local reimplementation.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class RepeatLast(nn.Module):
    """Parameter-free persistence forecast."""

    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:].repeat(1, self.pred_len)


class LinearForecast(nn.Module):
    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NLinearForecast(nn.Module):
    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:].detach()
        return self.linear(x - last) + last


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        left = x[:, :1].repeat(1, padding)
        right = x[:, -1:].repeat(1, padding)
        padded = torch.cat([left, x, right], dim=1).unsqueeze(1)
        return self.pool(padded).squeeze(1)


class DLinearForecast(nn.Module):
    """Univariate DLinear with a centered moving-average decomposition."""

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        seasonal = x - trend
        return self.seasonal(seasonal) + self.trend(trend)


class DLinearKANResidual(nn.Module):
    """DLinear forecast augmented by a validation-gated KAN residual."""

    def __init__(
        self,
        dlinear: DLinearForecast,
        residual_model: nn.Module,
        *,
        freeze_dlinear: bool,
        initial_residual_gain: float = 0.0,
    ) -> None:
        super().__init__()
        self.dlinear = dlinear
        self.residual_model = residual_model
        self.residual_gain = nn.Parameter(
            torch.tensor(float(initial_residual_gain), dtype=torch.float32)
        )
        self.freeze_dlinear = bool(freeze_dlinear)
        if self.freeze_dlinear:
            for parameter in self.dlinear.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        base = self.dlinear(x)
        residual = self.residual_model(x, update_grid=update_grid)
        return base + self.residual_gain * residual

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        return self.residual_model.regularization_loss(
            regularize_activation, regularize_entropy
        )


class HuggingFacePatchTST(nn.Module):
    """Thin adapter around ``transformers.PatchTSTForPrediction``."""

    def __init__(self, seq_len: int, pred_len: int, config: Dict[str, Any]) -> None:
        super().__init__()
        try:
            from transformers import PatchTSTConfig, PatchTSTForPrediction
        except ImportError as exc:
            raise RuntimeError(
                "PatchTST requires transformers. Install transformers==4.48.3."
            ) from exc

        model_config = PatchTSTConfig(
            num_input_channels=1,
            context_length=int(seq_len),
            prediction_length=int(pred_len),
            patch_length=int(config.get("patch_length", 16)),
            patch_stride=int(config.get("patch_stride", 8)),
            d_model=int(config.get("d_model", 64)),
            num_hidden_layers=int(config.get("num_hidden_layers", 3)),
            num_attention_heads=int(config.get("num_attention_heads", 4)),
            ffn_dim=int(config.get("ffn_dim", 128)),
            dropout=float(config.get("dropout", 0.1)),
            head_dropout=float(config.get("head_dropout", 0.1)),
            scaling=config.get("scaling", None),
            loss="mse",
        )
        self.model = PatchTSTForPrediction(model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(past_values=x.unsqueeze(-1))
        prediction = output.prediction_outputs
        if prediction.ndim == 3 and prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        return prediction


BASELINE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "RepeatLast": {"trainable": False},
    "Linear": {"lr": 1e-3, "weight_decay": 0.0},
    "NLinear": {"lr": 1e-3, "weight_decay": 0.0},
    "DLinear": {"lr": 1e-3, "weight_decay": 0.0, "kernel_size": 25},
    "PatchTST": {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patch_length": 16,
        "patch_stride": 8,
        "d_model": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "ffn_dim": 128,
        "dropout": 0.1,
        "head_dropout": 0.1,
        "scaling": None,
    },
}


def build_baseline(name: str, seq_len: int, pred_len: int, config: Dict[str, Any]) -> nn.Module:
    if name == "RepeatLast":
        return RepeatLast(pred_len)
    if name == "Linear":
        return LinearForecast(seq_len, pred_len)
    if name == "NLinear":
        return NLinearForecast(seq_len, pred_len)
    if name == "DLinear":
        return DLinearForecast(seq_len, pred_len, int(config.get("kernel_size", 25)))
    if name == "PatchTST":
        return HuggingFacePatchTST(seq_len, pred_len, config)
    raise ValueError(f"Unknown baseline model: {name}")
