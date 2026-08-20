"""Low-cost follow-up experiments for the KAN HPO paper.

This runner does not repeat the six-algorithm HPO campaign.  It adds:

1. Fair forecasting baselines under the exact historical 60/20/20 protocol.
2. A controlled Grid Update ablation using one validation-selected KAN config.

Every task is checkpointed as an individual JSON file and can be resumed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline_models import BASELINE_DEFAULTS, DLinearKANResidual, build_baseline
from experiment_protocols import (
    PROTOCOL_PILOT,
    SUPPORTED_PROTOCOLS,
    build_protocol_loaders,
    file_sha256,
    get_input_dim,
)


LOGGER = logging.getLogger("targeted_experiments")
PROJECT_ROOT = Path(__file__).resolve().parent
EFFICIENT_KAN_COMMIT = "7b6ce1c87f18c8bc90c208f6b494042344216b11"
DEFAULT_SEEDS = (1000, 1001, 1002)
DEFAULT_HORIZONS = (96, 192, 336, 720)
DEFAULT_MODELS = (
    "RepeatLast",
    "Linear",
    "NLinear",
    "DLinear",
    "PatchTST",
    "KAN_Vanilla",
    "KAN_WOA_Selected",
)

# Frozen from WOA_run02.json.  Selection rule: lowest HPO validation fitness
# across the historical campaign, without looking at test MSE.
WOA_SELECTED_CONFIG: Dict[str, Any] = {
    "grid_size": 3,
    "spline_order": 2,
    "grid_eps": 0.011726312650002712,
    "warmup_epochs": 0,
    "update_every": 0,
    "hidden_size": 8,
    "n_layers": 1,
    "seq_len": 96,
    "lr": 0.0012595329117212266,
    "weight_decay": 0.00016360005892202873,
    "batch_size": 16,
    "scale_noise": 0.05046546573371035,
    "scale_base": 0.2018618629348414,
    "scale_spline": 0.2018618629348414,
    "reg_lambda": 9.403346617567799e-05,
}

VANILLA_KAN_CONFIG: Dict[str, Any] = {
    "grid_size": 5,
    "spline_order": 3,
    "grid_eps": 0.02,
    "warmup_epochs": 0,
    "update_every": 0,
    "hidden_size": 64,
    "n_layers": 1,
    "seq_len": 96,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "reg_lambda": 1e-3,
}

# Best validation configuration among historical records that actually enabled
# grid updates.  Using this for the ablation avoids structurally favoring the
# no-update condition selected by WOA_run02.
GRID_ABLATION_CONFIG: Dict[str, Any] = {
    "grid_size": 3,
    "spline_order": 3,
    "grid_eps": 0.11606629207416591,
    "warmup_epochs": 5,
    "update_every": 1,
    "hidden_size": 10,
    "n_layers": 1,
    "seq_len": 96,
    "lr": 0.020632987341920997,
    "weight_decay": 3.2593146144503852e-05,
    "batch_size": 32,
    "scale_noise": 0.22350508394019297,
    "scale_base": 1.684736410071365,
    "scale_spline": 0.864612822959884,
    "reg_lambda": 0.009332961380394173,
}

GRID_SCHEDULES: Dict[str, Dict[str, int]] = {
    "none": {"warmup_epochs": 0, "update_every": 0},
    "conservative": {"warmup_epochs": 2, "update_every": 2},
    "moderate": {"warmup_epochs": 5, "update_every": 1},
    "aggressive": {"warmup_epochs": 10, "update_every": 1},
}

OPTIMIZER_STATE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "none_retain": {"schedule": "none", "reset_optimizer_state": False},
    "conservative_retain": {
        "schedule": "conservative",
        "reset_optimizer_state": False,
    },
    "moderate_retain": {
        "schedule": "moderate",
        "reset_optimizer_state": False,
    },
    "moderate_reset": {
        "schedule": "moderate",
        "reset_optimizer_state": True,
    },
    "aggressive_retain": {
        "schedule": "aggressive",
        "reset_optimizer_state": False,
    },
    "aggressive_reset": {
        "schedule": "aggressive",
        "reset_optimizer_state": True,
    },
}

# A compact, validation-only improvement matrix.  These variants do not launch
# another metaheuristic search and never select a model using test performance.
IMPROVEMENT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "KAN_WOA_CurrentControl": {
        "kind": "direct_kan",
        "seq_len": 96,
        "use_regularization": False,
        "best_checkpoint": False,
        "gradient_clip_norm": None,
    },
    "KAN_WOA_Reg": {
        "kind": "direct_kan",
        "seq_len": 96,
        "use_regularization": True,
        "best_checkpoint": False,
        "gradient_clip_norm": None,
    },
    "KAN_WOA_RegBest": {
        "kind": "direct_kan",
        "seq_len": 96,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": None,
    },
    "KAN_WOA_RegBestClip": {
        "kind": "direct_kan",
        "seq_len": 96,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "DLinear_KAN_Frozen_s96": {
        "kind": "residual_kan",
        "seq_len": 96,
        "freeze_dlinear": True,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "DLinear_KAN_Joint_s96": {
        "kind": "residual_kan",
        "seq_len": 96,
        "freeze_dlinear": False,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "DLinear_KAN_Frozen_s336": {
        "kind": "residual_kan",
        "seq_len": 336,
        "freeze_dlinear": True,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "DLinear_KAN_Joint_s336": {
        "kind": "residual_kan",
        "seq_len": 336,
        "freeze_dlinear": False,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "KAN_WOA_RegBestClip_s336": {
        "kind": "direct_kan",
        "seq_len": 336,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "DLinear_KAN_ResidualTarget_s96": {
        "kind": "residual_target",
        "seq_len": 96,
        "freeze_dlinear": True,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
    "DLinear_KAN_ResidualTarget_s336": {
        "kind": "residual_target",
        "seq_len": 336,
        "freeze_dlinear": True,
        "use_regularization": True,
        "best_checkpoint": True,
        "gradient_clip_norm": 1.0,
    },
}

_KAN_CLASS = None
_KAN_LINEAR_CLASS = None
_KAN_IMPLEMENTATION: Dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata(device: torch.device) -> dict:
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit_learn": _package_version("scikit-learn"),
        "transformers": _package_version("transformers"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def source_manifest() -> dict:
    files = [
        PROJECT_ROOT / "targeted_experiments.py",
        PROJECT_ROOT / "experiment_protocols.py",
        PROJECT_ROOT / "baseline_models.py",
    ]
    hashes = {
        path.name: file_sha256(path)
        for path in files
        if path.is_file()
    }
    joined = "".join(f"{name}:{value}\n" for name, value in sorted(hashes.items()))
    return {
        "files": hashes,
        "combined_sha256": sha256(joined.encode("utf-8")).hexdigest(),
    }


def find_or_download_ett_root(explicit_root: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.extend(
        [
            PROJECT_ROOT / "ETDataset" / "ETT-small",
            Path("/kaggle/input/ett-small/ETT-small"),
            Path("/kaggle/input/etdataset/ETT-small"),
            Path("/kaggle/input/ett-dataset/ETT-small"),
            Path("/kaggle/working/ETDataset/ETT-small"),
        ]
    )
    for candidate in candidates:
        if (candidate / "ETTh1.csv").is_file():
            return candidate.resolve()

    if not Path("/kaggle/working").exists():
        raise FileNotFoundError(
            "ETT data not found. Pass --data-root pointing to ETT-small."
        )

    destination = Path("/kaggle/working/ETDataset/ETT-small")
    destination.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
    for name in ("ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"):
        target = destination / name
        if not target.is_file():
            LOGGER.info("Downloading %s", name)
            urllib.request.urlretrieve(f"{base_url}/{name}", target)
    return destination.resolve()


def ensure_transformers(allow_install: bool) -> None:
    required_version = "4.48.3"
    installed_version = _package_version("transformers")
    if installed_version == required_version:
        return
    if not allow_install:
        raise RuntimeError(
            "PatchTST requires transformers==4.48.3; found "
            f"{installed_version or 'no installation'}. Use --install-deps."
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            f"transformers=={required_version}",
        ]
    )
    if _package_version("transformers") != required_version:
        raise RuntimeError(
            f"Failed to establish transformers=={required_version} for PatchTST"
        )


def load_historical_kan(allow_install: bool):
    """Load efficient-kan and apply the same constructor patch as the old run."""
    global _KAN_CLASS, _KAN_LINEAR_CLASS, _KAN_IMPLEMENTATION
    if _KAN_CLASS is not None:
        return _KAN_CLASS

    local_src = PROJECT_ROOT / "efficient-kan" / "src"
    source_kind = "local_checkout"
    if local_src.is_dir():
        sys.path.insert(0, str(local_src))
    else:
        source_kind = "pinned_pip_commit"
        if not allow_install:
            raise RuntimeError(
                "efficient-kan is unavailable. Use --install-deps on Kaggle."
            )
        package = (
            "git+https://github.com/Blealtan/efficient-kan.git@"
            f"{EFFICIENT_KAN_COMMIT}"
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    from efficient_kan import KAN as EfficientKAN
    from efficient_kan.kan import KANLinear

    def patched_init(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=(-4, 4),
        enable_standalone_scale_spline=False,
    ):
        nn.Module.__init__(self)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    EfficientKAN.__init__ = patched_init
    _KAN_CLASS = EfficientKAN
    _KAN_LINEAR_CLASS = KANLinear
    kan_source = Path(sys.modules[KANLinear.__module__].__file__).resolve()
    _KAN_IMPLEMENTATION = {
        "upstream_repository": "https://github.com/Blealtan/efficient-kan",
        "intended_upstream_commit": EFFICIENT_KAN_COMMIT,
        "source_kind": source_kind,
        "loaded_source_path": str(kan_source),
        "loaded_source_sha256": file_sha256(kan_source),
        "constructor_patch": {
            "grid_range": [-4, 4],
            "enable_standalone_scale_spline": False,
        },
    }
    return _KAN_CLASS


def validate_frozen_config(record_name: str, expected_config: Dict[str, Any]) -> dict:
    historical = (
        PROJECT_ROOT
        / "my_results"
        / "kaggle"
        / "working"
        / "experiments"
        / "results"
        / f"{record_name}.json"
    )
    expected_fitness = {
        "WOA_run02": 0.143690080432554,
        "GWO_run00": 0.15178766484648634,
    }[record_name]
    result = {
        "selected_record": record_name,
        "historical_hpo_fitness": expected_fitness,
        "embedded_config": copy.deepcopy(expected_config),
        "local_record_checked": False,
    }
    if historical.is_file():
        with historical.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("best_config") != expected_config:
            raise RuntimeError(f"Embedded config differs from {record_name}.json")
        result["local_record_checked"] = True
        result["historical_record_sha256"] = file_sha256(historical)
    return result


def build_kan(config: Dict[str, Any], pred_len: int, device: torch.device, allow_install: bool):
    kan_class = load_historical_kan(allow_install)
    input_dim = get_input_dim(config["seq_len"], "S")
    width = [input_dim] + [config["hidden_size"]] * config["n_layers"] + [pred_len]
    return kan_class(
        layers_hidden=width,
        grid_size=config["grid_size"],
        spline_order=config["spline_order"],
        scale_noise=config["scale_noise"],
        scale_base=config["scale_base"],
        scale_spline=config["scale_spline"],
        grid_eps=config["grid_eps"],
    ).to(device)


def count_parameters(model: nn.Module) -> dict:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    samples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        prediction = model(x_batch)
        if prediction.shape != y_batch.shape:
            raise RuntimeError(
                f"Prediction shape {tuple(prediction.shape)} != target {tuple(y_batch.shape)}"
            )
        delta = prediction - y_batch
        squared_error += delta.square().sum().item()
        absolute_error += delta.abs().sum().item()
        elements += delta.numel()
        samples += len(x_batch)
    if elements == 0:
        raise RuntimeError("Evaluation loader produced no elements")
    return {
        "mse": squared_error / elements,
        "mae": absolute_error / elements,
        "samples": samples,
        "elements": elements,
    }


@torch.no_grad()
def evaluate_with_predictions(
    model: nn.Module, loader, device: torch.device
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate once while retaining arrays and end-to-end inference timing."""
    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        prediction = model(x_batch)
        if prediction.shape != y_batch.shape:
            raise RuntimeError(
                f"Prediction shape {tuple(prediction.shape)} != target "
                f"{tuple(y_batch.shape)}"
            )
        predictions.append(prediction.detach().cpu().numpy().astype(np.float32))
        targets.append(y_batch.detach().cpu().numpy().astype(np.float32))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started
    if not predictions:
        raise RuntimeError("Evaluation loader produced no elements")
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    delta = y_pred.astype(np.float64) - y_true.astype(np.float64)
    metrics = {
        "mse": float(np.mean(np.square(delta))),
        "mae": float(np.mean(np.abs(delta))),
        "samples": int(y_true.shape[0]),
        "elements": int(y_true.size),
        "inference_elapsed_s": float(elapsed_s),
        "inference_ms_per_sample": float(1000.0 * elapsed_s / y_true.shape[0]),
        "inference_measurement_scope": (
            "single full test-loader pass including loader overhead, "
            "host-to-device transfer and prediction copy to host"
        ),
    }
    return metrics, y_true, y_pred


def _model_is_finite(model: nn.Module) -> bool:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            return False
    return True


def _state_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> dict:
    total_delta_sq = total_before_sq = 0.0
    grid_delta_sq = spline_delta_sq = base_delta_sq = 0.0
    changed_tensors = 0
    for name, before_tensor in before.items():
        after_tensor = after[name]
        if not before_tensor.is_floating_point():
            continue
        delta_sq = (after_tensor - before_tensor).double().square().sum().item()
        total_delta_sq += delta_sq
        total_before_sq += before_tensor.double().square().sum().item()
        changed_tensors += delta_sq > 0.0
        if name.endswith(".grid"):
            grid_delta_sq += delta_sq
        elif name.endswith(".spline_weight") or name.endswith(".spline_scaler"):
            spline_delta_sq += delta_sq
        elif name.endswith(".base_weight"):
            base_delta_sq += delta_sq
    total_l2 = math.sqrt(total_delta_sq)
    before_l2 = math.sqrt(total_before_sq)
    return {
        "changed_state_tensors": int(changed_tensors),
        "state_delta_l2": total_l2,
        "state_relative_delta_l2": total_l2 / before_l2 if before_l2 else None,
        "grid_delta_l2": math.sqrt(grid_delta_sq),
        "spline_parameter_delta_l2": math.sqrt(spline_delta_sq),
        "base_weight_delta_l2": math.sqrt(base_delta_sq),
    }


def _optimizer_state_summary(optimizer: torch.optim.Optimizer) -> dict:
    exp_avg_sq_sum = exp_avg_sum = 0.0
    parameter_states = 0
    optimizer_tensors = 0
    for state in optimizer.state.values():
        if state:
            parameter_states += 1
        for key, value in state.items():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            optimizer_tensors += 1
            if key == "exp_avg":
                exp_avg_sum += value.double().square().sum().item()
            elif key == "exp_avg_sq":
                exp_avg_sq_sum += value.double().square().sum().item()
    return {
        "parameter_states": parameter_states,
        "floating_state_tensors": optimizer_tensors,
        "adam_exp_avg_l2": math.sqrt(exp_avg_sum),
        "adam_exp_avg_sq_l2": math.sqrt(exp_avg_sq_sum),
    }


@torch.no_grad()
def safe_grid_update(
    model: nn.Module,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    reset_optimizer_state: bool = False,
) -> dict:
    """Apply one transactional grid update and record its immediate disruption."""
    criterion = nn.MSELoss()
    model.eval()
    before = criterion(model(probe_x), probe_y).item()
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    event = {
        "epoch_zero_based": int(epoch),
        "attempted": True,
        "success": False,
        "probe_mse_before": before,
        "optimizer_state_before": _optimizer_state_summary(optimizer),
    }
    try:
        model(probe_x, update_grid=True)
        if not _model_is_finite(model):
            raise FloatingPointError("non-finite parameter or buffer after grid update")
        after = criterion(model(probe_x), probe_y).item()
        if not math.isfinite(after):
            raise FloatingPointError("non-finite probe loss after grid update")
        event.update(
            {
                "success": True,
                "probe_mse_after": after,
                "probe_mse_delta": after - before,
                "state_change": _state_delta(snapshot, model.state_dict()),
                "optimizer_state_after_grid_update": _optimizer_state_summary(optimizer),
            }
        )
        if reset_optimizer_state:
            optimizer.state.clear()
        event["optimizer_state_reset_after_grid_update"] = bool(
            reset_optimizer_state
        )
        event["optimizer_state_after_reset"] = _optimizer_state_summary(optimizer)
    except Exception as exc:
        model.load_state_dict(snapshot)
        event.update(
            {
                "rolled_back": True,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )
    event["elapsed_s"] = time.perf_counter() - started
    return event


def collect_probe(loader, device: torch.device, max_samples: int = 256):
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    total = 0
    for x_batch, y_batch in loader:
        take = min(len(x_batch), max_samples - total)
        xs.append(x_batch[:take])
        ys.append(y_batch[:take])
        total += take
        if total >= max_samples:
            break
    if not xs:
        raise RuntimeError("Training loader is empty")
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grid_schedule: Optional[Dict[str, int]] = None,
    *,
    kan_regularization_lambda: float = 0.0,
    restore_best_validation: bool = False,
    gradient_clip_norm: Optional[float] = None,
    reset_optimizer_after_grid_update: bool = False,
    include_initial_validation_checkpoint: bool = False,
    activate_residual_after_initial_checkpoint: bool = False,
) -> dict:
    if count_parameters(model)["trainable"] == 0:
        return {"epochs": [], "grid_updates": [], "selection_rule": "parameter_free"}

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=float(learning_rate) * 0.01
    )
    criterion = nn.MSELoss()
    probe_x, probe_y = collect_probe(train_loader, device)
    history: List[dict] = []
    grid_events: List[dict] = []
    best_val_mse = math.inf
    best_epoch: Optional[int] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    initial_validation: Optional[dict] = None
    if include_initial_validation_checkpoint:
        initial_validation = evaluate(model, val_loader, device)
        best_val_mse = float(initial_validation["mse"])
        best_epoch = 0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    if activate_residual_after_initial_checkpoint:
        residual_gain = getattr(model, "residual_gain", None)
        if residual_gain is None:
            raise TypeError(
                "Residual activation requested for a model without residual_gain"
            )
        with torch.no_grad():
            residual_gain.fill_(1.0)
        residual_gain.requires_grad_(False)

    for epoch in range(int(epochs)):
        if grid_schedule is not None:
            warmup = int(grid_schedule["warmup_epochs"])
            every = int(grid_schedule["update_every"])
            should_update = epoch < warmup and (
                every == 0 or epoch % max(every, 1) == 0
            )
            if should_update:
                event = safe_grid_update(
                    model,
                    probe_x,
                    probe_y,
                    epoch,
                    optimizer,
                    reset_optimizer_state=reset_optimizer_after_grid_update,
                )
                grid_events.append(event)
                if not event["success"]:
                    LOGGER.warning(
                        "Grid update failed at epoch %d: %s: %s",
                        epoch,
                        event.get("exception_type"),
                        event.get("exception_message"),
                    )

        model.train()
        mse_sum = 0.0
        objective_sum = 0.0
        regularization_sum = 0.0
        sample_count = 0
        skipped_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            mse = criterion(prediction, y_batch)
            regularization = torch.zeros((), device=device)
            if kan_regularization_lambda > 0.0:
                regularization_fn = getattr(model, "regularization_loss", None)
                if regularization_fn is None:
                    raise TypeError(
                        "KAN regularization requested for a model without "
                        "regularization_loss()"
                    )
                regularization = regularization_fn(
                    kan_regularization_lambda, kan_regularization_lambda
                )
            loss = mse + regularization
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(gradient_clip_norm)
                )
            optimizer.step()
            mse_sum += mse.item() * len(x_batch)
            objective_sum += loss.item() * len(x_batch)
            regularization_sum += regularization.item() * len(x_batch)
            sample_count += len(x_batch)
        scheduler.step()
        if sample_count == 0:
            raise FloatingPointError(f"All training batches were non-finite at epoch {epoch}")
        validation = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_mse_batch_mean": mse_sum / sample_count,
            "train_objective_batch_mean": objective_sum / sample_count,
            "train_regularization_batch_mean": regularization_sum / sample_count,
            "val_mse": validation["mse"],
            "val_mae": validation["mae"],
            "lr": scheduler.get_last_lr()[0],
            "skipped_nonfinite_batches": skipped_batches,
        }
        history.append(row)
        if row["val_mse"] < best_val_mse:
            best_val_mse = float(row["val_mse"])
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        LOGGER.info(
            "%s epoch %02d/%02d train=%.6f val=%.6f",
            model_name,
            epoch + 1,
            epochs,
            row["train_mse_batch_mean"],
            row["val_mse"],
        )

    final_epoch_val_mse = float(history[-1]["val_mse"])
    if restore_best_validation:
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint was captured")
        model.load_state_dict(best_state)

    return {
        "epochs": history,
        "grid_updates": grid_events,
        "selection_rule": (
            "minimum_validation_mse_checkpoint"
            if restore_best_validation
            else "final_epoch_to_match_historical_kaggle_evaluation"
        ),
        "selected_epoch": int(best_epoch) if restore_best_validation else int(epochs),
        "best_validation_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "initial_validation_metrics": initial_validation,
        "final_epoch_validation_mse": final_epoch_val_mse,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "gradient_clipping": gradient_clip_norm,
        "explicit_kan_regularization_in_final_training": bool(
            kan_regularization_lambda > 0.0
        ),
        "kan_regularization_lambda": float(kan_regularization_lambda),
        "optimizer_state_reset_after_grid_update": bool(
            reset_optimizer_after_grid_update
        ),
        "residual_activated_after_epoch_zero_checkpoint": bool(
            activate_residual_after_initial_checkpoint
        ),
    }


def atomic_write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=True)
    temporary.replace(path)


def task_path(results_dir: Path, task_id: str) -> Path:
    return results_dir / "records" / f"{task_id}.json"


def task_complete(results_dir: Path, task_id: str) -> bool:
    path = task_path(results_dir, task_id)
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def run_task(
    *,
    experiment: str,
    variant: str,
    dataset: str,
    horizon: int,
    seed: int,
    protocol: str,
    epochs: int,
    data_root: Path,
    results_dir: Path,
    device: torch.device,
    allow_install: bool,
    reset_optimizer_after_grid_update: bool = False,
    uniform_best_checkpoint: bool = False,
    save_diagnostics: bool = False,
    campaign_id: Optional[str] = None,
    smoke: bool = False,
) -> dict:
    optimizer_state_spec = None
    if experiment == "optimizer_state":
        if variant not in OPTIMIZER_STATE_VARIANTS:
            raise ValueError(f"Unknown optimizer-state variant: {variant}")
        optimizer_state_spec = copy.deepcopy(OPTIMIZER_STATE_VARIANTS[variant])
        reset_optimizer_after_grid_update = bool(
            optimizer_state_spec["reset_optimizer_state"]
        )
    task_parts = [
        experiment,
        variant,
        dataset,
        f"h{horizon}",
        f"s{seed}",
        protocol,
    ]
    if campaign_id:
        task_parts.append(campaign_id)
    task_id = "__".join(task_parts)
    if task_complete(results_dir, task_id):
        LOGGER.info("SKIP completed task %s", task_id)
        return {"task_id": task_id, "status": "skipped"}

    started = time.time()
    record: Dict[str, Any] = {
        "schema_version": (
            "targeted_experiment_v4"
            if uniform_best_checkpoint or save_diagnostics or campaign_id
            else (
                "targeted_experiment_v3"
                if experiment == "optimizer_state"
                else (
                    "targeted_experiment_v2"
                    if experiment == "improvement"
                    else "targeted_experiment_v1"
                )
            )
        ),
        "task_id": task_id,
        "status": "running",
        "started_utc": utc_now(),
        "experiment": experiment,
        "variant": variant,
        "dataset": dataset,
        "horizon": int(horizon),
        "seed": int(seed),
        "protocol_name": protocol,
        "epochs": int(epochs),
        "device": environment_metadata(device),
        "source": source_manifest(),
        "campaign_id": campaign_id,
        "uniform_best_checkpoint": bool(uniform_best_checkpoint),
        "save_diagnostics": bool(save_diagnostics),
        "reset_optimizer_after_grid_update": bool(
            reset_optimizer_after_grid_update
        ),
    }
    if optimizer_state_spec is not None:
        record["optimizer_state_protocol"] = copy.deepcopy(optimizer_state_spec)
    atomic_write_json(task_path(results_dir, task_id), record)

    try:
        set_seed(seed)
        improvement_spec = None
        seq_len = 96
        if experiment == "improvement":
            if variant not in IMPROVEMENT_VARIANTS:
                raise ValueError(f"Unknown improvement variant: {variant}")
            improvement_spec = copy.deepcopy(IMPROVEMENT_VARIANTS[variant])
            seq_len = int(improvement_spec["seq_len"])
            config = copy.deepcopy(WOA_SELECTED_CONFIG)
            record["kan_selection"] = validate_frozen_config(
                "WOA_run02", WOA_SELECTED_CONFIG
            )
            record["kan_selection"]["selection_rule"] = (
                "historical WOA validation-selected configuration; seq_len is "
                "explicitly transferred or changed only by the declared variant"
            )
            config["seq_len"] = seq_len
            batch_size = int(config["batch_size"])
            record["improvement_protocol"] = improvement_spec
        elif experiment == "baseline" and variant in BASELINE_DEFAULTS:
            config = copy.deepcopy(BASELINE_DEFAULTS[variant])
            batch_size = 64
        else:
            if experiment in {"ablation", "optimizer_state"}:
                config = copy.deepcopy(GRID_ABLATION_CONFIG)
                record["kan_selection"] = validate_frozen_config(
                    "GWO_run00", GRID_ABLATION_CONFIG
                )
                record["kan_selection"]["selection_rule"] = (
                    "minimum historical HPO validation fitness among configurations "
                    "with warmup_epochs > 0"
                )
            elif variant == "KAN_Vanilla":
                config = copy.deepcopy(VANILLA_KAN_CONFIG)
                record["kan_selection"] = {
                    "selection_rule": "fixed efficient-kan-style default baseline; no HPO",
                    "selected_record": None,
                    "embedded_config": copy.deepcopy(VANILLA_KAN_CONFIG),
                }
            else:
                config = copy.deepcopy(WOA_SELECTED_CONFIG)
                record["kan_selection"] = validate_frozen_config(
                    "WOA_run02", WOA_SELECTED_CONFIG
                )
                record["kan_selection"]["selection_rule"] = (
                    "minimum historical HPO validation fitness across all algorithms"
                )
            batch_size = int(config["batch_size"])

        train_loader, val_loader, test_loader, scaler, protocol_metadata = (
            build_protocol_loaders(
                data_root,
                dataset,
                seq_len=seq_len,
                pred_len=horizon,
                features="S",
                batch_size=batch_size,
                protocol=protocol,
                seed=seed,
            )
        )
        if smoke:
            epochs = 1
            record["smoke_test"] = True
            record["epochs"] = 1

        grid_schedule = None
        if experiment == "baseline" and variant in BASELINE_DEFAULTS:
            if variant == "PatchTST":
                ensure_transformers(allow_install)
            model = build_baseline(variant, seq_len, horizon, config).to(device)
            learning_rate = float(config.get("lr", 0.0))
            weight_decay = float(config.get("weight_decay", 0.0))
        elif experiment == "improvement":
            grid_schedule = {
                "warmup_epochs": int(config["warmup_epochs"]),
                "update_every": int(config["update_every"]),
            }
            learning_rate = float(config["lr"])
            weight_decay = float(config["weight_decay"])
            if improvement_spec["kind"] in {"residual_kan", "residual_target"}:
                # Pretrain DLinear under the same protocol, then let the KAN
                # residual start from an exact DLinear forecast (gain = 0).
                base_train_loader, base_val_loader, base_test_loader, _, _ = (
                    build_protocol_loaders(
                        data_root,
                        dataset,
                        seq_len=seq_len,
                        pred_len=horizon,
                        features="S",
                        batch_size=64,
                        protocol=protocol,
                        seed=seed,
                    )
                )
                base_model = build_baseline(
                    "DLinear", seq_len, horizon, BASELINE_DEFAULTS["DLinear"]
                ).to(device)
                record["base_model_config"] = copy.deepcopy(
                    BASELINE_DEFAULTS["DLinear"]
                )
                record["base_parameter_count"] = count_parameters(base_model)
                record["base_training"] = train_model(
                    base_model,
                    f"{variant}__DLinearPretrain",
                    base_train_loader,
                    base_val_loader,
                    device,
                    epochs,
                    float(BASELINE_DEFAULTS["DLinear"]["lr"]),
                    float(BASELINE_DEFAULTS["DLinear"]["weight_decay"]),
                    None,
                    restore_best_validation=True,
                )
                record["base_validation_metrics"] = evaluate(
                    base_model, base_val_loader, device
                )
                record["base_test_metrics"] = evaluate(
                    base_model, base_test_loader, device
                )
                residual_kan = build_kan(config, horizon, device, allow_install)
                model = DLinearKANResidual(
                    base_model,
                    residual_kan,
                    freeze_dlinear=bool(improvement_spec["freeze_dlinear"]),
                    initial_residual_gain=0.0,
                ).to(device)
                if improvement_spec["kind"] == "residual_target":
                    model.residual_gain.requires_grad_(False)
            else:
                model = build_kan(config, horizon, device, allow_install)
            record["kan_implementation"] = copy.deepcopy(_KAN_IMPLEMENTATION)
        else:
            if experiment in {"ablation", "optimizer_state"}:
                schedule_name = (
                    optimizer_state_spec["schedule"]
                    if optimizer_state_spec is not None
                    else variant
                )
                if schedule_name not in GRID_SCHEDULES:
                    raise ValueError(f"Unknown grid schedule: {schedule_name}")
                grid_schedule = copy.deepcopy(GRID_SCHEDULES[schedule_name])
                config.update(grid_schedule)
            else:
                grid_schedule = {
                    "warmup_epochs": int(config["warmup_epochs"]),
                    "update_every": int(config["update_every"]),
                }
            model = build_kan(config, horizon, device, allow_install)
            learning_rate = float(config["lr"])
            weight_decay = float(config["weight_decay"])
            record["kan_implementation"] = copy.deepcopy(_KAN_IMPLEMENTATION)

        record["protocol"] = protocol_metadata
        record["model_config"] = config
        record["parameter_count"] = count_parameters(model)
        training_kwargs: Dict[str, Any] = {}
        if improvement_spec is not None:
            training_kwargs = {
                "kan_regularization_lambda": (
                    float(config["reg_lambda"])
                    if improvement_spec["use_regularization"]
                    else 0.0
                ),
                "restore_best_validation": bool(improvement_spec["best_checkpoint"]),
                "gradient_clip_norm": improvement_spec["gradient_clip_norm"],
                "include_initial_validation_checkpoint": improvement_spec["kind"]
                in {"residual_kan", "residual_target"},
                "activate_residual_after_initial_checkpoint": (
                    improvement_spec["kind"] == "residual_target"
                ),
            }
        training_kwargs["reset_optimizer_after_grid_update"] = bool(
            reset_optimizer_after_grid_update
        )
        if uniform_best_checkpoint:
            training_kwargs["restore_best_validation"] = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        record["training"] = train_model(
            model,
            variant,
            train_loader,
            val_loader,
            device,
            epochs,
            learning_rate,
            weight_decay,
            grid_schedule,
            **training_kwargs,
        )
        record["compute_diagnostics"] = {
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
                if device.type == "cuda"
                else None
            ),
            "peak_gpu_memory_scope": (
                "model training after loader construction and before evaluation"
            ),
        }
        record["validation_metrics"] = evaluate(model, val_loader, device)
        if save_diagnostics:
            test_metrics, y_true, y_pred = evaluate_with_predictions(
                model, test_loader, device
            )
            record["test_metrics"] = test_metrics
            target_index = list(protocol_metadata["columns"]).index("OT")
            target_mean = float(scaler.mean_[target_index])
            target_scale = float(scaler.scale_[target_index])
            predictions_dir = results_dir / "predictions"
            checkpoints_dir = results_dir / "checkpoints"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = predictions_dir / f"{task_id}.npz"
            prediction_tmp = prediction_path.with_suffix(".npz.tmp")
            with prediction_tmp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    y_true_standardized=y_true,
                    y_pred_standardized=y_pred,
                    y_true_ot=(y_true * target_scale + target_mean).astype(np.float32),
                    y_pred_ot=(y_pred * target_scale + target_mean).astype(np.float32),
                    target_mean=np.asarray(target_mean, dtype=np.float64),
                    target_scale=np.asarray(target_scale, dtype=np.float64),
                )
            prediction_tmp.replace(prediction_path)
            checkpoint_path = checkpoints_dir / f"{task_id}.pt"
            checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "task_id": task_id,
                    "campaign_id": campaign_id,
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_config": config,
                    "training": record["training"],
                    "protocol": protocol_metadata,
                },
                checkpoint_tmp,
            )
            checkpoint_tmp.replace(checkpoint_path)
            record["artifacts"] = {
                "prediction_npz": {
                    "path": str(prediction_path.relative_to(results_dir)),
                    "sha256": file_sha256(prediction_path),
                    "arrays": [
                        "y_true_standardized",
                        "y_pred_standardized",
                        "y_true_ot",
                        "y_pred_ot",
                    ],
                },
                "checkpoint": {
                    "path": str(checkpoint_path.relative_to(results_dir)),
                    "sha256": file_sha256(checkpoint_path),
                },
            }
        else:
            record["test_metrics"] = evaluate(model, test_loader, device)
        if isinstance(model, DLinearKANResidual):
            record["hybrid_diagnostics"] = {
                "freeze_dlinear": model.freeze_dlinear,
                "residual_gain": float(model.residual_gain.detach().cpu().item()),
                "validation_selected_over_epoch_zero": (
                    record["training"].get("selected_epoch", 0) > 0
                ),
                "test_mse_delta_vs_internal_dlinear": (
                    record["test_metrics"]["mse"]
                    - record["base_test_metrics"]["mse"]
                ),
            }
        # Refresh after optional dependency installation so the record captures
        # the package versions actually used by this task.
        record["device"] = environment_metadata(device)
        record["status"] = "completed"
        record["completed_utc"] = utc_now()
        record["elapsed_s"] = time.time() - started
        atomic_write_json(task_path(results_dir, task_id), record)
        LOGGER.info(
            "DONE %s MSE=%.6f MAE=%.6f elapsed=%.1fs",
            task_id,
            record["test_metrics"]["mse"],
            record["test_metrics"]["mae"],
            record["elapsed_s"],
        )
        return record
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_utc": utc_now(),
                "elapsed_s": time.time() - started,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        atomic_write_json(task_path(results_dir, task_id), record)
        LOGGER.exception("FAILED %s", task_id)
        return record


def iter_tasks(args) -> Iterable[dict]:
    if args.mode in {"baselines", "all"}:
        for model_name in args.models:
            if model_name not in DEFAULT_MODELS:
                raise ValueError(f"Unsupported model: {model_name}")
            for dataset in args.datasets:
                for horizon in args.horizons:
                    for seed in args.seeds:
                        yield {
                            "experiment": "baseline",
                            "variant": model_name,
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
    if args.mode in {"ablation", "all"}:
        for schedule in args.schedules:
            if schedule not in GRID_SCHEDULES:
                raise ValueError(f"Unsupported schedule: {schedule}")
            for dataset in args.datasets:
                for horizon in args.ablation_horizons:
                    for seed in args.seeds:
                        yield {
                            "experiment": "ablation",
                            "variant": schedule,
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
    if args.mode == "improvement":
        for variant in args.improvement_variants:
            if variant not in IMPROVEMENT_VARIANTS:
                raise ValueError(f"Unsupported improvement variant: {variant}")
            for dataset in args.datasets:
                for horizon in args.improvement_horizons:
                    for seed in args.seeds:
                        yield {
                            "experiment": "improvement",
                            "variant": variant,
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
    if args.mode in {"optimizer-state", "confirmation"}:
        if args.mode == "confirmation":
            for dataset in args.datasets:
                for horizon in args.horizons:
                    for seed in args.seeds:
                        yield {
                            "experiment": "baseline",
                            "variant": "DLinear",
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
                        yield {
                            "experiment": "improvement",
                            "variant": "KAN_WOA_RegBestClip",
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
        for variant in args.optimizer_state_variants:
            if variant not in OPTIMIZER_STATE_VARIANTS:
                raise ValueError(f"Unsupported optimizer-state variant: {variant}")
            for dataset in args.datasets:
                for horizon in args.horizons:
                    for seed in args.seeds:
                        yield {
                            "experiment": "optimizer_state",
                            "variant": variant,
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
    if args.mode == "canonical-multihorizon":
        # K10 is intentionally isolated from the optimizer-state branch above.
        # K09 remains the dedicated h96 Grid Update mechanism experiment.
        for dataset in args.datasets:
            for horizon in args.horizons:
                for seed in args.seeds:
                    for model_name in ("DLinear", "PatchTST"):
                        yield {
                            "experiment": "baseline",
                            "variant": model_name,
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                        }
                    yield {
                        "experiment": "improvement",
                        "variant": "KAN_WOA_RegBestClip",
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                    }


def write_summary(results_dir: Path) -> Path:
    rows: List[dict] = []
    records_dir = results_dir / "records"
    for path in sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []:
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "completed":
            continue
        test = record.get("test_metrics", {})
        grid_events = record.get("training", {}).get("grid_updates", [])
        rows.append(
            {
                "task_id": record["task_id"],
                "experiment": record["experiment"],
                "variant": record["variant"],
                "dataset": record["dataset"],
                "horizon": record["horizon"],
                "seed": record["seed"],
                "protocol": record["protocol_name"],
                "grid_schedule": record.get("optimizer_state_protocol", {}).get(
                    "schedule"
                ),
                "optimizer_state_policy": (
                    "reset"
                    if record.get("optimizer_state_protocol", {}).get(
                        "reset_optimizer_state"
                    )
                    else (
                        "retain"
                        if record.get("experiment") == "optimizer_state"
                        else None
                    )
                ),
                "seq_len": record.get("protocol", {}).get("seq_len"),
                "val_mse": record.get("validation_metrics", {}).get("mse"),
                "test_mse": test.get("mse"),
                "test_mae": test.get("mae"),
                "parameters": record.get("parameter_count", {}).get("total"),
                "grid_attempts": len(grid_events),
                "grid_successes": sum(bool(event.get("success")) for event in grid_events),
                "grid_failures": sum(not bool(event.get("success")) for event in grid_events),
                "selected_epoch": record.get("training", {}).get("selected_epoch"),
                "selection_rule": record.get("training", {}).get("selection_rule"),
                "inference_ms_per_sample": record.get("test_metrics", {}).get(
                    "inference_ms_per_sample"
                ),
                "peak_gpu_memory_mb": record.get("compute_diagnostics", {}).get(
                    "peak_gpu_memory_mb"
                ),
                "prediction_sha256": record.get("artifacts", {})
                .get("prediction_npz", {})
                .get("sha256"),
                "base_test_mse": record.get("base_test_metrics", {}).get("mse"),
                "residual_gain": record.get("hybrid_diagnostics", {}).get(
                    "residual_gain"
                ),
                "elapsed_s": record.get("elapsed_s"),
            }
        )
    output = results_dir / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["task_id"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "baselines",
            "ablation",
            "all",
            "improvement",
            "optimizer-state",
            "confirmation",
            "canonical-multihorizon",
        ),
        default="smoke",
    )
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=PROTOCOL_PILOT)
    parser.add_argument("--data-root")
    parser.add_argument("--results-dir", default="targeted_results")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"])
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--ablation-horizons", nargs="+", type=int, default=[96])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--schedules", nargs="+", default=list(GRID_SCHEDULES))
    parser.add_argument(
        "--improvement-horizons", nargs="+", type=int, default=[96]
    )
    parser.add_argument(
        "--improvement-variants", nargs="+", default=list(IMPROVEMENT_VARIANTS)
    )
    parser.add_argument(
        "--optimizer-state-variants",
        nargs="+",
        default=list(OPTIMIZER_STATE_VARIANTS),
    )
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument(
        "--reset-optimizer-after-grid-update", action="store_true"
    )
    parser.add_argument("--uniform-best-checkpoint", action="store_true")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--campaign-id")
    parser.add_argument("--max-tasks", type=int)
    return parser


def validate_frozen_campaign_args(args) -> None:
    """Reject accidental edits to preregistered campaign contracts."""
    if args.mode != "canonical-multihorizon":
        return
    expected = {
        "protocol": "ett_canonical_12_4_4_v1",
        "datasets": ["ETTh1", "ETTh2"],
        "horizons": [96, 192, 336, 720],
        "seeds": [1000, 1001, 1002, 1003, 1004],
        "epochs": 30,
        "campaign_id": "canonical_multihorizon_baselines_v2",
    }
    observed = {
        "protocol": args.protocol,
        "datasets": args.datasets,
        "horizons": args.horizons,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "campaign_id": args.campaign_id,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if not args.uniform_best_checkpoint:
        mismatches["uniform_best_checkpoint"] = {
            "expected": True,
            "observed": False,
        }
    if not args.save_diagnostics:
        mismatches["save_diagnostics"] = {"expected": True, "observed": False}
    if mismatches:
        raise ValueError(
            "K10 arguments differ from the frozen preregistration: "
            + json.dumps(mismatches, sort_keys=True)
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_frozen_campaign_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    data_root = find_or_download_ett_root(args.data_root)
    results_dir = Path(args.results_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Data: %s", data_root)
    LOGGER.info("Results: %s", results_dir)

    if args.mode == "smoke":
        smoke_tasks = [
            {
                "experiment": "baseline",
                "variant": "DLinear",
                "dataset": "ETTh1",
                "horizon": 96,
                "seed": args.seeds[0],
            },
            {
                "experiment": "ablation",
                "variant": "moderate",
                "dataset": "ETTh1",
                "horizon": 96,
                "seed": args.seeds[0],
            },
        ]
        tasks = smoke_tasks
        smoke = True
    else:
        tasks = list(iter_tasks(args))
        smoke = False
    if args.mode == "canonical-multihorizon" and len(tasks) != 120:
        raise RuntimeError(
            f"K10 frozen task matrix must contain 120 tasks, found {len(tasks)}"
        )
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]

    completed = failed = skipped = 0
    overall_start = time.time()
    LOGGER.info("Planned tasks: %d", len(tasks))
    for index, task in enumerate(tasks, start=1):
        LOGGER.info("Task %d/%d: %s", index, len(tasks), task)
        result = run_task(
            **task,
            protocol=args.protocol,
            epochs=args.epochs,
            data_root=data_root,
            results_dir=results_dir,
            device=device,
            allow_install=args.install_deps,
            reset_optimizer_after_grid_update=(
                args.reset_optimizer_after_grid_update
            ),
            uniform_best_checkpoint=args.uniform_best_checkpoint,
            save_diagnostics=args.save_diagnostics,
            campaign_id=args.campaign_id,
            smoke=smoke,
        )
        status = result.get("status")
        completed += status == "completed"
        failed += status == "failed"
        skipped += status == "skipped"
        write_summary(results_dir)

    summary_path = write_summary(results_dir)
    archive_path = Path(
        shutil.make_archive(str(results_dir), "zip", root_dir=str(results_dir))
    )
    LOGGER.info(
        "Finished in %.2fh: completed=%d failed=%d skipped=%d summary=%s archive=%s",
        (time.time() - overall_start) / 3600,
        completed,
        failed,
        skipped,
        summary_path,
        archive_path,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
