"""Versioned data protocols for the KAN HPO follow-up experiments.

The original Kaggle campaign used a chronological 60/20/20 split with a
StandardScaler and no history shared across split boundaries.  That behavior is
preserved as ``pilot_60_20_20_v1`` so new baselines can be compared fairly with
the 18 historical JSON files.

The canonical ETT 12/4/4-month protocol is also implemented, but its results
must never be mixed with the pilot protocol in the same comparison table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PROTOCOL_PILOT = "pilot_60_20_20_v1"
PROTOCOL_CANONICAL = "ett_canonical_12_4_4_v1"
SUPPORTED_PROTOCOLS = (PROTOCOL_PILOT, PROTOCOL_CANONICAL)

_ETT_BOUNDARIES = {
    "ETTh1": (8640, 11520, 14400),
    "ETTh2": (8640, 11520, 14400),
    "ETTm1": (34560, 46080, 57600),
    "ETTm2": (34560, 46080, 57600),
}


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    split_rule: str
    scaler: str
    validation_history_overlap: bool
    test_history_overlap: bool
    target: str = "OT"


PROTOCOL_SPECS: Dict[str, ProtocolSpec] = {
    PROTOCOL_PILOT: ProtocolSpec(
        name=PROTOCOL_PILOT,
        split_rule="chronological 60/20/20 using int(N*fraction)",
        scaler="StandardScaler fit on the training rows only",
        validation_history_overlap=False,
        test_history_overlap=False,
    ),
    PROTOCOL_CANONICAL: ProtocolSpec(
        name=PROTOCOL_CANONICAL,
        split_rule="official ETT 12/4/4-month boundaries",
        scaler="StandardScaler fit on the training rows only",
        validation_history_overlap=True,
        test_history_overlap=True,
    ),
}


class SlidingWindowDataset(Dataset):
    """Flattened look-back window with an OT forecasting target."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        features: str,
        target_index: int,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected a 2-D array, received shape={data.shape}")
        if features not in {"S", "M"}:
            raise ValueError("features must be 'S' or 'M'")
        self.data = np.asarray(data, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.features = features
        self.target_index = int(target_index)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(index)
        split = start + self.seq_len
        end = split + self.pred_len
        if self.features == "S":
            x = self.data[start:split, self.target_index : self.target_index + 1]
        else:
            x = self.data[start:split]
        y = self.data[split:end, self.target_index]
        return torch.from_numpy(x.copy()).flatten(), torch.from_numpy(y.copy())


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ett(root: str | Path, dataset_name: str) -> Tuple[pd.DataFrame, Path]:
    csv_path = Path(root) / f"{dataset_name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"ETT CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path).drop(columns=["date"], errors="ignore")
    if "OT" not in frame.columns:
        raise ValueError(f"{csv_path} does not contain the OT target column")
    return frame, csv_path


def _split_ranges(
    protocol: str,
    dataset_name: str,
    n_rows: int,
    seq_len: int,
) -> Dict[str, Tuple[int, int]]:
    if protocol == PROTOCOL_PILOT:
        train_end = int(n_rows * 0.6)
        val_end = int(n_rows * 0.8)
        return {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, n_rows),
        }
    if protocol == PROTOCOL_CANONICAL:
        if dataset_name not in _ETT_BOUNDARIES:
            raise ValueError(f"No canonical ETT boundaries for {dataset_name}")
        train_end, val_end, test_end = _ETT_BOUNDARIES[dataset_name]
        if n_rows < test_end:
            raise ValueError(
                f"{dataset_name} has {n_rows} rows; canonical protocol needs {test_end}"
            )
        return {
            "train": (0, train_end),
            "val": (train_end - seq_len, val_end),
            "test": (val_end - seq_len, test_end),
        }
    raise ValueError(f"Unknown protocol {protocol!r}; choose from {SUPPORTED_PROTOCOLS}")


def build_protocol_loaders(
    root: str | Path,
    dataset_name: str = "ETTh1",
    seq_len: int = 96,
    pred_len: int = 96,
    features: str = "S",
    batch_size: int = 64,
    protocol: str = PROTOCOL_PILOT,
    seed: int = 1000,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, dict]:
    """Build loaders and a complete protocol manifest.

    Metrics are computed in standardized space, matching the historical Kaggle
    campaign and the usual ETT benchmark reporting convention.
    """
    if protocol not in PROTOCOL_SPECS:
        raise ValueError(f"Unsupported protocol: {protocol}")
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")

    frame, csv_path = _read_ett(root, dataset_name)
    values = frame.to_numpy(dtype=np.float64)
    target_index = frame.columns.get_loc("OT")
    ranges = _split_ranges(protocol, dataset_name, len(frame), seq_len)

    train_fit_end = ranges["train"][1]
    scaler = StandardScaler()
    scaler.fit(values[:train_fit_end])
    scaled = scaler.transform(values).astype(np.float32)

    datasets = {
        split: SlidingWindowDataset(
            scaled[start:end], seq_len, pred_len, features, target_index
        )
        for split, (start, end) in ranges.items()
    }
    empty = [name for name, dataset in datasets.items() if len(dataset) == 0]
    if empty:
        raise ValueError(
            f"No windows for splits {empty}; seq_len={seq_len}, pred_len={pred_len}"
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }
    train_loader = DataLoader(
        datasets["train"], shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(datasets["val"], shuffle=False, **common)
    test_loader = DataLoader(datasets["test"], shuffle=False, **common)

    spec = PROTOCOL_SPECS[protocol]
    metadata = {
        "protocol": asdict(spec),
        "dataset": dataset_name,
        "dataset_path": str(csv_path.resolve()),
        "dataset_sha256": file_sha256(csv_path),
        "n_rows": len(frame),
        "columns": list(frame.columns),
        "features": features,
        "target": "OT",
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "split_row_ranges": {
            key: {"start_inclusive": value[0], "end_exclusive": value[1]}
            for key, value in ranges.items()
        },
        "scaler_fit_rows": {"start_inclusive": 0, "end_exclusive": train_fit_end},
        "window_counts": {key: len(value) for key, value in datasets.items()},
    }
    return train_loader, val_loader, test_loader, scaler, metadata


def get_input_dim(seq_len: int, features: str = "S", n_features: int = 7) -> int:
    if features not in {"S", "M"}:
        raise ValueError("features must be 'S' or 'M'")
    return int(seq_len) * (1 if features == "S" else int(n_features))

