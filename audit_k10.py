"""Audit the preregistered K10 canonical multi-horizon archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np


CAMPAIGN_ID = "canonical_multihorizon_baselines_v2"
PROTOCOL_ID = "ett_canonical_12_4_4_v1"
EXPECTED_DATASET_HASHES = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
}
EXPECTED_CONDITIONS = {
    ("baseline", "DLinear"),
    ("baseline", "PatchTST"),
    ("improvement", "KAN_WOA_RegBestClip"),
}
EXPECTED_HORIZONS = {96, 192, 336, 720}
EXPECTED_SEEDS = set(range(1000, 1005))
EXPECTED_RECORDS = 120


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class EvidenceSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None
        self._names = set(self.archive.namelist()) if self.archive is not None else None

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def names(self) -> Iterable[str]:
        if self._names is not None:
            return self._names
        return (
            item.relative_to(self.path).as_posix()
            for item in self.path.rglob("*")
            if item.is_file()
        )

    def read(self, relative: str) -> bytes:
        normalized = PurePosixPath(relative).as_posix()
        if self.archive is not None:
            return self.archive.read(normalized)
        return (self.path / Path(normalized)).read_bytes()

    def exists(self, relative: str) -> bool:
        normalized = PurePosixPath(relative).as_posix()
        if self._names is not None:
            return normalized in self._names
        return (self.path / Path(normalized)).is_file()


def close_enough(actual: float, expected: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def audit(path: Path) -> list[str]:
    errors: list[str] = []
    source = EvidenceSource(path)
    try:
        record_names = sorted(
            name
            for name in source.names()
            if name.startswith("records/") and name.endswith(".json")
        )
        if len(record_names) != EXPECTED_RECORDS:
            errors.append(
                f"expected {EXPECTED_RECORDS} JSON records, found {len(record_names)}"
            )

        observed: set[tuple[str, int, str, str, int]] = set()
        source_hashes: set[str] = set()
        for record_name in record_names:
            record = json.loads(source.read(record_name))
            task_id = record.get("task_id", record_name)
            if record.get("status") != "completed":
                errors.append(f"{task_id}: status is not completed")
                continue
            if record.get("campaign_id") != CAMPAIGN_ID:
                errors.append(f"{task_id}: wrong campaign_id")
            if record.get("protocol_name") != PROTOCOL_ID:
                errors.append(f"{task_id}: wrong protocol")
            if not record.get("uniform_best_checkpoint"):
                errors.append(f"{task_id}: uniform_best_checkpoint is false")
            if not record.get("save_diagnostics"):
                errors.append(f"{task_id}: save_diagnostics is false")

            dataset = record.get("dataset")
            horizon = record.get("horizon")
            seed = record.get("seed")
            condition = (record.get("experiment"), record.get("variant"))
            if dataset not in EXPECTED_DATASET_HASHES:
                errors.append(f"{task_id}: unexpected dataset {dataset}")
                continue
            if horizon not in EXPECTED_HORIZONS:
                errors.append(f"{task_id}: unexpected horizon {horizon}")
            if seed not in EXPECTED_SEEDS:
                errors.append(f"{task_id}: unexpected seed {seed}")
            if condition not in EXPECTED_CONDITIONS:
                errors.append(f"{task_id}: unexpected condition {condition}")
            observed.add((dataset, horizon, condition[0], condition[1], seed))

            if record.get("protocol", {}).get("dataset_sha256") != EXPECTED_DATASET_HASHES[dataset]:
                errors.append(f"{task_id}: dataset hash mismatch")
            if record.get("protocol", {}).get("seq_len") != 96:
                errors.append(f"{task_id}: lookback is not 96")
            combined_hash = record.get("source", {}).get("combined_sha256")
            if not combined_hash:
                errors.append(f"{task_id}: missing source combined hash")
            else:
                source_hashes.add(combined_hash)
            if record.get("training", {}).get("selection_rule") != "minimum_validation_mse_checkpoint":
                errors.append(f"{task_id}: checkpoint rule is not validation-best")

            artifacts = record.get("artifacts", {})
            prediction = artifacts.get("prediction_npz", {})
            checkpoint = artifacts.get("checkpoint", {})
            for artifact, label in ((prediction, "prediction"), (checkpoint, "checkpoint")):
                artifact_path = artifact.get("path")
                if not artifact_path or not source.exists(artifact_path):
                    errors.append(f"{task_id}: missing {label} artifact")
                    continue
                if sha256_bytes(source.read(artifact_path)) != artifact.get("sha256"):
                    errors.append(f"{task_id}: {label} SHA-256 mismatch")

            prediction_path = prediction.get("path")
            if not prediction_path or not source.exists(prediction_path):
                continue
            try:
                arrays_context = np.load(io.BytesIO(source.read(prediction_path)))
            except Exception as exc:
                errors.append(f"{task_id}: prediction NPZ cannot be loaded: {exc}")
                continue
            with arrays_context as arrays:
                required = {
                    "y_true_standardized", "y_pred_standardized",
                    "y_true_ot", "y_pred_ot", "target_mean", "target_scale",
                }
                if not required.issubset(arrays.files):
                    errors.append(f"{task_id}: prediction NPZ keys are incomplete")
                    continue
                y_true = arrays["y_true_standardized"]
                y_pred = arrays["y_pred_standardized"]
                if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != horizon:
                    errors.append(f"{task_id}: prediction shapes are invalid")
                    continue
                if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                    errors.append(f"{task_id}: non-finite standardized predictions")
                mse = float(np.mean(np.square(y_pred.astype(np.float64) - y_true.astype(np.float64))))
                mae = float(np.mean(np.abs(y_pred.astype(np.float64) - y_true.astype(np.float64))))
                metrics = record.get("test_metrics", {})
                if not close_enough(mse, float(metrics.get("mse", math.nan))):
                    errors.append(f"{task_id}: recomputed MSE mismatch")
                if not close_enough(mae, float(metrics.get("mae", math.nan))):
                    errors.append(f"{task_id}: recomputed MAE mismatch")
                mean = float(arrays["target_mean"])
                scale = float(arrays["target_scale"])
                if not np.allclose(arrays["y_true_ot"], y_true * scale + mean, rtol=1e-6, atol=1e-6):
                    errors.append(f"{task_id}: y_true OT inverse scaling mismatch")
                if not np.allclose(arrays["y_pred_ot"], y_pred * scale + mean, rtol=1e-6, atol=1e-6):
                    errors.append(f"{task_id}: y_pred OT inverse scaling mismatch")

        expected = {
            (dataset, horizon, experiment, variant, seed)
            for dataset in EXPECTED_DATASET_HASHES
            for horizon in EXPECTED_HORIZONS
            for experiment, variant in EXPECTED_CONDITIONS
            for seed in EXPECTED_SEEDS
        }
        if observed != expected:
            errors.append(
                f"task matrix mismatch: missing={sorted(expected - observed)} "
                f"extra={sorted(observed - expected)}"
            )
        if len(source_hashes) != 1:
            errors.append(f"expected one source combined hash, found {len(source_hashes)}")

        if not source.exists("summary.csv"):
            errors.append("summary.csv is missing")
        else:
            rows = list(csv.DictReader(io.StringIO(source.read("summary.csv").decode("utf-8"))))
            if len(rows) != EXPECTED_RECORDS:
                errors.append(
                    f"summary.csv has {len(rows)} rows instead of {EXPECTED_RECORDS}"
                )
    finally:
        source.close()
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", default="canonical_multihorizon_baselines_v2.zip", type=Path
    )
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"evidence path does not exist: {args.path}")
    errors = audit(args.path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("K10 audit passed: 120 canonical multi-horizon diagnostic records are intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
