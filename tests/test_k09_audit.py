from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from audit_k09 import (
        CAMPAIGN_ID,
        EXPECTED_CONDITIONS,
        EXPECTED_DATASET_HASHES,
        audit,
    )
except ImportError:  # Dependency-light default Python may not include NumPy.
    np = None


@unittest.skipIf(np is None, "NumPy is required for K09 artifact-audit tests")
class K09AuditTests(unittest.TestCase):
    def test_complete_synthetic_matrix_passes_and_corruption_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records"
            predictions = root / "predictions"
            checkpoints = root / "checkpoints"
            for path in (records, predictions, checkpoints):
                path.mkdir()
            summary_rows = []
            for dataset, dataset_hash in EXPECTED_DATASET_HASHES.items():
                for experiment, variant in EXPECTED_CONDITIONS:
                    for seed in range(1000, 1005):
                        task_id = (
                            f"{experiment}__{variant}__{dataset}__h96__s{seed}__"
                            f"ett_canonical_12_4_4_v1__{CAMPAIGN_ID}"
                        )
                        y_true = np.zeros((2, 96), dtype=np.float32)
                        y_pred = np.full((2, 96), 0.25, dtype=np.float32)
                        prediction_path = predictions / f"{task_id}.npz"
                        np.savez_compressed(
                            prediction_path,
                            y_true_standardized=y_true,
                            y_pred_standardized=y_pred,
                            y_true_ot=y_true * 2.0 + 10.0,
                            y_pred_ot=y_pred * 2.0 + 10.0,
                            target_mean=np.asarray(10.0),
                            target_scale=np.asarray(2.0),
                        )
                        checkpoint_path = checkpoints / f"{task_id}.pt"
                        checkpoint_path.write_bytes(task_id.encode("utf-8"))
                        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                        record = {
                            "task_id": task_id,
                            "status": "completed",
                            "campaign_id": CAMPAIGN_ID,
                            "protocol_name": "ett_canonical_12_4_4_v1",
                            "uniform_best_checkpoint": True,
                            "save_diagnostics": True,
                            "horizon": 96,
                            "dataset": dataset,
                            "seed": seed,
                            "experiment": experiment,
                            "variant": variant,
                            "protocol": {"dataset_sha256": dataset_hash},
                            "source": {"combined_sha256": "source-hash"},
                            "training": {
                                "selection_rule": "minimum_validation_mse_checkpoint"
                            },
                            "test_metrics": {"mse": 0.0625, "mae": 0.25},
                            "artifacts": {
                                "prediction_npz": {
                                    "path": prediction_path.relative_to(root).as_posix(),
                                    "sha256": sha(prediction_path),
                                },
                                "checkpoint": {
                                    "path": checkpoint_path.relative_to(root).as_posix(),
                                    "sha256": sha(checkpoint_path),
                                },
                            },
                        }
                        (records / f"{task_id}.json").write_text(
                            json.dumps(record), encoding="utf-8"
                        )
                        summary_rows.append({"task_id": task_id})
            with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["task_id"])
                writer.writeheader()
                writer.writerows(summary_rows)

            self.assertEqual(audit(root), [])
            first_prediction = next(predictions.glob("*.npz"))
            first_prediction.write_bytes(b"corrupt")
            self.assertTrue(any("SHA-256 mismatch" in error for error in audit(root)))


if __name__ == "__main__":
    unittest.main()
