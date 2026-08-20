"""Dependency-light integrity checks for the frozen ETT evidence package.

These checks intentionally read the CSV/JSON artifacts directly. They can run
without PyTorch or pandas and catch accidental edits to the evidence boundary
before publication tables or the manuscript are regenerated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ETDataset" / "ETT-small"
CANONICAL = ROOT / "canonical_confirmation_results" / "records"

EXPECTED_HASHES = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenProtocolTests(unittest.TestCase):
    def test_dataset_identity_and_hourly_continuity(self) -> None:
        for dataset, expected_hash in EXPECTED_HASHES.items():
            path = DATA / f"{dataset}.csv"
            # The frozen Kaggle files use LF. A Windows checkout may represent
            # the same CSV rows with CRLF, so hash the canonical LF bytes.
            canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n")
            digest = hashlib.sha256(canonical_bytes).hexdigest()
            self.assertEqual(digest, expected_hash)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 17420)
            self.assertEqual(
                list(rows[0]),
                ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            )
            timestamps = [datetime.fromisoformat(row["date"]) for row in rows]
            self.assertEqual(len(set(timestamps)), len(timestamps))
            for before, after in zip(timestamps, timestamps[1:]):
                self.assertEqual(after - before, timedelta(hours=1))
            for row in rows:
                for key in row:
                    self.assertNotEqual(row[key], "")

    def test_canonical_matrix_has_expected_seeds_and_contracts(self) -> None:
        records = [load_json(path) for path in sorted(CANONICAL.glob("*.json"))]
        self.assertEqual(len(records), 40)
        self.assertEqual({record["seed"] for record in records}, {1000, 1001, 1002, 1003, 1004})
        self.assertEqual({record["protocol_name"] for record in records}, {"ett_canonical_12_4_4_v1"})
        self.assertEqual({record["protocol"]["features"] for record in records}, {"S"})
        self.assertEqual({record["protocol"]["target"] for record in records}, {"OT"})

        selections = [record["training"]["selection_rule"] for record in records]
        self.assertEqual(selections.count("minimum_validation_mse_checkpoint"), 10)
        self.assertEqual(selections.count("final_epoch_to_match_historical_kaggle_evaluation"), 30)

        for record in records:
            protocol = record["protocol"]
            self.assertEqual(protocol["n_rows"], 17420)
            self.assertEqual(protocol["scaler_fit_rows"], {"start_inclusive": 0, "end_exclusive": 8640})
            self.assertEqual(protocol["window_counts"], {"train": 8449, "val": 2785, "test": 2785})
            self.assertEqual(record["training"]["epochs"][-1]["epoch"], 30)
            self.assertNotIn("predictions", record)
            self.assertNotIn("y_true", record)
            self.assertNotIn("y_pred", record)

    def test_canonical_moderate_schedule_has_five_successful_events(self) -> None:
        paths = sorted(CANONICAL.glob("optimizer_state__moderate_retain__*.json"))
        self.assertEqual(len(paths), 10)
        for path in paths:
            record = load_json(path)
            events = record["training"]["grid_updates"]
            self.assertEqual(len(events), 5)
            self.assertTrue(all(event["success"] for event in events))
            self.assertEqual(
                [event["epoch_zero_based"] for event in events],
                [0, 1, 2, 3, 4],
            )


if __name__ == "__main__":
    unittest.main()
