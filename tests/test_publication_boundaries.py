from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicationBoundaryTests(unittest.TestCase):
    def test_k09_supplement_has_exact_uniform_matrix(self) -> None:
        path = ROOT / "analysis" / "publication" / "k09_per_seed_supplement.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 40)
        self.assertEqual({row["dataset"] for row in rows}, {"ETTh1", "ETTh2"})
        self.assertEqual({int(row["seed"]) for row in rows}, set(range(1000, 1005)))
        self.assertEqual({row["selection_rule"] for row in rows}, {"minimum_validation_mse_checkpoint"})

    def test_public_manifest_excludes_private_workspace_classes(self) -> None:
        manifest = json.loads((ROOT / "PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
        paths = [item["path"].lower() for item in manifest["files"]]
        forbidden = (".venv", "ultraviewer", "lo_profile", "docx_qa", "img_", ".exe")
        for path in paths:
            self.assertFalse(any(token in path for token in forbidden), path)

    def test_k08_k09_publication_tables_are_separate(self) -> None:
        publication = ROOT / "analysis" / "publication"
        with (publication / "canonical_summary.csv").open(encoding="utf-8", newline="") as handle:
            k09 = list(csv.DictReader(handle))
        with (publication / "canonical_condition_specific_checkpoint_summary.csv").open(encoding="utf-8", newline="") as handle:
            k08 = list(csv.DictReader(handle))
        self.assertEqual({row["campaign_id"] for row in k09}, {"canonical_uniform_best_diagnostics_v1"})
        self.assertNotIn("canonical_uniform_best_diagnostics_v1", {row["campaign_id"] for row in k08})


if __name__ == "__main__":
    unittest.main()
