from __future__ import annotations

import ast
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "canonical_multihorizon_baselines_v2"


class K10ProtocolTests(unittest.TestCase):
    def test_builder_freezes_canonical_multihorizon_matrix(self) -> None:
        outputs = runpy.run_path(str(ROOT / "build_kaggle_bundle.py"))["OUTPUTS"]
        args = outputs["KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py"]
        self.assertEqual(args[args.index("--mode") + 1], "canonical-multihorizon")
        self.assertEqual(args[args.index("--protocol") + 1], "ett_canonical_12_4_4_v1")
        self.assertEqual(args[args.index("--campaign-id") + 1], CAMPAIGN_ID)
        self.assertIn("--uniform-best-checkpoint", args)
        self.assertIn("--save-diagnostics", args)
        datasets = args[args.index("--datasets") + 1 : args.index("--horizons")]
        horizons = args[args.index("--horizons") + 1 : args.index("--seeds")]
        seeds = args[args.index("--seeds") + 1 : args.index("--epochs")]
        self.assertEqual(datasets, ["ETTh1", "ETTh2"])
        self.assertEqual(horizons, ["96", "192", "336", "720"])
        self.assertEqual(seeds, ["1000", "1001", "1002", "1003", "1004"])
        self.assertEqual(2 * 4 * 3 * 5, 120)

    def test_runner_has_isolated_k10_task_matrix(self) -> None:
        source = (ROOT / "targeted_experiments.py").read_text(encoding="utf-8")
        self.assertIn('"canonical-multihorizon"', source)
        self.assertIn('for model_name in ("DLinear", "PatchTST")', source)
        self.assertIn('"variant": "KAN_WOA_RegBestClip"', source)
        self.assertIn("validate_frozen_campaign_args", source)
        self.assertIn('required_version = "4.48.3"', source)
        self.assertIn('len(tasks) != 120', source)
        self.assertIn('K10 frozen task matrix must contain 120 tasks', source)

    def test_runner_contains_frozen_contract_values(self) -> None:
        source = (ROOT / "targeted_experiments.py").read_text(encoding="utf-8")
        required = (
            '"protocol": "ett_canonical_12_4_4_v1"',
            '"datasets": ["ETTh1", "ETTh2"]',
            '"horizons": [96, 192, 336, 720]',
            '"seeds": [1000, 1001, 1002, 1003, 1004]',
            '"campaign_id": "canonical_multihorizon_baselines_v2"',
            '"K10 arguments differ from the frozen preregistration',
        )
        for token in required:
            self.assertIn(token, source)

    def test_generated_bundle_embeds_runtime_sources(self) -> None:
        source = (ROOT / "KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"KAGGLE_ARGS", "EMBEDDED"}
        }
        self.assertEqual(
            assignments["KAGGLE_ARGS"][assignments["KAGGLE_ARGS"].index("--campaign-id") + 1],
            CAMPAIGN_ID,
        )
        self.assertEqual(
            set(assignments["EMBEDDED"]),
            {"experiment_protocols.py", "baseline_models.py", "targeted_experiments.py"},
        )
        self.assertIn("FileLink", source)
        self.assertNotIn("data:application/zip;base64", source)


if __name__ == "__main__":
    unittest.main()
