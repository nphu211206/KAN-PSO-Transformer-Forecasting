from __future__ import annotations

import ast
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class K09ProtocolTests(unittest.TestCase):
    def test_builder_freezes_uniform_best_diagnostics_matrix(self) -> None:
        outputs = runpy.run_path(str(ROOT / "build_kaggle_bundle.py"))["OUTPUTS"]
        args = outputs["KAGGLE_09_UNIFORM_CANONICAL_DIAGNOSTICS.py"]
        self.assertIn("--uniform-best-checkpoint", args)
        self.assertIn("--save-diagnostics", args)
        self.assertEqual(
            args[args.index("--campaign-id") + 1],
            "canonical_uniform_best_diagnostics_v1",
        )
        datasets = args[args.index("--datasets") + 1 : args.index("--horizons")]
        seeds = args[args.index("--seeds") + 1 : args.index("--optimizer-state-variants")]
        variants = args[
            args.index("--optimizer-state-variants") + 1 : args.index("--epochs")
        ]
        self.assertEqual(datasets, ["ETTh1", "ETTh2"])
        self.assertEqual(seeds, ["1000", "1001", "1002", "1003", "1004"])
        self.assertEqual(variants, ["none_retain", "moderate_retain"])
        self.assertEqual(2 * 5 * (2 + len(variants)), 40)

    def test_generated_bundle_embeds_only_declared_sources(self) -> None:
        source = (ROOT / "KAGGLE_09_UNIFORM_CANONICAL_DIAGNOSTICS.py").read_text(
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
        self.assertIn("--uniform-best-checkpoint", assignments["KAGGLE_ARGS"])
        self.assertIn("--save-diagnostics", assignments["KAGGLE_ARGS"])
        self.assertEqual(
            set(assignments["EMBEDDED"]),
            {"experiment_protocols.py", "baseline_models.py", "targeted_experiments.py"},
        )

    def test_runner_contains_required_artifact_contract(self) -> None:
        source = (ROOT / "targeted_experiments.py").read_text(encoding="utf-8")
        required = (
            "evaluate_with_predictions",
            "y_true_standardized",
            "y_pred_standardized",
            "y_true_ot",
            "y_pred_ot",
            "inference_ms_per_sample",
            "peak_gpu_memory_mb",
            "uniform_best_checkpoint",
            "prediction_npz",
            "checkpoint",
        )
        for token in required:
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
