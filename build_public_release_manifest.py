"""Build and audit an allowlisted public-release manifest without copying files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "canonical_uniform_best_diagnostics.zip": "0685deae0d453fe2135ab8855fc68cb635d53663f09699f1fa537a014d124004",
    "canonical_confirmation_results.zip": "7f0a3da091857b0caf8bd72dad1850ea2a3a1a005079b05d639d540fa48da378",
    "optimizer_state_causal_results.zip": "c6ac363e337a58d9638011caca055f3861ac3f498757cc6a6877c72eb35028c3",
    "etth2_confirmation_results.zip": "3f4f3198b9d5deb2dee2ae17037397fe5a7226a19c0cd7de6d1a9a915e72e5ab",
    "canonical_multihorizon_baselines_v2.zip": "bd3bea46869c555f89243c954533b827a5151f6f8bd006baf9693a321281d281",
}

ALLOWLIST = [
    "paper_draft.md", "paper_submission.docx", "LICENSE", "THIRD_PARTY_NOTICES.md", "COVER_LETTER.md",
    "DECLARATIONS.md", "PROJECT_HANDOVER.md", "MASTER_UPGRADE_ROADMAP.md",
    "EXPERIMENT_REGISTRY.md", "CLAIM_EVIDENCE_MATRIX.md", "REPRODUCIBILITY.md",
    "ENVIRONMENT_MANIFEST.md", "FINAL_QA_REPORT.md", "SUBMISSION_CHECKLIST.md",
    "REFERENCE_AUDIT.md", "JEAS_SUBMISSION_REQUIREMENTS.md",
    "JEAS_WRITING_STYLE_AUDIT.md", "GITHUB_ZENODO_RELEASE_PLAN.md",
    "PUBLICATION_READINESS_AUDIT.md", "requirements-canonical.txt",
    "requirements-publication.txt", "datasets.py", "data_loader.py",
    "experiment_protocols.py", "baseline_models.py", "kan_search_space.py",
    "kan_hpo_problem.py", "experiment_runner.py", "targeted_experiments.py",
    "analysis.py", "analysis/analyze_k09_diagnostics.py", "analysis/analyze_k10_multihorizon.py", "analysis/checkpoint_contract_sensitivity.py", "publication_tables.py",
    "audit_evidence.py", "audit_k09.py", "audit_k10.py", "audit_manuscript.py",
    "audit_submission_docx.py", "format_submission_docx.py",
    "strip_docx_comments.py", "build_kaggle_bundle.py", "run_release_qa.ps1",
    "build_public_release_manifest.py", "stage_public_release.py", "audit_public_release.py", "PUBLIC_RELEASE_README.md",
    "CANONICAL_CONFIRMATION_PREREGISTRATION.md",
    "CANONICAL_UNIFORM_BEST_DIAGNOSTICS_PREREGISTRATION.md",
    "CANONICAL_MULTIHORIZON_BASELINE_PREREGISTRATION.md",
    "CANONICAL_MULTIHORIZON_BASELINE_PREREGISTRATION_V2.md",
    "KAGGLE_09_README.md", "KAGGLE_10_README.md",
    "KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py",
    "tests/test_frozen_protocol.py",
    "tests/test_k09_protocol.py", "tests/test_k09_audit.py",
    "tests/test_k10_protocol.py", "tests/test_k10_audit.py",
    "tests/test_publication_boundaries.py",
    "analysis/publication/PUBLICATION_TABLES.md",
    "analysis/publication/k09_per_seed_supplement.csv",
    "analysis/publication/canonical_summary.csv",
    "analysis/publication/canonical_original_unit_rmse.csv",
    "analysis/publication/CHECKPOINT_CONTRACT_APPENDIX.md",
    "analysis/publication/checkpoint_contract_sensitivity_summary.csv",
    "analysis/publication/checkpoint_contract_sensitivity_per_seed.csv",
    "analysis/k10_multihorizon/K10_MULTIHORIZON_REPORT.md",
    "analysis/k10_multihorizon/k10_condition_summary.csv",
    "analysis/k10_multihorizon/k10_per_seed.csv",
    "analysis/k10_multihorizon/k10_forecast_step_rmse.csv",
]

FIGURES = [
    "study_workflow.png", "ett_canonical_distribution_shift.png",
    "hpo_fitness_three_runs.png", "grid_update_ablation_h96.png",
    "grid_update_dynamics.png", "k09_actual_vs_predicted.png",
    "k09_residual_diagnostics.png", "k09_forecast_step_rmse.png",
    "k09_compute_checkpoints.png", "k10_canonical_multihorizon.png",
    "k10_multihorizon_compute.png", "k10_h720_forecasts.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    failures = []
    entries = []
    paths = ALLOWLIST + [f"figures/{name}" for name in FIGURES] + list(REQUIRED)
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing allowlisted file: {relative}")
            continue
        digest = sha256(path)
        expected = REQUIRED.get(relative)
        if expected and digest.lower() != expected:
            failures.append(f"SHA-256 mismatch: {relative}")
        entries.append({"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": digest})
    if failures:
        raise SystemExit("Public release manifest failed:\n- " + "\n- ".join(failures))
    return {
        "schema": "kan_hpo_public_release_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_records": 380,
        "release_status": "staging; author identity, repository DOI and project license pending",
        "exclusion_policy": [
            "No virtual environments, caches, browser/LibreOffice profiles or credentials.",
            "No unrelated PDFs, images, installers, remote-access or signing utilities.",
            "No nested upstream Git metadata; third-party licenses must be supplied separately.",
        ],
        "files": sorted(entries, key=lambda item: item["path"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "PUBLIC_RELEASE_MANIFEST.json")
    args = parser.parse_args()
    manifest = build()
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Public release manifest passed: {len(manifest['files'])} allowlisted files")
