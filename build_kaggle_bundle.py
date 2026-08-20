"""Build one self-contained Kaggle script from the targeted experiment modules."""

from __future__ import annotations

import base64
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = (
    "experiment_protocols.py",
    "baseline_models.py",
    "targeted_experiments.py",
)
OUTPUTS = {
    "KAGGLE_01_SMOKE.py": [
        "--mode", "smoke",
        "--install-deps",
        "--results-dir", "/kaggle/working/targeted_results_smoke",
    ],
    "KAGGLE_02_MAIN_H96.py": [
        "--mode", "all",
        "--horizons", "96",
        "--ablation-horizons", "96",
        "--install-deps",
        "--results-dir", "/kaggle/working/targeted_results",
    ],
    "KAGGLE_03_REMAINING_HORIZONS.py": [
        "--mode", "baselines",
        "--horizons", "192", "336", "720",
        "--install-deps",
        "--results-dir", "/kaggle/working/targeted_results",
    ],
    "KAGGLE_04_KAN_IMPROVEMENT.py": [
        "--mode", "improvement",
        "--improvement-horizons", "96",
        "--improvement-variants",
        "KAN_WOA_Reg",
        "KAN_WOA_RegBest",
        "KAN_WOA_RegBestClip",
        "DLinear_KAN_Frozen_s96",
        "DLinear_KAN_Joint_s96",
        "DLinear_KAN_Frozen_s336",
        "DLinear_KAN_Joint_s336",
        "--install-deps",
        "--results-dir", "/kaggle/working/kan_improvement_results",
    ],
    "KAGGLE_05_KAN_IMPROVEMENT_STAGE2.py": [
        "--mode", "improvement",
        "--improvement-horizons", "96",
        "--improvement-variants",
        "KAN_WOA_RegBestClip_s336",
        "DLinear_KAN_ResidualTarget_s96",
        "DLinear_KAN_ResidualTarget_s336",
        "--install-deps",
        "--results-dir", "/kaggle/working/kan_improvement_stage2_results",
    ],
    "KAGGLE_06_OPTIMIZER_STATE_CAUSAL.py": [
        "--mode", "optimizer-state",
        "--datasets", "ETTh1",
        "--horizons", "96",
        "--optimizer-state-variants",
        "moderate_retain",
        "moderate_reset",
        "aggressive_retain",
        "aggressive_reset",
        "--install-deps",
        "--results-dir", "/kaggle/working/optimizer_state_causal_results",
    ],
    "KAGGLE_07_ETTH2_CONFIRMATION.py": [
        "--mode", "confirmation",
        "--datasets", "ETTh2",
        "--horizons", "96",
        "--optimizer-state-variants",
        "none_retain",
        "conservative_retain",
        "moderate_retain",
        "moderate_reset",
        "aggressive_retain",
        "aggressive_reset",
        "--install-deps",
        "--results-dir", "/kaggle/working/etth2_confirmation_results",
    ],
    "KAGGLE_08_CANONICAL_CONFIRMATION.py": [
        "--mode", "confirmation",
        "--protocol", "ett_canonical_12_4_4_v1",
        "--datasets", "ETTh1", "ETTh2",
        "--horizons", "96",
        "--seeds", "1000", "1001", "1002", "1003", "1004",
        "--optimizer-state-variants",
        "none_retain",
        "moderate_retain",
        "--epochs", "30",
        "--install-deps",
        "--results-dir", "/kaggle/working/canonical_confirmation_results",
    ],
    "KAGGLE_09_UNIFORM_CANONICAL_DIAGNOSTICS.py": [
        "--mode", "confirmation",
        "--protocol", "ett_canonical_12_4_4_v1",
        "--datasets", "ETTh1", "ETTh2",
        "--horizons", "96",
        "--seeds", "1000", "1001", "1002", "1003", "1004",
        "--optimizer-state-variants", "none_retain", "moderate_retain",
        "--epochs", "30",
        "--uniform-best-checkpoint",
        "--save-diagnostics",
        "--campaign-id", "canonical_uniform_best_diagnostics_v1",
        "--install-deps",
        "--results-dir", "/kaggle/working/canonical_uniform_best_diagnostics",
    ],
    "KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py": [
        "--mode", "canonical-multihorizon",
        "--protocol", "ett_canonical_12_4_4_v1",
        "--datasets", "ETTh1", "ETTh2",
        "--horizons", "96", "192", "336", "720",
        "--seeds", "1000", "1001", "1002", "1003", "1004",
        "--epochs", "30",
        "--uniform-best-checkpoint",
        "--save-diagnostics",
        "--campaign-id", "canonical_multihorizon_baselines_v2",
        "--install-deps",
        "--results-dir", "/kaggle/working/canonical_multihorizon_baselines_v2",
    ],
}


def build(output: Path, arguments: list[str], encoded: dict[str, str]) -> None:
    argument_lines = [f"    {argument!r}," for argument in arguments]
    results_index = arguments.index("--results-dir") + 1
    archive_path = f"{arguments[results_index]}.zip"
    lines = [
        '"""Generated single-file Kaggle runner. Paste into one cell and Run All."""',
        "",
        "import base64",
        "import runpy",
        "import sys",
        "from pathlib import Path",
        "",
        "KAGGLE_ARGS = [",
        *argument_lines,
        "]",
        "",
        f"EMBEDDED = {encoded!r}",
        'RUNTIME = Path("/kaggle/working/kan_hpo_targeted_runtime")',
        "RUNTIME.mkdir(parents=True, exist_ok=True)",
        "for name, payload in EMBEDDED.items():",
        "    content = base64.b64decode(payload)",
        "    path = RUNTIME / name",
        "    if not path.is_file() or path.read_bytes() != content:",
        "        path.write_bytes(content)",
        "sys.path.insert(0, str(RUNTIME))",
        'sys.argv = [str(RUNTIME / "targeted_experiments.py"), *KAGGLE_ARGS]',
        "try:",
        '    runpy.run_path(str(RUNTIME / "targeted_experiments.py"), run_name="__main__")',
        "except SystemExit as exc:",
        "    if exc.code not in (None, 0):",
        "        raise",
        '    print("Kaggle runner completed successfully.")',
        "from hashlib import sha256",
        "from IPython.display import FileLink, display",
        f"archive = Path({archive_path!r})",
        "if archive.is_file():",
        "    digest = sha256()",
        "    with archive.open('rb') as handle:",
        "        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):",
        "            digest.update(chunk)",
        "    size_gib = archive.stat().st_size / (1024 ** 3)",
        "    print(f'ARCHIVE_READY path={archive} size={size_gib:.3f} GiB sha256={digest.hexdigest()}')",
        "    print('Download immediately while this Kaggle session is still active:')",
        "    display(FileLink(str(archive)))",
        "    print('If the link is not visible, use the Kaggle Files pane and refresh /kaggle/working.')",
        "else:",
        "    raise FileNotFoundError(f'Expected archive was not created: {archive}')",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Built {output} ({output.stat().st_size:,} bytes)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=tuple(OUTPUTS))
    selected = parser.parse_args().only
    encoded = {
        name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii")
        for name in SOURCES
    }
    outputs = OUTPUTS if not selected else {
        name: OUTPUTS[name] for name in selected
    }
    for name, arguments in outputs.items():
        build(ROOT / name, arguments, encoded)


if __name__ == "__main__":
    main()
