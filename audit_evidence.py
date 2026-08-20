"""Audit the frozen K01-K10 experiment evidence and archive integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATASET_HASHES = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
}

FROZEN_CAMPAIGNS = {
    "historical": ("my_results/kaggle/working/experiments/results", 18, None),
    "targeted": (
        "targeted_results_all/records",
        96,
        "f0b1d0a65405db8b0ed33bcbc8cae6e01358ee6e1e765b37b9f54b42786834e3",
    ),
    "improvement_stage1": (
        "kan_improvement_results/records",
        21,
        "4bcab98a0cea9785689e505486a2321f4d70c9ffc806f99bbadc08f2fb7718ce",
    ),
    "improvement_stage2": (
        "kan_improvement_stage2_results/records",
        9,
        "362b1edfb67e6c69b66d6956f8c76ffed4c605ab2e793326773c2fa6ba076416",
    ),
    "optimizer_state_causal": (
        "optimizer_state_causal_results/records",
        12,
        "bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516",
    ),
    "etth2_confirmation": (
        "etth2_confirmation_results/records",
        24,
        "bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516",
    ),
    "canonical_confirmation": (
        "canonical_confirmation_results/records",
        40,
        "bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516",
    ),
    "canonical_uniform_diagnostics": (
        "analysis/k09_diagnostics/records",
        40,
        "138ddf1b1feffc4fa03cb75c3e6746b67ace7c401cf397f3503f7d8635aa8a49",
    ),
    "canonical_multihorizon": (
        "analysis/k10_multihorizon/records",
        120,
        "edb30890ea8ba8097e6072b620f6c13c5cef9d26281d094b329cdcbd9b700fb5",
    ),
}

ARCHIVES = {
    "my_results.zip": "11E031023C8A75EAA2EA13A2DA29F4F624B84EF6AAD6D694139EA1E800F08023",
    "targeted_results.zip": "54122E8AC41D33154ACC311937D8E63EE2448C6B399478BAC2343120758BFE4B",
    "targeted_results_03.zip": "5FBD15FD93148F29AAA7C1F03A7CBA286482E4FDAECFB6179F221A01F5CA183F",
    "kan_improvement_results.zip": "FA46D6E44E6CCE17CDD60252E1D88E4F21676A988EAE9A011167E6833BF1AFAC",
    "kan_improvement_stage2_results.zip": "7E75ABCB636022E34E9E2E2C3C9ECB9A3D7846812340E7BAB01476694BD30EFD",
    "optimizer_state_causal_results.zip": "C6AC363E337A58D9638011CACA055F3861AC3F498757CC6A6877C72EB35028C3",
    "etth2_confirmation_results.zip": "3F4F3198B9D5DEB2DEE2AE17037397FE5A7226A19C0CD7DE6D1A9A915E72E5AB",
    "canonical_confirmation_results.zip": "7F0A3DA091857B0CAF8BD72DAD1850EA2A3A1A005079B05D639D540FA48DA378",
    "canonical_uniform_best_diagnostics.zip": "0685DEAE0D453FE2135AB8855FC68CB635D53663F09699F1FA537A014D124004",
    "canonical_multihorizon_baselines_v2.zip": "BD3BEA46869C555F89243C954533B827A5151F6F8BD006BAF9693A321281D281",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_text_sha256(path: Path) -> str:
    """Hash a text artifact using the LF representation used on Kaggle."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_records(path: Path) -> list[dict]:
    records = []
    for record_path in sorted(path.glob("*.json")):
        with record_path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def audit_frozen() -> tuple[list[str], dict[str, list[dict]]]:
    errors: list[str] = []
    loaded: dict[str, list[dict]] = {}
    task_ids: list[str] = []
    for dataset, expected_hash in DATASET_HASHES.items():
        dataset_path = ROOT / "ETDataset" / "ETT-small" / f"{dataset}.csv"
        if not dataset_path.is_file():
            errors.append(f"missing local dataset: {dataset_path}")
        elif canonical_text_sha256(dataset_path) != expected_hash:
            errors.append(f"local dataset content mismatch: {dataset}")
    for name, (relative, expected_count, expected_source) in FROZEN_CAMPAIGNS.items():
        records = load_records(ROOT / relative)
        loaded[name] = records
        if len(records) != expected_count:
            errors.append(f"{name}: expected {expected_count} records, found {len(records)}")
        if name == "historical":
            if {record.get("seed") for record in records} != {1000, 1001, 1002}:
                errors.append("historical: seed set mismatch")
            continue
        task_ids.extend(record.get("task_id", "") for record in records)
        if any(record.get("status") != "completed" for record in records):
            errors.append(f"{name}: contains a non-completed record")
        expected_seeds = (
            {1000, 1001, 1002, 1003, 1004}
            if name in {"canonical_confirmation", "canonical_uniform_diagnostics", "canonical_multihorizon"}
            else {1000, 1001, 1002}
        )
        if {record.get("seed") for record in records} != expected_seeds:
            errors.append(f"{name}: seed set mismatch")
        sources = {
            record.get("source", {}).get("combined_sha256") for record in records
        }
        if sources != {expected_source}:
            errors.append(f"{name}: source hash mismatch: {sorted(sources)}")
        protocols = {record.get("protocol_name") for record in records}
        expected_protocol = (
            "ett_canonical_12_4_4_v1"
            if name in {"canonical_confirmation", "canonical_uniform_diagnostics", "canonical_multihorizon"}
            else "pilot_60_20_20_v1"
        )
        if protocols != {expected_protocol}:
            errors.append(f"{name}: unexpected protocols: {sorted(protocols)}")
        for record in records:
            dataset = record.get("dataset")
            expected_dataset_hash = DATASET_HASHES.get(dataset)
            actual_dataset_hash = record.get("protocol", {}).get("dataset_sha256")
            if expected_dataset_hash is None:
                errors.append(f"{name}: unsupported dataset {dataset!r}")
            elif actual_dataset_hash != expected_dataset_hash:
                errors.append(
                    f"{name}: dataset hash mismatch for {record.get('task_id')}"
                )
        if name in {"canonical_confirmation", "canonical_uniform_diagnostics"}:
            expected_conditions = {
                (dataset, experiment, variant, seed)
                for dataset in ("ETTh1", "ETTh2")
                for experiment, variant in (
                    ("baseline", "DLinear"),
                    ("improvement", "KAN_WOA_RegBestClip"),
                    ("optimizer_state", "none_retain"),
                    ("optimizer_state", "moderate_retain"),
                )
                for seed in expected_seeds
            }
            actual_conditions = {
                (
                    record.get("dataset"),
                    record.get("experiment"),
                    record.get("variant"),
                    record.get("seed"),
                )
                for record in records
            }
            if actual_conditions != expected_conditions:
                errors.append(f"{name}: condition matrix mismatch")
        if name == "canonical_multihorizon":
            expected_conditions = {
                (dataset, horizon, experiment, variant, seed)
                for dataset in ("ETTh1", "ETTh2")
                for horizon in (96, 192, 336, 720)
                for experiment, variant in (
                    ("baseline", "DLinear"),
                    ("baseline", "PatchTST"),
                    ("improvement", "KAN_WOA_RegBestClip"),
                )
                for seed in expected_seeds
            }
            actual_conditions = {
                (
                    record.get("dataset"), record.get("horizon"),
                    record.get("experiment"), record.get("variant"),
                    record.get("seed"),
                )
                for record in records
            }
            if actual_conditions != expected_conditions:
                errors.append(f"{name}: condition matrix mismatch")
    if len(task_ids) != len(set(task_ids)):
        errors.append("modern campaigns contain duplicate task IDs")
    if sum(len(records) for records in loaded.values()) != 380:
        errors.append("frozen campaign total is not 380")
    for archive, expected in ARCHIVES.items():
        path = ROOT / archive
        if not path.is_file():
            errors.append(f"missing archive: {archive}")
        elif sha256(path) != expected:
            errors.append(f"archive hash mismatch: {archive}")
        else:
            with zipfile.ZipFile(path) as handle:
                if handle.testzip() is not None:
                    errors.append(f"archive CRC failure: {archive}")
    return errors, loaded


def audit_optimizer_reset(records: list[dict], label: str) -> list[str]:
    errors: list[str] = []
    optimizer_records = [
        record for record in records if record.get("experiment") == "optimizer_state"
    ]
    for should_reset in (True, False):
        events = [
            event
            for record in optimizer_records
            if record.get("optimizer_state_protocol", {}).get(
                "reset_optimizer_state"
            )
            == should_reset
            for event in record.get("training", {}).get("grid_updates", [])
            if event.get("epoch_zero_based", 0) > 0
        ]
        if not events:
            errors.append(f"{label}: no non-initial optimizer-state events")
            continue
        if any(not event.get("success") for event in events):
            errors.append(f"{label}: failed non-initial Grid Update event")
        if any(
            event.get("optimizer_state_before", {}).get("parameter_states", 0) <= 0
            for event in events
        ):
            errors.append(f"{label}: missing Adam state before a causal update")
        after_counts = [
            event.get("optimizer_state_after_reset", {}).get("parameter_states")
            for event in events
        ]
        if should_reset and any(count != 0 for count in after_counts):
            errors.append(f"{label}: reset condition retained optimizer state")
        if not should_reset and any(count == 0 for count in after_counts):
            errors.append(f"{label}: retain condition cleared optimizer state")
    return errors


def summarize_canonical(path: Path) -> list[str]:
    records = load_records(path / "records")
    errors: list[str] = []
    if not records:
        return [f"canonical: no records found under {path / 'records'}"]
    expected_seeds = {1000, 1001, 1002, 1003, 1004}
    if any(record.get("status") != "completed" for record in records):
        errors.append("canonical: contains a non-completed record")
    if {record.get("protocol_name") for record in records} != {
        "ett_canonical_12_4_4_v1"
    }:
        errors.append("canonical: protocol mismatch")
    grouped: dict[tuple, list[float]] = {}
    for record in records:
        key = (record["dataset"], record["experiment"], record["variant"])
        grouped.setdefault(key, []).append(float(record["test_metrics"]["mse"]))
    for key, values in sorted(grouped.items()):
        if len(values) != len(expected_seeds):
            errors.append(f"canonical: {key} has {len(values)} rather than 5 runs")
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{key}: n={len(values)} mean={statistics.mean(values):.6f} sd={sd:.6f}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path)
    args = parser.parse_args()
    errors, loaded = audit_frozen()
    errors.extend(
        audit_optimizer_reset(loaded["optimizer_state_causal"], "ETTh1 causal")
    )
    errors.extend(
        audit_optimizer_reset(loaded["etth2_confirmation"], "ETTh2 confirmation")
    )
    if args.canonical_dir is not None:
        errors.extend(summarize_canonical(args.canonical_dir))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Evidence audit passed: 380 frozen records and all archives are intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
