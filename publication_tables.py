"""Generate manuscript tables from the frozen JSON evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGNS = [
    "targeted_results_all/records",
    "kan_improvement_results/records",
    "kan_improvement_stage2_results/records",
    "optimizer_state_causal_results/records",
    "etth2_confirmation_results/records",
    "canonical_confirmation_results/records",
    "analysis/k09_diagnostics/records",
]

K09_CAMPAIGN_ID = "canonical_uniform_best_diagnostics_v1"


def records(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for record_path in sorted(path.glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") == "completed":
                rows.append(record)
    return rows


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def grouped_table(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(key) for key in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        mse = stats([float(row["test_metrics"]["mse"]) for row in group])
        mae = stats([float(row["test_metrics"]["mae"]) for row in group])
        output.append(
            {
                **dict(zip(keys, key)),
                "n": mse["n"],
                "mse_mean": mse["mean"],
                "mse_sample_sd": mse["sample_sd"],
                "mse_median": mse["median"],
                "mse_min": mse["min"],
                "mse_max": mse["max"],
                "mae_mean": mae["mean"],
                "mae_sample_sd": mae["sample_sd"],
            }
        )
    return output


def canonical_original_unit_table(rows: list[dict]) -> list[dict]:
    canonical = [row for row in rows if row.get("campaign_id") == K09_CAMPAIGN_ID]
    scales: dict[str, float] = {}
    for dataset in sorted({row["dataset"] for row in canonical}):
        exemplar = next(row for row in canonical if row["dataset"] == dataset)
        fit_range = exemplar["protocol"]["scaler_fit_rows"]
        csv_path = ROOT / "ETDataset" / "ETT-small" / f"{dataset}.csv"
        with csv_path.open(encoding="utf-8", newline="") as handle:
            values = [
                float(row["OT"])
                for row in csv.DictReader(handle)
            ][fit_range["start_inclusive"] : fit_range["end_exclusive"]]
        mean = statistics.mean(values)
        scales[dataset] = math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        )

    groups: dict[tuple[str, str], list[float]] = {}
    for row in canonical:
        key = (row["dataset"], row["variant"])
        rmse = math.sqrt(float(row["test_metrics"]["mse"])) * scales[row["dataset"]]
        groups.setdefault(key, []).append(rmse)

    output = []
    for (dataset, variant), values in sorted(groups.items()):
        summary = stats(values)
        output.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n": summary["n"],
                "ot_scale": scales[dataset],
                "rmse_mean": summary["mean"],
                "rmse_sample_sd": summary["sample_sd"],
                "rmse_min": summary["min"],
                "rmse_max": summary["max"],
            }
        )
    return output


def k09_per_seed_table(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        if row.get("campaign_id") != K09_CAMPAIGN_ID:
            continue
        scale = next(
            item["ot_scale"]
            for item in canonical_original_unit_table(rows)
            if item["dataset"] == row["dataset"]
        )
        metrics = row["test_metrics"]
        training = row["training"]
        compute = row.get("compute_diagnostics", {})
        output.append(
            {
                "dataset": row["dataset"],
                "variant": row["variant"],
                "seed": row["seed"],
                "selected_epoch": training["selected_epoch"],
                "validation_mse": row["validation_metrics"]["mse"],
                "test_mse": metrics["mse"],
                "test_mae": metrics["mae"],
                "ot_rmse": math.sqrt(float(metrics["mse"])) * scale,
                "training_elapsed_s": row["elapsed_s"],
                "inference_ms_per_sample": metrics["inference_ms_per_sample"],
                "peak_gpu_memory_mb": compute.get("peak_gpu_memory_mb"),
                "parameter_count": row["parameter_count"]["trainable"],
                "selection_rule": training["selection_rule"],
            }
        )
    return sorted(output, key=lambda item: (item["dataset"], item["variant"], item["seed"]))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def markdown_table(rows: list[dict], columns: list[str], rename: dict[str, str]) -> str:
    headers = [rename.get(column, column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif column in {"n", "horizon"}:
                values.append(str(int(value)))
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "publication")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = records([ROOT / path for path in CAMPAIGNS])

    grouped = grouped_table(
        rows,
        ["campaign_id", "experiment", "variant", "dataset", "horizon", "protocol_name"],
    )
    columns = list(grouped[0]) if grouped else []
    write_csv(args.output_dir / "all_condition_summary.csv", grouped, columns)

    canonical = [row for row in grouped if row["campaign_id"] == K09_CAMPAIGN_ID]
    canonical_legacy = [
        row for row in grouped
        if row["protocol_name"] == "ett_canonical_12_4_4_v1"
        and row["campaign_id"] != K09_CAMPAIGN_ID
    ]
    pilot_baselines = [
        row for row in grouped
        if row["experiment"] == "baseline" and row["protocol_name"] == "pilot_60_20_20_v1"
    ]
    pilot_kan = [
        row for row in grouped
        if row["variant"] in {"KAN_WOA_Selected", "KAN_Vanilla"}
        and row["protocol_name"] == "pilot_60_20_20_v1"
    ]
    write_csv(args.output_dir / "canonical_summary.csv", canonical, columns)
    write_csv(args.output_dir / "canonical_condition_specific_checkpoint_summary.csv", canonical_legacy, columns)
    write_csv(args.output_dir / "pilot_baseline_summary.csv", pilot_baselines, columns)
    write_csv(args.output_dir / "pilot_kan_summary.csv", pilot_kan, columns)
    original_unit = canonical_original_unit_table(rows)
    original_columns = list(original_unit[0]) if original_unit else []
    write_csv(
        args.output_dir / "canonical_original_unit_rmse.csv",
        original_unit,
        original_columns,
    )
    per_seed = k09_per_seed_table(rows)
    per_seed_columns = list(per_seed[0]) if per_seed else []
    write_csv(args.output_dir / "k09_per_seed_supplement.csv", per_seed, per_seed_columns)

    lines = [
        "# Publication Tables",
        "",
        "Generated directly from completed JSON records. Values are test MSE in standardized space.",
        "",
        "## Canonical Uniform-Checkpoint Confirmation (K09)",
        "",
        markdown_table(
            canonical,
            ["dataset", "variant", "n", "mse_mean", "mse_sample_sd", "mse_median", "mse_min", "mse_max"],
            {
                "dataset": "Dataset",
                "variant": "Condition",
                "n": "n",
                "mse_mean": "Mean",
                "mse_sample_sd": "Sample SD",
                "mse_median": "Median",
                "mse_min": "Min",
                "mse_max": "Max",
            },
        ),
        "",
        "## K09 Per-Seed Supplement",
        "",
        "Uniform validation-best results. Timing and peak memory are comparative measurements from the recorded Tesla T4 environment.",
        "",
        markdown_table(
            per_seed,
            ["dataset", "variant", "seed", "selected_epoch", "test_mse", "test_mae", "ot_rmse", "training_elapsed_s", "inference_ms_per_sample", "peak_gpu_memory_mb", "parameter_count"],
            {
                "dataset": "Dataset", "variant": "Condition", "seed": "Seed",
                "selected_epoch": "Selected epoch", "test_mse": "Test MSE",
                "test_mae": "Test MAE", "ot_rmse": "OT RMSE",
                "training_elapsed_s": "Train s", "inference_ms_per_sample": "Inference ms/sample",
                "peak_gpu_memory_mb": "Peak GPU MB", "parameter_count": "Parameters",
            },
        ),
        "",
        "## Canonical RMSE in Recorded OT Units",
        "",
        "Derived seed by seed as sqrt(standardized MSE) multiplied by the training-only OT scale.",
        "",
        markdown_table(
            original_unit,
            ["dataset", "variant", "n", "ot_scale", "rmse_mean", "rmse_sample_sd", "rmse_min", "rmse_max"],
            {
                "dataset": "Dataset",
                "variant": "Condition",
                "n": "n",
                "ot_scale": "OT scale",
                "rmse_mean": "RMSE mean",
                "rmse_sample_sd": "RMSE sample SD",
                "rmse_min": "RMSE min",
                "rmse_max": "RMSE max",
            },
        ),
        "",
        "## Pilot Baselines",
        "",
        markdown_table(
            pilot_baselines,
            ["dataset", "variant", "horizon", "n", "mse_mean", "mse_sample_sd"],
            {
                "dataset": "Dataset",
                "variant": "Model",
                "horizon": "Horizon",
                "n": "n",
                "mse_mean": "Mean",
                "mse_sample_sd": "Sample SD",
            },
        ),
        "",
        "## Corrected KAN and Negative Results",
        "",
        markdown_table(
            [row for row in grouped if row["variant"] in {
                "KAN_WOA_Reg",
                "KAN_WOA_RegBest",
                "KAN_WOA_RegBestClip",
                "KAN_WOA_RegBestClip_s336",
                "DLinear_KAN_Frozen_s336",
                "DLinear_KAN_ResidualTarget_s336",
            }],
            ["dataset", "variant", "horizon", "protocol_name", "n", "mse_mean", "mse_sample_sd"],
            {
                "dataset": "Dataset",
                "variant": "Variant",
                "horizon": "Horizon",
                "protocol_name": "Protocol",
                "n": "n",
                "mse_mean": "Mean",
                "mse_sample_sd": "Sample SD",
            },
        ),
    ]
    (args.output_dir / "PUBLICATION_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote publication tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
