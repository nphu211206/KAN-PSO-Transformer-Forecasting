"""Derive K09 publication diagnostics directly from the frozen result archive.

The script never modifies the ZIP. It copies the 40 JSON records for convenient
inspection, writes derived CSV/Markdown tables, and creates deterministic PNG
figures using NumPy and Pillow only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "canonical_uniform_best_diagnostics_v1"
ARCHIVE_SHA256 = "0685DEAE0D453FE2135AB8855FC68CB635D53663F09699F1FA537A014D124004"
SOURCE_SHA256 = "138ddf1b1feffc4fa03cb75c3e6746b67ace7c401cf397f3503f7d8635aa8a49"
CONDITIONS = {
    ("baseline", "DLinear"): "DLinear",
    ("improvement", "KAN_WOA_RegBestClip"): "Corrected KAN",
    ("optimizer_state", "none_retain"): "No-update KAN",
    ("optimizer_state", "moderate_retain"): "Moderate-update KAN",
}
COLORS = {
    "Actual": "#111827",
    "DLinear": "#2E86AB",
    "Corrected KAN": "#D1495B",
    "No-update KAN": "#2A9D8F",
    "Moderate-update KAN": "#F4A261",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("C:/Windows/Fonts") / ("calibrib.ttf" if bold else "calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(34, bold=True)
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - box[2]) / 2, 24), title, font=title_font, fill="#111827")
    return image, draw


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x_label: str, y_label: str,
         x_ticks: list[tuple[float, str]], y_ticks: list[tuple[float, str]], x_range: tuple[float, float],
         y_range: tuple[float, float]) -> tuple[callable, callable]:
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill="#374151", width=2)
    draw.line((left, bottom, right, bottom), fill="#374151", width=2)
    label_font = font(21)
    tick_font = font(18)

    def xp(value: float) -> float:
        return left + (value - x_range[0]) / (x_range[1] - x_range[0]) * (right - left)

    def yp(value: float) -> float:
        return bottom - (value - y_range[0]) / (y_range[1] - y_range[0]) * (bottom - top)

    for value, label in y_ticks:
        y = yp(value)
        draw.line((left, y, right, y), fill="#D1D5DB", width=1)
        text_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - text_box[2] - 10, y - 10), label, font=tick_font, fill="#4B5563")
    for value, label in x_ticks:
        x = xp(value)
        draw.line((x, bottom, x, bottom + 6), fill="#374151", width=2)
        text_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - text_box[2] / 2, bottom + 10), label, font=tick_font, fill="#4B5563")
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((left + right - x_box[2]) / 2, bottom + 45), x_label, font=label_font, fill="#111827")
    draw.text((left, top - 35), y_label, font=label_font, fill="#111827")
    return xp, yp


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[str]) -> None:
    legend_font = font(18)
    cursor = x
    for label in labels:
        color = COLORS[label]
        draw.line((cursor, y + 10, cursor + 34, y + 10), fill=color, width=5)
        draw.text((cursor + 42, y), label, font=legend_font, fill="#111827")
        cursor += 56 + draw.textbbox((0, 0), label, font=legend_font)[2]


def load_archive(path: Path) -> tuple[list[dict], dict[str, dict[str, np.ndarray]], list[dict[str, str]]]:
    records: list[dict] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(path) as archive:
        summary = list(csv.DictReader(io.StringIO(archive.read("summary.csv").decode("utf-8"))))
        summary_by_task = {row["task_id"]: row for row in summary}
        for name in sorted(archive.namelist()):
            if not (name.startswith("records/") and name.endswith(".json")):
                continue
            record = json.loads(archive.read(name))
            record["_archive_record_path"] = name
            record["_summary"] = summary_by_task[record["task_id"]]
            prediction_path = record["artifacts"]["prediction_npz"]["path"]
            with np.load(io.BytesIO(archive.read(prediction_path))) as arrays:
                predictions[record["task_id"]] = {key: arrays[key].copy() for key in arrays.files}
            records.append(record)
    return records, predictions, summary


def copy_records(archive_path: Path, records: list[dict], output_dir: Path) -> None:
    target = output_dir / "records"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            name = record["_archive_record_path"]
            (target / Path(name).name).write_bytes(archive.read(name))


def derive(records: list[dict], predictions: dict[str, dict[str, np.ndarray]]) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        condition = CONDITIONS[(record["experiment"], record["variant"])]
        arrays = predictions[record["task_id"]]
        residual_ot = arrays["y_pred_ot"].astype(np.float64) - arrays["y_true_ot"].astype(np.float64)
        record["_condition"] = condition
        record["_ot_rmse"] = float(np.sqrt(np.mean(residual_ot**2)))
        record["_ot_mae"] = float(np.mean(np.abs(residual_ot)))
        record["_step_rmse"] = np.sqrt(np.mean(residual_ot**2, axis=0))
        record["_residual_ot"] = residual_ot
        groups[(record["dataset"], condition)].append(record)

    condition_rows: list[dict] = []
    step_rows: list[dict] = []
    for (dataset, condition), runs in sorted(groups.items()):
        runs.sort(key=lambda item: item["seed"])
        fields = {
            "standardized_mse": [float(run["test_metrics"]["mse"]) for run in runs],
            "standardized_mae": [float(run["test_metrics"]["mae"]) for run in runs],
            "ot_rmse": [run["_ot_rmse"] for run in runs],
            "ot_mae": [run["_ot_mae"] for run in runs],
            "training_s": [float(run["elapsed_s"]) for run in runs],
            "inference_ms_per_sample": [float(run["_summary"]["inference_ms_per_sample"]) for run in runs],
            "peak_gpu_memory_mb": [float(run["_summary"]["peak_gpu_memory_mb"]) for run in runs],
            "selected_epoch": [float(run["training"]["selected_epoch"]) for run in runs],
        }
        row = {"dataset": dataset, "condition": condition, "n": len(runs),
               "parameters": int(runs[0]["parameter_count"]["trainable"])}
        for name, values in fields.items():
            row[f"{name}_mean"], row[f"{name}_sample_sd"] = mean_sd(values)
        condition_rows.append(row)
        for step in range(96):
            values = [float(run["_step_rmse"][step]) for run in runs]
            mean, sample_sd = mean_sd(values)
            step_rows.append({"dataset": dataset, "condition": condition, "forecast_step": step + 1,
                              "ot_rmse_mean": mean, "ot_rmse_sample_sd": sample_sd})

    paired_rows: list[dict] = []
    comparisons = [("DLinear", "Corrected KAN"), ("No-update KAN", "Moderate-update KAN")]
    for dataset in ("ETTh1", "ETTh2"):
        for reference, comparison in comparisons:
            left = {run["seed"]: run for run in groups[(dataset, reference)]}
            right = {run["seed"]: run for run in groups[(dataset, comparison)]}
            deltas = []
            for seed in sorted(left):
                delta = float(right[seed]["test_metrics"]["mse"] - left[seed]["test_metrics"]["mse"])
                deltas.append(delta)
                paired_rows.append({"dataset": dataset, "reference": reference, "comparison": comparison,
                                    "seed": seed, "mse_delta_comparison_minus_reference": delta})
            paired_rows.append({"dataset": dataset, "reference": reference, "comparison": comparison,
                                "seed": "mean", "mse_delta_comparison_minus_reference": statistics.mean(deltas)})
    return condition_rows, step_rows, paired_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_actual(records: list[dict], predictions: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    image, draw = canvas(2400, 1500, "Uniform-checkpoint forecasts at the fixed midpoint test origin")
    panel_font = font(25, bold=True)
    origin = 1392
    for panel, dataset in enumerate(("ETTh1", "ETTh2")):
        left, right = 170, 2300
        top = 150 + panel * 650
        bottom = top + 500
        selected = [r for r in records if r["dataset"] == dataset and r["_condition"] in {"DLinear", "Corrected KAN"}]
        actual = predictions[selected[0]["task_id"]]["y_true_ot"][origin].astype(float)
        series = {"Actual": (actual, np.zeros_like(actual))}
        for condition in ("DLinear", "Corrected KAN"):
            runs = sorted((r for r in selected if r["_condition"] == condition), key=lambda r: r["seed"])
            values = np.asarray([predictions[r["task_id"]]["y_pred_ot"][origin] for r in runs], dtype=float)
            series[condition] = (values.mean(axis=0), values.std(axis=0, ddof=1))
        low = min(float(np.min(mean - sd)) for mean, sd in series.values())
        high = max(float(np.max(mean + sd)) for mean, sd in series.values())
        padding = max(0.5, (high - low) * 0.08)
        y_min, y_max = low - padding, high + padding
        ticks = [(value, f"{value:.0f}") for value in np.linspace(y_min, y_max, 5)]
        xp, yp = axes(draw, (left, top, right, bottom), "Forecast step", "OT", [(1, "1"), (24, "24"), (48, "48"), (72, "72"), (96, "96")], ticks, (1, 96), (y_min, y_max))
        draw.text((left, top - 76), dataset, font=panel_font, fill="#111827")
        for condition in ("DLinear", "Corrected KAN"):
            mean, sd = series[condition]
            upper = [(xp(i + 1), yp(float(mean[i] + sd[i]))) for i in range(96)]
            lower = [(xp(i + 1), yp(float(mean[i] - sd[i]))) for i in range(95, -1, -1)]
            draw.polygon(upper + lower, fill=COLORS[condition] + "28")
        for condition in ("Actual", "DLinear", "Corrected KAN"):
            mean, _ = series[condition]
            draw.line([(xp(i + 1), yp(float(mean[i]))) for i in range(96)], fill=COLORS[condition], width=5 if condition == "Actual" else 4)
        legend(draw, right - 790, top - 74, ["Actual", "DLinear", "Corrected KAN"])
    image.save(output, dpi=(300, 300))


def plot_residuals(records: list[dict], output: Path) -> None:
    image, draw = canvas(2400, 1550, "Residual distributions under the uniform checkpoint rule")
    panel_font = font(24, bold=True)
    legend(draw, 920, 94, ["DLinear", "Corrected KAN"])
    for column, dataset in enumerate(("ETTh1", "ETTh2")):
        subset = [r for r in records if r["dataset"] == dataset and r["_condition"] in {"DLinear", "Corrected KAN"}]
        residuals = {condition: np.concatenate([r["_residual_ot"].ravel() for r in subset if r["_condition"] == condition])
                     for condition in ("DLinear", "Corrected KAN")}
        limit = max(float(np.quantile(np.abs(values), 0.99)) for values in residuals.values())
        left = 160 + column * 1170
        right = left + 1010
        top, bottom = 170, 690
        xp, yp = axes(draw, (left, top, right, bottom), "Residual (predicted - actual OT)", "Density",
                      [(-limit, f"{-limit:.0f}"), (0, "0"), (limit, f"{limit:.0f}")],
                      [(0, "0"), (0.5, "0.5"), (1, "1")], (-limit, limit), (0, 1))
        for condition in ("DLinear", "Corrected KAN"):
            hist, edges = np.histogram(residuals[condition], bins=100, range=(-limit, limit), density=True)
            hist = hist / max(hist.max(), 1e-12)
            draw.line([(xp(float((edges[i] + edges[i + 1]) / 2)), yp(float(hist[i]))) for i in range(len(hist))], fill=COLORS[condition], width=4)
        draw.text((left, top - 76), f"{dataset}: signed residual density (peak-normalized)", font=panel_font, fill="#111827")

        top, bottom = 920, 1410
        absolute = {condition: np.abs(values) for condition, values in residuals.items()}
        x_max = max(float(np.quantile(values, 0.99)) for values in absolute.values())
        xp, yp = axes(draw, (left, top, right, bottom), "Absolute error (OT)", "Empirical probability",
                      [(0, "0"), (x_max / 2, f"{x_max/2:.1f}"), (x_max, f"{x_max:.1f}")],
                      [(0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1, "1")], (0, x_max), (0, 1))
        quantiles = np.linspace(0, 0.99, 500)
        for condition in ("DLinear", "Corrected KAN"):
            values = np.quantile(absolute[condition], quantiles)
            draw.line([(xp(float(value)), yp(float(q))) for value, q in zip(values, quantiles)], fill=COLORS[condition], width=4)
        draw.text((left, top - 76), f"{dataset}: absolute-error ECDF", font=panel_font, fill="#111827")
    image.save(output, dpi=(300, 300))


def plot_steps(step_rows: list[dict], output: Path) -> None:
    image, draw = canvas(2400, 1500, "Error growth across the 96-step forecast horizon")
    panel_font = font(25, bold=True)
    for panel, dataset in enumerate(("ETTh1", "ETTh2")):
        selected = [row for row in step_rows if row["dataset"] == dataset]
        y_max = max(row["ot_rmse_mean"] + row["ot_rmse_sample_sd"] for row in selected) * 1.08
        left, right = 170, 2300
        top = 150 + panel * 650
        bottom = top + 500
        xp, yp = axes(draw, (left, top, right, bottom), "Forecast step", "OT RMSE",
                      [(1, "1"), (24, "24"), (48, "48"), (72, "72"), (96, "96")],
                      [(value, f"{value:.1f}") for value in np.linspace(0, y_max, 5)], (1, 96), (0, y_max))
        draw.text((left, top - 76), dataset, font=panel_font, fill="#111827")
        for condition in ("DLinear", "Corrected KAN", "No-update KAN", "Moderate-update KAN"):
            rows = sorted((row for row in selected if row["condition"] == condition), key=lambda row: row["forecast_step"])
            upper = [(xp(row["forecast_step"]), yp(row["ot_rmse_mean"] + row["ot_rmse_sample_sd"])) for row in rows]
            lower = [(xp(row["forecast_step"]), yp(max(0, row["ot_rmse_mean"] - row["ot_rmse_sample_sd"]))) for row in reversed(rows)]
            draw.polygon(upper + lower, fill=COLORS[condition] + "1C")
            draw.line([(xp(row["forecast_step"]), yp(row["ot_rmse_mean"])) for row in rows], fill=COLORS[condition], width=4)
        legend(draw, right - 1360, top - 74, ["DLinear", "Corrected KAN", "No-update KAN", "Moderate-update KAN"])
    image.save(output, dpi=(300, 300))


def plot_compute(records: list[dict], condition_rows: list[dict], output: Path) -> None:
    image, draw = canvas(2400, 1500, "Compute cost and validation-selected checkpoints")
    panel_font = font(25, bold=True)
    conditions = ["DLinear", "Corrected KAN", "No-update KAN", "Moderate-update KAN"]
    short = {"DLinear": "DLinear", "Corrected KAN": "Corrected\nKAN",
             "No-update KAN": "No-update\nKAN", "Moderate-update KAN": "Moderate\nKAN"}
    for panel, dataset in enumerate(("ETTh1", "ETTh2")):
        top = 165 + panel * 665
        bottom = top + 480
        subset = [row for row in condition_rows if row["dataset"] == dataset]
        max_time = max(row["training_s_mean"] + row["training_s_sample_sd"] for row in subset) * 1.12
        left, right = 160, 1280
        xp, yp = axes(draw, (left, top, right, bottom), "Condition", "Training time (s)",
                      [(i + 1, short[condition]) for i, condition in enumerate(conditions)],
                      [(value, f"{value:.0f}") for value in np.linspace(0, max_time, 5)], (0.5, 4.5), (0, max_time))
        for index, condition in enumerate(conditions, 1):
            row = next(item for item in subset if item["condition"] == condition)
            x0, x1 = xp(index - 0.28), xp(index + 0.28)
            draw.rectangle((x0, yp(row["training_s_mean"]), x1, yp(0)), fill=COLORS[condition])
            draw.line((xp(index), yp(row["training_s_mean"] - row["training_s_sample_sd"]), xp(index), yp(row["training_s_mean"] + row["training_s_sample_sd"])), fill="#111827", width=3)
        draw.text((left, top - 76), f"{dataset}: training cost", font=panel_font, fill="#111827")

        left, right = 1510, 2290
        xp, yp = axes(draw, (left, top, right, bottom), "Condition", "Selected epoch",
                      [(i + 1, short[condition]) for i, condition in enumerate(conditions)],
                      [(0, "0"), (5, "5"), (10, "10"), (15, "15"), (20, "20"), (25, "25"), (30, "30")],
                      (0.5, 4.5), (0, 30))
        for index, condition in enumerate(conditions, 1):
            epochs = sorted(r["training"]["selected_epoch"] for r in records if r["dataset"] == dataset and r["_condition"] == condition)
            for offset, epoch in zip((-0.12, -0.06, 0, 0.06, 0.12), epochs):
                x, y = xp(index + offset), yp(epoch)
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=COLORS[condition], outline="#111827", width=2)
        draw.text((left, top - 76), f"{dataset}: checkpoint selection", font=panel_font, fill="#111827")
    image.save(output, dpi=(300, 300))


def write_report(path: Path, condition_rows: list[dict], paired_rows: list[dict]) -> None:
    lines = [
        "# K09 Uniform-Checkpoint Diagnostic Report", "",
        f"Campaign: `{CAMPAIGN_ID}`", "",
        f"Result archive SHA-256: `{ARCHIVE_SHA256}`", "",
        f"Executed source combined SHA-256: `{SOURCE_SHA256}`", "",
        "All values below are derived directly from the frozen ZIP. K08 and K09 are not pooled.", "",
        "| Dataset | Condition | MSE | OT RMSE | OT MAE | Train s | Inference ms/sample | Peak GPU MB | Params | Selected epoch |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in condition_rows:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['standardized_mse_mean']:.6f} +/- {row['standardized_mse_sample_sd']:.6f} | "
            f"{row['ot_rmse_mean']:.4f} +/- {row['ot_rmse_sample_sd']:.4f} | {row['ot_mae_mean']:.4f} +/- {row['ot_mae_sample_sd']:.4f} | "
            f"{row['training_s_mean']:.2f} +/- {row['training_s_sample_sd']:.2f} | {row['inference_ms_per_sample_mean']:.5f} +/- {row['inference_ms_per_sample_sample_sd']:.5f} | "
            f"{row['peak_gpu_memory_mb_mean']:.2f} | {row['parameters']} | {row['selected_epoch_mean']:.1f} +/- {row['selected_epoch_sample_sd']:.1f} |"
        )
    lines.extend(["", "## Paired conclusions", ""])
    for dataset in ("ETTh1", "ETTh2"):
        for reference, comparison in (("DLinear", "Corrected KAN"), ("No-update KAN", "Moderate-update KAN")):
            rows = [row for row in paired_rows if row["dataset"] == dataset and row["reference"] == reference and row["comparison"] == comparison and row["seed"] != "mean"]
            worse = sum(row["mse_delta_comparison_minus_reference"] > 0 for row in rows)
            mean_delta = statistics.mean(row["mse_delta_comparison_minus_reference"] for row in rows)
            lines.append(f"- {dataset}: {comparison} minus {reference} mean MSE delta `{mean_delta:.6f}`; comparison is worse in `{worse}/5` paired seeds.")
    lines.extend(["", "## Interpretation boundary", "", "K09 answers checkpoint-contract and diagnostic questions under the official ETT h96 protocol. It does not establish state-of-the-art performance, universal KAN inferiority, multivariate validity or external-domain generalization.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=ROOT / "canonical_uniform_best_diagnostics.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "k09_diagnostics")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    if sha256(args.archive) != ARCHIVE_SHA256:
        raise SystemExit("K09 archive SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    records, predictions, summary = load_archive(args.archive)
    if len(records) != 40:
        raise SystemExit(f"expected 40 records, found {len(records)}")
    copy_records(args.archive, records, args.output_dir)
    condition_rows, step_rows, paired_rows = derive(records, predictions)
    write_csv(args.output_dir / "condition_summary.csv", condition_rows)
    write_csv(args.output_dir / "forecast_step_rmse_ot.csv", step_rows)
    write_csv(args.output_dir / "paired_mse_differences.csv", paired_rows)
    write_csv(args.output_dir / "archive_summary.csv", summary)
    plot_actual(records, predictions, args.figures_dir / "k09_actual_vs_predicted.png")
    plot_residuals(records, args.figures_dir / "k09_residual_diagnostics.png")
    plot_steps(step_rows, args.figures_dir / "k09_forecast_step_rmse.png")
    plot_compute(records, condition_rows, args.figures_dir / "k09_compute_checkpoints.png")
    write_report(args.output_dir / "K09_DIAGNOSTIC_REPORT.md", condition_rows, paired_rows)
    print(f"K09 diagnostics written to {args.output_dir} and {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

