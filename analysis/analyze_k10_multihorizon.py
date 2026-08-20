"""Derive K10 multi-horizon tables and figures from the immutable ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "canonical_multihorizon_baselines_v2"
ARCHIVE_SHA256 = "BD3BEA46869C555F89243C954533B827A5151F6F8BD006BAF9693A321281D281"
MODELS = {
    ("baseline", "DLinear"): "DLinear",
    ("baseline", "PatchTST"): "PatchTST",
    ("improvement", "KAN_WOA_RegBestClip"): "Corrected KAN",
}
COLORS = {
    "Actual": "#111827",
    "DLinear": "#2E86AB",
    "PatchTST": "#2A9D8F",
    "Corrected KAN": "#D1495B",
}
HORIZONS = (96, 192, 336, 720)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("C:/Windows/Fonts") / ("calibrib.ttf" if bold else "calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def canvas(width: int, height: int, title: str):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(34, True)
    bounds = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - bounds[2]) / 2, 24), title, font=title_font, fill="#111827")
    return image, draw


def axes(draw, box, x_label, y_label, x_ticks, y_ticks, x_range, y_range):
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill="#374151", width=2)
    draw.line((left, bottom, right, bottom), fill="#374151", width=2)
    tick_font = font(18)
    label_font = font(21)

    def xp(value):
        return left + (value - x_range[0]) / (x_range[1] - x_range[0]) * (right - left)

    def yp(value):
        return bottom - (value - y_range[0]) / (y_range[1] - y_range[0]) * (bottom - top)

    for value, label in y_ticks:
        y = yp(value)
        draw.line((left, y, right, y), fill="#D1D5DB", width=1)
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - bounds[2] - 10, y - 10), label, font=tick_font, fill="#4B5563")
    for value, label in x_ticks:
        x = xp(value)
        draw.line((x, bottom, x, bottom + 6), fill="#374151", width=2)
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - bounds[2] / 2, bottom + 10), label, font=tick_font, fill="#4B5563")
    bounds = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((left + right - bounds[2]) / 2, bottom + 45), x_label, font=label_font, fill="#111827")
    draw.text((left, top - 35), y_label, font=label_font, fill="#111827")
    return xp, yp


def legend(draw, x: int, y: int, labels: list[str]):
    item_font = font(18)
    cursor = x
    for label in labels:
        draw.line((cursor, y + 10, cursor + 34, y + 10), fill=COLORS[label], width=5)
        draw.text((cursor + 42, y), label, font=item_font, fill="#111827")
        cursor += 58 + draw.textbbox((0, 0), label, font=item_font)[2]


def load_archive(path: Path):
    records = []
    predictions = {}
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
    return records, predictions


def copy_records(archive_path: Path, records: list[dict], output_dir: Path):
    target = output_dir / "records"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            name = record["_archive_record_path"]
            (target / Path(name).name).write_bytes(archive.read(name))


def derive(records, predictions):
    groups = defaultdict(list)
    per_seed = []
    for record in records:
        model = MODELS[(record["experiment"], record["variant"])]
        arrays = predictions[record["task_id"]]
        residual = arrays["y_pred_ot"].astype(np.float64) - arrays["y_true_ot"].astype(np.float64)
        record["_model"] = model
        record["_ot_rmse"] = float(np.sqrt(np.mean(residual**2)))
        record["_ot_mae"] = float(np.mean(np.abs(residual)))
        record["_step_rmse"] = np.sqrt(np.mean(residual**2, axis=0))
        groups[(record["dataset"], int(record["horizon"]), model)].append(record)
        per_seed.append({
            "dataset": record["dataset"], "horizon": int(record["horizon"]),
            "model": model, "seed": int(record["seed"]),
            "selected_epoch": int(record["training"]["selected_epoch"]),
            "standardized_mse": float(record["test_metrics"]["mse"]),
            "standardized_mae": float(record["test_metrics"]["mae"]),
            "ot_rmse": record["_ot_rmse"], "ot_mae": record["_ot_mae"],
            "training_s": float(record["elapsed_s"]),
            "inference_ms_per_sample": float(record["test_metrics"]["inference_ms_per_sample"]),
            "peak_gpu_memory_mb": float(record["compute_diagnostics"]["peak_gpu_memory_mb"]),
            "parameters": int(record["parameter_count"]["trainable"]),
        })

    summary_rows = []
    step_rows = []
    for (dataset, horizon, model), runs in sorted(groups.items()):
        runs.sort(key=lambda item: item["seed"])
        row = {"dataset": dataset, "horizon": horizon, "model": model, "n": len(runs),
               "parameters": int(runs[0]["parameter_count"]["trainable"])}
        fields = {
            "standardized_mse": [float(run["test_metrics"]["mse"]) for run in runs],
            "standardized_mae": [float(run["test_metrics"]["mae"]) for run in runs],
            "ot_rmse": [run["_ot_rmse"] for run in runs],
            "ot_mae": [run["_ot_mae"] for run in runs],
            "training_s": [float(run["elapsed_s"]) for run in runs],
            "inference_ms_per_sample": [float(run["test_metrics"]["inference_ms_per_sample"]) for run in runs],
            "peak_gpu_memory_mb": [float(run["compute_diagnostics"]["peak_gpu_memory_mb"]) for run in runs],
            "selected_epoch": [float(run["training"]["selected_epoch"]) for run in runs],
        }
        for name, values in fields.items():
            row[f"{name}_mean"], row[f"{name}_sample_sd"] = mean_sd(values)
        summary_rows.append(row)
        for step in range(horizon):
            values = [float(run["_step_rmse"][step]) for run in runs]
            mean, sd = mean_sd(values)
            step_rows.append({"dataset": dataset, "horizon": horizon, "model": model,
                              "forecast_step": step + 1, "ot_rmse_mean": mean,
                              "ot_rmse_sample_sd": sd})
    return summary_rows, sorted(per_seed, key=lambda row: (row["dataset"], row["horizon"], row["model"], row["seed"])), step_rows


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows: list[dict], output: Path):
    image, draw = canvas(2400, 1500, "Canonical multi-horizon forecasting accuracy")
    panel_font = font(25, True)
    for panel, dataset in enumerate(("ETTh1", "ETTh2")):
        subset = [row for row in rows if row["dataset"] == dataset]
        y_max = max(row["standardized_mse_mean"] + row["standardized_mse_sample_sd"] for row in subset) * 1.1
        left, right = 170, 2300
        top = 150 + panel * 650
        bottom = top + 500
        xp, yp = axes(draw, (left, top, right, bottom), "Forecast horizon", "Test MSE",
                      [(h, str(h)) for h in HORIZONS],
                      [(value, f"{value:.2f}") for value in np.linspace(0, y_max, 6)],
                      (80, 736), (0, y_max))
        draw.text((left, top - 76), dataset, font=panel_font, fill="#111827")
        for model in ("DLinear", "PatchTST", "Corrected KAN"):
            model_rows = sorted((row for row in subset if row["model"] == model), key=lambda row: row["horizon"])
            upper = [(xp(row["horizon"]), yp(row["standardized_mse_mean"] + row["standardized_mse_sample_sd"])) for row in model_rows]
            lower = [(xp(row["horizon"]), yp(max(0, row["standardized_mse_mean"] - row["standardized_mse_sample_sd"]))) for row in reversed(model_rows)]
            draw.polygon(upper + lower, fill=COLORS[model] + "24")
            points = [(xp(row["horizon"]), yp(row["standardized_mse_mean"])) for row in model_rows]
            draw.line(points, fill=COLORS[model], width=5)
            for x, y in points:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=COLORS[model], outline="white", width=2)
        legend(draw, right - 1040, top - 74, ["DLinear", "PatchTST", "Corrected KAN"])
    image.save(output, dpi=(300, 300))


def plot_compute(rows: list[dict], output: Path):
    image, draw = canvas(2400, 1600, "Canonical multi-horizon compute profiles on Tesla T4")
    panel_font = font(23, True)
    legend(draw, 875, 94, ["DLinear", "PatchTST", "Corrected KAN"])
    for column, dataset in enumerate(("ETTh1", "ETTh2")):
        subset = [row for row in rows if row["dataset"] == dataset]
        left = 150 + column * 1170
        right = left + 1020
        for row_index, (field, ylabel) in enumerate((
            ("training_s_mean", "Training time (s)"),
            ("inference_ms_per_sample_mean", "Inference (ms/sample)"),
        )):
            top = 190 + row_index * 680
            bottom = top + 500
            y_max = max(row[field] for row in subset) * 1.12
            xp, yp = axes(draw, (left, top, right, bottom), "Forecast horizon", ylabel,
                          [(h, str(h)) for h in HORIZONS],
                          [(value, f"{value:.2f}" if y_max < 10 else f"{value:.0f}") for value in np.linspace(0, y_max, 5)],
                          (80, 736), (0, y_max))
            draw.text((left, top - 76), f"{dataset}: {ylabel.lower()}", font=panel_font, fill="#111827")
            for model in ("DLinear", "PatchTST", "Corrected KAN"):
                model_rows = sorted((row for row in subset if row["model"] == model), key=lambda row: row["horizon"])
                points = [(xp(row["horizon"]), yp(row[field])) for row in model_rows]
                draw.line(points, fill=COLORS[model], width=5)
                for x, y in points:
                    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[model])
    image.save(output, dpi=(300, 300))


def plot_long_forecast(records, predictions, output: Path):
    image, draw = canvas(2400, 1500, "Canonical 720-step forecasts at the fixed midpoint test origin")
    panel_font = font(25, True)
    for panel, dataset in enumerate(("ETTh1", "ETTh2")):
        selected = [record for record in records if record["dataset"] == dataset and int(record["horizon"]) == 720]
        origin = predictions[selected[0]["task_id"]]["y_true_ot"].shape[0] // 2
        actual = predictions[selected[0]["task_id"]]["y_true_ot"][origin].astype(float)
        series = {"Actual": (actual, np.zeros_like(actual))}
        for model in ("DLinear", "PatchTST", "Corrected KAN"):
            runs = sorted((record for record in selected if record["_model"] == model), key=lambda record: record["seed"])
            values = np.asarray([predictions[record["task_id"]]["y_pred_ot"][origin] for record in runs], dtype=float)
            series[model] = (values.mean(axis=0), values.std(axis=0, ddof=1))
        low = min(float(np.min(mean - sd)) for mean, sd in series.values())
        high = max(float(np.max(mean + sd)) for mean, sd in series.values())
        padding = max(0.5, (high - low) * 0.06)
        left, right = 170, 2300
        top = 150 + panel * 650
        bottom = top + 500
        xp, yp = axes(draw, (left, top, right, bottom), "Forecast step", "OT",
                      [(1, "1"), (180, "180"), (360, "360"), (540, "540"), (720, "720")],
                      [(value, f"{value:.0f}") for value in np.linspace(low - padding, high + padding, 5)],
                      (1, 720), (low - padding, high + padding))
        draw.text((left, top - 76), dataset, font=panel_font, fill="#111827")
        for model in ("DLinear", "PatchTST", "Corrected KAN"):
            mean, sd = series[model]
            upper = [(xp(i + 1), yp(float(mean[i] + sd[i]))) for i in range(720)]
            lower = [(xp(i + 1), yp(float(mean[i] - sd[i]))) for i in range(719, -1, -1)]
            draw.polygon(upper + lower, fill=COLORS[model] + "18")
        for model in ("Actual", "DLinear", "PatchTST", "Corrected KAN"):
            mean, _ = series[model]
            draw.line([(xp(i + 1), yp(float(mean[i]))) for i in range(720)],
                      fill=COLORS[model], width=5 if model == "Actual" else 3)
        legend(draw, right - 1360, top - 74, ["Actual", "DLinear", "PatchTST", "Corrected KAN"])
    image.save(output, dpi=(300, 300))


def write_report(path: Path, rows: list[dict], source_hash: str):
    lines = [
        "# K10 Canonical Multi-Horizon Report", "",
        f"Campaign: `{CAMPAIGN_ID}`", "",
        f"Archive SHA-256: `{ARCHIVE_SHA256}`", "",
        f"Executed source combined SHA-256: `{source_hash}`", "",
        "Values are mean +/- sample SD over five seeds; MSE is in standardized space.", "",
        "| Dataset | Horizon | Model | MSE | OT RMSE | Train s | Inference ms/sample | Peak GPU MB | Params |", "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {row['model']} | "
            f"{row['standardized_mse_mean']:.6f} +/- {row['standardized_mse_sample_sd']:.6f} | "
            f"{row['ot_rmse_mean']:.4f} +/- {row['ot_rmse_sample_sd']:.4f} | "
            f"{row['training_s_mean']:.2f} +/- {row['training_s_sample_sd']:.2f} | "
            f"{row['inference_ms_per_sample_mean']:.5f} +/- {row['inference_ms_per_sample_sample_sd']:.5f} | "
            f"{row['peak_gpu_memory_mb_mean']:.1f} +/- {row['peak_gpu_memory_mb_sample_sd']:.1f} | {row['parameters']} |"
        )
    lines.extend(["", "## Pairwise outcome counts", ""])
    for dataset in ("ETTh1", "ETTh2"):
        for left, right in (("DLinear", "Corrected KAN"), ("DLinear", "PatchTST"), ("PatchTST", "Corrected KAN")):
            wins = total = 0
            for horizon in HORIZONS:
                left_runs = {row["seed"]: row for row in PER_SEED_CACHE if row["dataset"] == dataset and row["horizon"] == horizon and row["model"] == left}
                right_runs = {row["seed"]: row for row in PER_SEED_CACHE if row["dataset"] == dataset and row["horizon"] == horizon and row["model"] == right}
                for seed in left_runs:
                    wins += left_runs[seed]["standardized_mse"] < right_runs[seed]["standardized_mse"]
                    total += 1
            lines.append(f"- {dataset}: {left} has lower MSE than {right} in {wins}/{total} paired horizon-seed comparisons.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


PER_SEED_CACHE: list[dict] = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=ROOT / "canonical_multihorizon_baselines_v2.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "k10_multihorizon")
    args = parser.parse_args()
    if file_sha256(args.archive) != ARCHIVE_SHA256:
        raise SystemExit("K10 archive SHA-256 does not match the registered archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, predictions = load_archive(args.archive)
    source_hashes = {record["source"]["combined_sha256"] for record in records}
    if len(source_hashes) != 1:
        raise SystemExit(f"Expected one source hash, found {source_hashes}")
    rows, per_seed, step_rows = derive(records, predictions)
    global PER_SEED_CACHE
    PER_SEED_CACHE = per_seed
    copy_records(args.archive, records, args.output_dir)
    write_csv(args.output_dir / "k10_condition_summary.csv", rows)
    write_csv(args.output_dir / "k10_per_seed.csv", per_seed)
    write_csv(args.output_dir / "k10_forecast_step_rmse.csv", step_rows)
    write_report(args.output_dir / "K10_MULTIHORIZON_REPORT.md", rows, next(iter(source_hashes)))
    plot_accuracy(rows, ROOT / "figures" / "k10_canonical_multihorizon.png")
    plot_compute(rows, ROOT / "figures" / "k10_multihorizon_compute.png")
    plot_long_forecast(records, predictions, ROOT / "figures" / "k10_h720_forecasts.png")
    print(f"K10 analysis complete: {len(records)} records, {len(rows)} summary cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
