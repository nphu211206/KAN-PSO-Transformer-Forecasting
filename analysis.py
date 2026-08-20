"""Audit and summarize historical HPO plus targeted follow-up results.

With three runs per HPO algorithm, this script reports descriptive statistics
only.  It deliberately does not manufacture Wilcoxon/Friedman significance
claims from an underpowered sample.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORICAL = (
    PROJECT_ROOT / "my_results" / "kaggle" / "working" / "experiments" / "results"
)
DEFAULT_TARGETED = PROJECT_ROOT / "targeted_results" / "records"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis"
DEFAULT_FIGURES = PROJECT_ROOT / "figures"

COLORS = {
    "PSO": "#2E86AB",
    "GWO": "#D1495B",
    "DE": "#F79256",
    "GA": "#2A9D8F",
    "WOA": "#7656A8",
    "Random": "#6B7280",
}


def load_json_records(directory: Path) -> List[dict]:
    records: List[dict] = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["_source_file"] = path.name
            records.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: cannot read {path}: {exc}")
    return records


def finite_values(values: Iterable[object]) -> List[float]:
    output: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            output.append(numeric)
    return output


def describe(values: Iterable[object]) -> dict:
    clean = np.asarray(finite_values(values), dtype=float)
    n = int(clean.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "median": None,
            "best_min": None,
            "worst_max": None,
            "coefficient_of_variation": None,
        }
    mean = float(clean.mean())
    sample_std = float(clean.std(ddof=1)) if n >= 2 else None
    cv = None if sample_std is None or mean == 0 else abs(sample_std / mean)
    return {
        "n": n,
        "mean": mean,
        "sample_std": sample_std,
        "median": float(np.median(clean)),
        "best_min": float(clean.min()),
        "worst_max": float(clean.max()),
        "coefficient_of_variation": cv,
    }


def group_by(records: Iterable[dict], key: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        grouped.setdefault(value, []).append(record)
    return grouped


def historical_hpo_summary(records: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for algorithm, runs in group_by(records, "algorithm").items():
        stats = describe(run.get("hpo_fitness") for run in runs)
        rows.append(
            {
                "algorithm": algorithm,
                **stats,
                "expected_runs": 3,
                "missing_or_nonfinite_runs": max(0, 3 - stats["n"]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", na_position="last")


def _historical_metric(record: dict, horizon: int, metric: str):
    return record.get("test_metrics", {}).get(f"ETTh1_{horizon}", {}).get(metric)


def historical_test_summary(records: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for algorithm, runs in group_by(records, "algorithm").items():
        for horizon in (96, 192, 336, 720):
            mse = describe(_historical_metric(run, horizon, "mse") for run in runs)
            mae = describe(_historical_metric(run, horizon, "mae") for run in runs)
            rows.append(
                {
                    "algorithm": algorithm,
                    "dataset": "ETTh1",
                    "horizon": horizon,
                    "mse_mean": mse["mean"],
                    "mse_sample_std": mse["sample_std"],
                    "mse_median": mse["median"],
                    "mse_best_min": mse["best_min"],
                    "mse_worst_max": mse["worst_max"],
                    "mse_cv": mse["coefficient_of_variation"],
                    "mae_mean": mae["mean"],
                    "mae_sample_std": mae["sample_std"],
                    "mae_median": mae["median"],
                    "valid_runs": mse["n"],
                    "missing_or_nonfinite_runs": max(0, 3 - mse["n"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "mse_mean"], na_position="last")


def plot_hpo_fitness(records: List[dict], figures_dir: Path) -> None:
    grouped = group_by(records, "algorithm")
    algorithms = sorted(
        grouped,
        key=lambda name: np.mean(finite_values(r.get("hpo_fitness") for r in grouped[name])),
    )
    values = [finite_values(record.get("hpo_fitness") for record in grouped[name]) for name in algorithms]
    if not values or any(not item for item in values):
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    boxes = axis.boxplot(values, patch_artist=True, widths=0.55, showfliers=False)
    rng = np.random.default_rng(2026)
    for index, (box, name, points) in enumerate(zip(boxes["boxes"], algorithms, values), 1):
        box.set_facecolor(COLORS.get(name, "#777777"))
        box.set_alpha(0.65)
        jitter = rng.uniform(-0.06, 0.06, len(points))
        axis.scatter(np.full(len(points), index) + jitter, points, color="#111111", s=28, zorder=3)
    axis.set_xticks(range(1, len(algorithms) + 1), algorithms)
    axis.set_xlabel("HPO algorithm")
    axis.set_ylabel("Best validation fitness (MSE, lower is better)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "hpo_fitness_three_runs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(records: List[dict], figures_dir: Path) -> None:
    grouped = group_by(records, "algorithm")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    plotted = False
    for algorithm, runs in grouped.items():
        curves = [finite_values(run.get("convergence", [])) for run in runs]
        curves = [curve for curve in curves if curve]
        if not curves:
            continue
        length = max(map(len, curves))
        padded = np.asarray([curve + [curve[-1]] * (length - len(curve)) for curve in curves])
        mean = padded.mean(axis=0)
        sample_std = padded.std(axis=0, ddof=1) if len(padded) >= 2 else np.zeros(length)
        x_axis = np.arange(1, length + 1)
        color = COLORS.get(algorithm, "#777777")
        axis.plot(x_axis, mean, label=algorithm, color=color, linewidth=2)
        axis.fill_between(x_axis, mean - sample_std, mean + sample_std, color=color, alpha=0.12)
        plotted = True
    if plotted:
        axis.set_xlabel("HPO iteration")
        axis.set_ylabel("Global-best validation fitness")
        axis.set_title("HPO convergence, mean and sample SD across three runs")
        axis.legend(ncol=2)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "hpo_convergence_three_runs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def targeted_summary(records: List[dict]) -> pd.DataFrame:
    completed = [record for record in records if record.get("status") == "completed"]
    groups: Dict[tuple, List[dict]] = {}
    for record in completed:
        key = (
            record.get("experiment"),
            record.get("variant"),
            record.get("dataset"),
            record.get("horizon"),
            record.get("protocol_name"),
        )
        groups.setdefault(key, []).append(record)
    rows: List[dict] = []
    for key, runs in groups.items():
        mse = describe(run.get("test_metrics", {}).get("mse") for run in runs)
        mae = describe(run.get("test_metrics", {}).get("mae") for run in runs)
        grid_events = [
            event
            for run in runs
            for event in run.get("training", {}).get("grid_updates", [])
        ]
        rows.append(
            {
                "experiment": key[0],
                "variant": key[1],
                "dataset": key[2],
                "horizon": key[3],
                "protocol": key[4],
                "runs": mse["n"],
                "test_mse_mean": mse["mean"],
                "test_mse_sample_std": mse["sample_std"],
                "test_mse_median": mse["median"],
                "test_mse_best_min": mse["best_min"],
                "test_mse_worst_max": mse["worst_max"],
                "test_mae_mean": mae["mean"],
                "test_mae_sample_std": mae["sample_std"],
                "grid_update_attempts": len(grid_events),
                "grid_update_successes": sum(bool(event.get("success")) for event in grid_events),
                "grid_update_failures": sum(not bool(event.get("success")) for event in grid_events),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["experiment", "horizon", "test_mse_mean"], na_position="last"
    )


def improvement_diagnostics(records: List[dict]) -> pd.DataFrame:
    completed = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("experiment") == "improvement"
    ]
    rows: List[dict] = []
    groups: Dict[tuple, List[dict]] = {}
    for record in completed:
        key = (
            record.get("dataset"),
            record.get("protocol_name"),
            record.get("horizon"),
            record.get("variant"),
        )
        groups.setdefault(key, []).append(record)
    for (dataset, protocol, horizon, variant), runs in groups.items():
        test = describe(run.get("test_metrics", {}).get("mse") for run in runs)
        base = describe(run.get("base_test_metrics", {}).get("mse") for run in runs)
        deltas = finite_values(
            run.get("hybrid_diagnostics", {}).get(
                "test_mse_delta_vs_internal_dlinear"
            )
            for run in runs
        )
        gains = finite_values(
            run.get("hybrid_diagnostics", {}).get("residual_gain") for run in runs
        )
        selected_epochs = [
            run.get("training", {}).get("selected_epoch") for run in runs
        ]
        spec = runs[0].get("improvement_protocol", {})
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "horizon": horizon,
                "variant": variant,
                "seq_len": runs[0].get("protocol", {}).get("seq_len"),
                "runs": test["n"],
                "test_mse_mean": test["mean"],
                "test_mse_sample_std": test["sample_std"],
                "test_mse_min": test["best_min"],
                "test_mse_max": test["worst_max"],
                "base_test_mse_mean": base["mean"],
                "hybrid_minus_base_mse_mean": (
                    float(np.mean(deltas)) if deltas else None
                ),
                "selected_epoch_zero_runs": sum(epoch == 0 for epoch in selected_epochs),
                "selected_after_epoch_zero_runs": sum(
                    isinstance(epoch, int) and epoch > 0 for epoch in selected_epochs
                ),
                "residual_gain_mean": float(np.mean(gains)) if gains else None,
                "uses_regularization": spec.get("use_regularization"),
                "uses_best_checkpoint": spec.get("best_checkpoint"),
                "gradient_clip_norm": spec.get("gradient_clip_norm"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "protocol", "horizon", "test_mse_mean"],
        na_position="last",
    )


def optimizer_state_diagnostics(records: List[dict]) -> pd.DataFrame:
    completed = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("experiment") == "optimizer_state"
    ]
    rows: List[dict] = []
    groups: Dict[tuple, List[dict]] = {}
    for record in completed:
        key = (
            record.get("dataset"),
            record.get("protocol_name"),
            record.get("horizon"),
            record.get("variant"),
        )
        groups.setdefault(key, []).append(record)
    for (dataset, protocol, horizon, variant), runs in groups.items():
        test = describe(run.get("test_metrics", {}).get("mse") for run in runs)
        post_warmup_means: List[float] = []
        final_validation: List[float] = []
        maximum_validation: List[float] = []
        catastrophic_runs = 0
        grid_events: List[dict] = []
        for run in runs:
            history = run.get("training", {}).get("epochs", [])
            warmup = int(run.get("model_config", {}).get("warmup_epochs", 0))
            post_warmup = finite_values(
                row.get("val_mse") for row in history[warmup:]
            )
            all_validation = finite_values(row.get("val_mse") for row in history)
            if post_warmup:
                post_warmup_means.append(float(np.mean(post_warmup)))
            if all_validation:
                final_validation.append(all_validation[-1])
                maximum_validation.append(max(all_validation))
            test_mse = run.get("test_metrics", {}).get("mse")
            if any(value >= 1.0 for value in all_validation) or (
                isinstance(test_mse, (int, float)) and test_mse >= 1.0
            ):
                catastrophic_runs += 1
            grid_events.extend(run.get("training", {}).get("grid_updates", []))
        spec = runs[0].get("optimizer_state_protocol", {})
        reset_expected = bool(spec.get("reset_optimizer_state"))
        noninitial_events = [
            event for event in grid_events if event.get("epoch_zero_based", 0) > 0
        ]
        cleared_events = sum(
            event.get("optimizer_state_after_reset", {}).get("parameter_states") == 0
            for event in noninitial_events
        )
        rows.append(
            {
                "variant": variant,
                "schedule": spec.get("schedule"),
                "optimizer_state_policy": "reset" if reset_expected else "retain",
                "dataset": dataset,
                "protocol": protocol,
                "horizon": horizon,
                "runs": test["n"],
                "test_mse_mean": test["mean"],
                "test_mse_sample_std": test["sample_std"],
                "post_warmup_val_mse_mean": (
                    float(np.mean(post_warmup_means))
                    if post_warmup_means else None
                ),
                "final_val_mse_mean": (
                    float(np.mean(final_validation))
                    if final_validation else None
                ),
                "maximum_val_mse_mean": (
                    float(np.mean(maximum_validation))
                    if maximum_validation else None
                ),
                "catastrophic_runs": catastrophic_runs,
                "grid_update_events": len(grid_events),
                "noninitial_grid_update_events": len(noninitial_events),
                "noninitial_events_with_zero_optimizer_state": cleared_events,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "protocol", "horizon", "schedule", "optimizer_state_policy"]
    )


def plot_targeted(summary: pd.DataFrame, records: List[dict], figures_dir: Path) -> None:
    if summary.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    baselines = summary[summary["experiment"] == "baseline"]
    if not baselines.empty:
        multiple_datasets = baselines["dataset"].nunique() > 1
        for (dataset, horizon), frame in baselines.groupby(["dataset", "horizon"]):
            frame = frame.sort_values("test_mse_mean")
            errors = frame["test_mse_sample_std"].fillna(0.0).to_numpy()
            fig, axis = plt.subplots(figsize=(8.5, 4.8))
            positions = np.arange(len(frame))
            axis.bar(
                positions,
                frame["test_mse_mean"],
                yerr=errors,
                capsize=4,
                color="#3B82A0",
                alpha=0.8,
            )
            labels = frame["variant"].str.replace("_", " ", regex=False)
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            for position, (_, row) in zip(positions, frame.iterrows()):
                axis.text(
                    position,
                    row["test_mse_mean"] + errors[position] + 0.004,
                    f"{row['test_mse_mean']:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_ylabel("Test MSE (mean +/- sample SD)")
            axis.set_title(f"Same-protocol {dataset} baseline comparison, horizon {horizon}")
            axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                figures_dir
                / (
                    f"targeted_baselines_{dataset}_h{int(horizon)}.png"
                    if multiple_datasets
                    else f"targeted_baselines_h{int(horizon)}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    ablation = summary[summary["experiment"] == "ablation"]
    if not ablation.empty:
        order = ["none", "conservative", "moderate", "aggressive"]
        multiple_datasets = ablation["dataset"].nunique() > 1
        for (dataset, horizon), frame in ablation.groupby(["dataset", "horizon"]):
            frame = frame.assign(
                _order=frame["variant"].map({name: i for i, name in enumerate(order)})
            ).sort_values("_order")
            fig, axis = plt.subplots(figsize=(7.5, 4.8))
            positions = np.arange(len(frame))
            axis.plot(
                positions,
                frame["test_mse_mean"],
                marker="D",
                linewidth=2.2,
                color="#A33E4D",
                label="mean",
            )
            for position, variant in zip(positions, frame["variant"]):
                seed_values = finite_values(
                    record.get("test_metrics", {}).get("mse")
                    for record in records
                    if record.get("status") == "completed"
                    and record.get("experiment") == "ablation"
                    and record.get("variant") == variant
                    and record.get("dataset") == dataset
                    and record.get("horizon") == horizon
                )
                axis.scatter(
                    np.full(len(seed_values), position),
                    seed_values,
                    color="#111111",
                    s=30,
                    zorder=3,
                    label="individual seed" if position == 0 else None,
                )
            axis.set_xticks(positions, frame["variant"])
            axis.set_xlabel("Grid update schedule")
            axis.set_ylabel("Test MSE (log scale)")
            axis.set_yscale("log")
            axis.legend()
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                figures_dir
                / (
                    f"grid_update_ablation_{dataset}_h{int(horizon)}.png"
                    if multiple_datasets
                    else f"grid_update_ablation_h{int(horizon)}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_grid_dynamics(records: List[dict], figures_dir: Path) -> None:
    ablation = [
        record
        for record in records
        if record.get("status") == "completed" and record.get("experiment") == "ablation"
    ]
    if not ablation:
        return
    grouped = group_by(ablation, "variant")
    colors = {
        "none": "#4B5563",
        "conservative": "#2E86AB",
        "moderate": "#E09F3E",
        "aggressive": "#A33E4D",
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for variant, runs in grouped.items():
        curves = [
            finite_values(epoch.get("val_mse") for epoch in run.get("training", {}).get("epochs", []))
            for run in runs
        ]
        curves = [curve for curve in curves if curve]
        if not curves:
            continue
        length = min(map(len, curves))
        values = np.asarray([curve[:length] for curve in curves])
        mean = values.mean(axis=0)
        sample_std = values.std(axis=0, ddof=1) if len(values) >= 2 else np.zeros(length)
        epochs = np.arange(1, length + 1)
        color = colors.get(variant, "#777777")
        axes[0].plot(epochs, mean, label=variant, color=color, linewidth=2)
        lower = np.maximum(mean - sample_std, 1e-4)
        upper = mean + sample_std
        axes[0].fill_between(epochs, lower, upper, color=color, alpha=0.12)

    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("Validation trajectory")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    variants = [name for name in ["conservative", "moderate", "aggressive"] if name in grouped]
    disruptions = []
    labels = []
    for variant in variants:
        values = finite_values(
            event.get("probe_mse_delta")
            for run in grouped[variant]
            for event in run.get("training", {}).get("grid_updates", [])
            if event.get("success")
        )
        if values:
            disruptions.append(values)
            labels.append(variant)
    if disruptions:
        boxes = axes[1].boxplot(disruptions, patch_artist=True, showfliers=True)
        for box, label in zip(boxes["boxes"], labels):
            box.set_facecolor(colors.get(label, "#777777"))
            box.set_alpha(0.65)
        axes[1].set_xticks(range(1, len(labels) + 1), labels, rotation=15)
    axes[1].axhline(0.0, color="#111111", linewidth=1)
    axes[1].set_xlabel("Grid update schedule")
    axes[1].set_ylabel("Probe MSE after - before update")
    axes[1].set_title("Immediate grid-update disruption")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(figures_dir / "grid_update_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multi_horizon_baselines(summary: pd.DataFrame, figures_dir: Path) -> None:
    baselines = summary[summary["experiment"] == "baseline"].copy()
    if baselines.empty or baselines["horizon"].nunique() < 2:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "DLinear": "#111111",
        "Linear": "#2E86AB",
        "NLinear": "#3A9D5D",
        "PatchTST": "#E09F3E",
        "RepeatLast": "#777777",
        "KAN_WOA_Selected": "#A33E4D",
        "KAN_Vanilla": "#7656A8",
    }
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for variant, frame in baselines.groupby("variant"):
        frame = frame.sort_values("horizon")
        label = variant.replace("_", " ")
        axis.errorbar(
            frame["horizon"],
            frame["test_mse_mean"],
            yerr=frame["test_mse_sample_std"].fillna(0.0),
            marker="o",
            linewidth=2.4 if variant in {"DLinear", "KAN_WOA_Selected"} else 1.6,
            capsize=3,
            color=colors.get(variant, "#777777"),
            label=label,
        )
    axis.set_xticks(sorted(baselines["horizon"].unique()))
    axis.set_xlabel("Forecast horizon")
    axis.set_ylabel("Test MSE (mean +/- sample SD)")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "targeted_baselines_all_horizons.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confirmation_comparison(summary: pd.DataFrame, figures_dir: Path) -> None:
    canonical = summary[
        summary["protocol"].eq("ett_canonical_12_4_4_v1")
        & summary["variant"].isin(
            ["DLinear", "KAN_WOA_RegBestClip", "none_retain", "moderate_retain"]
        )
    ].copy()
    if canonical.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    order = ["DLinear", "KAN_WOA_RegBestClip", "none_retain", "moderate_retain"]
    labels = {
        "DLinear": "DLinear",
        "KAN_WOA_RegBestClip": "Corrected KAN",
        "none_retain": "KAN, no update",
        "moderate_retain": "KAN, moderate update",
    }
    colors = {
        "DLinear": "#111111",
        "KAN_WOA_RegBestClip": "#2E86AB",
        "none_retain": "#6B7280",
        "moderate_retain": "#A33E4D",
    }
    datasets = sorted(canonical["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.9))
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets):
        frame = canonical[canonical["dataset"].eq(dataset)].set_index("variant")
        variants = [variant for variant in order if variant in frame.index]
        means = [frame.loc[variant, "test_mse_mean"] for variant in variants]
        errors = [
            frame.loc[variant, "test_mse_sample_std"]
            if pd.notna(frame.loc[variant, "test_mse_sample_std"])
            else 0.0
            for variant in variants
        ]
        positions = np.arange(len(variants))
        lower_errors = [min(error, mean * 0.95) for mean, error in zip(means, errors)]
        axis.bar(
            positions,
            means,
            yerr=np.asarray([lower_errors, errors]),
            capsize=4,
            color=[colors[variant] for variant in variants],
            alpha=0.88,
        )
        axis.set_xticks(
            positions,
            [labels[variant] for variant in variants],
            rotation=22,
            ha="right",
        )
        axis.set_title(f"{dataset}, canonical h96")
        axis.set_ylabel("Test MSE (mean +/- sample SD)")
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "canonical_confirmation_h96.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def write_audit(records: List[dict], output_dir: Path) -> None:
    algorithms = group_by(records, "algorithm")
    audit = {
        "historical_record_count": len(records),
        "runs_per_algorithm": {key: len(value) for key, value in algorithms.items()},
        "seeds_per_algorithm": {
            key: sorted(record.get("seed") for record in value) for key, value in algorithms.items()
        },
        "statistical_policy": (
            "Descriptive mean, sample SD, median, min, max, and CV only. "
            "No inferential significance claim is produced for n=3."
        ),
        "protocol_warning": (
            "Historical JSON files use pilot_60_20_20_v1 and must not be compared "
            "directly with canonical 12/4/4 ETT literature values."
        ),
    }
    with (output_dir / "historical_results_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--targeted-dir", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical = load_json_records(args.historical_dir)
    if not historical:
        raise SystemExit(f"No historical JSON records found in {args.historical_dir}")

    hpo = historical_hpo_summary(historical)
    tests = historical_test_summary(historical)
    hpo.to_csv(args.output_dir / "historical_hpo_summary.csv", index=False)
    tests.to_csv(args.output_dir / "historical_test_summary.csv", index=False)
    write_audit(historical, args.output_dir)
    plot_hpo_fitness(historical, args.figures_dir)
    plot_convergence(historical, args.figures_dir)

    targeted_records = load_json_records(args.targeted_dir)
    targeted = targeted_summary(targeted_records)
    if not targeted.empty:
        targeted.to_csv(args.output_dir / "targeted_summary.csv", index=False)
        plot_targeted(targeted, targeted_records, args.figures_dir)
        plot_grid_dynamics(targeted_records, args.figures_dir)
        plot_multi_horizon_baselines(targeted, args.figures_dir)
        plot_confirmation_comparison(targeted, args.figures_dir)
        baselines = targeted[targeted["experiment"] == "baseline"]
        if not baselines.empty:
            baseline_matrix = baselines.pivot(
                index=["dataset", "variant"],
                columns="horizon",
                values="test_mse_mean",
            )
            baseline_matrix = baseline_matrix.sort_values(
                list(sorted(baseline_matrix.columns))
            )
            baseline_matrix.to_csv(
                args.output_dir / "targeted_baseline_mse_matrix.csv"
            )
        improvements = improvement_diagnostics(targeted_records)
        if not improvements.empty:
            improvements.to_csv(
                args.output_dir / "improvement_diagnostics.csv", index=False
            )
        optimizer_states = optimizer_state_diagnostics(targeted_records)
        if not optimizer_states.empty:
            optimizer_states.to_csv(
                args.output_dir / "optimizer_state_diagnostics.csv", index=False
            )

    print("Historical HPO summary")
    print(hpo.to_string(index=False))
    print(f"\nWrote analysis files to {args.output_dir}")
    if targeted.empty:
        print(f"No completed targeted records yet in {args.targeted_dir}")
    else:
        print("\nTargeted summary")
        print(targeted.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
