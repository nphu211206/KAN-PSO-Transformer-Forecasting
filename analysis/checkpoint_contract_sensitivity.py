"""Generate a descriptive K08-to-K09 checkpoint-contract appendix."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "publication"


def load(folder: Path, campaign: str) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "completed":
            continue
        rows.append(
            {
                "campaign": campaign,
                "dataset": row["dataset"],
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "test_mse": float(row["test_metrics"]["mse"]),
                "selected_epoch": int(row["training"]["selected_epoch"]),
                "selection_rule": row["training"]["selection_rule"],
                "source_sha256": row["source"]["combined_sha256"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    k08 = load(ROOT / "canonical_confirmation_results" / "records", "K08")
    k09 = load(ROOT / "analysis" / "k09_diagnostics" / "records", "K09")
    indexed = {(row["campaign"], row["dataset"], row["variant"], row["seed"]): row for row in k08 + k09}
    per_seed = []
    for dataset in ("ETTh1", "ETTh2"):
        for variant in ("DLinear", "KAN_WOA_RegBestClip", "none_retain", "moderate_retain"):
            for seed in range(1000, 1005):
                old = indexed[("K08", dataset, variant, seed)]
                new = indexed[("K09", dataset, variant, seed)]
                per_seed.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "k08_test_mse": old["test_mse"],
                        "k09_test_mse": new["test_mse"],
                        "delta_k09_minus_k08": new["test_mse"] - old["test_mse"],
                        "k08_selected_epoch": old["selected_epoch"],
                        "k09_selected_epoch": new["selected_epoch"],
                        "k08_selection_rule": old["selection_rule"],
                        "k09_selection_rule": new["selection_rule"],
                    }
                )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["dataset"], row["variant"])].append(row)
    summary = []
    for (dataset, variant), rows in sorted(grouped.items()):
        old = [row["k08_test_mse"] for row in rows]
        new = [row["k09_test_mse"] for row in rows]
        delta = [row["delta_k09_minus_k08"] for row in rows]
        summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_paired_seeds": len(rows),
                "k08_mean_mse": statistics.mean(old),
                "k08_sample_sd": statistics.stdev(old),
                "k09_mean_mse": statistics.mean(new),
                "k09_sample_sd": statistics.stdev(new),
                "mean_delta_k09_minus_k08": statistics.mean(delta),
                "median_delta_k09_minus_k08": statistics.median(delta),
                "k09_lower_seed_count": sum(value < -1e-6 for value in delta),
                "effectively_equal_seed_count": sum(abs(value) <= 1e-6 for value in delta),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "checkpoint_contract_sensitivity_per_seed.csv", per_seed)
    write_csv(OUTPUT / "checkpoint_contract_sensitivity_summary.csv", summary)

    lines = [
        "# K08-K09 Checkpoint-Contract Sensitivity Appendix",
        "",
        "K08 and K09 are shown side by side and are never pooled. The difference is descriptive, not a causal estimate of checkpoint selection alone: K09 is a separately preregistered campaign with a distinct source hash and added diagnostic instrumentation.",
        "",
        "| Dataset | Condition | K08 MSE | K09 MSE | Mean delta K09-K08 | K09 lower seeds | Equal within 1e-6 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | "
            f"{row['k08_mean_mse']:.6f} +/- {row['k08_sample_sd']:.6f} | "
            f"{row['k09_mean_mse']:.6f} +/- {row['k09_sample_sd']:.6f} | "
            f"{row['mean_delta_k09_minus_k08']:.6f} | {row['k09_lower_seed_count']}/5 | "
            f"{row['effectively_equal_seed_count']}/5 |"
        )
    lines += [
        "",
        "Interpretation: validation-best selection reduces late-epoch damage for several Grid Update conditions, but it does not reverse the DLinear-versus-KAN ranking. DLinear's K08 final-epoch and K09 validation-best results should not be treated as identical evaluation contracts.",
    ]
    (OUTPUT / "CHECKPOINT_CONTRACT_APPENDIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote K08-K09 checkpoint-contract appendix")


if __name__ == "__main__":
    main()
