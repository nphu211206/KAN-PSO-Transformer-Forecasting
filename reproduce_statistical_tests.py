"""
reproduce_statistical_tests.py
================================
Exact reproduction of the two-sided Wilcoxon signed-rank tests and
Holm-Bonferroni corrections across the 16 canonical (Dataset, L, H)
benchmark configurations reported in Section 4.4 and Table 4.

The script reports both:
  1. the rounded 4-decimal values printed in Table 4, and
  2. the full-precision MSE means in TABLE1_BENCHMARK_FINAL.csv.

Expected full-precision result for PSO-TCWavKAN vs NLinear:
  W = 28.0, uncorrected p = 0.0386, Holm-corrected p = 0.116
(not significant at alpha = 0.05 after correction).

Usage:
  python reproduce_statistical_tests.py
  python reproduce_statistical_tests.py --csv results/TABLE1_BENCHMARK_FINAL.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def run_tests(k_mse, d_mse, n_mse, p_mse, label=""):
    print(f"\n--- {label} ---")
    print(f"  Paired conditions: {len(k_mse)}")
    res_nl = stats.wilcoxon(k_mse, n_mse, alternative="two-sided")
    res_pt = stats.wilcoxon(k_mse, p_mse, alternative="two-sided")
    res_dl = stats.wilcoxon(k_mse, d_mse, alternative="two-sided")

    raw = [
        ("vs NLinear", res_nl.statistic, res_nl.pvalue),
        ("vs PatchTST Lite", res_pt.statistic, res_pt.pvalue),
        ("vs DLinear", res_dl.statistic, res_dl.pvalue),
    ]
    sorted_raw = sorted(raw, key=lambda x: x[2])
    m = len(sorted_raw)

    print(f"  {'Comparison':<18} | {'W stat':<8} | {'Raw p':<10} | "
          f"{'Holm p':<10} | {'Sig (alpha=0.05)':<18}")
    print("  " + "-" * 70)
    running_max = 0.0
    for rank, (comp, w, p) in enumerate(sorted_raw):
        mult = m - rank
        adj_p = min(1.0, max(running_max, p * mult))
        running_max = adj_p
        sig_str = "Yes (p < 0.05)" if adj_p < 0.05 else "No (p >= 0.05)"
        print(f"  {comp:<18} | {w:<8.1f} | {p:<10.4f} | "
              f"{adj_p:<10.4f} | {sig_str:<18}")


def resolve_csv(explicit=None):
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "results" / "TABLE1_BENCHMARK_FINAL.csv",
        here / "TABLE1_BENCHMARK_FINAL.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "TABLE1_BENCHMARK_FINAL.csv not found; pass --csv /path/to/file.csv"
    )


def load_model(df, model_name):
    candidates = [model_name, f"{model_name}_v91", f"{model_name}_v90"]
    for name in candidates:
        sub = df[df["Model"] == name]
        if len(sub):
            return sub.set_index(["Dataset", "L", "H"]).sort_index()
    raise KeyError(f"Model {model_name!r} not found in CSV")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to TABLE1_BENCHMARK_FINAL.csv")
    args = parser.parse_args()

    print("=" * 75)
    print("  EXACT REPRODUCTION OF BENCHMARK STATISTICAL TESTS (TABLE 4)")
    print("=" * 75)

    # 1. Rounded values exactly as printed in Table 4.
    d_round = np.array([0.1121, 0.1528, 0.1464, 0.1759, 0.1086, 0.1147, 0.1572, 0.1659,
                        0.2806, 0.3531, 0.3854, 0.5628, 0.2520, 0.3017, 0.3420, 0.4351])
    n_round = np.array([0.1106, 0.1375, 0.1459, 0.2213, 0.1031, 0.1244, 0.1789, 0.2205,
                        0.2756, 0.3501, 0.3607, 0.5025, 0.2654, 0.3430, 0.3966, 0.5014])
    p_round = np.array([0.1110, 0.1486, 0.1480, 0.2049, 0.0987, 0.1163, 0.1614, 0.2332,
                        0.2744, 0.3412, 0.3545, 0.5048, 0.2735, 0.3782, 0.5073, 0.6228])
    k_round = np.array([0.1127, 0.1320, 0.1500, 0.2193, 0.1002, 0.1328, 0.1325, 0.1817,
                        0.2630, 0.3312, 0.3527, 0.4860, 0.2360, 0.2924, 0.3162, 0.6656])

    run_tests(k_round, d_round, n_round, p_round,
              "Table 4 formatted values (4 decimals; manuscript basis)")

    # 2. Full-precision CSV means, aligned on (Dataset, L, H).
    csv_path = resolve_csv(args.csv)
    df = pd.read_csv(csv_path)
    kan = load_model(df, "PSO-TCWavKAN")
    dlin = load_model(df, "DLinear").reindex(kan.index)
    nlin = load_model(df, "NLinear").reindex(kan.index)
    patch = load_model(df, "PatchTST_Lite").reindex(kan.index)
    if kan.shape[0] != 16:
        raise ValueError(f"Expected 16 benchmark conditions, found {kan.shape[0]}")

    run_tests(kan["MSE_mean"].values, dlin["MSE_mean"].values,
              nlin["MSE_mean"].values, patch["MSE_mean"].values,
              f"Full-precision CSV means ({csv_path.name})")

    print("\n" + "=" * 75)
    print("  Note: rounded-table Holm p is 0.1329 (reported as 0.133);")
    print("        full-precision Holm p is 0.116. Both are non-significant.")
    print("=" * 75)


if __name__ == "__main__":
    main()
