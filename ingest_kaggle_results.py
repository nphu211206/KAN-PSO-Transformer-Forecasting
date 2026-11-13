#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AUTOMATED KAGGLE BENCHMARK INGESTION PIPELINE (ingest_kaggle_results.py)
================================================================================
Venue: Neural Computing and Applications (Springer Nature)
Manuscript: PSO-TCWavKAN
Author: Phu Toan Nguyen (HUBT, Vietnam)

This pipeline provides automated end-to-end ingestion of verified experimental
results from the dual-GPU Kaggle benchmark run (ncaa_option_a_results.zip /
results/). Zero synthetic fallbacks are permitted.
================================================================================
"""

import os
import sys
import re
import zipfile
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_kaggle_results")


class SecurityError(Exception):
    """Raised when an insecure archive operation or directory traversal is detected."""
    pass


CANDIDATE_ZIP_PATHS = [
    "/kaggle/working/ncaa_option_a_results.zip",
    "ncaa_option_a_results.zip",
    "results.zip",
    "../ncaa_option_a_results.zip",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Kaggle Ingestion Pipeline for NCAA Paper")
    parser.add_argument("--zip_path", type=str, default=None,
                        help="Path to ncaa_option_a_results.zip (auto-detected if omitted)")
    parser.add_argument("--extract_dir", type=str, default=".",
                        help="Directory where results/ will be unpacked or located")
    parser.add_argument("--paper_tex", type=str, default="paper_ncaa.tex",
                        help="Path to master LaTeX manuscript")
    parser.add_argument("--paper_md", type=str, default="paper_v91_FINAL.md",
                        help="Path to Markdown manuscript")
    parser.add_argument("--render_figures", action="store_true", default=False,
                        help="Re-render 600 DPI publication figures")
    parser.add_argument("--verify_only", action="store_true", default=False,
                        help="Run verification on existing local results without mutating files")
    return parser.parse_args()


def find_archive(explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for p in CANDIDATE_ZIP_PATHS:
        if os.path.exists(p):
            return p
    return None


def secure_unpack(zip_path, target_dir):
    """Safely extracts zip archive preventing path traversal vulnerabilities."""
    print(f"[INGEST] Securely extracting: {zip_path} -> {target_dir}")
    target_abs = os.path.abspath(target_dir)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            member_path = os.path.abspath(os.path.join(target_dir, member.filename))
            try:
                if os.path.commonpath([target_abs, member_path]) != target_abs:
                    raise SecurityError(f"Directory traversal attack detected in zip member: {member.filename}")
            except ValueError:
                raise SecurityError(f"Cross-drive path traversal attack detected in zip member: {member.filename}")
        zf.extractall(target_dir)
    print(f"[INGEST] Successfully unpacked {len(zf.namelist())} artifacts.")


def compute_bessel_stats(values):
    """Computes sample mean and sample standard deviation with Bessel correction (ddof=1)."""
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def compute_wilcoxon_from_df(df_bench):
    """Computes real paired Wilcoxon signed-rank & Holm-Bonferroni stats directly from df_bench."""
    configs = [(ds, l, h) for ds in ['ETTh1', 'ETTh2'] for l in [96, 336] for h in [96, 192, 336, 720]]
    df_ours = df_bench[df_bench['model'] == 'tcwavkan_master'].set_index(['dataset', 'lookback', 'horizon'])
    comparators = ['dlinear', 'patchtst_lite', 'bspline_kan', 'tcwavkan_no_pso', 'nlinear']

    pre = []
    for base in comparators:
        df_b = df_bench[df_bench['model'] == base].set_index(['dataset', 'lookback', 'horizon'])
        ours_vals = np.array([float(df_ours.loc[c, 'mean_mse']) for c in configs])
        base_vals = np.array([float(df_b.loc[c, 'mean_mse']) for c in configs])
        diffs = base_vals - ours_vals
        pcts = (base_vals - ours_vals) / base_vals * 100.0
        wins = int(np.sum(diffs > 0))
        draws = int(np.sum(diffs == 0))
        losses = int(np.sum(diffs < 0))
        ranks = stats.rankdata(np.abs(diffs))
        w_plus = float(np.sum(ranks[diffs > 0]))
        w_minus = float(np.sum(ranks[diffs < 0]))
        res_ex = stats.wilcoxon(diffs, alternative='two-sided', method='exact')
        res_as = stats.wilcoxon(diffs, alternative='two-sided', method='approx')
        pre.append({
            'baseline_model': base,
            'n_configs': len(configs),
            'win': wins,
            'draw': draws,
            'loss': losses,
            'mean_pct_reduction': round(float(np.mean(pcts)), 3),
            'w_plus': w_plus,
            'w_minus': w_minus,
            'w_stat': min(w_plus, w_minus),
            'p_exact': float(res_ex.pvalue),
            'p_asymptotic': float(res_as.pvalue),
        })

    # Step-down Holm-Bonferroni adjustment
    pre.sort(key=lambda r: r['p_exact'])
    m = len(pre)
    prev_adj = 0.0
    for i, r in enumerate(pre):
        adj = max(prev_adj, min(1.0, r['p_exact'] * (m - i)))
        prev_adj = adj
        r['holm_p_exact'] = adj
        r['significant_005'] = bool(r['p_exact'] < 0.05)
        r['significant_001'] = bool(r['p_exact'] < 0.01)
    return pd.DataFrame(pre)


def update_table1_latex(df_bench, paper_tex_content):
    """Updates Table 1 (tab:master_benchmark_16_canonical) in paper_ncaa.tex."""
    df_pso = df_bench[df_bench['model'] == 'tcwavkan_master'].sort_values(['dataset', 'lookback', 'horizon'])
    df_dlin = df_bench[df_bench['model'] == 'dlinear']
    dlin_baselines = {
        (str(dr['dataset']), int(dr['lookback']), int(dr['horizon'])): float(dr['mean_mse'])
        for _, dr in df_dlin.iterrows()
    }

    rows = []
    cur_ds, cur_l = None, None
    for _, r in df_pso.iterrows():
        ds, l, h = str(r['dataset']), int(r['lookback']), int(r['horizon'])
        bs = int(r['batch_size'])
        m_mse, s_mse = float(r['mean_mse']), float(r['std_mse'])
        m_mae, s_mae = float(r['mean_mae']), float(r['std_mae'])

        seed_mses_str = str(r['seed_mses'])
        rt = float(r['total_runtime_sec'])
        dl_mse = dlin_baselines[(ds, l, h)]
        gain_pct = ((dl_mse - m_mse) / dl_mse) * 100.0
        gain_str = f"{gain_pct:+.2f}\\%" if gain_pct < 0 else f"\\textbf{{{gain_pct:+.2f}\\%}}"
        if ds == 'ETTh1' and l == 96 and h == 96:
            gain_str = f"\\textbf{{{gain_pct:+.2f}\\%}} (vs {dl_mse:.4f})"

        if cur_ds is not None and cur_ds != ds:
            rows.append(r"\midrule")
        elif cur_l is not None and cur_l != l and cur_ds == ds:
            rows.append(r"\addlinespace")
        cur_ds, cur_l = ds, l

        row_tex = (
            f"{ds} & {l} & {h} & {bs} & "
            f"{m_mse:.4f} $\\pm$ {s_mse:.4f} & {m_mae:.4f} $\\pm$ {s_mae:.4f} & "
            f"{seed_mses_str} & {rt:.2f} & {gain_str} \\\\"
        )
        rows.append(row_tex)

    table_block = "\n".join(rows)
    pattern = r"(\\begin\{table\*\}\[htbp\].*?\\label\{tab:master_benchmark_16_canonical\}.*?\\midrule)(.*?)(\\bottomrule)"
    new_tex, n_subs = re.subn(pattern, lambda m: m.group(1) + "\n" + table_block + "\n" + m.group(3),
                              paper_tex_content, flags=re.DOTALL)
    if n_subs > 0:
        print("[INGEST] Table 1 updated successfully in LaTeX.")
    else:
        print("[WARNING] Table 1 pattern not matched in LaTeX.")
    return new_tex


def _format_cell_with_ranks(val_str, val, best_val, sec_val):
    if abs(val - best_val) < 1e-12:
        return f"\\textbf{{{val_str}}}"
    if sec_val is not None and abs(val - sec_val) < 1e-12:
        return f"\\underline{{{val_str}}}"
    return val_str


def update_table2_latex(df_bench, paper_tex_content):
    """Updates Table 2 (tab:cross_model_comparison) with true row-wise best bolded and 2nd-best underlined."""
    configs = [
        ('ETTh1', 96, 96), ('ETTh1', 96, 192), ('ETTh1', 96, 336), ('ETTh1', 96, 720),
        ('ETTh1', 336, 96), ('ETTh1', 336, 192), ('ETTh1', 336, 336), ('ETTh1', 336, 720),
        ('ETTh2', 96, 96), ('ETTh2', 96, 192), ('ETTh2', 96, 336), ('ETTh2', 96, 720),
        ('ETTh2', 336, 96), ('ETTh2', 336, 192), ('ETTh2', 336, 336), ('ETTh2', 336, 720),
    ]
    model_cols = ['dlinear', 'nlinear', 'patchtst_lite', 'bspline_kan', 'tcwavkan_no_pso', 'tcwavkan_master']

    rows = []
    cur_ds, cur_l = None, None
    for ds, l, h in configs:
        sub_cfg = df_bench[(df_bench['dataset'] == ds) & (df_bench['lookback'] == l) & (df_bench['horizon'] == h)]
        vals = {}
        for m_name in model_cols:
            r = sub_cfg[sub_cfg['model'] == m_name].iloc[0]
            vals[m_name] = (round(float(r['mean_mse']), 4), round(float(r['mean_mae']), 4))

        sorted_mse = sorted(set(v[0] for v in vals.values()))
        sorted_mae = sorted(set(v[1] for v in vals.values()))
        best_mse, sec_mse = sorted_mse[0], (sorted_mse[1] if len(sorted_mse) > 1 else None)
        best_mae, sec_mae = sorted_mae[0], (sorted_mae[1] if len(sorted_mae) > 1 else None)

        cells = []
        for m_name in model_cols:
            mse_v, mae_v = vals[m_name]
            mse_f = _format_cell_with_ranks(f"{mse_v:.4f}", mse_v, best_mse, sec_mse)
            mae_f = _format_cell_with_ranks(f"{mae_v:.4f}", mae_v, best_mae, sec_mae)
            cells.append(f"{mse_f} / {mae_f}")

        if cur_ds is not None and cur_ds != ds:
            rows.append(r"\midrule")
        elif cur_l is not None and cur_l != l and cur_ds == ds:
            rows.append(r"\addlinespace")
        cur_ds, cur_l = ds, l

        row_tex = f"{ds} & ({l}, {h}) & " + " & ".join(cells) + r" \\"
        rows.append(row_tex)

    table_block = "\n".join(rows)
    pattern = r"(\\begin\{table\*\}\[htbp\].*?\\label\{tab:cross_model_comparison\}.*?\\midrule)(.*?)(\\bottomrule)"
    new_tex, n_subs = re.subn(pattern, lambda m: m.group(1) + "\n" + table_block + "\n" + m.group(3),
                              paper_tex_content, flags=re.DOTALL)
    if n_subs > 0:
        print("[INGEST] Table 2 updated successfully in LaTeX (true row-wise best bold, 2nd-best underlined).")
    else:
        print("[WARNING] Table 2 pattern not matched in LaTeX.")
    return new_tex


def update_table3_latex(df_wilcox, paper_tex_content):
    """Updates Table 3 (tab:wilcoxon_significance) with exact + asymptotic + Holm p-values."""
    rows = []
    display_names = {
        'dlinear': 'DLinear',
        'nlinear': 'NLinear',
        'patchtst_lite': 'PatchTST Lite',
        'bspline_kan': 'B-Spline KAN',
        'tcwavkan_no_pso': 'TCWavKAN (w/o PSO)',
    }
    for _, r in df_wilcox.iterrows():
        base = str(r['baseline_model'])
        clean_name = display_names.get(base, base.replace('_', ' ').title())
        comp = f"PSO-TCWavKAN vs. {clean_name}"
        w_plus = float(r['w_plus'])
        w_minus = float(r['w_minus'])
        z_stat = (w_plus - 68.0) / 19.339079605813716
        p_exact = float(r['p_exact'])
        p_asymp = float(r['p_asymptotic'])
        p_holm = float(r['holm_p_exact'])

        if p_exact < 0.01:
            decision = "Reject $H_0$ ($p < 0.01$)"
        elif p_exact < 0.05:
            decision = "Reject $H_0$ ($p < 0.05$)"
        else:
            decision = "Fail to reject $H_0$"

        record = f"{int(r['win'])} / {int(r['draw'])} / {int(r['loss'])}"
        p_exact_tex = f"\\textbf{{{p_exact:.4f}}}" if p_exact < 0.05 else f"{p_exact:.4f}"
        p_holm_tex = f"\\textbf{{{p_holm:.4f}}}" if p_holm < 0.05 else f"{p_holm:.4f}"
        comp_tex = f"\\textbf{{{comp}}}" if p_exact < 0.05 else comp

        row_tex = (
            f"{comp_tex} & {w_plus:.1f} & {w_minus:.1f} & {z_stat:+.3f} & "
            f"{p_exact_tex} & {p_asymp:.2e} & {p_holm_tex} & {decision} & {record} \\\\"
        )
        rows.append(row_tex)

    table_block = "\n".join(rows)
    pattern = r"(\\begin\{table\*?\}\[htbp\].*?\\label\{tab:wilcoxon_significance\}.*?\\midrule)(.*?)(\\bottomrule)"
    new_tex, n_subs = re.subn(pattern, lambda m: m.group(1) + "\n" + table_block + "\n" + m.group(3),
                              paper_tex_content, flags=re.DOTALL)
    if n_subs > 0:
        print("[INGEST] Table 3 updated successfully in LaTeX (including Holm-Bonferroni p_adj).")
    else:
        print("[WARNING] Table 3 pattern not matched in LaTeX.")
    return new_tex


def update_table4_latex(df_ablation, paper_tex_content):
    """Updates Table 4 (tab:component_ablation) in paper_ncaa.tex."""
    if df_ablation is None or df_ablation.empty:
        return paper_tex_content

    variant_names = {
        'full': "\\textbf{Full PSO-TCWavKAN Master}",
        'no_revin': "Variant 1: w/o RevIN (No Norm)",
        'no_tcd': "Variant 2: w/o Multi-Kernel Conv (No TCD)",
        'spline_kan_block': "Variant 3: Spline-KAN instead of WavKAN",
        'no_clpso': "Variant 4: w/o CLPSO (Fixed Default Init)",
    }

    rows = []
    for v_key, v_label in variant_names.items():
        sub = df_ablation[df_ablation['variant'] == v_key]
        if sub.empty:
            continue
        r = sub.iloc[0]
        m_mse, s_mse = float(r['mean_mse']), float(r['std_mse'])
        m_mae, s_mae = float(r['mean_mae']), float(r['std_mae'])
        delta = float(r['delta_mse_pct'])
        rt = float(r['phase1_mean_epoch_sec'])
        delta_str = "\\textbf{0.00\\%}" if v_key == 'full' else f"{delta:+.2f}\\%"
        if v_key == 'full':
            row_tex = f"{v_label} & \\textbf{{{m_mse:.4f} $\\pm$ {s_mse:.4f}}} & \\textbf{{{m_mae:.4f} $\\pm$ {s_mae:.4f}}} & {delta_str} & {rt:.2f} \\\\"
        else:
            row_tex = f"{v_label} & {m_mse:.4f} $\\pm$ {s_mse:.4f} & {m_mae:.4f} $\\pm$ {s_mae:.4f} & {delta_str} & {rt:.2f} \\\\"
        rows.append(row_tex)

    if rows:
        table_block = "\n".join(rows)
        pattern = r"(\\begin\{table\*?\}\[htbp\].*?\\label\{tab:component_ablation\}.*?\\midrule)(.*?)(\\bottomrule)"
        new_tex, n_subs = re.subn(pattern, lambda m: m.group(1) + "\n" + table_block + "\n" + m.group(3),
                                  paper_tex_content, flags=re.DOTALL)
        if n_subs > 0:
            print("[INGEST] Table 4 updated successfully in LaTeX.")
        else:
            print("[WARNING] Table 4 pattern not matched in LaTeX.")
        return new_tex
    return paper_tex_content


def execute_pipeline(args):
    print("=" * 80)
    print("STARTING AUTOMATED KAGGLE BENCHMARK INGESTION PIPELINE")
    print("=" * 80)

    if getattr(args, 'verify_only', False):
        print("[INGEST] Executing pipeline in VERIFY_ONLY mode: Validating local results without modifying files.")
        master_csv = os.path.join(args.extract_dir, "results", "MASTER_BENCHMARK_RESULTS.csv")
        if not os.path.exists(master_csv):
            master_csv = os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv")
        if not os.path.exists(master_csv):
            raise FileNotFoundError(f"Could not locate benchmark metrics CSV: {master_csv}")

        df_bench = pd.read_csv(master_csv)
        print(f"[INGEST-VERIFY] Loaded {len(df_bench)} benchmark evaluation records (expected 96 = 16 configs x 6 models).")
        if len(df_bench) != 96:
            raise ValueError(f"Expected 96 records (16 configs x 6 models), got {len(df_bench)}")

        for fname in ["WILCOXON_STATS.csv", "ABLATION_RESULTS.csv",
                      "CLPSO_CONVERGENCE_LOG.csv", "CLPSO_SENSITIVITY_LOG.csv",
                      "CLPSO_PER_CONFIG_LOG.csv", "EDGE_PROFILE.csv", "ENVIRONMENT_MANIFEST.json"]:
            fpath = os.path.join("results", fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing required verified artifact: {fpath}")
            print(f"[INGEST-VERIFY] Verified {fpath}")

        pred_dir = os.path.join("predictions")
        npz_files = [f for f in os.listdir(pred_dir) if f.endswith(".npz")] if os.path.exists(pred_dir) else []
        print(f"[INGEST-VERIFY] Validated {len(npz_files)} prediction/ground-truth NPZ files in {pred_dir}.")
        if len(npz_files) != 144:
            raise ValueError(f"Expected 144 Option A NPZ files in predictions/, found {len(npz_files)}")

        print("\n" + "=" * 80)
        print("VERIFY_ONLY VALIDATION COMPLETED SUCCESSFULLY (0 FILE MUTATIONS)")
        print("=" * 80)
        return 0

    archive_path = find_archive(args.zip_path)
    if archive_path:
        print(f"[INGEST] Found Kaggle results archive: {archive_path}")
        secure_unpack(archive_path, args.extract_dir)
    else:
        print("[INGEST] Using verified local results/ directory...")

    master_csv = os.path.join(args.extract_dir, "results", "MASTER_BENCHMARK_RESULTS.csv")
    if not os.path.exists(master_csv):
        master_csv = os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv")
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Could not locate benchmark metrics CSV: {master_csv}")

    df_bench = pd.read_csv(master_csv)
    print(f"[INGEST] Loaded {len(df_bench)} benchmark evaluation records.")

    wilcox_csv = os.path.join("results", "WILCOXON_STATS.csv")
    if os.path.exists(wilcox_csv):
        print(f"[INGEST] Loading verified Wilcoxon stats from: {wilcox_csv}")
        df_wilcox = pd.read_csv(wilcox_csv)
    else:
        print("[INGEST] Computing real paired Wilcoxon & Holm-Bonferroni statistics from df_bench...")
        df_wilcox = compute_wilcoxon_from_df(df_bench)
        os.makedirs("results", exist_ok=True)
        df_wilcox.to_csv(wilcox_csv, index=False)

    ablation_csv = os.path.join("results", "ABLATION_RESULTS.csv")
    df_ablation = pd.read_csv(ablation_csv) if os.path.exists(ablation_csv) else None

    if os.path.exists(args.paper_tex):
        print(f"[INGEST] Synchronizing LaTeX manuscript: {args.paper_tex}")
        with open(args.paper_tex, 'r', encoding='utf-8') as f:
            tex_content = f.read()

        tex_content = update_table1_latex(df_bench, tex_content)
        tex_content = update_table2_latex(df_bench, tex_content)
        tex_content = update_table3_latex(df_wilcox, tex_content)
        tex_content = update_table4_latex(df_ablation, tex_content)

        with open(args.paper_tex, 'w', encoding='utf-8') as f:
            f.write(tex_content)
        print(f"[INGEST] Successfully synchronized Tables 1-4 in {args.paper_tex}.")
    else:
        print(f"[WARNING] LaTeX file not found: {args.paper_tex}")

    if args.render_figures:
        print("\n[INGEST] Triggering 600 DPI publication figure re-rendering...")
        import generate_ncaa_figures
        generate_ncaa_figures.main()

    print("\n" + "=" * 80)
    print("INGESTION PIPELINE EXECUTION COMPLETED WITH 0 ERRORS")
    print("=" * 80)
    return 0


def main():
    args = parse_args()
    return execute_pipeline(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
