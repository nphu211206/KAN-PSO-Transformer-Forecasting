#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_paper_markdown.py
Synchronizes Tables 1-5 and runtime summary notes in paper_v91_FINAL.md
directly from verified Option A CSV/JSON outputs (results/). Zero synthetic fallbacks.
"""

import json
import os
import re
import numpy as np
import pandas as pd


def build_table1_md():
    df = pd.read_csv(os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv"))
    pso = df[df["model"] == "tcwavkan_master"].sort_values(["dataset", "lookback", "horizon"])
    dlin = df[df["model"] == "dlinear"].set_index(["dataset", "lookback", "horizon"])

    lines = [
        r"| Dataset | Lookback $L$ | Horizon $H$ | Batch Size | Test MSE ($\text{Mean} \pm \text{SD}$) | Test MAE ($\text{Mean} \pm \text{SD}$) | Seed MSEs $[2024, 2025, 2026]$ | Runtime (s) | Gain vs DLinear |",
        r"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in pso.iterrows():
        ds = str(r["dataset"])
        l = int(r["lookback"])
        h = int(r["horizon"])
        bs = int(r["batch_size"])
        m_mse, s_mse = float(r["mean_mse"]), float(r["std_mse"])
        m_mae, s_mae = float(r["mean_mae"]), float(r["std_mae"])
        seed_mses = str(r["seed_mses"])
        rt = float(r["total_runtime_sec"])
        d_mse = float(dlin.loc[(ds, l, h), "mean_mse"])
        gain = ((d_mse - m_mse) / d_mse) * 100.0
        gain_str = f"{gain:+.2f}%" if gain < 0 else f"**{gain:+.2f}%**"
        if ds == "ETTh1" and l == 96 and h == 96:
            gain_str = f"**{gain:+.2f}%** (vs {d_mse:.4f})"
        lines.append(
            f"| {ds} | {l} | {h} | {bs} | {m_mse:.4f} $\\pm$ {s_mse:.4f} | "
            f"{m_mae:.4f} $\\pm$ {s_mae:.4f} | {seed_mses} | {rt:.2f} | {gain_str} |"
        )
    return "\n".join(lines)


def _fmt_md_rank(val_str, val, best_val, sec_val):
    if abs(val - best_val) < 1e-12:
        return f"**{val_str}**"
    if sec_val is not None and abs(val - sec_val) < 1e-12:
        return f"<u>{val_str}</u>"
    return val_str


def build_table2_md():
    df = pd.read_csv(os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv"))
    configs = [
        ("ETTh1", 96, 96), ("ETTh1", 96, 192), ("ETTh1", 96, 336), ("ETTh1", 96, 720),
        ("ETTh1", 336, 96), ("ETTh1", 336, 192), ("ETTh1", 336, 336), ("ETTh1", 336, 720),
        ("ETTh2", 96, 96), ("ETTh2", 96, 192), ("ETTh2", 96, 336), ("ETTh2", 96, 720),
        ("ETTh2", 336, 96), ("ETTh2", 336, 192), ("ETTh2", 336, 336), ("ETTh2", 336, 720),
    ]
    model_cols = ["dlinear", "nlinear", "patchtst_lite", "bspline_kan", "tcwavkan_no_pso", "tcwavkan_master"]

    lines = [
        "| Dataset | $(L, H)$ | DLinear | NLinear | PatchTST Lite | B-Spline KAN | TCWavKAN (No PSO) | PSO-TCWavKAN Master |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ds, l, h in configs:
        sub_cfg = df[(df.dataset == ds) & (df.lookback == l) & (df.horizon == h)]
        vals = {}
        for m in model_cols:
            r = sub_cfg[sub_cfg.model == m].iloc[0]
            vals[m] = (round(float(r.mean_mse), 4), round(float(r.mean_mae), 4))

        sorted_mse = sorted(set(v[0] for v in vals.values()))
        sorted_mae = sorted(set(v[1] for v in vals.values()))
        best_mse, sec_mse = sorted_mse[0], (sorted_mse[1] if len(sorted_mse) > 1 else None)
        best_mae, sec_mae = sorted_mae[0], (sorted_mae[1] if len(sorted_mae) > 1 else None)

        cells = []
        for m in model_cols:
            mse_v, mae_v = vals[m]
            mse_s = _fmt_md_rank(f"{mse_v:.4f}", mse_v, best_mse, sec_mse)
            mae_s = _fmt_md_rank(f"{mae_v:.4f}", mae_v, best_mae, sec_mae)
            cells.append(f"{mse_s} / {mae_s}")

        lines.append(f"| {ds} | ({l}, {h}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_table3_md():
    df = pd.read_csv(os.path.join("results", "WILCOXON_STATS.csv"))
    name_map = {
        "dlinear": "DLinear",
        "nlinear": "NLinear",
        "patchtst_lite": "PatchTST Lite",
        "bspline_kan": "B-Spline KAN",
        "tcwavkan_no_pso": "TCWavKAN (w/o PSO)",
    }
    lines = [
        r"| Paired Comparison | $W^+$ | $W^-$ | $z$-stat | Exact $p$ | Asymp. $p$ | Holm $p_{\mathrm{adj}}$ | Decision | W / D / L |",
        r"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df.iterrows():
        base = str(r["baseline_model"])
        clean_name = name_map.get(base, base.replace("_", " ").title())
        comp = f"PSO-TCWavKAN vs. {clean_name}"
        w_p = float(r["w_plus"])
        w_m = float(r["w_minus"])
        z = (w_p - 68.0) / 19.339079605813716
        p_ex = float(r["p_exact"])
        p_as = float(r["p_asymptotic"])
        p_holm = float(r["holm_p_exact"])
        if p_ex < 0.01:
            dec = "Reject $H_0$ ($p < 0.01$)"
        elif p_ex < 0.05:
            dec = "Reject $H_0$ ($p < 0.05$)"
        else:
            dec = "Fail to reject $H_0$"
        rec = f"{int(r['win'])} / {int(r['draw'])} / {int(r['loss'])}"
        comp_str = f"**{comp}**" if p_ex < 0.05 else comp
        p_ex_str = f"**{p_ex:.4f}**" if p_ex < 0.05 else f"{p_ex:.4f}"
        p_holm_str = f"**{p_holm:.4f}**" if p_holm < 0.05 else f"{p_holm:.4f}"
        lines.append(
            f"| {comp_str} | {w_p:.1f} | {w_m:.1f} | {z:+.3f} | {p_ex_str} | "
            f"{p_as:.2e} | {p_holm_str} | {dec} | {rec} |"
        )
    return "\n".join(lines)


def build_table4_md():
    df = pd.read_csv(os.path.join("results", "ABLATION_RESULTS.csv"))
    variant_names = {
        "full": "**Full PSO-TCWavKAN Master**",
        "no_revin": "Variant 1: w/o RevIN (No Norm)",
        "no_tcd": "Variant 2: w/o Multi-Kernel Conv (No TCD)",
        "spline_kan_block": "Variant 3: Spline-KAN instead of WavKAN",
        "no_clpso": "Variant 4: w/o CLPSO (Fixed Default Init)",
    }
    lines = [
        r"| Ablation Variant | Test MSE ($\text{Mean} \pm \text{SD}$) | Test MAE ($\text{Mean} \pm \text{SD}$) | $\Delta$ MSE (%) | Mean Epoch Time (s) |",
        r"| :--- | :---: | :---: | :---: | :---: |",
    ]
    for v_key, v_label in variant_names.items():
        sub = df[df.variant == v_key]
        if sub.empty:
            continue
        r = sub.iloc[0]
        m_mse, s_mse = float(r.mean_mse), float(r.std_mse)
        m_mae, s_mae = float(r.mean_mae), float(r.std_mae)
        delta = float(r.delta_mse_pct)
        rt = float(r.phase1_mean_epoch_sec)
        delta_str = "**0.00%**" if v_key == "full" else f"{delta:+.2f}%"
        if v_key == "full":
            lines.append(
                f"| {v_label} | **{m_mse:.4f} $\\pm$ {s_mse:.4f}** | "
                f"**{m_mae:.4f} $\\pm$ {s_mae:.4f}** | {delta_str} | {rt:.2f} |"
            )
        else:
            lines.append(
                f"| {v_label} | {m_mse:.4f} $\\pm$ {s_mse:.4f} | "
                f"{m_mae:.4f} $\\pm$ {s_mae:.4f} | {delta_str} | {rt:.2f} |"
            )
    return "\n".join(lines)


def build_table5_md():
    df = pd.read_csv(os.path.join("results", "EDGE_PROFILE.csv"))
    models = [
        ("dlinear", "DLinear"),
        ("nlinear", "NLinear"),
        ("patchtst_lite", "PatchTST Lite"),
        ("bspline_kan", "B-Spline KAN"),
        ("tcwavkan_master", "**PSO-TCWavKAN Master**"),
    ]
    lines = [
        r"| Model | Config $(L, H, B_{\mathrm{tr}})$ | Total Params | Train VRAM (MB) | Infer VRAM $B=1$ (MB) | Latency $B=1$ (ms) |",
        r"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_key, m_label in models:
        r0 = df[(df.model == m_key) & (df.lookback == 96) & (df.horizon == 96)].iloc[0]
        r1 = df[(df.model == m_key) & (df.lookback == 336) & (df.horizon == 720)].iloc[0]
        p0, p1 = int(r0.total_params) / 1000.0, int(r1.total_params) / 1000.0
        p_str = f"{p0:,.1f}K / {p1:,.1f}K"
        tv_str = f"{float(r0.peak_train_vram_mb):.3f} / {float(r1.peak_train_vram_mb):.3f}"
        iv_str = f"{float(r0.peak_infer_vram_b1_mb):.3f} / {float(r1.peak_infer_vram_b1_mb):.3f}"
        lt_str = f"{float(r0.infer_latency_b1_ms_mean):.2f} / {float(r1.infer_latency_b1_ms_mean):.2f}"
        if m_key == "tcwavkan_master":
            p_str, tv_str, iv_str, lt_str = f"**{p_str}**", f"**{tv_str}**", f"**{iv_str}**", f"**{lt_str}**"
        lines.append(f"| {m_label} | (96, 96, 128) / (336, 720, 32) | {p_str} | {tv_str} | {iv_str} | {lt_str} |")
    return "\n".join(lines)


def update_markdown():
    md_path = "paper_v91_FINAL.md"
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    t1_md = build_table1_md()
    content, n1 = re.subn(
        r"(\*\*Table 1:\*\*.*?\n\n)(\| Dataset \|.*?\n(?:\|.*?\|\n)+)",
        lambda m: m.group(1) + t1_md + "\n",
        content,
    )
    df_bench = pd.read_csv(os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv"))
    df_pso = df_bench[df_bench["model"] == "tcwavkan_master"]
    df_clpso = pd.read_csv(os.path.join("results", "CLPSO_PER_CONFIG_LOG.csv"))
    main_rt = float(df_pso["total_runtime_sec"].sum())
    clpso_rt = float(df_clpso["pso_time_sec"].sum())
    df_w = pd.read_csv(os.path.join("results", "WILCOXON_STATS.csv"))
    dlin_gain = float(df_w[df_w["baseline_model"] == "dlinear"].iloc[0]["mean_pct_reduction"])
    max_gain = max(
        ((float(df_bench[(df_bench.model == "dlinear") & (df_bench.dataset == r.dataset) & (df_bench.lookback == r.lookback) & (df_bench.horizon == r.horizon)].iloc[0]["mean_mse"]) - float(r.mean_mse))
         / float(df_bench[(df_bench.model == "dlinear") & (df_bench.dataset == r.dataset) & (df_bench.lookback == r.lookback) & (df_bench.horizon == r.horizon)].iloc[0]["mean_mse"]) * 100.0)
        for _, r in df_pso.iterrows()
    )

    new_note1 = (
        f"*Note: Total main training runtime across 16 configs (3 seeds each): {main_rt:,.2f} s; "
        f"total CLPSO pre-optimization runtime: {clpso_rt:,.2f} s. "
        f"Mean MSE reduction vs. DLinear across 16 configs: +{dlin_gain:.2f}% (reaching up to +{max_gain:.2f}% on ETTh1 $L=96, H=720$).*"
    )
    content = re.sub(
        r"^\*Note:.*?(?:DLinear|16 configs).*$",
        lambda m: new_note1,
        content,
        count=1,
        flags=re.MULTILINE,
    )

    t2_md = build_table2_md()
    content, n2 = re.subn(
        r"(\*\*Table 2:\*\*.*?\n\n)(\| Dataset \|.*?\n(?:\|.*?\|\n)+)",
        lambda m: m.group(1) + t2_md + "\n",
        content,
    )

    t3_md = build_table3_md()
    content, n3 = re.subn(
        r"(\*\*Table 3:\*\*.*?\n\n)(\| Paired Comparison \|.*?\n(?:\|.*?\|\n)+)",
        lambda m: m.group(1) + t3_md + "\n",
        content,
    )

    t4_md = build_table4_md()
    content, n4 = re.subn(
        r"(\*\*Table 4:\*\*.*?\n\n)(\| Ablation Variant \|.*?\n(?:\|.*?\|\n)+)",
        lambda m: m.group(1) + t4_md + "\n",
        content,
    )

    t5_md = build_table5_md()
    content, n5 = re.subn(
        r"(\*\*Table 5:\*\*.*?\n\n)(\| Model \|.*?\n(?:\|.*?\|\n)+)",
        lambda m: m.group(1) + t5_md + "\n",
        content,
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] Synchronized paper_v91_FINAL.md (Tables matched: T1={n1}, T2={n2}, T3={n3}, T4={n4}, T5={n5}).")


if __name__ == "__main__":
    update_markdown()
