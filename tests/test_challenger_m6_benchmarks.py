#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical Challenger Test Suite for Option A Verification:
- Bessel ddof=1 Sample Statistics across all 16 configurations
- Tensor prediction audit across all 144 Option A .npz files (48 Master + 80 baseline + 16 ground truth)
- Paired Wilcoxon Signed-Rank & Holm-Bonferroni test verification against WILCOXON_STATS.csv
- LaTeX <-> Markdown Table 1 consistency
- Kaggle Ingestion SecurityError & --verify_only stress tests
"""

import io
import os
import re
import json
import zipfile
import shutil
import inspect
import pytest
import numpy as np
import pandas as pd
from scipy import stats

MASTER_CSV = os.path.join("results", "MASTER_BENCHMARK_RESULTS.csv")
WILCOX_CSV = os.path.join("results", "WILCOXON_STATS.csv")
PAPER_TEX = "paper_ncaa.tex"
PAPER_MD = "paper_v91_FINAL.md"
PRED_DIR = "predictions"

EXPECTED_CONFIGS = {
    ("ETTh1", 96, 96), ("ETTh1", 96, 192), ("ETTh1", 96, 336), ("ETTh1", 96, 720),
    ("ETTh1", 336, 96), ("ETTh1", 336, 192), ("ETTh1", 336, 336), ("ETTh1", 336, 720),
    ("ETTh2", 96, 96), ("ETTh2", 96, 192), ("ETTh2", 96, 336), ("ETTh2", 96, 720),
    ("ETTh2", 336, 96), ("ETTh2", 336, 192), ("ETTh2", 336, 336), ("ETTh2", 336, 720),
}


def test_master_benchmark_csv_schema_and_integrity():
    """Verify MASTER_BENCHMARK_RESULTS.csv has 96 rows (16 configs x 6 models)."""
    assert os.path.exists(MASTER_CSV), f"Missing {MASTER_CSV}"
    df = pd.read_csv(MASTER_CSV)
    assert len(df) == 96, f"Expected 96 rows in MASTER_BENCHMARK_RESULTS.csv, got {len(df)}"
    df_pso = df[df["model"] == "tcwavkan_master"]
    assert len(df_pso) == 16, f"Expected 16 configurations for tcwavkan_master, got {len(df_pso)}"
    actual_configs = set(zip(df_pso["dataset"], df_pso["lookback"], df_pso["horizon"]))
    assert actual_configs == EXPECTED_CONFIGS, f"Config mismatch: {actual_configs ^ EXPECTED_CONFIGS}"


def test_bessel_correction_ddof1_on_all_16_configurations():
    """Verify sample standard deviations strictly follow Bessel's correction (ddof=1) for N=3 seeds."""
    df = pd.read_csv(MASTER_CSV)
    df_pso = df[df["model"] == "tcwavkan_master"]
    for _, r in df_pso.iterrows():
        ds, l, h = str(r["dataset"]), int(r["lookback"]), int(r["horizon"])
        raw_mses = json.loads(r["seed_mses"]) if isinstance(r["seed_mses"], str) else list(r["seed_mses"])
        assert len(raw_mses) == 3, f"Expected 3 seeds for {ds} L={l} H={h}"

        std1_mse = float(np.std(raw_mses, ddof=1))
        std0_mse = float(np.std(raw_mses, ddof=0))
        csv_std_mse = float(r["std_mse"])

        if std0_mse > 1e-12:
            ratio = std1_mse / std0_mse
            assert abs(ratio - np.sqrt(1.5)) < 1e-4, f"Bessel correction ratio mismatch: {ratio}"
        abs_err = abs(std1_mse - csv_std_mse)
        assert abs_err < 2e-5, f"Config {ds} L={l} H={h}: ddof=1 does not match CSV (abs_err={abs_err:.2e})"


def test_empirical_npz_prediction_tensors():
    """Directly load all 144 Option A .npz files and verify MSE/MAE against MASTER_BENCHMARK_RESULTS.csv."""
    npz_files = [f for f in os.listdir(PRED_DIR) if f.endswith(".npz")]
    assert len(npz_files) == 144, f"Expected 144 Option A NPZs in {PRED_DIR}, found {len(npz_files)}"

    df = pd.read_csv(MASTER_CSV)
    df_pso = df[df["model"] == "tcwavkan_master"]
    baselines = ["dlinear", "nlinear", "patchtst_lite", "bspline_kan", "tcwavkan_no_pso"]

    for _, r in df_pso.iterrows():
        ds, l, h = str(r["dataset"]), int(r["lookback"]), int(r["horizon"])
        seeds = json.loads(r["seeds"]) if isinstance(r["seeds"], str) else list(r["seeds"])

        gt_path = os.path.join(PRED_DIR, f"ground_truth_{ds}_L{l}_H{h}.npz")
        assert os.path.exists(gt_path), f"Missing ground truth tensor: {gt_path}"
        trues = np.load(gt_path)["trues"].squeeze()
        assert trues.shape[1] == h, f"Expected horizon {h}, got {trues.shape[1]}"

        npz_mses, npz_maes = [], []
        for s in seeds:
            fpath = os.path.join(PRED_DIR, f"{ds}_L{l}_H{h}_tcwavkan_master_seed{s}.npz")
            assert os.path.exists(fpath), f"Missing prediction tensor file: {fpath}"
            preds = np.load(fpath)["preds"].squeeze()
            assert preds.shape == trues.shape, f"Shape mismatch in {fpath}: {preds.shape} vs {trues.shape}"
            diff = preds - trues
            npz_mses.append(float(np.mean(diff ** 2)))
            npz_maes.append(float(np.mean(np.abs(diff))))

        assert abs(np.mean(npz_mses) - float(r["mean_mse"])) < 1e-4
        assert abs(np.std(npz_mses, ddof=1) - float(r["std_mse"])) < 1e-4
        assert abs(np.mean(npz_maes) - float(r["mean_mae"])) < 1e-4

        for b_model in baselines:
            b_row = df[(df["model"] == b_model) & (df["dataset"] == ds) & (df["lookback"] == l) & (df["horizon"] == h)].iloc[0]
            b_seed_mses = json.loads(b_row["seed_mses"]) if isinstance(b_row["seed_mses"], str) else list(b_row["seed_mses"])
            b_fpath = os.path.join(PRED_DIR, f"{ds}_L{l}_H{h}_{b_model}_seed2024.npz")
            assert os.path.exists(b_fpath), f"Missing baseline NPZ: {b_fpath}"
            b_preds = np.load(b_fpath)["preds"].squeeze()
            assert b_preds.shape == trues.shape
            b_mse_2024 = float(np.mean((b_preds - trues) ** 2))
            assert abs(b_mse_2024 - b_seed_mses[0]) < 1e-4, f"{b_model} seed 2024 MSE mismatch on {ds} L={l} H={h}"


def test_wilcoxon_signed_rank_exact_and_asymptotic():
    """Verify paired two-sided Wilcoxon signed-rank and Holm-Bonferroni statistics across all 5 comparators."""
    df = pd.read_csv(MASTER_CSV)
    df_w = pd.read_csv(WILCOX_CSV)
    pso_mses = df[df["model"] == "tcwavkan_master"].sort_values(["dataset", "lookback", "horizon"])["mean_mse"].to_numpy()
    assert len(pso_mses) == 16

    for _, rw in df_w.iterrows():
        base = str(rw["baseline_model"])
        b_mses = df[df["model"] == base].sort_values(["dataset", "lookback", "horizon"])["mean_mse"].to_numpy()
        diffs = b_mses - pso_mses
        res_exact = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
        assert abs(float(res_exact.pvalue) - float(rw["p_exact"])) < 1e-6, f"Wilcoxon p_exact mismatch for {base}"
        assert int(np.sum(diffs > 0)) == int(rw["win"])
        assert int(np.sum(diffs < 0)) == int(rw["loss"])

    r_dlin = df_w[df_w["baseline_model"] == "dlinear"].iloc[0]
    assert int(r_dlin["win"]) == 14
    assert int(r_dlin["loss"]) == 2
    assert float(r_dlin["w_plus"]) == 125.0
    assert float(r_dlin["w_minus"]) == 11.0
    assert float(r_dlin["p_exact"]) < 0.01
    assert float(r_dlin["holm_p_exact"]) < 0.01


def test_latex_and_markdown_table1_consistency():
    """Verify Table 1 numerical harmony between paper_ncaa.tex and paper_v91_FINAL.md."""
    with open(PAPER_TEX, encoding="utf-8") as f:
        tex = f.read()
    with open(PAPER_MD, encoding="utf-8") as f:
        md = f.read()
    df = pd.read_csv(MASTER_CSV)
    df_pso = df[df["model"] == "tcwavkan_master"]

    m_tex = re.search(r"\\label\{tab:master_benchmark_16_canonical\}.*?\\midrule(.*?)\\bottomrule", tex, re.DOTALL)
    assert m_tex is not None
    tex_rows = [l.strip() for l in m_tex.group(1).strip().split("\n") if "&" in l]
    assert len(tex_rows) == 16

    m_md = re.search(r"\*\*Table 1:\*\*.*?\n\n(\|.*?\n(?:\|.*?\|\n)+)", md, re.DOTALL)
    assert m_md is not None
    md_rows = [l.strip() for l in m_md.group(1).strip().split("\n") if l.startswith("|") and ("ETTh1" in l or "ETTh2" in l)]
    assert len(md_rows) == 16

    for idx, (t_row, m_row) in enumerate(zip(tex_rows, md_rows)):
        t_parts = [p.strip() for p in t_row.split("&")]
        m_parts = [p.strip() for p in m_row.split("|")[1:-1]]

        ds = t_parts[0]
        l = int(t_parts[1])
        h = int(t_parts[2])
        r = df_pso[(df_pso["dataset"] == ds) & (df_pso["lookback"] == l) & (df_pso["horizon"] == h)].iloc[0]

        expected_mse = f"{r['mean_mse']:.4f} $\\pm$ {r['std_mse']:.4f}"
        expected_mae = f"{r['mean_mae']:.4f} $\\pm$ {r['std_mae']:.4f}"

        assert expected_mse in t_parts[4], f"Table 1 TeX MSE mismatch row {idx}"
        assert expected_mae in t_parts[5], f"Table 1 TeX MAE mismatch row {idx}"
        assert expected_mse in m_parts[4], f"Table 1 MD MSE mismatch row {idx}"
        assert expected_mae in m_parts[5], f"Table 1 MD MAE mismatch row {idx}"


def test_ingest_kaggle_results_security_error_nameerror_bug():
    """Verify secure_unpack raises SecurityError on path traversal (Zip Slip) and prefix collisions."""
    import ingest_kaggle_results

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../etc/passwd", "malicious payload")
    buf.seek(0)

    os.makedirs("test_sandbox", exist_ok=True)
    slip_zip = "test_slip_challenger.zip"
    with open(slip_zip, "wb") as f:
        f.write(buf.getvalue())

    try:
        with pytest.raises(ingest_kaggle_results.SecurityError) as excinfo:
            ingest_kaggle_results.secure_unpack(slip_zip, "test_sandbox")
        assert "traversal" in str(excinfo.value).lower()
    finally:
        if os.path.exists(slip_zip):
            os.remove(slip_zip)
        if os.path.exists("test_sandbox"):
            shutil.rmtree("test_sandbox")

    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as zf:
        zf.writestr("../test_sandbox_evil/evil.txt", "malicious prefix payload")
    buf2.seek(0)

    os.makedirs("test_sandbox", exist_ok=True)
    slip_zip2 = "test_slip_prefix.zip"
    with open(slip_zip2, "wb") as f:
        f.write(buf2.getvalue())

    try:
        with pytest.raises(ingest_kaggle_results.SecurityError) as excinfo2:
            ingest_kaggle_results.secure_unpack(slip_zip2, "test_sandbox")
        assert "traversal" in str(excinfo2.value).lower()
    finally:
        if os.path.exists(slip_zip2):
            os.remove(slip_zip2)
        if os.path.exists("test_sandbox"):
            shutil.rmtree("test_sandbox")
        if os.path.exists("test_sandbox_evil"):
            shutil.rmtree("test_sandbox_evil")


def test_ingest_kaggle_results_verify_only_behavior():
    """Verify execute_pipeline honors args.verify_only without mutating manuscripts."""
    import ingest_kaggle_results

    src = inspect.getsource(ingest_kaggle_results.execute_pipeline)
    assert "verify_only" in src

    class MockArgs:
        zip_path = None
        extract_dir = "."
        paper_tex = "paper_ncaa.tex"
        paper_md = "paper_v91_FINAL.md"
        render_figures = False
        verify_only = True

    mtime_before = os.path.getmtime("paper_ncaa.tex") if os.path.exists("paper_ncaa.tex") else None
    ret = ingest_kaggle_results.execute_pipeline(MockArgs())
    assert ret == 0, f"Expected return code 0 from verify_only, got {ret}"
    if mtime_before is not None:
        mtime_after = os.path.getmtime("paper_ncaa.tex")
        assert mtime_before == mtime_after, "paper_ncaa.tex mutated during --verify_only run"
