"""
Experiment Runner — KAN HPO Benchmark Study.

Chạy 6 thuật toán (PSO / GWO / DE / GA / WOA / Random) × N_RUNS lần,
mỗi lần tối ưu trên ETTh1, sau đó evaluate config tốt nhất trên test set
ở cả 4 horizon (96 / 192 / 336 / 720).

Kết quả mỗi run → experiments/results/<ALG>_run<i>.json
Bảng tổng hợp   → experiments/summary.csv     (cho Wilcoxon / Friedman)
Đường hội tụ    → experiments/convergence.csv  (cho Figure trong bài báo)

Checkpoint/resume:
    Mỗi run được lưu thành file JSON riêng ngay khi xong. Nếu file đã tồn tại,
    run đó được bỏ qua — chạy lại script sẽ tiếp tục từ điểm dừng.

ƯTiming (CPU, HPO_EPOCHS=8, POP=10, EPOCH=10):
    ~1 eval ~ 20-25s  ->  1 run ~ 100 x 24s / 3600 ~ 40 min
    6 algs x 5 runs = 30 runs x 40 min = 20h (overnight feasible)
    Dung --fast de override constants tu dong lenh.
"""

import os
import sys
import json
import time
import logging
import argparse
import math
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "efficient-kan", "src"))

from efficient_kan import KAN                                  # noqa: E402
from datasets    import build_loaders, get_input_dim           # noqa: E402
from kan_search_space import BOUNDS, decode_solution, DEFAULT_CFG, N_DIMS  # noqa: E402
from kan_hpo_problem  import KANHPOProblem, train_one_config   # noqa: E402
from mealpy import IntegerVar                                  # noqa: E402

from mealpy import PSO, GWO, DE, GA, WOA                      # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Hằng số thí nghiệm (override bằng --fast hoặc đối số CLI) ────────────────
N_RUNS        = 5           # so lan chay moi thuat toan (Wilcoxon can >= 5)
MEALPY_EPOCH  = 10          # so epoch MEALPY (vong lap ngoai)
POP_SIZE      = 10          # kich thuoc quan the
FINAL_EPOCHS  = 30          # epoch train lai sau khi tim duoc config tot nhat
HORIZONS      = [96, 192, 336, 720]
ETT_ROOT      = os.path.join(_ROOT, "ETDataset", "ETT-small")
HPO_DATASET   = "ETTh1"          # dataset dùng cho vòng lặp HPO (nhanh)
EVAL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]  # đánh giá final đủ 4 dataset
FEATURES      = "M"
RESULTS_DIR   = os.path.join(_ROOT, "experiments", "results")
SEED_BASE     = 1000        # seed run i = SEED_BASE + i

# ── Thuật toán MEALPY ─────────────────────────────────────────────────────────
ALGORITHMS: Dict[str, Any] = {
    "PSO": PSO.OriginalPSO,
    "GWO": GWO.OriginalGWO,
    "DE":  DE.OriginalDE,
    "GA":  GA.BaseGA,
    "WOA": WOA.OriginalWOA,
    # Random được thêm bằng RandomSearchRunner (không phải MEALPY)
}


# ── Random Search baseline ─────────────────────────────────────────────────────
class RandomSearchRunner:
    """Random search với cùng ngân sách eval = MEALPY_EPOCH × POP_SIZE.

    Tương thích MEALPY 3.x: tạo solution dict {name: value} thay vì ndarray.
    """

    def __init__(self, epoch: int = MEALPY_EPOCH, pop_size: int = POP_SIZE):
        self.n_evals = epoch * pop_size

    def solve(self, problem: KANHPOProblem, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        best_sol, best_fit = None, float("inf")
        hist: List[float] = []

        for _ in range(self.n_evals):
            # Tạo solution dict theo đúng format — b.lb/b.ub là numpy arrays trong MEALPY 3.x
            sol: dict = {}
            for b in BOUNDS:
                lb = float(b.lb[0]) if hasattr(b.lb, '__len__') else float(b.lb)
                ub = float(b.ub[0]) if hasattr(b.ub, '__len__') else float(b.ub)
                v = rng.uniform(lb, ub)
                sol[b.name] = int(round(v)) if isinstance(b, IntegerVar) else float(v)

            fit = problem.obj_func(sol)
            if fit < best_fit:
                best_fit = fit
                best_sol = dict(sol)
            hist.append(best_fit)

        # Convert best_sol dict → ndarray (theo thu tu BOUNDS) de decode_solution nhan duoc
        best_pos_arr = np.array([
            best_sol[b.name] for b in BOUNDS
        ], dtype=float)

        # Expose g_best (MEALPY 3.x interface)
        self.g_best = SimpleNamespace(
            solution=best_pos_arr,
            target=SimpleNamespace(fitness=best_fit),
        )
        # Legacy attribute (backward compat)
        self.solution = self.g_best
        self.history = SimpleNamespace(list_global_best_fit=hist)


# ── Build + train helper ───────────────────────────────────────────────────────
def _build_and_train_on(
    cfg: Dict[str, Any],
    dataset: str,
    pred_len: int,
    n_epochs: int,
    device: torch.device,
    seed: int,
) -> nn.Module:
    """Train KAN với cfg trên DATASET CHỈ ĐỊNH, trả về model đã train.

    FIX: tham số dataset tường minh — không hard-code HPO_DATASET.
    Đảm bảo train-split và test-split cùng nguồn.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_len   = cfg["seq_len"]
    input_dim = get_input_dim(seq_len, FEATURES)

    train_loader, _, _, _ = build_loaders(
        ETT_ROOT, dataset,              # ← dataset tường minh
        seq_len=seq_len, pred_len=pred_len,
        features=FEATURES, batch_size=cfg["batch_size"],
    )

    width = [input_dim] + [cfg["hidden_size"]] * cfg["n_layers"] + [pred_len]
    model = KAN(
        layers_hidden=width, grid_size=cfg["grid_size"],
        spline_order=cfg["spline_order"], scale_noise=cfg["scale_noise"],
        scale_base=cfg["scale_base"], scale_spline=cfg["scale_spline"],
        base_activation=nn.SiLU, grid_eps=cfg["grid_eps"], grid_range=[-1.0, 1.0],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.MSELoss()

    x_repr, _ = next(iter(train_loader))
    x_repr = x_repr.to(device)

    for epoch in range(n_epochs):
        in_warmup = epoch < cfg["warmup_epochs"]
        ue = cfg["update_every"]
        if in_warmup and (ue == 0 or epoch % max(ue, 1) == 0):
            with torch.no_grad():
                model(x_repr, update_grid=True)

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, update_grid=False)
            loss = criterion(pred, yb) + model.regularization_loss(
                cfg["reg_lambda"], cfg["reg_lambda"]
            )
            if math.isfinite(loss.item()):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    return model


def evaluate_final_config(
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Retrain + test trên CÙNG dataset, đủ 4 horizon.

    FIX (lỗi #5): train và test phải trên cùng 1 dataset.
    Phiên bản cũ train trên ETTh1 rồi test trên ETTh2/m1/m2 = transfer learning,
    không phải benchmark. Phiên bản mới: với mỗi dataset, train trên train-split
    của dataset đó rồi test trên test-split của dataset đó.

    Returns:
        {"ETTh1": {"96": {"mse":..., "mae":...}, ...}, "ETTh2": {...}, ...}
    """
    all_results: Dict[str, Any] = {}
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    for dataset in EVAL_DATASETS:
        ds_results: Dict[str, Any] = {}
        for h in HORIZONS:
            # Train VÀ test đều dùng cùng dataset → đúng benchmark chuẩn
            model = _build_and_train_on(
                cfg, dataset=dataset, pred_len=h,
                n_epochs=FINAL_EPOCHS, device=device, seed=seed,
            )
            _, _, test_loader, _ = build_loaders(
                ETT_ROOT, dataset,
                seq_len=cfg["seq_len"], pred_len=h,
                features=FEATURES, batch_size=256,
            )
            model.eval()
            mse_sum = mae_sum = 0.0
            cnt = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb, update_grid=False)
                    mse_sum += mse_fn(pred, yb).item() * len(xb)
                    mae_sum += mae_fn(pred, yb).item() * len(xb)
                    cnt += len(xb)
            ds_results[str(h)] = {
                "mse": round(mse_sum / cnt, 6) if cnt else None,
                "mae": round(mae_sum / cnt, 6) if cnt else None,
            }
            logger.info("  %s  H=%d  MSE=%.4f  MAE=%.4f",
                        dataset, h, ds_results[str(h)]["mse"], ds_results[str(h)]["mae"])
        all_results[dataset] = ds_results
    return all_results


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _result_path(alg: str, run_idx: int) -> str:
    return os.path.join(RESULTS_DIR, f"{alg}_run{run_idx:02d}.json")


def _is_done(alg: str, run_idx: int) -> bool:
    return os.path.isfile(_result_path(alg, run_idx))


def _save_result(alg: str, run_idx: int, record: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(_result_path(alg, run_idx), "w") as f:
        json.dump(record, f, indent=2)


# ── Chạy 1 thuật toán × 1 run ─────────────────────────────────────────────────
def run_one(alg_name: str, run_idx: int, device: torch.device) -> Optional[dict]:
    """Chạy 1 algorithm 1 lần. Bỏ qua nếu checkpoint đã tồn tại.

    alg_name special values:
      "Default" — không HPO, dùng DEFAULT_CFG (baseline so sánh)
      "Random"  — random search với cùng ngân sách eval
    """
    if _is_done(alg_name, run_idx):
        logger.info("SKIP  %s run%02d (exists)", alg_name, run_idx)
        return None

    seed = SEED_BASE + run_idx
    logger.info("START %s run%02d  seed=%d", alg_name, run_idx, seed)
    t0 = time.time()

    if alg_name == "Default":
        # Baseline: không HPO, dùng efficient-kan defaults
        best_cfg  = dict(DEFAULT_CFG)
        best_fit  = train_one_config(best_cfg, device=device, verbose=False)
        convergence = []
    else:
        problem = KANHPOProblem(bounds=BOUNDS, minmax="min")

        if alg_name == "Random":
            optimizer = RandomSearchRunner(epoch=MEALPY_EPOCH, pop_size=POP_SIZE)
        else:
            optimizer = ALGORITHMS[alg_name](epoch=MEALPY_EPOCH, pop_size=POP_SIZE)

        optimizer.solve(problem, seed=seed)
        # MEALPY 3.x: best solution in optimizer.g_best
        g_best   = optimizer.g_best
        best_pos = g_best.solution          # ndarray position
        best_fit = float(g_best.target.fitness)
        best_cfg = decode_solution(best_pos)
        # Convergence history: try list_global_best_fit (MEALPY 3.x)
        try:
            convergence = [float(v) for v in optimizer.history.list_global_best_fit]
        except AttributeError:
            convergence = []

    logger.info("HPO done %s run%02d  fit=%.6f  (%.0fs)",
                alg_name, run_idx, best_fit, time.time() - t0)

    test_metrics = evaluate_final_config(best_cfg, seed=seed, device=device)

    record = {
        "algorithm": alg_name, "run": run_idx, "seed": seed,
        "hpo_fitness": best_fit, "best_config": best_cfg,
        "convergence": convergence, "test_metrics": test_metrics,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _save_result(alg_name, run_idx, record)
    logger.info("DONE  %s run%02d  elapsed=%.0fs", alg_name, run_idx, time.time() - t0)
    return record


# ── Chạy tất cả experiments với resume ────────────────────────────────────────
def run_all(alg_filter: Optional[List[str]] = None) -> None:
    """Chạy toàn bộ matrix (algorithm × run), tự động bỏ qua run đã hoàn thành."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    all_algs = list(ALGORITHMS.keys()) + ["Random", "Default"]
    if alg_filter:
        all_algs = [a for a in all_algs if a in alg_filter]

    total = len(all_algs) * N_RUNS
    done  = sum(_is_done(a, i) for a in all_algs for i in range(N_RUNS))
    logger.info("Progress: %d / %d runs done", done, total)

    for alg in all_algs:
        for run_idx in range(N_RUNS):
            run_one(alg, run_idx, device)

    logger.info("All runs complete. Collecting CSV...")
    collect_to_csv()


# ── Tổng hợp kết quả → CSV ────────────────────────────────────────────────────
def collect_to_csv() -> None:
    """Đọc tất cả JSON trong RESULTS_DIR → summary.csv + convergence.csv."""
    import csv

    all_algs = list(ALGORITHMS.keys()) + ["Random", "Default"]
    summary_rows: List[dict] = []
    conv_rows:    List[dict] = []

    for alg in all_algs:
        for run_idx in range(N_RUNS):
            path = _result_path(alg, run_idx)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)

            row: Dict[str, Any] = {
                "algorithm": alg, "run": run_idx, "seed": rec["seed"],
                "hpo_fitness": rec["hpo_fitness"],
                "elapsed_s":   rec["elapsed_s"],
            }
            # test_metrics: {"ETTh1": {"96": {"mse":..., "mae":...}, ...}, ...}
            tm = rec.get("test_metrics", {})
            for ds in EVAL_DATASETS:
                ds_m = tm.get(ds, {})
                for h in HORIZONS:
                    m = ds_m.get(str(h), {})
                    row[f"mse_{ds}_{h}"] = m.get("mse")
                    row[f"mae_{ds}_{h}"] = m.get("mae")
            summary_rows.append(row)

            for ep, fit in enumerate(rec.get("convergence", [])):
                conv_rows.append({"algorithm": alg, "run": run_idx,
                                   "epoch": ep, "best_fit": fit})

    # summary.csv
    os.makedirs(os.path.join(_ROOT, "experiments"), exist_ok=True)
    s_path = os.path.join(_ROOT, "experiments", "summary.csv")
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with open(s_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(summary_rows)
        logger.info("Wrote %s (%d rows)", s_path, len(summary_rows))

    # convergence.csv
    c_path = os.path.join(_ROOT, "experiments", "convergence.csv")
    if conv_rows:
        with open(c_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["algorithm", "run", "epoch", "best_fit"])
            w.writeheader(); w.writerows(conv_rows)
        logger.info("Wrote %s (%d rows)", c_path, len(conv_rows))


# ── Ước lượng thời gian chạy ──────────────────────────────────────────────────
def estimate_timing() -> None:
    """Chạy 1 fitness eval và ngoại suy tổng thời gian."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MEALPY 3.x: mid-point dùng dict, không dùng ndarray
    # b.lb và b.ub là numpy arrays → cần float() trước khi round
    mid_sol: dict = {}
    for b in BOUNDS:
        lb = float(b.lb[0]) if hasattr(b.lb, '__len__') else float(b.lb)
        ub = float(b.ub[0]) if hasattr(b.ub, '__len__') else float(b.ub)
        v = (lb + ub) / 2
        mid_sol[b.name] = int(round(v)) if isinstance(b, IntegerVar) else v
    cfg = decode_solution(mid_sol)

    logger.info("Timing 1 fitness eval on %s ...", device)
    t0 = time.time()
    fit = train_one_config(cfg, device=device, verbose=False)
    t1 = time.time()
    per_eval = t1 - t0

    n_evals   = MEALPY_EPOCH * POP_SIZE
    per_run   = per_eval * n_evals
    total     = per_run * N_RUNS * (len(ALGORITHMS) + 1)  # +1 for Random
    per_final = per_eval * (FINAL_EPOCHS / 40) * len(HORIZONS)  # rough estimate

    print(f"\n{'='*55}")
    print(f"Device          : {device}")
    print(f"1 fitness eval  : {per_eval:.1f}s")
    print(f"Evals per run   : {n_evals}  ({MEALPY_EPOCH} epochs × {POP_SIZE} pop)")
    print(f"Time per HPO run: {per_run/60:.0f} min")
    print(f"Final eval/run  : {per_final/60:.0f} min (×{len(HORIZONS)} horizons)")
    print(f"Total (est.)    : {total/3600:.1f}h  "
          f"({len(ALGORITHMS)+1} algs × {N_RUNS} runs)")
    print(f"SSW fitness     : {fit:.6f}")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="KAN HPO Experiment Runner")
    parser.add_argument("--alg",    nargs="+", help="Chỉ chạy các thuật toán này (PSO GWO DE GA WOA Random)")
    parser.add_argument("--run",    type=int,  help="Chỉ chạy run index này (0-based)")
    parser.add_argument("--csv",    action="store_true", help="Chỉ collect CSV từ các JSON đã có")
    parser.add_argument("--timing", action="store_true", help="Ước lượng thời gian chạy rồi thoát")
    parser.add_argument("--fast",   action="store_true",
                        help="Override: epoch=15, pop=10, runs=3, final_epochs=50 (để test nhanh)")
    args = parser.parse_args()

    if args.fast:
        global MEALPY_EPOCH, POP_SIZE, N_RUNS, FINAL_EPOCHS
        MEALPY_EPOCH  = 15
        POP_SIZE      = 10
        N_RUNS        = 3
        FINAL_EPOCHS  = 50
        logger.info("FAST mode: epoch=%d pop=%d runs=%d final=%d",
                    MEALPY_EPOCH, POP_SIZE, N_RUNS, FINAL_EPOCHS)

    if args.timing:
        estimate_timing()
        return

    if args.csv:
        collect_to_csv()
        return

    if args.run is not None and args.alg:
        # chạy 1 run cụ thể cho 1 hoặc nhiều thuật toán
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for alg in args.alg:
            run_one(alg, args.run, device)
        collect_to_csv()
        return

    run_all(alg_filter=args.alg)


if __name__ == "__main__":
    main()
