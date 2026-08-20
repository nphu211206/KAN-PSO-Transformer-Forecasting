"""
MEALPY 3.x Problem class cho KAN Hyperparameter Optimization.

Thay đổi so với phiên bản cũ (MEALPY 2.x):
  - obj_func(solution) nhận DICT thay vì np.ndarray
  - Không còn gọi decode_solution(np.ndarray) với indexing số
  - BOUNDS từ kan_search_space là list[IntegerVar|FloatVar]

Fitness = Stabilization Sliding Window (SSW) val-MSE:
    Thay vì val_loss của epoch cuối, lấy mean(val_MSE[-SSW_K:]).
    Grid shift gây bước nhảy bất thường → epoch cuối có thể rơi vào đỉnh nhiễu.
    SSW ổn định hơn và phản ánh hiệu năng trung bình sau khi grid đã ổn định.

Quy trình:
    1. MEALPY gọi obj_func(solution: dict)
    2. decode_solution(dict) → hyperparams dict
    3. Build KAN([input_dim, h1, ..., pred_len])
    4. Train HPO_EPOCHS epoch với AdamW
       - Trong warmup: gọi update_grid 1 lần / epoch (không phải mỗi batch)
       - Sau warmup: đóng băng lưới
    5. Trả về mean(val_MSE[-SSW_K:])
"""

import os
import sys
import json
import time
import math
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# ── sys.path: thêm efficient-kan ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_KAN_SRC = os.path.join(_PROJECT_ROOT, "efficient-kan", "src")
if _KAN_SRC not in sys.path:
    sys.path.insert(0, _KAN_SRC)
from efficient_kan import KAN  # noqa: E402

from mealpy import Problem     # noqa: E402
from datasets import build_loaders, get_input_dim          # noqa: E402
from kan_search_space import decode_solution, BOUNDS, SSW_K, HPO_EPOCHS  # noqa: E402

logger = logging.getLogger(__name__)

# ── Hằng số thí nghiệm ───────────────────────────────────────────────────────
HPO_DATASET  = "ETTh1"    # dataset dùng cho phase HPO
HPO_PRED_LEN = 96         # horizon ngắn nhất → đánh giá nhanh hơn
ETT_ROOT     = os.path.join(_PROJECT_ROOT, "ETDataset", "ETT-small")
FEATURES     = "M"        # multivariate: 7 cột
PENALTY      = 1e6        # giá trị trả về khi có NaN/inf/crash
SEED         = 42

# ── Checkpoint ───────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(_PROJECT_ROOT, "hpo_checkpoint.json")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: Dict[str, Any], input_dim: int, pred_len: int,
                 device: torch.device) -> KAN:
    """Xây dựng KAN model từ config dict.
    Architecture: [input_dim, hidden_size × n_layers, pred_len]
    """
    width = [input_dim] + [cfg["hidden_size"]] * cfg["n_layers"] + [pred_len]
    model = KAN(
        layers_hidden=width,
        grid_size=cfg["grid_size"],
        spline_order=cfg["spline_order"],
        scale_noise=cfg["scale_noise"],
        scale_base=cfg["scale_base"],
        scale_spline=cfg["scale_spline"],
        base_activation=nn.SiLU,
        grid_eps=cfg["grid_eps"],
        grid_range=[-1.0, 1.0],   # khớp MinMaxScaler(feature_range=(-1, 1))
    ).to(device)
    return model


def train_one_config(
    cfg: Dict[str, Any],
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> float:
    """Train KAN với config cho trước, trả về SSW val-MSE (thấp hơn = tốt hơn).

    Args:
        cfg    : dict từ decode_solution()
        device : torch device; None → tự chọn cuda nếu có
        verbose: in loss mỗi epoch

    Returns:
        float: SSW fitness = mean(val_mse[-SSW_K:]), hoặc PENALTY nếu crash.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _set_seed(SEED)

    seq_len   = cfg["seq_len"]
    pred_len  = HPO_PRED_LEN
    input_dim = get_input_dim(seq_len, FEATURES)

    # ── Dataloader ──────────────────────────────────────────────────────────
    try:
        train_loader, val_loader, _, _ = build_loaders(
            ETT_ROOT, HPO_DATASET,
            seq_len=seq_len, pred_len=pred_len,
            features=FEATURES, batch_size=cfg["batch_size"],
        )
    except Exception as e:
        logger.warning("DataLoader error: %s", e)
        return PENALTY

    # ── Model ────────────────────────────────────────────────────────────────
    try:
        model = _build_model(cfg, input_dim, pred_len, device)
    except Exception as e:
        logger.warning("Model build error: %s", e)
        return PENALTY

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.MSELoss()

    # Lấy 1 batch đại diện để update_grid
    try:
        x_repr, _ = next(iter(train_loader))
        x_repr = x_repr.to(device)
    except StopIteration:
        return PENALTY

    # ── Training loop ────────────────────────────────────────────────────────
    val_mse_history = []

    for epoch in range(HPO_EPOCHS):

        # --- Grid update schedule (1 lần/epoch, KHÔNG phải mỗi batch) --------
        in_warmup = epoch < cfg["warmup_epochs"]
        ue = cfg["update_every"]
        should_update = in_warmup and (ue == 0 or epoch % max(ue, 1) == 0)

        if should_update:
            with torch.no_grad():
                model(x_repr, update_grid=True)

        # --- Train batches ──────────────────────────────────────────────────
        model.train()
        consecutive_nan = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, update_grid=False)
            mse  = criterion(pred, yb)
            reg  = model.regularization_loss(cfg["reg_lambda"], cfg["reg_lambda"])
            loss = mse + reg
            if not math.isfinite(loss.item()):
                consecutive_nan += 1
                if consecutive_nan >= 5:
                    logger.warning(
                        "epoch %d: 5+ consecutive nan batches, PENALTY  cfg=%s",
                        epoch + 1, cfg
                    )
                    return PENALTY
                continue
            consecutive_nan = 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation ─────────────────────────────────────────────────────
        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, update_grid=False)
                val_sum += criterion(pred, yb).item() * len(xb)
                val_cnt += len(xb)

        if val_cnt == 0:
            return PENALTY

        val_mse = val_sum / val_cnt
        if not math.isfinite(val_mse):
            return PENALTY

        val_mse_history.append(val_mse)
        if verbose:
            print(f"  epoch {epoch+1:3d}/{HPO_EPOCHS}  val_mse={val_mse:.6f}")

    # ── SSW fitness ──────────────────────────────────────────────────────────
    ssw_fitness = float(np.mean(val_mse_history[-SSW_K:]))
    return ssw_fitness


# ── MEALPY 3.x Problem class ──────────────────────────────────────────────────

class KANHPOProblem(Problem):
    """Single-objective minimization problem cho MEALPY 3.x.

    Fitness = SSW val-MSE (thấp hơn = tốt hơn).

    Thay đổi vs MEALPY 2.x:
      - bounds = list[IntegerVar|FloatVar] (từ kan_search_space.BOUNDS)
      - obj_func nhận solution: dict {name: value}  ← MEALPY 3.x tự decode

    Cách dùng:
        from mealpy import PSO
        from kan_search_space import BOUNDS
        from kan_hpo_problem import KANHPOProblem

        problem = KANHPOProblem(bounds=BOUNDS, minmax="min")
        optimizer = PSO.OriginalPSO(epoch=50, pop_size=30)
        optimizer.solve(problem)
        best_cfg = decode_solution(optimizer.g_best.solution)
    """

    def __init__(self, bounds=None, minmax: str = "min", **kwargs):
        self._best_fitness: float = float("inf")
        self._best_cfg: Optional[Dict[str, Any]] = None
        self._eval_count: int = 0
        self._t0: float = time.time()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(bounds=bounds, minmax=minmax, **kwargs)

    def obj_func(self, solution) -> float:
        """MEALPY 3.x gọi hàm này với solution là numpy ndarray (shape=14,).

        decode_solution() nhận cả ndarray lẫn dict — hoạt động trong cả hai trường hợp.

        Returns:
            SSW val-MSE (scalar float), thấp hơn = tốt hơn.
        """
        self._eval_count += 1
        cfg = decode_solution(solution)   # dict → dict

        fitness = train_one_config(cfg, device=self._device, verbose=False)

        # ── Checkpoint: lưu config tốt nhất ─────────────────────────────────
        if fitness < self._best_fitness:
            self._best_fitness = fitness
            self._best_cfg = cfg
            elapsed = time.time() - self._t0
            record = {
                "eval":      self._eval_count,
                "fitness":   fitness,
                "elapsed_s": round(elapsed, 1),
                "config":    cfg,
            }
            try:
                with open(CKPT_PATH, "w") as f:
                    json.dump(record, f, indent=2)
            except OSError:
                pass
            logger.info(
                "[HPO] eval=%d  fitness=%.6f  elapsed=%.0fs  cfg=%s",
                self._eval_count, fitness, elapsed, cfg,
            )

        return fitness


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from kan_search_space import BOUNDS, PARAM_NAMES
    from mealpy import IntegerVar

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Search space: {len(BOUNDS)} dims")
    for b in BOUNDS:
        print(f"  {b.name:20s}  [{b.lb:.4g}, {b.ub:.4g}]  {type(b).__name__}")

    # Tạo mid-point solution (giả lập MEALPY 3.x dict)
    mid_sol: dict = {}
    for b in BOUNDS:
        v = (b.lb + b.ub) / 2
        mid_sol[b.name] = int(round(v)) if isinstance(b, IntegerVar) else float(v)

    cfg = decode_solution(mid_sol)
    print("\nMid-point config:", json.dumps(cfg, indent=2))

    print("\nRunning 1 fitness evaluation (may take ~2 min on CPU)...")
    fitness = train_one_config(cfg, verbose=True)
    print(f"\nSSW fitness = {fitness:.6f}")
