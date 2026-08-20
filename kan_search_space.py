"""
Không gian tìm kiếm 14 chiều cho KAN Hyperparameter Optimization.
Tương thích MEALPY 3.0.2+ (dùng IntegerVar/FloatVar thay vì list bounds).

seq_len được CỐ ĐỊNH ở 96 — không đưa vào search space.
Thay đổi so với phiên bản cũ:
  - BOUNDS là list[IntegerVar|FloatVar] cho MEALPY 3.x
  - reg_lambda → log_reg_lambda ∈ [-5, -1], decode: 10^x  (R9: bắt buộc log scale)
  - warmup_epochs → warmup_ratio ∈ [0.0, 0.5], decode: int(ratio × HPO_EPOCHS)
    (bất biến với budget thay đổi)
  - update_every đổi sang [0, 10] (giới hạn thực tế hơn khi HPO_EPOCHS=30)
  - decode_solution nhận dict (MEALPY 3.x) thay vì np.ndarray

Căn cứ khoa học mỗi chiều:
  grid_size, spline_order : tham số lõi B-spline, chưa có benchmark hệ thống
  log_grid_eps            : sweet spot 0.02 (R5/R9), phụ thuộc domain
  warmup_ratio            : Grid Update Dilemma (R1/R9) — warmup duration tối ưu
  update_every            : Grid Update Dilemma (R1/R9) — tần suất cập nhật tối ưu
  hidden_size, n_layers   : capacity vs. overfitting KAN
  log_lr, log_wd          : AdamW — tương tác với grid dynamics
  batch_exp               : kích thước batch → grid update stability
  scale_noise             : hội tụ vào 0.27-0.30 (R5)
  scale_base              : khởi tạo base activation
  scale_spline            : khởi tạo spline activation
  log_reg_lambda          : L1+entropy reg, PHẢI log scale (R9)
"""

from mealpy import FloatVar, IntegerVar

# ── Cố định seq_len theo chuẩn ETT benchmark ──────────────────────────────────
SEQ_LEN = 96   # look-back window — cố định để so sánh được với văn liệu

# ── SSW fitness constants ─────────────────────────────────────────────────────
SSW_K      = 3    # Stabilization Sliding Window — final 3 epochs
HPO_EPOCHS = 8    # proxy training epochs (mỗi lần đánh giá fitness ~20s trên CPU)
              # Đủ để phân biệt config tốt/xấu; config tốt nhất được retrain 50 epoch

# ── 14 chiều: MEALPY 3.x IntegerVar / FloatVar ───────────────────────────────
BOUNDS = [
    # --- Lưới B-spline (đóng góp cốt lõi) ---
    IntegerVar(lb=3,    ub=20,   name="grid_size"),      # số ô lưới B-spline
    IntegerVar(lb=2,    ub=5,    name="spline_order"),   # bậc đa thức

    # --- Grid Update Schedule — "Grid Update Dilemma" ---
    FloatVar(lb=-3.0,  ub=-0.3,  name="log_grid_eps"),   # grid_eps = 10^x
    FloatVar(lb=0.0,   ub=0.5,   name="warmup_ratio"),   # warmup = ratio×HPO_EPOCHS
    IntegerVar(lb=0,   ub=4,     name="update_every"),   # mỗi N epoch update (0=mỗi epoch)

    # --- Kiến trúc ---
    IntegerVar(lb=16,  ub=128,   name="hidden_size"),    # chiều rộng hidden layer
    IntegerVar(lb=1,   ub=3,     name="n_layers"),       # số hidden layer

    # --- AdamW ---
    FloatVar(lb=-4.0,  ub=-1.5,  name="log_lr"),         # lr = 10^x
    FloatVar(lb=-5.0,  ub=-2.0,  name="log_wd"),         # weight_decay = 10^x
    IntegerVar(lb=5,   ub=8,     name="batch_exp"),      # batch_size = 2^x → {32,64,128,256}

    # --- Khởi tạo spline ---
    FloatVar(lb=0.01,  ub=0.50,  name="scale_noise"),
    FloatVar(lb=0.10,  ub=2.00,  name="scale_base"),
    FloatVar(lb=0.10,  ub=2.00,  name="scale_spline"),

    # --- Regularisation (LOG SCALE — bắt buộc theo R9) ---
    FloatVar(lb=-5.0,  ub=-1.0,  name="log_reg_lambda"), # reg_lambda = 10^x
]

# Tên tham số để log / debug
PARAM_NAMES = [b.name for b in BOUNDS]
N_DIMS = len(BOUNDS)   # 14

# ── Default config (baseline "không HPO") ─────────────────────────────────────
DEFAULT_CFG = {
    "grid_size":     5,
    "spline_order":  3,
    "grid_eps":      0.02,
    "warmup_epochs": 0,
    "update_every":  0,
    "hidden_size":   64,
    "n_layers":      1,
    "seq_len":       SEQ_LEN,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "batch_size":    64,
    "scale_noise":   0.1,
    "scale_base":    1.0,
    "scale_spline":  1.0,
    "reg_lambda":    1e-3,
}


def decode_solution(solution) -> dict:
    """Chuyển solution vector của MEALPY 3.x → dict hyperparams KAN.

    MEALPY 3.x vẫn truyền numpy ndarray (không phải dict) vào obj_func.
    Giá trị trong array tương ứng theo đúng thứ tự khai báo trong BOUNDS.
    IntegerVar: MEALPY encode float nội bộ, but truyền raw float → ta round.

    Index map (theo thứ tự BOUNDS):
      0: grid_size      (int)
      1: spline_order   (int)
      2: log_grid_eps   (float) → grid_eps = 10^x
      3: warmup_ratio   (float) → warmup_epochs = int(ratio * HPO_EPOCHS)
      4: update_every   (int)
      5: hidden_size    (int)
      6: n_layers       (int)
      7: log_lr         (float) → lr = 10^x
      8: log_wd         (float) → weight_decay = 10^x
      9: batch_exp      (int)   → batch_size = 2^x
     10: scale_noise    (float)
     11: scale_base     (float)
     12: scale_spline   (float)
     13: log_reg_lambda (float) → reg_lambda = 10^x

    Args:
        solution: np.ndarray shape (14,) từ MEALPY obj_func,
                  hoặc dict {name: value} (cho smoke test / mã ngoài).

    Returns:
        dict hyperparams KAN sẵn sàng dùng cho train_one_config().
    """
    import numpy as np
    # Hỗ trợ cả ndarray (MEALPY runtime) và dict (smoke test)
    if isinstance(solution, dict):
        s = np.array([solution[b.name] for b in BOUNDS], dtype=float)
    else:
        s = np.asarray(solution, dtype=float)

    grid_size     = int(round(s[0]))
    spline_order  = int(round(s[1]))
    grid_eps      = float(10 ** s[2])
    warmup_epochs = int(round(float(s[3]) * HPO_EPOCHS))
    update_every  = int(round(s[4]))
    hidden_size   = int(round(s[5]))
    n_layers      = int(round(s[6]))
    lr            = float(10 ** s[7])
    weight_decay  = float(10 ** s[8])
    batch_size    = int(2 ** int(round(s[9])))
    scale_noise   = float(s[10])
    scale_base    = float(s[11])
    scale_spline  = float(s[12])
    reg_lambda    = float(10 ** s[13])

    return {
        "grid_size":     grid_size,
        "spline_order":  spline_order,
        "grid_eps":      grid_eps,
        "warmup_epochs": warmup_epochs,
        "update_every":  update_every,
        "hidden_size":   hidden_size,
        "n_layers":      n_layers,
        "seq_len":       SEQ_LEN,
        "lr":            lr,
        "weight_decay":  weight_decay,
        "batch_size":    batch_size,
        "scale_noise":   scale_noise,
        "scale_base":    scale_base,
        "scale_spline":  scale_spline,
        "reg_lambda":    reg_lambda,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import numpy as np
    print(f"Search space: {N_DIMS} dims")
    for b in BOUNDS:
        lb_val = float(b.lb[0]) if hasattr(b.lb, '__len__') else float(b.lb)
        ub_val = float(b.ub[0]) if hasattr(b.ub, '__len__') else float(b.ub)
        print(f"  {b.name:20s}  [{lb_val:.4g}, {ub_val:.4g}]  {type(b).__name__}")

    # MEALPY 3.x truyền ndarray → test với ndarray mid-point
    mid_arr = np.array([
        (float(b.lb[0] if hasattr(b.lb, '__len__') else b.lb) +
         float(b.ub[0] if hasattr(b.ub, '__len__') else b.ub)) / 2
        for b in BOUNDS
    ])
    cfg = decode_solution(mid_arr)
    print("\nMid-point config:", json.dumps(cfg, indent=2))
