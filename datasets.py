"""
ETT Dataset Loader — chuẩn split văn liệu (Autoformer / iTransformer).

Phân chia:
  ETTh (hourly):   train 12 tháng / val 4 tháng / test 4 tháng
                   ranh giới: 8640 / 11520 / 14400 (hàng)
  ETTm (15-min):   ×4 hơn ETTh  → 34560 / 46080 / 57600

Quy tắc chống rò rỉ:
  Scaler FIT chỉ trên tập train. Val/test chỉ được TRANSFORM.
  Khi gọi build_loaders(), scaler được chia sẻ qua 3 split.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ── Ranh giới split chuẩn (theo Autoformer official code) ─────────────────────
# ETTh: 1 tick = 1 giờ; ETTm: 1 tick = 15 phút (×4 so với ETTh)
_H = 30 * 24          # ~720 rows / tháng (hourly)
_M = 30 * 24 * 4      # ~2880 rows / tháng (15-min)

_BORDERS = {
    "h": {"train": (0, 12 * _H), "val": (12 * _H, 16 * _H), "test": (16 * _H, 20 * _H)},
    "m": {"train": (0, 12 * _M), "val": (12 * _M, 16 * _M), "test": (16 * _M, 20 * _M)},
}
# ETTh1/ETTh2 = 17 420 hàng  →  ranh giới [0, 8640, 11520, 14400]
# ETTm1/ETTm2 = 69 680 hàng  →  ranh giới [0, 34560, 46080, 57600]

_ALL_COLS   = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
_TARGET_COL = "OT"


class ETTDataset(Dataset):
    """Sliding-window dataset cho chuỗi ETT.

    Args:
        root_path   : thư mục chứa ETTh1.csv … ETTm2.csv
        dataset_name: 'ETTh1' | 'ETTh2' | 'ETTm1' | 'ETTm2'
        split       : 'train' | 'val' | 'test'
        seq_len     : độ dài cửa sổ đầu vào (look-back)
        pred_len    : horizon dự báo (96 / 192 / 336 / 720)
        features    : 'M' = multivariate (7 cột); 'S' = univariate (OT only)
        target      : tên cột mục tiêu (mặc định 'OT')
        scaler      : scaler đã fit sẵn; None → tự fit trên train split
    """

    def __init__(
        self,
        root_path: str,
        dataset_name: str = "ETTh1",
        split: str = "train",
        seq_len: int = 96,
        pred_len: int = 96,
        features: str = "M",
        target: str = _TARGET_COL,
        scaler: Optional[MinMaxScaler] = None,
    ):
        freq = "m" if "m" in dataset_name.lower()[-2:] else "h"
        b_start, b_end = _BORDERS[freq][split]

        # ── Load CSV + drop timestamp column ───────────────────────────────────
        csv_path = os.path.join(root_path, f"{dataset_name}.csv")
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["date"])  # giữ lại 7 cột số

        # ── Select features ────────────────────────────────────────────────────
        if features == "S":
            data = df[[target]].values  # (N, 1)
            self.n_features = 1
        else:  # "M"
            data = df[_ALL_COLS].values  # (N, 7)
            self.n_features = len(_ALL_COLS)

        # ── No-leak scaling: fit scaler CHỈ trên tập train ────────────────────
        if scaler is None:
            train_start, train_end = _BORDERS[freq]["train"]
            scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
            scaler.fit(data[train_start:train_end])
        self.scaler = scaler

        data_scaled = scaler.transform(data)

        # ── Extract split segment ──────────────────────────────────────────────
        segment = data_scaled[b_start:b_end]

        self.data = segment
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = 0 if features == "S" else _ALL_COLS.index(target)

    def __len__(self):
        """Số cửa sổ trượt hợp lệ."""
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (seq_len × n_features,)  — input flattened
            y: (pred_len,)               — target horizon univariate
        """
        x = self.data[idx : idx + self.seq_len]  # (seq_len, n_feat)
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len,
            self.target_idx,
        ]  # (pred_len,)
        return (
            torch.tensor(x, dtype=torch.float32).flatten(),  # (seq_len × n_feat,)
            torch.tensor(y, dtype=torch.float32),            # (pred_len,)
        )


def build_loaders(
    root_path: str,
    dataset_name: str = "ETTh1",
    seq_len: int = 96,
    pred_len: int = 96,
    features: str = "M",
    target: str = _TARGET_COL,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]:
    """Tạo train/val/test DataLoader với scaler dùng chung (không rò rỉ).

    Returns:
        (train_loader, val_loader, test_loader, scaler)
        scaler cần lưu lại để inverse-transform khi tính MAE/MSE thực tế.
    """
    train_ds = ETTDataset(root_path, dataset_name, "train", seq_len, pred_len, features, target)
    scaler = train_ds.scaler  # đã fit trên train

    val_ds  = ETTDataset(root_path, dataset_name, "val",  seq_len, pred_len, features, target, scaler=scaler)
    test_ds = ETTDataset(root_path, dataset_name, "test", seq_len, pred_len, features, target, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler


def get_input_dim(seq_len: int, features: str = "M") -> int:
    """Trả về chiều đầu vào phẳng của KAN: seq_len × n_features."""
    return seq_len * (7 if features == "M" else 1)


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = "ETDataset/ETT-small"
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        tr, va, te, sc = build_loaders(ROOT, name, seq_len=96, pred_len=96)
        xb, yb = next(iter(tr))
        print(f"{name}: train={len(tr.dataset):6d}  val={len(va.dataset):6d}  "
              f"test={len(te.dataset):6d}  x={tuple(xb.shape)}  y={tuple(yb.shape)}")
