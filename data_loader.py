"""
Pipeline xử lý dữ liệu Time Series cho mô hình KAN.

THIẾT KẾ CHỐNG RÒ RỈ DỮ LIỆU (data leakage):
    Thứ tự bắt buộc là  CHIA TRƯỚC -> FIT SAU.
    Scaler chỉ được học min/max từ tập train. Bản cũ fit_transform trên toàn bộ
    chuỗi rồi mới chia, khiến min/max của tập test rò rỉ vào tập train — đây là
    lỗi bị reviewer bắt ngay ở các tạp chí bình duyệt.

CHỐNG SUY BIẾN MA TRẬN (rank-deficient lstsq trong KANLinear.update_grid):
    Có HAI nguyên nhân độc lập, tài liệu nghiên cứu ban đầu chỉ nêu nguyên nhân 1:

    1. Dữ liệu lệch khỏi grid_range mặc định [-1, 1] của efficient-kan.
       Ví dụ dữ liệu nằm trong [0, 1]: các hàm cơ sở B-spline phụ trách khoảng
       [-1, 0) không bao giờ được kích hoạt -> cột toàn zero -> thiếu hạng.
       Kiểm chứng: rank = 6/8 với dữ liệu [0,1] so với 8/8 với dữ liệu [-1,1].
       Khắc phục: MinMaxScaler(feature_range=(-1, 1)).

    2. Đặc trưng hằng số hoặc gần như hằng số (cảm biến nằm im, cột one-hot,
       biến ít thay đổi). Toàn bộ điểm dữ liệu của kênh đó dồn về một chỗ ->
       rank tụt xuống 1. Chuẩn hóa [-1, 1] KHÔNG cứu được trường hợp này.
       Khắc phục: lọc bỏ đặc trưng có phương sai dưới ngưỡng (drop_low_variance).

    Lưu ý: lỗi chỉ thực sự ném exception trên GPU, vì torch.linalg.lstsq mặc định
    dùng driver 'gels' trên CUDA (đòi hỏi ma trận đủ hạng), trong khi trên CPU mặc
    định là 'gelsy' (xử lý được ma trận thiếu hạng). Chạy tốt trên CPU không có
    nghĩa là sẽ chạy tốt trên GPU.
"""

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


class TimeSeriesDataProcessor:
    def __init__(
        self,
        window_size: int,
        forecast_size: int = 1,
        target_col: int = 0,
        feature_range=(-1.0, 1.0),
        var_threshold: float = 1e-8,
    ):
        """
        Args:
            window_size: Số bước thời gian dùng làm input (cửa sổ trượt).
            forecast_size: Số bước thời gian cần dự báo.
            target_col: Chỉ số cột mục tiêu trong dữ liệu gốc.
            feature_range: Dải chuẩn hóa. Giữ (-1, 1) để khớp grid_range của efficient-kan.
            var_threshold: Ngưỡng phương sai để loại đặc trưng suy biến.
        """
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.target_col = target_col
        self.feature_range = feature_range
        self.var_threshold = var_threshold

        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.keep_cols_ = None      # chỉ số các cột giữ lại sau khi lọc phương sai
        self.target_col_ = None     # chỉ số cột mục tiêu SAU khi lọc
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # Bước 1: chọn đặc trưng + fit scaler — CHỈ trên tập train
    # ------------------------------------------------------------------ #
    def fit(self, train_raw: np.ndarray):
        """Học bộ lọc đặc trưng và tham số scaler từ DUY NHẤT tập train."""
        train_raw = self._as_2d(train_raw)

        variances = train_raw.var(axis=0)
        keep = np.where(variances > self.var_threshold)[0]

        if self.target_col not in keep:
            raise ValueError(
                f"Cột mục tiêu {self.target_col} có phương sai {variances[self.target_col]:.3e} "
                f"<= ngưỡng {self.var_threshold:.3e}. Chuỗi mục tiêu gần như hằng số."
            )

        dropped = sorted(set(range(train_raw.shape[1])) - set(keep.tolist()))
        if dropped:
            print(f"[data_loader] Loại {len(dropped)} đặc trưng phương sai thấp (chống suy biến ma trận): {dropped}")

        self.keep_cols_ = keep
        self.target_col_ = int(np.where(keep == self.target_col)[0][0])
        self.scaler.fit(train_raw[:, keep])
        self.fitted_ = True
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Áp dụng bộ lọc + scaler đã học trên train. Không học lại bất cứ điều gì."""
        if not self.fitted_:
            raise RuntimeError("Phải gọi fit() trên tập train trước khi transform().")
        return self.scaler.transform(self._as_2d(raw)[:, self.keep_cols_])

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Đảo chuẩn hóa riêng cột mục tiêu để báo cáo RMSE/MAE ở đơn vị gốc."""
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        n_feat = len(self.keep_cols_)
        buf = np.zeros((y_scaled.shape[0], n_feat))
        buf[:, self.target_col_] = y_scaled[:, 0]
        return self.scaler.inverse_transform(buf)[:, self.target_col_]

    # ------------------------------------------------------------------ #
    # Bước 2: tạo cửa sổ trượt
    # ------------------------------------------------------------------ #
    def create_sliding_window(self, scaled: np.ndarray):
        """
        Trả về:
            X: (samples, window_size * n_features) — KAN nhận input 2D nên phải flatten.
            y: (samples, forecast_size)
        """
        n = len(scaled) - self.window_size - self.forecast_size + 1
        if n <= 0:
            raise ValueError(
                f"Đoạn dữ liệu dài {len(scaled)} quá ngắn cho "
                f"window_size={self.window_size} + forecast_size={self.forecast_size}."
            )
        X = np.stack([scaled[i:i + self.window_size] for i in range(n)])
        y = np.stack([
            scaled[i + self.window_size: i + self.window_size + self.forecast_size, self.target_col_]
            for i in range(n)
        ])
        return (
            torch.tensor(X.reshape(n, -1), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    # ------------------------------------------------------------------ #
    # Bước 3: quy trình đầy đủ
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        shuffle_train: bool = True,
    ):
        """
        Chia theo THỜI GIAN (không xáo trộn), fit scaler chỉ trên train, rồi tạo cửa sổ.

        Biên các tập được lùi lại window_size bước theo quy ước của TSlib: cửa sổ đầu
        tiên của tập val/test lấy ngữ cảnh đầu vào từ phần đuôi của tập liền trước.
        Đây là dùng QUÁ KHỨ để dự báo TƯƠNG LAI nên không phải rò rỉ dữ liệu, đồng thời
        tránh vứt bỏ mẫu ở mỗi ranh giới.
        """
        data = self._as_2d(data)
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Fit CHỈ trên phần train thô, trước khi đụng tới val/test.
        self.fit(data[:n_train])
        scaled = self.transform(data)

        w = self.window_size
        segments = {
            "train": scaled[:n_train],
            "val": scaled[max(0, n_train - w): n_train + n_val],
            "test": scaled[max(0, n_train + n_val - w):],
        }

        loaders, sizes = {}, {}
        for name, seg in segments.items():
            X, y = self.create_sliding_window(seg)
            self._assert_in_range(X, name)
            loaders[name] = DataLoader(
                TensorDataset(X, y),
                batch_size=batch_size,
                shuffle=(shuffle_train and name == "train"),
            )
            sizes[name] = len(X)

        input_dim = w * len(self.keep_cols_)
        return {
            "loaders": loaders,
            "sizes": sizes,
            "input_dim": input_dim,
            "output_dim": self.forecast_size,
            "n_features": len(self.keep_cols_),
            "processor": self,
        }

    # ------------------------------------------------------------------ #
    # Tiện ích
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_2d(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        return data.reshape(-1, 1) if data.ndim == 1 else data

    def _assert_in_range(self, X: torch.Tensor, name: str):
        """
        Guardrail. Train phải nằm trọn trong feature_range. Val/test CÓ THỂ vượt biên
        (giá trị tương lai lớn hơn mọi giá trị đã thấy trong train) — đó là hành vi đúng,
        không được kẹp lại vì kẹp sẽ làm rò rỉ thông tin về tương lai.
        """
        lo, hi = self.feature_range
        mn, mx = X.min().item(), X.max().item()
        if name == "train":
            eps = 1e-5
            assert lo - eps <= mn and mx <= hi + eps, \
                f"Tập train ngoài dải {self.feature_range}: [{mn:.4f}, {mx:.4f}]"
        elif mn < lo or mx > hi:
            print(f"[data_loader] Lưu ý: tập {name} vượt dải train [{mn:.4f}, {mx:.4f}] "
                  f"— bình thường với chuỗi thời gian phi trạm, không kẹp giá trị.")


# ---------------------------------------------------------------------- #
# KIỂM THỬ
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print(">>> KIỂM THỬ DATA LOADER <<<\n")
    rng = np.random.default_rng(0)

    t = np.arange(2000)
    data = np.column_stack([
        np.sin(t / 24) * 50 + 100 + rng.normal(0, 2, 2000),   # cột 0: mục tiêu, có chu kỳ
        np.cos(t / 168) * 20 + rng.normal(0, 1, 2000),        # cột 1: chu kỳ tuần
        rng.normal(0, 5, 2000),                               # cột 2: nhiễu
        np.full(2000, 7.0),                                   # cột 3: HẰNG SỐ -> phải bị loại
    ])

    proc = TimeSeriesDataProcessor(window_size=24, forecast_size=1, target_col=0)
    out = proc.prepare(data, train_ratio=0.7, val_ratio=0.1, batch_size=32)

    print(f"\nSố mẫu: {out['sizes']}")
    print(f"Đặc trưng giữ lại: {len(proc.keep_cols_)}/4 -> {proc.keep_cols_.tolist()}")
    print(f"Chiều đầu vào KAN: {out['input_dim']} (= 24 bước x 3 đặc trưng)")

    Xb, yb = next(iter(out["loaders"]["train"]))
    print(f"Batch: X={tuple(Xb.shape)}, y={tuple(yb.shape)}")

    # Kiểm chứng không rò rỉ: scaler chỉ thấy tập train
    n_train = int(len(data) * 0.7)
    assert np.allclose(proc.scaler.data_min_, data[:n_train][:, proc.keep_cols_].min(axis=0)), \
        "RÒ RỈ: scaler đã nhìn thấy dữ liệu ngoài tập train"
    print("\nOK  Scaler chỉ fit trên tập train — không rò rỉ dữ liệu.")
    print("OK  Đặc trưng hằng số đã bị loại — chống suy biến ma trận.")

    # Kiểm chứng ma trận đủ hạng khi đưa vào KAN
    import sys
    sys.path.insert(0, "efficient-kan/src")
    from efficient_kan import KAN

    model = KAN([out["input_dim"], 8, 1])
    A = model.layers[0].b_splines(Xb).transpose(0, 1)
    full = model.layers[0].grid_size + model.layers[0].spline_order
    ranks = torch.linalg.matrix_rank(A)
    n_bad = int((ranks < full).sum())
    print(f"OK  Hạng ma trận B-spline: {n_bad}/{len(ranks)} kênh thiếu hạng (đủ hạng = {full}).")

    model(Xb, update_grid=True)
    print("OK  update_grid chạy được, không lỗi Singular Matrix.")
