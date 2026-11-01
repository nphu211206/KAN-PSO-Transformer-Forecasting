"""Reproduce the analytically counted parameter totals in Table 7."""
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from pso_tcwavkan_v91 import (
        Decomp_TCWavKAN_v91,
        DLinear,
        NLinear,
        PatchTST_Lite,
    )


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    for lookback, horizon in ((96, 96), (336, 720)):
        print(f"L={lookback}, H={horizon}")
        models = (
            ("DLinear", DLinear(lookback, horizon)),
            ("NLinear", NLinear(lookback, horizon)),
            ("PatchTST Lite", PatchTST_Lite(lookback, horizon)),
            ("PSO-TCWavKAN", Decomp_TCWavKAN_v91(lookback, horizon)),
        )
        for name, model in models:
            print(f"  {name:16s} {count_trainable(model):>10,d}")


if __name__ == "__main__":
    main()
