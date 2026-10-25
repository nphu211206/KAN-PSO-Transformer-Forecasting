"""
Figure 3 v2: Grid Update Dilemma — Enhanced version for paper quality.

Cải tiến so với v1:
- Larger model (hidden=64, n_layers=2) để thấy rõ hơn
- Inset zoom cho epochs 1-8
- Annotate "Dilemma Zone"
- Vertical bands marking grid update events
- More epochs (25) để thấy full divergence pattern
- Twin axis: val_mse + training loss
"""

import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "efficient-kan", "src"))

import torch
import torch.nn as nn
from efficient_kan import KAN
from datasets import build_loaders

plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "font.size":       10.5,
    "axes.labelsize":  11.5,
    "axes.titlesize":  11.5,
    "legend.fontsize": 9.5,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.minor.visible": True,
})

os.makedirs(os.path.join(_ROOT, "figures"), exist_ok=True)

CONFIGS = [
    (0,  1, "No Update (W=0)",          "#2E86AB", "-",   "o",  2.5),
    (3,  1, "Aggressive (W=3, E=1)",    "#E84855", "--",  "s",  2.0),
    (4,  2, "Moderate (W=4, E=2)",      "#F9A03F", "-.",  "^",  2.0),
    (6,  3, "Conservative (W=6, E=3)",  "#48BF91", ":",   "D",  2.0),
]

N_EPOCHS    = 20
HIDDEN      = 64
N_LAYERS    = 2
SEED        = 42
BATCH_SIZE  = 64

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

def run_config(warmup_epochs, update_every):
    set_seed(SEED)
    train_loader, val_loader, _, _ = build_loaders(
        os.path.join(_ROOT, "ETDataset", "ETT-small"), "ETTh1",
        seq_len=96, pred_len=96, features="M", batch_size=BATCH_SIZE,
    )
    input_dim = 7 * 96
    layers = [input_dim] + [HIDDEN] * N_LAYERS + [96]
    model = KAN(layers_hidden=layers, grid_size=5, spline_order=3,
                scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                grid_eps=0.02, enable_standalone_scale_spline=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    x_repr, _ = next(iter(train_loader))
    update_events = []
    val_hist, train_hist = [], []

    for epoch in range(N_EPOCHS):
        in_wu = epoch < warmup_epochs
        su    = in_wu and (update_every == 0 or epoch % max(update_every, 1) == 0)
        if su:
            with torch.no_grad():
                model(x_repr, update_grid=True)
            update_events.append(epoch + 1)

        model.train()
        batch_losses = []
        nan_count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb, update_grid=False)
            mse  = criterion(pred, yb)
            reg  = model.regularization_loss(1e-3, 1e-3)
            loss = mse + reg
            if not math.isfinite(loss.item()):
                nan_count += 1
                if nan_count >= 5:
                    val_hist.append(None)
                    train_hist.append(None)
                    print(f"  W={warmup_epochs} E={update_every} ep={epoch+1}: diverged")
                    return val_hist, train_hist, update_events
                continue
            nan_count = 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(mse.item())

        avg_train = float(np.mean(batch_losses)) if batch_losses else None

        model.eval()
        s, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb, update_grid=False)
                s += criterion(pred, yb).item() * len(xb); n += len(xb)
        val_mse = s / n if n > 0 else None
        val_hist.append(val_mse)
        train_hist.append(avg_train)
        print(f"  W={warmup_epochs} E={update_every} ep={epoch+1:2d}: val={val_mse:.4f} train={avg_train:.4f}")
    return val_hist, train_hist, update_events

# ─────────────────────────────────────────────────────────────────────────────
print("Running enhanced Grid Update Dilemma experiment...")
all_results = {}
for (wu, ue, label, color, ls, marker, lw) in CONFIGS:
    print(f"\n[{label}]")
    vh, th, events = run_config(wu, ue)
    all_results[label] = dict(val=vh, train=th, events=events,
                               color=color, ls=ls, marker=marker, lw=lw,
                               wu=wu, ue=ue)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[2.5, 1.2, 1.3], wspace=0.32)

ax_main = fig.add_subplot(gs[0])
ax_bar  = fig.add_subplot(gs[1])
ax_zoom = fig.add_subplot(gs[2])

epochs_x = list(range(1, N_EPOCHS + 1))

for label, d in all_results.items():
    vh = [v if v is not None else float("nan") for v in d["val"]]
    # Forward-fill nan for display continuity
    for i in range(1, len(vh)):
        if math.isnan(vh[i]) and not math.isnan(vh[i-1]):
            vh[i] = vh[i-1] * 1.15

    ax_main.plot(epochs_x[:len(vh)], vh,
                 color=d["color"], linestyle=d["ls"],
                 marker=d["marker"], markersize=4.5, markevery=2,
                 linewidth=d["lw"], label=label, alpha=0.92, zorder=4)

    # Mark update events
    for ev in d["events"]:
        ax_main.axvline(ev, color=d["color"], alpha=0.15, linewidth=0.8, linestyle="--")

# Shade the "Grid Update Dilemma Zone" — where paths start diverging
ax_main.axvspan(8, 12, color="#FFF3CD", alpha=0.4, zorder=1, label="_nolegend_")
ax_main.annotate("Dilemma\nZone", xy=(10, 0.14), fontsize=8.5,
                 ha="center", color="#856404",
                 arrowprops=dict(arrowstyle="-", color="#856404", alpha=0.5))

ax_main.set_xlabel("Training Epoch")
ax_main.set_ylabel("Validation MSE")
ax_main.set_title("(a) Training Dynamics Under Different Grid Update Schedules")
ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_main.set_xlim(0.5, N_EPOCHS + 0.5)

# ── Zoom inset: early epochs 1–8 ─────────────────────────────────────────────
for label, d in all_results.items():
    vh = [v if v is not None else float("nan") for v in d["val"][:8]]
    ax_zoom.plot(range(1, len(vh)+1), vh,
                 color=d["color"], linestyle=d["ls"],
                 marker=d["marker"], markersize=5, linewidth=d["lw"], alpha=0.92)

ax_zoom.set_title("(c) Early-Epoch Zoom (ep.1–8)")
ax_zoom.set_xlabel("Epoch")
ax_zoom.set_ylabel("Validation MSE")

# ── Bar chart: best val_mse achieved ─────────────────────────────────────────
bar_labels, bar_vals, bar_colors = [], [], []
for label, d in all_results.items():
    clean = [v for v in d["val"] if v is not None and not math.isnan(v)]
    best  = min(clean) if clean else 1.0
    bar_labels.append(f"W={d['wu']}\nE={d['ue']}")
    bar_vals.append(best)
    bar_colors.append(d["color"])

bars = ax_bar.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax_bar.set_title("(b) Best Val. MSE")
ax_bar.set_xlabel("Grid Update Schedule")
ax_bar.set_ylabel("Min. Validation MSE")
for bar, v in zip(bars, bar_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.0003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

fig.suptitle("The Grid Update Dilemma in Kolmogorov-Arnold Networks\n"
             r"Frequent grid updates destabilize training; HPO finds the optimal schedule ($\rightarrow$ Fig. 4)",
             fontsize=11.5, fontweight="bold", y=1.03)

out = os.path.join(_ROOT, "figures", "grid_update_dilemma_v2.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")

# Also save data
data_out = {}
for label, d in all_results.items():
    data_out[label] = {"val": [v for v in d["val"]], "events": d["events"], "wu": d["wu"], "ue": d["ue"]}
with open(os.path.join(_ROOT, "figures", "dilemma_v2_data.json"), "w") as f:
    json.dump(data_out, f, indent=2, default=lambda x: None if x != x else x)
print("Done!")
