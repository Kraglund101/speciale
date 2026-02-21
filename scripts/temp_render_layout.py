"""Render a static snapshot of the 2x3 live loss layout."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def parse_losses(filepath):
    losses = []
    tqdm_pattern = re.compile(r"loss=([0-9]+\.[0-9]+)")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                losses.append(float(line))
                continue
            except ValueError:
                pass
            m = tqdm_pattern.search(line)
            if m:
                losses.append(float(m.group(1)))
    return losses


def parse_stats(filepath):
    steps, diff_losses = [], []
    attn_gates, ff_gates = [], []
    l2sp_vals = []
    core_losses, band_losses = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            cols = header.split(",")
            n_cols = len(cols)
            for line in f:
                parts = line.strip().split(",")
                if n_cols >= 9 and len(parts) >= 9:
                    steps.append(int(parts[0]))
                    diff_losses.append(float(parts[1]))
                    attn_gates.append(float(parts[4]))
                    ff_gates.append(float(parts[5]))
                    l2sp_vals.append(float(parts[6]))
                    core_losses.append(float(parts[7]))
                    band_losses.append(float(parts[8]))
                elif n_cols >= 7 and len(parts) >= 7:
                    steps.append(int(parts[0]))
                    diff_losses.append(float(parts[1]))
                    attn_gates.append(float(parts[4]))
                    ff_gates.append(float(parts[5]))
                    l2sp_vals.append(float(parts[6]))
                    core_losses.append(0.0)
                    band_losses.append(0.0)
                elif len(parts) >= 5:
                    steps.append(int(parts[0]))
                    diff_losses.append(float(parts[1]))
                    attn_gates.append(float(parts[3]))
                    ff_gates.append(float(parts[4]))
                    l2sp_vals.append(0.0)
                    core_losses.append(0.0)
                    band_losses.append(0.0)
    except FileNotFoundError:
        pass
    return {
        "steps": np.array(steps), "diff_loss": np.array(diff_losses),
        "attn_gate": np.array(attn_gates), "ff_gate": np.array(ff_gates),
        "l2sp": np.array(l2sp_vals),
        "core_loss": np.array(core_losses), "band_loss": np.array(band_losses),
    }


def ema_smooth(values, smoothing=0.6):
    arr = np.array(values, dtype=np.float64)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


SMOOTH_FAST = 0.6
SMOOTH_SLOW = 0.99

data_dir = Path(r"C:\Users\frede\desktop\kandidat\speciale\results\stress_test_20k_no_multicrop")
losses = parse_losses(str(data_dir / "losses.txt"))
stats = parse_stats(str(data_dir / "stats.csv"))

fig, axes = plt.subplots(2, 3, figsize=(21, 10))
ax_trend, ax_cb, ax_ratio = axes[0]
ax_gates, ax_l2sp, ax_l2sp_div = axes[1]

st = np.arange(len(losses))
arr = np.array(losses)
ema_fast = ema_smooth(arr, SMOOTH_FAST)
ema_slow = ema_smooth(arr, SMOOTH_SLOW)

# [0,0] Smooth Trend
ax_trend.plot(st, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SMOOTH_SLOW})")
ax_trend.plot(st, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SMOOTH_FAST})")
ax_trend.set_xlabel("Step")
ax_trend.set_ylabel("Loss")
ax_trend.set_title(f"Smooth Trend \u2014 step {len(losses)}, trend: {ema_slow[-1]:.4f}")
ax_trend.legend(loc="upper right", fontsize=8)
ax_trend.grid(True, alpha=0.3)

has_cb = (len(stats["steps"]) > 0 and len(stats["core_loss"]) > 0 and stats["core_loss"].any())

# [0,1] Core vs Band stacked
if has_cb:
    s = stats["steps"]
    ce = ema_smooth(stats["core_loss"], SMOOTH_SLOW)
    be = ema_smooth(stats["band_loss"], SMOOTH_SLOW)
    wt = 0.8 * stats["core_loss"] + 0.2 * stats["band_loss"]
    we = ema_smooth(wt, SMOOTH_SLOW)
    ax_cb.fill_between(s, 0, ce, color="#2196F3", alpha=0.4)
    ax_cb.fill_between(s, ce, ce + be, color="#FF9800", alpha=0.4)
    ax_cb.plot(s, ce, color="#1565C0", linewidth=1.5, label="Core")
    ax_cb.plot(s, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
    ax_cb.plot(s, we, color="black", linewidth=2, linestyle="--",
               label=f"0.8\u00d7core+0.2\u00d7band = {we[-1]:.4f}")
    ax_cb.annotate(f"{ce[-1]:.3f}", xy=(s[-1], ce[-1] / 2),
                   fontsize=9, fontweight="bold", color="#1565C0", ha="right")
    ax_cb.annotate(f"{be[-1]:.3f}", xy=(s[-1], ce[-1] + be[-1] / 2),
                   fontsize=9, fontweight="bold", color="#E65100", ha="right")
    ax_cb.legend(loc="upper right", fontsize=8)
    ax_cb.set_title(f"Core vs Band (stacked) \u2014 weighted sum: {we[-1]:.4f}")
else:
    ax_cb.set_title("Core vs Band (no data yet)")
ax_cb.set_xlabel("Step")
ax_cb.set_ylabel("Per-pixel MSE")
ax_cb.grid(True, alpha=0.3)

# [0,2] Band/Core ratio
if has_cb:
    safe_c = np.maximum(stats["core_loss"], 1e-10)
    cbr = stats["band_loss"] / safe_c
    cbre = ema_smooth(cbr, SMOOTH_SLOW)
    ax_ratio.plot(stats["steps"], cbr, color="gray", linewidth=0.8, alpha=0.5, label="Raw")
    ax_ratio.plot(stats["steps"], cbre, color="purple", linewidth=2, label=f"EMA ({SMOOTH_SLOW})")
    ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_ratio.set_title(f"Band/Core Ratio \u2014 {cbre[-1]:.2f}x")
    ax_ratio.legend(loc="upper left", fontsize=8)
else:
    ax_ratio.set_title("Band/Core Ratio (no data yet)")
ax_ratio.set_xlabel("Step")
ax_ratio.set_ylabel("Band / Core")
ax_ratio.grid(True, alpha=0.3)

# [1,0] Gates
if len(stats["steps"]) > 0:
    ax_gates.plot(stats["steps"], stats["attn_gate"], color="blue", linewidth=1.5, label="Attn gate")
    ax_gates.plot(stats["steps"], stats["ff_gate"], color="green", linewidth=1.5, label="FF gate")
    ax_gates.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, ff={stats['ff_gate'][-1]:.4f}")
    ax_gates.legend(loc="upper left", fontsize=8)
else:
    ax_gates.set_title("Gates (no data yet)")
ax_gates.set_xlabel("Step")
ax_gates.set_ylabel("Gate value")
ax_gates.grid(True, alpha=0.3)

# [1,1] L2-SP Loss
if len(stats["steps"]) > 0 and stats["l2sp"].any():
    ax_l2sp.plot(stats["steps"], stats["l2sp"], color="darkorange", linewidth=1.5)
    ax_l2sp.set_title(f"L2-SP Loss \u2014 current: {stats['l2sp'][-1]:.2e}")
else:
    ax_l2sp.set_title("L2-SP Loss (no data yet)")
ax_l2sp.set_xlabel("Step")
ax_l2sp.set_ylabel("L2-SP")
ax_l2sp.grid(True, alpha=0.3)

# [1,2] L2-SP / Diffusion
if len(stats["steps"]) > 0 and stats["l2sp"].any() and len(stats["diff_loss"]) > 0:
    safe_d = np.maximum(stats["diff_loss"], 1e-10)
    rat = stats["l2sp"] / safe_d
    rate = ema_smooth(rat, SMOOTH_SLOW)
    ax_l2sp_div.plot(stats["steps"], rat, color="gray", linewidth=0.8, alpha=0.5, label="Raw ratio")
    ax_l2sp_div.plot(stats["steps"], rate, color="crimson", linewidth=2, label=f"EMA ({SMOOTH_SLOW})")
    ax_l2sp_div.set_title(f"L2-SP / Diffusion \u2014 current: {rat[-1]:.1%}")
    ax_l2sp_div.legend(loc="upper left", fontsize=8)
    ax_l2sp_div.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
else:
    ax_l2sp_div.set_title("L2-SP / Diffusion (no data yet)")
ax_l2sp_div.set_xlabel("Step")
ax_l2sp_div.set_ylabel("Ratio")
ax_l2sp_div.grid(True, alpha=0.3)

fig.tight_layout()
out = Path(r"C:\Users\frede\desktop\kandidat\speciale\results\live_loss_preview.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
