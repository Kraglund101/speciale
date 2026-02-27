"""Live loss curve viewer for CFG-primary training — auto-refreshes every 10s.

Reads:
  - losses.txt (one float per line) for per-step diffusion loss
  - stats.csv (23 cols) for CFG diagnostics

Layout (2x3):
  [0,0] Loss trend (EMA)           [0,1] L_cfg vs s (5+2 curves)   [0,2] L_null standalone
  [1,0] Delta norm                 [1,1] Core/Band decomposition   [1,2] Gates or s stats

Usage:
  python live_loss_cfg.py <losses_file>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def parse_losses(filepath: str) -> list:
    """Parse loss values — auto-detects format."""
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


def parse_stats(filepath: str) -> dict:
    """Parse CFG stats.csv → dict of numpy arrays.

    Expected 23 columns:
    step,loss,L_null,L_cond,delta_norm,
    L_cfg_s1.0,L_cfg_s1.5,L_cfg_s2.0,L_cfg_s3.0,L_cfg_s5.0,
    s_mean,s_std,s_min,s_max,
    core_loss,band_loss,grad_norm,
    attn_gate,ff_gate,lr_pretrained,lr_scratch,l2sp,progress
    """
    result = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            cols = header.split(",")
            for col in cols:
                result[col] = []
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != len(cols):
                    continue
                for i, col in enumerate(cols):
                    try:
                        result[col].append(float(parts[i]))
                    except ValueError:
                        result[col].append(0.0)
        for col in result:
            result[col] = np.array(result[col])
    except FileNotFoundError:
        pass
    return result


def ema_smooth(values, smoothing=0.6):
    """Exponential moving average."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


SMOOTH_FAST = 0.6
SMOOTH_SLOW = 0.99
SM = SMOOTH_SLOW


def _load_run_title(run_dir: str) -> str:
    """Build a descriptive title from run_config.json."""
    config_path = Path(run_dir) / "run_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        parts = [
            f"CFG-{cfg.get('cfg_config', '?')}",
            f"BS={cfg.get('batch_size', '?')} (UNet {cfg.get('unet_batch_size', '?')})",
            f"lr={cfg.get('lr_pretrained', '?')}",
        ]
        cc = cfg.get("cfg_config", "")
        if cc == "A":
            parts.append(f"dropout={cfg.get('dropout_prob', 0.5)}")
        elif cc == "B":
            parts.append(f"s={cfg.get('cfg_scale', '?')}")
        elif cc in ("C", "D"):
            parts.append(f"s=[{cfg.get('cfg_s_min', '?')},{cfg.get('cfg_s_max', '?')}]")
            if cc == "D":
                parts.append("warmup")
        elif cc in ("E", "F"):
            parts.append("learned_s(t)")
            if cc == "F":
                parts.append("warmup")
        parts.append(f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H")
        parts.append(f"gates={'forced' if cfg.get('force_gates') else 'learnable' if cfg.get('learnable_gates', True) else 'no'}")
        return " | ".join(str(p) for p in parts)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "CFG-Primary Training"


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_loss_cfg.py <losses_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    run_dir = str(Path(filepath).parent)
    stats_path = str(Path(run_dir) / "stats.csv")
    run_title = _load_run_title(run_dir)

    # Detect cfg_config for [1,2] subplot selection
    cfg_config = "C"  # default
    try:
        with open(Path(run_dir) / "run_config.json") as f:
            cfg_config = json.load(f).get("cfg_config", "C")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.canvas.manager.set_window_title(run_title)
    fig.suptitle(run_title, fontsize=11, fontweight="bold")
    all_axes = list(axes.flat)

    def update(frame):
        losses = parse_losses(filepath)
        stats = parse_stats(stats_path)
        if not losses:
            return

        for ax in all_axes:
            ax.clear()

        steps = np.arange(len(losses))
        arr = np.array(losses)
        ema_fast = ema_smooth(arr, SMOOTH_FAST)
        ema_slow = ema_smooth(arr, SM)

        s_steps = stats.get("step", np.array([]))
        has_data = len(s_steps) > 0

        # === [0,0] Loss trend ===
        ax = axes[0, 0]
        ax.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax.plot(steps, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SMOOTH_FAST})")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title(f"Smooth Trend \u2014 step {len(losses)}, trend: {ema_slow[-1]:.4f}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # === [0,1] L_cfg vs s (THE diagnostic) ===
        ax = axes[0, 1]
        if has_data:
            cfg_colors = {
                "1.0": "#2196F3", "1.5": "#4CAF50", "2.0": "#FF9800",
                "3.0": "#F44336", "5.0": "#9C27B0",
            }
            for s_val in ["1.0", "1.5", "2.0", "3.0", "5.0"]:
                key = f"L_cfg_s{s_val}"
                if key in stats and len(stats[key]) > 1:
                    ema_val = ema_smooth(stats[key], SM)
                    ax.plot(s_steps, ema_val, color=cfg_colors[s_val], linewidth=2,
                            label=f"s={s_val} ({ema_val[-1]:.4f})")
            # Dashed L_null and L_cond
            if "L_null" in stats and len(stats["L_null"]) > 1:
                ema_null = ema_smooth(stats["L_null"], SM)
                ax.plot(s_steps, ema_null, color="black", linewidth=1.5, linestyle="--",
                        label=f"L_null ({ema_null[-1]:.4f})")
            if "L_cond" in stats and len(stats["L_cond"]) > 1:
                ema_cond = ema_smooth(stats["L_cond"], SM)
                ax.plot(s_steps, ema_cond, color="gray", linewidth=1.5, linestyle="--",
                        label=f"L_cond ({ema_cond[-1]:.4f})")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title("L_cfg vs s \u2014 want curves to flatten")
        else:
            ax.set_title("L_cfg vs s (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # === [0,2] Grad Norm ===
        ax = axes[0, 2]
        if has_data and "grad_norm" in stats and len(stats["grad_norm"]) > 1:
            gn = stats["grad_norm"]
            gn_ema_fast = ema_smooth(gn, SMOOTH_FAST)
            gn_ema_slow = ema_smooth(gn, SM)
            ax.plot(s_steps, gn_ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SMOOTH_FAST})")
            ax.plot(s_steps, gn_ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
            ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
            ax.set_title(f"Grad Norm \u2014 {gn_ema_slow[-1]:.2f}")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_title("Grad Norm (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Norm")
        ax.grid(True, alpha=0.3)

        # === [1,0] Delta norm ===
        ax = axes[1, 0]
        if has_data and "delta_norm" in stats and len(stats["delta_norm"]) > 1:
            dn = stats["delta_norm"]
            ema_dn = ema_smooth(dn, SM)
            ax.plot(s_steps, ema_smooth(dn, SMOOTH_FAST), color="red", linewidth=0.8, alpha=0.3,
                    label=f"EMA ({SMOOTH_FAST})")
            ax.plot(s_steps, ema_dn, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
            ax.set_title(f"Delta Norm \u2014 trend: {ema_dn[-1]:.4f}")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_title("Delta Norm (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("||eps_cond - eps_null||")
        ax.grid(True, alpha=0.3)

        # === [1,1] Core/Band decomposition ===
        ax = axes[1, 1]
        if has_data and "core_loss" in stats and len(stats["core_loss"]) > 1:
            s_core = stats["core_loss"]
            s_band = stats["band_loss"]
            if s_core.any():
                w_core_s = 0.8 * s_core
                w_band_s = 0.2 * s_band
                w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
                core_share = w_core_s / w_total_s
                core_share_ps = np.interp(steps, s_steps, core_share)
                core_ps = arr * core_share_ps / 0.8
                band_ps = arr * (1.0 - core_share_ps) / 0.2
                ce = ema_smooth(core_ps, SM)
                be = ema_smooth(band_ps, SM)
                ax.fill_between(steps, 0, ce, color="#2196F3", alpha=0.4)
                ax.fill_between(steps, ce, ce + be, color="#FF9800", alpha=0.4)
                ax.plot(steps, ce, color="#1565C0", linewidth=1.5, label="Core")
                ax.plot(steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
                ax.plot(steps, ema_slow, color="black", linewidth=2, linestyle="--",
                        label=f"0.8\u00d7core+0.2\u00d7band = {ema_slow[-1]:.4f}")
                ax.annotate(f"{ce[-1]:.3f}", xy=(steps[-1], ce[-1] / 2),
                            fontsize=9, fontweight="bold", color="#1565C0", ha="right")
                ax.annotate(f"{be[-1]:.3f}", xy=(steps[-1], ce[-1] + be[-1] / 2),
                            fontsize=9, fontweight="bold", color="#E65100", ha="right")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title(f"Core vs Band, EMA({SM}) \u2014 weighted: {ema_slow[-1]:.4f}")
            else:
                ax.set_title("Core vs Band (no data)")
        else:
            ax.set_title("Core vs Band (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Per-pixel MSE")
        ax.grid(True, alpha=0.3)

        # === [1,2] Gates or s stats (auto-detect) ===
        ax = axes[1, 2]
        if has_data and "s_mean" in stats and len(stats["s_mean"]) > 1:
            s_mean = stats["s_mean"]
            if s_mean.max() > 0:
                # Show s stats
                s_min_arr = stats.get("s_min", np.zeros_like(s_mean))
                s_max_arr = stats.get("s_max", np.zeros_like(s_mean))
                ax.plot(s_steps, s_mean, color="blue", linewidth=2, label=f"s_mean ({s_mean[-1]:.2f})")
                ax.fill_between(s_steps, s_min_arr, s_max_arr, color="blue", alpha=0.15, label="s range")
                ax.legend(loc="upper left", fontsize=8)
                ax.set_title(f"Guidance Scale \u2014 mean={s_mean[-1]:.2f}")
                ax.set_xlabel("Step"); ax.set_ylabel("s")
                ax.grid(True, alpha=0.3)
            else:
                _plot_gates_live(ax, stats, s_steps)
        elif has_data:
            _plot_gates_live(ax, stats, s_steps)
        else:
            ax.set_title("Gates / s stats (no data yet)")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
    plt.show()


def _plot_gates_live(ax, stats, s_steps):
    """Plot gate values on the given axis."""
    if "attn_gate" in stats and len(stats["attn_gate"]) > 1:
        ax.plot(s_steps, stats["attn_gate"], color="blue", linewidth=1.5, label="Attn gate")
        ax.plot(s_steps, stats["ff_gate"], color="green", linewidth=1.5, label="FF gate")
        ax.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, ff={stats['ff_gate'][-1]:.4f}")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_title("Gates (no data)")
    ax.set_xlabel("Step"); ax.set_ylabel("Gate value")
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    main()
