"""Live loss curve viewer — auto-refreshes every 10s.

Reads:
  - losses.txt (one float per line) for per-step diffusion loss
  - stats.csv for gate values, L2-SP loss, core/band breakdown (every 10 steps)

Layout (2x3):
  Row 1: Smooth Trend (0.99)  |  Core vs Band (stacked + weighted sum)  |  Band/Core ratio
  Row 2: Gates                |  L2-SP Loss                             |  L2-SP / Diffusion

Usage:
  python live_loss.py <losses_file>
"""
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


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
    """Parse stats.csv → dict of arrays.

    Supports formats:
    - Old (5 cols): step,loss,lr,attn_gate,ff_gate
    - v2  (7 cols): step,loss,lr_pretrained,lr_scratch,attn_gate,ff_gate,l2sp
    - v3  (9 cols): step,loss,lr_pretrained,lr_scratch,attn_gate,ff_gate,l2sp,core_loss,band_loss
    """
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
                    # v3 format
                    steps.append(int(parts[0]))
                    diff_losses.append(float(parts[1]))
                    attn_gates.append(float(parts[4]))
                    ff_gates.append(float(parts[5]))
                    l2sp_vals.append(float(parts[6]))
                    core_losses.append(float(parts[7]))
                    band_losses.append(float(parts[8]))
                elif n_cols >= 7 and len(parts) >= 7:
                    # v2 format
                    steps.append(int(parts[0]))
                    diff_losses.append(float(parts[1]))
                    attn_gates.append(float(parts[4]))
                    ff_gates.append(float(parts[5]))
                    l2sp_vals.append(float(parts[6]))
                    core_losses.append(0.0)
                    band_losses.append(0.0)
                elif len(parts) >= 5:
                    # Old format fallback
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
        "steps": np.array(steps),
        "diff_loss": np.array(diff_losses),
        "attn_gate": np.array(attn_gates),
        "ff_gate": np.array(ff_gates),
        "l2sp": np.array(l2sp_vals),
        "core_loss": np.array(core_losses),
        "band_loss": np.array(band_losses),
    }


def ema_smooth(values, smoothing=0.6):
    """Exponential moving average (TensorBoard convention).

    smoothing=0.6 is TensorBoard's default. Higher = smoother.
    EMA_t = smoothing * EMA_{t-1} + (1 - smoothing) * x_t
    """
    arr = np.array(values, dtype=np.float64)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def ema_with_std(values, smoothing=0.6):
    """EMA + exponentially-weighted standard deviation."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    ema = np.empty(n)
    ema_var = np.empty(n)
    ema[0] = arr[0]
    ema_var[0] = 0.0
    for i in range(1, n):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
        diff = arr[i] - ema[i]
        ema_var[i] = smoothing * ema_var[i - 1] + (1.0 - smoothing) * diff * diff
    ema_std = np.sqrt(ema_var)
    return ema, ema_std


SMOOTH_FAST = 0.6    # responsive — shows recent movement
SMOOTH_SLOW = 0.99   # smooth — per-step data (losses.txt)
SMOOTH_STATS = 0.99   # same as SMOOTH_SLOW — smoother trend for stats data (every 10 steps)


def _load_run_title(run_dir: str) -> str:
    """Build a descriptive title from run_config.json."""
    import json
    config_path = Path(run_dir) / "run_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        parts = [
            f"VM{cfg.get('visual_mode', '?')}",
            f"LoRA={cfg['lora_rank']}" if cfg.get('lora_rank', 0) > 0 else "no-LoRA",
            f"BS={cfg.get('batch_size', '?')}",
            f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H",
            f"drop={cfg.get('drop_image_prob', 0) + cfg.get('drop_text_prob', 0) + cfg.get('drop_both_prob', 0):.0%}",
            f"gates={'yes' if cfg.get('learnable_gates', True) else 'no'}",
            cfg.get('optimizer', 'adamw'),
            f"ts={cfg.get('timestep_sampling', '?')}",
            f"corrupt={cfg['corrupt_context']}" if cfg.get('corrupt_context', 0) > 0 else None,
            f"x0={cfg['x0_start_ratio']}->{cfg['x0_end_ratio']} w={cfg.get('x0_warmup_frac', 0)} ctx={'no' if cfg.get('x0_no_context') else 'yes'}" if cfg.get('x0_objective') else None,
        ]
        parts = [p for p in parts if p is not None]
        return " | ".join(parts)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "Live Training Loss"


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_loss.py <losses_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    run_dir = str(Path(filepath).parent)
    stats_path = str(Path(run_dir) / "stats.csv")
    run_title = _load_run_title(run_dir)

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.canvas.manager.set_window_title(run_title)
    fig.suptitle(run_title, fontsize=13, fontweight="bold")
    ax_trend, ax_cb, ax_ratio = axes[0]
    ax_gates, ax_l2sp, ax_l2sp_div = axes[1]
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
        ema_slow = ema_smooth(arr, SMOOTH_SLOW)

        # === [0,0] Smooth Trend ===
        ax_trend.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SMOOTH_SLOW})")
        ax_trend.plot(steps, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SMOOTH_FAST})")
        ax_trend.set_xlabel("Step")
        ax_trend.set_ylabel("Loss")
        ax_trend.set_title(f"Smooth Trend \u2014 step {len(losses)}, trend: {ema_slow[-1]:.4f}")
        ax_trend.legend(loc="upper right", fontsize=8)
        ax_trend.grid(True, alpha=0.3)

        # === Core vs Band shared setup ===
        has_core_band = (len(stats["steps"]) > 0
                         and len(stats["core_loss"]) > 0
                         and stats["core_loss"].any())

        # === [0,1] Core vs Band — disentangled from per-step total ===
        if has_core_band:
            s = stats["steps"]
            # Compute core's share of weighted total at each stats point
            w_core_s = 0.8 * stats["core_loss"]
            w_band_s = 0.2 * stats["band_loss"]
            w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
            core_share = w_core_s / w_total_s

            # Interpolate share to every step, decompose per-step total
            core_share_ps = np.interp(steps, s, core_share)
            core_ps = arr * core_share_ps / 0.8      # raw core per step
            band_ps = arr * (1.0 - core_share_ps) / 0.2  # raw band per step

            core_ema = ema_smooth(core_ps, SMOOTH_SLOW)
            band_ema = ema_smooth(band_ps, SMOOTH_SLOW)

            ax_cb.fill_between(steps, 0, core_ema, color="#2196F3", alpha=0.4)
            ax_cb.fill_between(steps, core_ema, core_ema + band_ema, color="#FF9800", alpha=0.4)
            ax_cb.plot(steps, core_ema, color="#1565C0", linewidth=1.5, label="Core")
            ax_cb.plot(steps, core_ema + band_ema, color="#E65100", linewidth=1.5, label="Band (stacked)")
            ax_cb.plot(steps, ema_slow, color="black", linewidth=2, linestyle="--",
                       label=f"0.8\u00d7core+0.2\u00d7band = {ema_slow[-1]:.4f}")
            last_x = steps[-1]
            ax_cb.annotate(f"{core_ema[-1]:.3f}", xy=(last_x, core_ema[-1] / 2),
                           fontsize=9, fontweight="bold", color="#1565C0", ha="right")
            ax_cb.annotate(f"{band_ema[-1]:.3f}",
                           xy=(last_x, core_ema[-1] + band_ema[-1] / 2),
                           fontsize=9, fontweight="bold", color="#E65100", ha="right")
            ax_cb.legend(loc="upper right", fontsize=8)
            ax_cb.set_title(f"Core vs Band, EMA({SMOOTH_SLOW}) \u2014 weighted: {ema_slow[-1]:.4f}")
        else:
            ax_cb.set_title("Core vs Band (no data yet)")
        ax_cb.set_xlabel("Step")
        ax_cb.set_ylabel("Per-pixel MSE")
        ax_cb.grid(True, alpha=0.3)

        # === [0,2] Band/Core ratio — disentangled from per-step total ===
        if has_core_band:
            # Reuse core_ps / band_ps from above
            safe_core_ps = np.maximum(core_ps, 1e-10)
            ratio_ps = band_ps / safe_core_ps
            ratio_ema = ema_smooth(ratio_ps, SMOOTH_SLOW)
            ax_ratio.plot(steps, ratio_ps, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
            ax_ratio.plot(steps, ratio_ema, color="purple", linewidth=2,
                          label=f"EMA ({SMOOTH_SLOW})")
            ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_ratio.set_title(f"Band/Core Ratio, EMA({SMOOTH_SLOW}) \u2014 {ratio_ema[-1]:.2f}x")
            ax_ratio.legend(loc="upper left", fontsize=8)
        else:
            ax_ratio.set_title("Band/Core Ratio (no data yet)")
        ax_ratio.set_xlabel("Step")
        ax_ratio.set_ylabel("Band / Core")
        ax_ratio.grid(True, alpha=0.3)

        # === [1,0] Gates ===
        if len(stats["steps"]) > 0:
            ax_gates.plot(stats["steps"], stats["attn_gate"], color="blue",
                          linewidth=1.5, label="Attn gate")
            ax_gates.plot(stats["steps"], stats["ff_gate"], color="green",
                          linewidth=1.5, label="FF gate")
            ax_gates.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, "
                               f"ff={stats['ff_gate'][-1]:.4f}")
            ax_gates.legend(loc="upper left", fontsize=8)
        else:
            ax_gates.set_title("Gates (no data yet)")
        ax_gates.set_xlabel("Step")
        ax_gates.set_ylabel("Gate value")
        ax_gates.grid(True, alpha=0.3)

        # === [1,1] L2-SP Loss ===
        if len(stats["steps"]) > 0 and stats["l2sp"].any():
            ax_l2sp.plot(stats["steps"], stats["l2sp"], color="darkorange", linewidth=1.5)
            ax_l2sp.set_title(f"L2-SP Loss \u2014 current: {stats['l2sp'][-1]:.2e}")
        else:
            ax_l2sp.set_title("L2-SP Loss (no data yet)")
        ax_l2sp.set_xlabel("Step")
        ax_l2sp.set_ylabel("L2-SP")
        ax_l2sp.grid(True, alpha=0.3)

        # === [1,2] L2-SP / Diffusion ratio ===
        if (len(stats["steps"]) > 0 and stats["l2sp"].any()
                and len(stats["diff_loss"]) > 0):
            safe_diff = np.maximum(stats["diff_loss"], 1e-10)
            ratio = stats["l2sp"] / safe_diff
            ratio_ema = ema_smooth(ratio, SMOOTH_STATS)
            ax_l2sp_div.plot(stats["steps"], ratio, color="gray", linewidth=0.8,
                             alpha=0.5, label="Raw ratio")
            ax_l2sp_div.plot(stats["steps"], ratio_ema, color="crimson", linewidth=2,
                             label=f"EMA ({SMOOTH_STATS})")
            ax_l2sp_div.set_title(f"L2-SP / Diffusion \u2014 current: {ratio[-1]:.1%}")
            ax_l2sp_div.legend(loc="upper left", fontsize=8)
            ax_l2sp_div.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        else:
            ax_l2sp_div.set_title("L2-SP / Diffusion (no data yet)")
        ax_l2sp_div.set_xlabel("Step")
        ax_l2sp_div.set_ylabel("Ratio")
        ax_l2sp_div.grid(True, alpha=0.3)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
