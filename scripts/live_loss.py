"""Live loss curve viewer — auto-refreshes every 10s.

Reads:
  - losses.txt (one float per line) for per-step diffusion loss
  - stats.csv (14 cols) for diagnostics

Three layouts (auto-detected, matches save_loss_plot):
  Default:       [Trend, Core/Band, Band/Core Ratio] / [Gates, Grad Norm, (empty)]
  x0 ablation:   [Trend, Core/Band, x0 vs eps + annealing] / [Gates, Band/Core Ratio, Grad Norm]
  ctx annealing:  [Trend, Core/Band, Ctx dropout schedule] / [Gates, Band/Core Ratio, Grad Norm]

Usage:
  python live_loss.py <losses_file>
"""
import json
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

    Supports 14-col format:
    step,loss,lr_pretrained,lr_scratch,attn_gate,ff_gate,l2sp,core_loss,band_loss,
    x0_ratio,x0_loss,eps_loss,grad_norm,ctx_drop
    """
    steps, diff_losses = [], []
    attn_gates, ff_gates, l2sp_vals = [], [], []
    core_losses, band_losses = [], []
    x0_ratios, x0_losses, eps_losses = [], [], []
    grad_norms, ctx_drops = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            cols = header.split(",")
            n_cols = len(cols)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                steps.append(int(parts[0]))
                diff_losses.append(float(parts[1]))
                attn_gates.append(float(parts[4]))
                ff_gates.append(float(parts[5]))
                l2sp_vals.append(float(parts[6]))
                core_losses.append(float(parts[7]) if len(parts) > 8 else 0.0)
                band_losses.append(float(parts[8]) if len(parts) > 8 else 0.0)
                x0_ratios.append(float(parts[9]) if len(parts) > 9 else 0.0)
                x0_losses.append(float(parts[10]) if len(parts) > 10 else 0.0)
                eps_losses.append(float(parts[11]) if len(parts) > 11 else 0.0)
                grad_norms.append(float(parts[12]) if len(parts) > 12 else 0.0)
                ctx_drops.append(float(parts[13]) if len(parts) > 13 else 0.0)
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
        "x0_ratio": np.array(x0_ratios),
        "x0_loss": np.array(x0_losses),
        "eps_loss": np.array(eps_losses),
        "grad_norm": np.array(grad_norms),
        "ctx_drop": np.array(ctx_drops),
    }


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
            f"VM{cfg.get('visual_mode', '?')}",
            f"LoRA={cfg['lora_rank']}" if cfg.get('lora_rank', 0) > 0 else "no-LoRA",
            f"BS={cfg.get('batch_size', '?')}",
            f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H",
            f"drop={cfg.get('drop_image_prob', 0) + cfg.get('drop_text_prob', 0) + cfg.get('drop_both_prob', 0):.0%}",
            f"gates={'yes' if cfg.get('learnable_gates', True) else 'no'}",
            f"UNet={'QO-' + cfg['unfreeze_qo'] if cfg.get('unfreeze_qo') else 'frozen'}",
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

        s_steps = stats["steps"]
        s_core = stats["core_loss"]
        s_band = stats["band_loss"]
        s_x0 = stats["x0_loss"]
        s_eps = stats["eps_loss"]
        s_x0r = stats["x0_ratio"]
        s_gn = stats["grad_norm"]
        s_ctx = stats["ctx_drop"]

        has_cb = len(s_steps) > 0 and len(s_core) > 0 and s_core.any()
        has_x0 = len(s_steps) > 0 and s_x0.any()
        has_gn = len(s_steps) > 0 and s_gn.any()
        has_ctx = (len(s_steps) > 0 and s_ctx.any()
                   and (s_ctx.max() - s_ctx.min()) > 0.01)

        # Precompute core/band decomposition
        core_ps = band_ps = ce = be = None
        if has_cb:
            w_core_s = 0.8 * s_core
            w_band_s = 0.2 * s_band
            w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
            core_share = w_core_s / w_total_s
            core_share_ps = np.interp(steps, s_steps, core_share)
            core_ps = arr * core_share_ps / 0.8
            band_ps = arr * (1.0 - core_share_ps) / 0.2
            ce = ema_smooth(core_ps, SM)
            be = ema_smooth(band_ps, SM)

        # =====================================================
        # Choose layout based on data presence
        # =====================================================
        ax_x0eps = None
        ax_ctx = None
        if has_x0:
            # x0 ABLATION LAYOUT
            ax_trend, ax_cb, ax_x0eps = axes[0, 0], axes[0, 1], axes[0, 2]
            ax_gates, ax_ratio, ax_gn = axes[1, 0], axes[1, 1], axes[1, 2]
        elif has_ctx:
            # CTX ANNEALING LAYOUT
            ax_trend, ax_cb, ax_ctx = axes[0, 0], axes[0, 1], axes[0, 2]
            ax_gates, ax_ratio, ax_gn = axes[1, 0], axes[1, 1], axes[1, 2]
        else:
            # DEFAULT LAYOUT
            ax_trend, ax_cb, ax_ratio = axes[0, 0], axes[0, 1], axes[0, 2]
            ax_gates, ax_gn = axes[1, 0], axes[1, 1]
            axes[1, 2].set_visible(False)

        # === [0,0] Smooth Trend ===
        ax_trend.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax_trend.plot(steps, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SMOOTH_FAST})")
        ax_trend.set_xlabel("Step")
        ax_trend.set_ylabel("Loss")
        ax_trend.set_title(f"Smooth Trend \u2014 step {len(losses)}, trend: {ema_slow[-1]:.4f}")
        ax_trend.legend(loc="upper right", fontsize=8)
        ax_trend.grid(True, alpha=0.3)

        # === [0,1] Core vs Band ===
        if has_cb and ce is not None:
            ax_cb.fill_between(steps, 0, ce, color="#2196F3", alpha=0.4)
            ax_cb.fill_between(steps, ce, ce + be, color="#FF9800", alpha=0.4)
            ax_cb.plot(steps, ce, color="#1565C0", linewidth=1.5, label="Core")
            ax_cb.plot(steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
            ax_cb.plot(steps, ema_slow, color="black", linewidth=2, linestyle="--",
                       label=f"0.8\u00d7core+0.2\u00d7band = {ema_slow[-1]:.4f}")
            last_x = steps[-1]
            ax_cb.annotate(f"{ce[-1]:.3f}", xy=(last_x, ce[-1] / 2),
                           fontsize=9, fontweight="bold", color="#1565C0", ha="right")
            ax_cb.annotate(f"{be[-1]:.3f}",
                           xy=(last_x, ce[-1] + be[-1] / 2),
                           fontsize=9, fontweight="bold", color="#E65100", ha="right")
            ax_cb.legend(loc="upper right", fontsize=8)
            ax_cb.set_title(f"Core vs Band, EMA({SM}) \u2014 weighted: {ema_slow[-1]:.4f}")
        else:
            ax_cb.set_title("Core vs Band (no data yet)")
        ax_cb.set_xlabel("Step")
        ax_cb.set_ylabel("Per-pixel MSE")
        ax_cb.grid(True, alpha=0.3)

        # === Gates ===
        if len(s_steps) > 0:
            ax_gates.plot(s_steps, stats["attn_gate"], color="blue",
                          linewidth=1.5, label="Attn gate")
            ax_gates.plot(s_steps, stats["ff_gate"], color="green",
                          linewidth=1.5, label="FF gate")
            ax_gates.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, "
                               f"ff={stats['ff_gate'][-1]:.4f}")
            ax_gates.legend(loc="upper left", fontsize=8)
        else:
            ax_gates.set_title("Gates (no data yet)")
        ax_gates.set_xlabel("Step")
        ax_gates.set_ylabel("Gate value")
        ax_gates.grid(True, alpha=0.3)

        # === Band/Core Ratio (used in x0 and ctx layouts at [1,1], default at [0,2]) ===
        if has_x0 or has_ctx:
            # ratio goes in [1,1]
            if has_cb and core_ps is not None:
                safe_core_ps = np.maximum(core_ps, 1e-10)
                ratio_ps = band_ps / safe_core_ps
                ratio_ema = ema_smooth(ratio_ps, SM)
                ax_ratio.plot(steps, ratio_ps, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
                ax_ratio.plot(steps, ratio_ema, color="purple", linewidth=2, label=f"EMA ({SM})")
                ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax_ratio.set_title(f"Band/Core Ratio, EMA({SM}) \u2014 {ratio_ema[-1]:.2f}x")
                ax_ratio.legend(loc="upper left", fontsize=8)
            else:
                ax_ratio.set_title("Band/Core Ratio (no data yet)")
            ax_ratio.set_xlabel("Step")
            ax_ratio.set_ylabel("Band / Core")
            ax_ratio.grid(True, alpha=0.3)
        else:
            # Default layout: ratio is at [0,2]
            if has_cb and core_ps is not None:
                safe_core_ps = np.maximum(core_ps, 1e-10)
                ratio_ps = band_ps / safe_core_ps
                ratio_ema = ema_smooth(ratio_ps, SM)
                ax_ratio.plot(steps, ratio_ps, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
                ax_ratio.plot(steps, ratio_ema, color="purple", linewidth=2, label=f"EMA ({SM})")
                ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax_ratio.set_title(f"Band/Core Ratio, EMA({SM}) \u2014 {ratio_ema[-1]:.2f}x")
                ax_ratio.legend(loc="upper left", fontsize=8)
            else:
                ax_ratio.set_title("Band/Core Ratio (no data yet)")
            ax_ratio.set_xlabel("Step")
            ax_ratio.set_ylabel("Band / Core")
            ax_ratio.grid(True, alpha=0.3)

        # === Grad Norm ===
        if has_gn:
            gn_ema = ema_smooth(s_gn, SM)
            ax_gn.plot(s_steps, s_gn, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
            ax_gn.plot(s_steps, gn_ema, color="#388E3C", linewidth=2.5, label=f"EMA ({SM})")
            ax_gn.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
            ax_gn.set_title(f"Grad Norm \u2014 {gn_ema[-1]:.2f}")
            ax_gn.legend(loc="upper right", fontsize=8)
        else:
            ax_gn.set_title("Grad Norm (no data yet)")
        ax_gn.set_xlabel("Step")
        ax_gn.set_ylabel("Gradient L2 norm")
        ax_gn.grid(True, alpha=0.3)

        # === x0 vs eps + annealing (x0 layout only) ===
        if ax_x0eps is not None:
            x0_vals_f = s_x0[s_x0 > 0]
            eps_vals_f = s_eps[s_eps > 0]
            x0_steps_f = s_steps[s_x0 > 0]
            eps_steps_f = s_steps[s_eps > 0]

            lines = []
            if len(x0_vals_f) > 1:
                x0_slow = ema_smooth(x0_vals_f, SM)
                l1, = ax_x0eps.plot(x0_steps_f, x0_slow, color="#D32F2F", linewidth=2.5, label="x0 loss")
                lines.append(l1)
            if len(eps_vals_f) > 1:
                eps_slow = ema_smooth(eps_vals_f, SM)
                l2, = ax_x0eps.plot(eps_steps_f, eps_slow, color="#1976D2", linewidth=2.5, label="\u03b5 loss")
                lines.append(l2)

            # Annealing schedule on twin axis
            ax_ann = ax_x0eps.twinx()
            l3, = ax_ann.plot(s_steps, s_x0r, color="#FF9800", linewidth=1.5,
                              linestyle="--", alpha=0.7, label="x0 ratio")
            ax_ann.fill_between(s_steps, 0, s_x0r, color="#FFE0B2", alpha=0.3)
            ax_ann.set_ylim(-0.05, 1.1)
            ax_ann.set_ylabel("x0 ratio", color="#FF9800")
            lines.append(l3)

            ax_x0eps.legend(handles=lines, loc="upper right", fontsize=7)
            title_parts = []
            if len(x0_vals_f) > 1:
                title_parts.append(f"x0={ema_smooth(x0_vals_f, SM)[-1]:.4f}")
            if len(eps_vals_f) > 1:
                title_parts.append(f"\u03b5={ema_smooth(eps_vals_f, SM)[-1]:.4f}")
            ax_x0eps.set_title(f"x0 vs \u03b5 Loss, EMA({SM}) \u2014 {', '.join(title_parts)}")
            ax_x0eps.set_xlabel("Step")
            ax_x0eps.set_ylabel("Loss")
            ax_x0eps.grid(True, alpha=0.3)

        # === Ctx dropout schedule (ctx layout only) ===
        if ax_ctx is not None:
            ax_ctx.plot(s_steps, s_ctx, color="#D32F2F", linewidth=2.5, label="Ctx dropout rate")
            ax_ctx.fill_between(s_steps, 0, s_ctx, color="#EF9A9A", alpha=0.3)
            ax_ctx.set_ylim(-0.05, 1.1)
            ax_ctx.set_title(f"Context Dropout Schedule \u2014 current: {s_ctx[-1]:.1%}")
            ax_ctx.set_xlabel("Step")
            ax_ctx.set_ylabel("Dropout rate")
            ax_ctx.legend(loc="upper right", fontsize=8)
            ax_ctx.grid(True, alpha=0.3)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
