"""Live loss curve viewer for CLIP consistency training — auto-refreshes every 10s.

Reads:
  - losses.txt (one float per line) for per-step total loss
  - stats.csv (~26 cols) for CLIP consistency diagnostics

Layout (3x3):
  [0,0] Loss Trend (EMA)          [0,1] L_diff vs L_clip         [0,2] Cosine Similarity
  [1,0] Core vs Band              [1,1] IP Ratio (A/F)           [1,2] Cross-Attn Entropy (H)
  [2,0] Feature Drift (B)         [2,1] Grad Norm Split (C)      [2,2] Gates

Usage:
  python live_loss_clip.py <losses_file>
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
    """Parse stats.csv -> dict of numpy arrays."""
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
    """Exponential moving average — NaN-aware (carries forward on NaN)."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            ema[i] = ema[i - 1]  # carry forward
        elif np.isnan(ema[i - 1]):
            ema[i] = arr[i]  # first valid value
        else:
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
            "CLIP-Consistency",
            f"\u03b3={cfg.get('clip_gamma', '?')}",
            f"BS={cfg.get('batch_size', '?')}",
            f"lr={cfg.get('lr_pretrained', '?')}",
            f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H",
            f"gates={'forced' if cfg.get('force_gates') else 'learnable' if cfg.get('learnable_gates', True) else 'no'}",
        ]
        return " | ".join(str(p) for p in parts)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "CLIP Consistency Training"


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_loss_clip.py <losses_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    run_dir = str(Path(filepath).parent)
    stats_path = str(Path(run_dir) / "stats.csv")
    run_title = _load_run_title(run_dir)

    fig, axes = plt.subplots(3, 3, figsize=(21, 15))
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

        # === [0,0] Loss Trend ===
        ax = axes[0, 0]
        ax.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax.plot(steps, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SMOOTH_FAST})")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title(f"Loss Trend \u2014 step {len(losses)}, trend: {ema_slow[-1]:.4f}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # === [0,1] L_diff vs L_clip ===
        ax = axes[0, 1]
        if has_data and "L_diffusion" in stats and len(stats["L_diffusion"]) > 1:
            ema_diff = ema_smooth(stats["L_diffusion"], SM)
            ax.plot(s_steps, ema_diff, color="blue", linewidth=2, label=f"L_diff ({ema_diff[-1]:.4f})")
            if "L_clip_scaled" in stats and len(stats["L_clip_scaled"]) > 1:
                ema_clip = ema_smooth(stats["L_clip_scaled"], SM)
                ax.plot(s_steps, ema_clip, color="red", linewidth=2, label=f"L_clip*\u03b3 ({ema_clip[-1]:.4f})")
            if "L_clip" in stats and len(stats["L_clip"]) > 1:
                ema_clip_raw = ema_smooth(stats["L_clip"], SM)
                ax.plot(s_steps, ema_clip_raw, color="red", linewidth=1, linestyle="--", alpha=0.5,
                        label=f"L_clip raw ({ema_clip_raw[-1]:.4f})")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title("L_diff vs L_clip \u2014 want both to decrease")
        else:
            ax.set_title("L_diff vs L_clip (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # === [0,2] Cosine Similarity ===
        ax = axes[0, 2]
        if has_data and "cos_sim_anomaly" in stats and len(stats["cos_sim_anomaly"]) > 1:
            cs_anom = stats["cos_sim_anomaly"]
            cs_all = stats.get("cos_sim_all", np.zeros_like(cs_anom))
            ema_anom = ema_smooth(cs_anom, SM)
            ema_all = ema_smooth(cs_all, SM)
            ax.plot(s_steps, ema_anom, color="green", linewidth=2.5, label=f"anomaly ({ema_anom[-1]:.3f})")
            ax.plot(s_steps, ema_all, color="gray", linewidth=1.5, linestyle="--",
                    label=f"all ({ema_all[-1]:.3f})")
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc="lower right", fontsize=8)
            ax.set_title(f"Cosine Similarity \u2014 anomaly: {ema_anom[-1]:.3f}")
        else:
            ax.set_title("Cosine Similarity (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("cos_sim")
        ax.grid(True, alpha=0.3)

        # === [1,0] Core vs Band (weighted, stacked) ===
        ax = axes[1, 0]
        if has_data and "core_loss" in stats and len(stats["core_loss"]) > 1:
            s_core = stats["core_loss"]
            s_band = stats["band_loss"]
            if s_core.any():
                # Weighted decomposition: 0.8×core + 0.2×band = total diffusion loss
                w_core_s = 0.8 * s_core
                w_band_s = 0.2 * s_band
                w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
                core_share = w_core_s / w_total_s

                # Interpolate to loss steps for stacked view
                L_diff = stats.get("L_diffusion", np.zeros_like(s_core))
                L_diff_ema = ema_smooth(L_diff, SM)
                core_share_ema = ema_smooth(core_share, SM)
                ce = L_diff_ema * core_share_ema / 0.8  # per-pixel core
                be = L_diff_ema * (1.0 - core_share_ema) / 0.2  # per-pixel band

                ax.fill_between(s_steps, 0, ce, color="#2196F3", alpha=0.4)
                ax.fill_between(s_steps, ce, ce + be, color="#FF9800", alpha=0.4)
                ax.plot(s_steps, ce, color="#1565C0", linewidth=1.5, label="Core")
                ax.plot(s_steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
                ax.plot(s_steps, L_diff_ema, color="black", linewidth=2, linestyle="--",
                        label=f"0.8\u00d7core+0.2\u00d7band = {L_diff_ema[-1]:.4f}")
                ax.annotate(f"{ce[-1]:.3f}", xy=(s_steps[-1], ce[-1] / 2),
                            fontsize=9, fontweight="bold", color="#1565C0", ha="right")
                ax.annotate(f"{be[-1]:.3f}", xy=(s_steps[-1], ce[-1] + be[-1] / 2),
                            fontsize=9, fontweight="bold", color="#E65100", ha="right")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title(f"Core vs Band, EMA({SM}) \u2014 weighted: {L_diff_ema[-1]:.4f}")
            else:
                ax.set_title("Core vs Band (no data)")
        else:
            ax.set_title("Core vs Band (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Per-pixel MSE")
        ax.grid(True, alpha=0.3)

        # === [1,1] IP Ratio (only at diag_interval steps) ===
        ax = axes[1, 1]
        if has_data and "ip_attn_norm" in stats and len(stats["ip_attn_norm"]) > 1:
            ip_norm = stats["ip_attn_norm"]
            h_pre = stats.get("h_pre_norm", np.ones_like(ip_norm))
            nonzero = ip_norm > 0
            if nonzero.any():
                ratio_nz = np.where(h_pre[nonzero] > 1e-8, ip_norm[nonzero] / h_pre[nonzero], 0.0)
                ema_ratio = ema_smooth(ratio_nz, SM)
                ax.plot(s_steps[nonzero], ema_ratio, color="purple", linewidth=2,
                        marker=".", markersize=3, label=f"ratio ({ema_ratio[-1]:.4f})")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title(f"IP Ratio (||ip_out|| / ||h_pre||) \u2014 {ema_ratio[-1]:.4f}")
            else:
                ax.set_title("IP Ratio (no nonzero data)")
        else:
            ax.set_title("IP Ratio (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)

        # === [1,2] Cross-Attn Entropy (only at diag_interval steps) ===
        ax = axes[1, 2]
        if has_data and "ip_entropy" in stats and len(stats["ip_entropy"]) > 1:
            ent = stats["ip_entropy"]
            nonzero = ent > 0
            if nonzero.any():
                ema_ent = ema_smooth(ent[nonzero], SM)
                ax.plot(s_steps[nonzero], ema_ent, color="teal", linewidth=2,
                        marker=".", markersize=3, label=f"entropy ({ema_ent[-1]:.2f})")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title(f"IP Attention Entropy \u2014 {ema_ent[-1]:.2f}")
            else:
                ax.set_title("IP Attention Entropy (no nonzero data)")
        else:
            ax.set_title("IP Attention Entropy (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)

        # === [2,0] Feature Drift ===
        ax = axes[2, 0]
        if has_data and "feature_drift" in stats and len(stats["feature_drift"]) > 1:
            fd = stats["feature_drift"]
            nonzero = fd > 0
            if nonzero.any():
                ax.plot(s_steps[nonzero], fd[nonzero], color="darkorange", linewidth=1.5,
                        marker=".", markersize=3, label=f"drift ({fd[nonzero][-1]:.4f})")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_title(f"Feature Drift (B) \u2014 want > 0 (IP has effect)")
            else:
                ax.set_title("Feature Drift (B) (no nonzero data)")
        else:
            ax.set_title("Feature Drift (B) (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("||delta|| / mask_area")
        ax.grid(True, alpha=0.3)

        # === [2,1] Grad Norm Split ===
        ax = axes[2, 1]
        if has_data and "grad_norm" in stats and len(stats["grad_norm"]) > 1:
            gn = stats["grad_norm"]
            gn_ema = ema_smooth(gn, SM)
            ax.plot(s_steps, gn_ema, color="darkblue", linewidth=2, label=f"total ({gn_ema[-1]:.2f})")

            if "g_diff_norm" in stats:
                g_diff = stats["g_diff_norm"]
                g_clip = stats.get("g_clip_norm", np.zeros_like(g_diff))
                nonzero = g_diff > 0
                if nonzero.any():
                    ax.plot(s_steps[nonzero], g_diff[nonzero], color="blue", linewidth=1,
                            marker=".", markersize=2, label=f"g_diff ({g_diff[nonzero][-1]:.2f})")
                    ax.plot(s_steps[nonzero], g_clip[nonzero], color="red", linewidth=1,
                            marker=".", markersize=2, label=f"g_clip ({g_clip[nonzero][-1]:.2f})")

            ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title(f"Grad Norm \u2014 {gn_ema[-1]:.2f}")
        else:
            ax.set_title("Grad Norm (no data yet)")
        ax.set_xlabel("Step"); ax.set_ylabel("Norm")
        ax.grid(True, alpha=0.3)

        # === [2,2] Gates ===
        ax = axes[2, 2]
        if has_data and "attn_gate" in stats and len(stats["attn_gate"]) > 1:
            ax.plot(s_steps, stats["attn_gate"], color="blue", linewidth=1.5, label="Attn gate")
            ax.plot(s_steps, stats["ff_gate"], color="green", linewidth=1.5, label="FF gate")
            ax.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, ff={stats['ff_gate'][-1]:.4f}")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.set_title("Gates (no data)")
        ax.set_xlabel("Step"); ax.set_ylabel("Gate value")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
