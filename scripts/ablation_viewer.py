"""Ablation comparison viewer — live or headless image save.

Layout (2x3): rows = optimizer, cols = corruption level.
Each panel has 2 lines: learnable (solid blue) vs forced (dashed orange).

Auto-detects run directories: scans for {optimizer}_{corrupt}_{gate} patterns.
Also supports server ablation layout with LoRA sweep (extra row groups).

Usage:
    # Live viewer (local)
    python scripts/ablation_viewer.py results/ablation_vm3

    # Headless save (server) — single snapshot
    python scripts/ablation_viewer.py results/ablation_vm3 --save comparison.png

    # Headless periodic save (server) — every 60s
    python scripts/ablation_viewer.py results/ablation_vm3 --save comparison.png --every 60
"""
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path


def ema_smooth(values, smoothing=0.99):
    arr = np.array(values, dtype=np.float64)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def parse_losses(filepath: Path) -> list:
    losses = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    losses.append(float(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return losses


SMOOTHING = 0.99
COLORS = {
    "learned": "#1565C0",    # blue
    "forced": "#E65100",     # orange
}


def detect_layout(ablation_dir: Path):
    """Auto-detect grid layout from existing run directories.

    Returns (row_labels, col_labels, gate_modes) where:
      row_labels: list of (label, dir_prefix) for rows
      col_labels: list of (label, dir_key) for columns (corruption levels)
      gate_modes: list of (dir_key, display_label) for line styles
    """
    subdirs = [d.name for d in ablation_dir.iterdir() if d.is_dir()] if ablation_dir.exists() else []

    # Check for LoRA sweep (server ablation)
    has_lora = any("lora64" in d or "lora128" in d or "nolora" in d for d in subdirs)

    gate_modes = [("learned", "learnable"), ("forced", "forced")]

    corruptions = [("c00", "corrupt=0.0"), ("c02", "corrupt=0.2"), ("c05", "corrupt=0.5")]

    if has_lora:
        # Server layout: rows = lora, cols = corruption
        # Auto-detect which optimizers are present
        lora_levels = [("nolora", "no-LoRA"), ("lora64", "LoRA=64"), ("lora128", "LoRA=128")]
        optimizers = []
        if any(d.startswith("prodigy_") for d in subdirs):
            optimizers.append(("prodigy", "Prodigy"))
        if any(d.startswith("adamw_") for d in subdirs):
            optimizers.append(("adamw", "AdamW"))
        if not optimizers:
            optimizers = [("adamw", "AdamW")]
        rows = []
        for opt_key, opt_label in optimizers:
            for lora_key, lora_label in lora_levels:
                rows.append((f"{opt_label} {lora_label}", f"{opt_key}_{lora_key}"))
        return rows, corruptions, gate_modes
    else:
        # Local layout: rows = optimizer, cols = corruption
        rows = [("Prodigy", "prodigy"), ("AdamW", "adamw")]
        return rows, corruptions, gate_modes


def render_plot(ablation_dir: Path, save_path: str = None):
    """Render the comparison plot. If save_path is given, save to file. Otherwise show."""
    if save_path:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols, gate_modes = detect_layout(ablation_dir)
    n_rows = len(rows)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle("VM3 Ablation: EMA Loss (smoothing=0.99)", fontsize=14, fontweight="bold")

    all_ema_vals = []

    for row_idx, (row_label, row_prefix) in enumerate(rows):
        for col_idx, (c_key, c_label) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            ax.set_title(f"{row_label} | {c_label}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)

            for g_key, g_label in gate_modes:
                run_name = f"{row_prefix}_{c_key}_{g_key}"
                run_dir = ablation_dir / run_name
                losses = parse_losses(run_dir / "losses.txt")

                if not losses:
                    continue

                steps = np.arange(len(losses))
                ema = ema_smooth(losses, SMOOTHING)
                all_ema_vals.extend(ema.tolist())

                style = "-" if g_key == "learned" else "--"
                label = f"{g_label} ({len(losses)}, {ema[-1]:.4f})"
                ax.plot(steps, ema, style, color=COLORS[g_key], linewidth=2, label=label)
                ax.plot(steps, losses, color=COLORS[g_key], linewidth=0.3, alpha=0.15)

            ax.legend(loc="upper right", fontsize=7)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Step")
            if col_idx == 0:
                ax.set_ylabel("Loss (EMA)")

    # Shared y-axis
    if all_ema_vals:
        y_lo = max(0, min(all_ema_vals) - 0.01)
        y_hi = max(all_ema_vals) + 0.02
        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()

    if save_path:
        out = ablation_dir / save_path
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out
    else:
        return fig


def main():
    parser = argparse.ArgumentParser(description="Ablation comparison viewer")
    parser.add_argument("ablation_dir", type=str, help="Directory containing ablation runs")
    parser.add_argument("--save", type=str, default=None,
                        help="Save to image file instead of showing (headless mode)")
    parser.add_argument("--every", type=int, default=0,
                        help="Re-save every N seconds (use with --save for periodic monitoring)")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)

    if args.save:
        if args.every > 0:
            # Periodic headless save
            print(f"Saving comparison to {ablation_dir / args.save} every {args.every}s (Ctrl+C to stop)")
            while True:
                out = render_plot(ablation_dir, save_path=args.save)
                print(f"  [{time.strftime('%H:%M:%S')}] Saved {out}")
                time.sleep(args.every)
        else:
            # Single snapshot
            out = render_plot(ablation_dir, save_path=args.save)
            print(f"Saved: {out}")
    else:
        # Live viewer with auto-refresh
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig = render_plot(ablation_dir)
        fig.canvas.manager.set_window_title("VM3 Ablation")

        def update(frame):
            fig.clear()
            # Re-render into same figure
            rows, cols, gate_modes = detect_layout(ablation_dir)
            n_rows = len(rows)
            n_cols = len(cols)
            axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
            fig.suptitle("VM3 Ablation: EMA Loss (smoothing=0.99)", fontsize=14, fontweight="bold")

            all_ema_vals = []
            for row_idx, (row_label, row_prefix) in enumerate(rows):
                for col_idx, (c_key, c_label) in enumerate(cols):
                    ax = axes[row_idx, col_idx]
                    ax.set_title(f"{row_label} | {c_label}", fontsize=10, fontweight="bold")
                    ax.grid(True, alpha=0.3)
                    for g_key, g_label in gate_modes:
                        run_name = f"{row_prefix}_{c_key}_{g_key}"
                        run_dir = ablation_dir / run_name
                        losses = parse_losses(run_dir / "losses.txt")
                        if not losses:
                            continue
                        steps = np.arange(len(losses))
                        ema = ema_smooth(losses, SMOOTHING)
                        all_ema_vals.extend(ema.tolist())
                        style = "-" if g_key == "learned" else "--"
                        label = f"{g_label} ({len(losses)}, {ema[-1]:.4f})"
                        ax.plot(steps, ema, style, color=COLORS[g_key], linewidth=2, label=label)
                        ax.plot(steps, losses, color=COLORS[g_key], linewidth=0.3, alpha=0.15)
                    ax.legend(loc="upper right", fontsize=7)
                    if row_idx == n_rows - 1:
                        ax.set_xlabel("Step")
                    if col_idx == 0:
                        ax.set_ylabel("Loss (EMA)")
            if all_ema_vals:
                y_lo = max(0, min(all_ema_vals) - 0.01)
                y_hi = max(all_ema_vals) + 0.02
                for ax_row in axes:
                    for ax in ax_row:
                        ax.set_ylim(y_lo, y_hi)
            fig.tight_layout()

        ani = FuncAnimation(fig, update, interval=15_000, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    main()
