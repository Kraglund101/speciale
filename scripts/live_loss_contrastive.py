"""Live loss viewer for contrastive training — auto-refreshes every 10s.

Reads stats.csv from contrastive training runs.

4x3 layout:
  [L_diff trend,        Core vs Band (stacked), Role Embedding Norms      ]
  [Contrastive Dists,   Gap Satisfaction,       Normalized Diagnostics    ]
  [L_inv / L_triplet,   L_rank typed/untyped,   Rank Satisfaction %       ]
  [Gates,               Grad Norm,              L_true vs L_wrong         ]

Usage:
  python live_loss_contrastive.py <stats_csv_path>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SM = 0.99
SF = 0.6


def ema_smooth(values, smoothing=0.99, seed=None):
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    if seed is not None:
        ema[0] = seed
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                ema[i] = ema[i - 1] if i > 0 else seed
            else:
                prev = ema[i - 1] if i > 0 else seed
                ema[i] = smoothing * prev + (1.0 - smoothing) * arr[i]
    else:
        first = 0
        while first < len(arr) and np.isnan(arr[first]):
            first += 1
        if first >= len(arr):
            return np.full_like(arr, np.nan)
        for i in range(first):
            ema[i] = arr[first]
        ema[first] = arr[first]
        for i in range(first + 1, len(arr)):
            if np.isnan(arr[i]):
                ema[i] = ema[i - 1]
            else:
                ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def parse_stats(filepath: str) -> dict:
    """Parse contrastive stats.csv -> dict of arrays."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})
    except (FileNotFoundError, ValueError):
        pass
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        result[key] = np.array([r.get(key, 0.0) for r in rows])
    return result


def _get(stats, key):
    return stats.get(key, np.array([]))


def _load_run_config(run_dir: str) -> dict:
    config_path = Path(run_dir) / "run_config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _build_title(cfg: dict) -> str:
    if not cfg:
        return "Contrastive Training \u2014 Live"
    parts = [
        f"Strategy {cfg.get('strategy', '?')}",
        f"HBS={cfg.get('host_batch_size', '?')}",
        f"VM{cfg.get('visual_mode', '?')}",
        f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H",
        f"gates={'forced' if cfg.get('force_gates') else ('learn' if cfg.get('learnable_gates', True) else 'fixed')}",
    ]
    if cfg.get("strategy") == "A":
        parts.append(f"\u03bb_inv={cfg.get('lambda_inv', '?')}")
        parts.append(f"\u03bb_rank={cfg.get('lambda_rank', '?')}")
        parts.append(f"\u03bb_trip={cfg.get('lambda_triplet', '?')}")
        parts.append(f"\u03b3_scale={cfg.get('rank_gamma_scale', '?')}")
    else:
        parts.append(f"\u03bb_trip={cfg.get('lambda_triplet', '?')}")
        parts.append(f"margin={cfg.get('triplet_margin_m', '?')}")
    parts.append(f"warmup={cfg.get('regularizer_warmup_frac', '?')}")
    parts.append(f"trip={cfg.get('triplet_mode', 'hinge')}")
    parts.append(f"neg={cfg.get('typed_neg_mode', '?')}")
    return " | ".join(parts)


def _add_warmup_vline(ax, warmup_step, steps):
    """Add a vertical warmup boundary line if within visible range."""
    if warmup_step > 0 and len(steps) > 0 and warmup_step <= steps[-1] * 1.5:
        ax.axvline(x=warmup_step, color="gray", linestyle=":", linewidth=1.5,
                   alpha=0.6, label=f"warmup end ({int(warmup_step)})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_loss_contrastive.py <stats_csv_path>")
        sys.exit(1)

    stats_path = sys.argv[1]
    run_dir = str(Path(stats_path).parent)
    cfg = _load_run_config(run_dir)
    run_title = _build_title(cfg)

    n_steps = cfg.get("n_steps", 0)
    warmup_frac = cfg.get("regularizer_warmup_frac", 0.05)
    warmup_step = int(n_steps * warmup_frac) if n_steps > 0 else 0

    fig, axes = plt.subplots(4, 3, figsize=(21, 20))
    fig.canvas.manager.set_window_title(run_title)
    fig.suptitle(run_title, fontsize=11, fontweight="bold")
    all_axes = list(axes.flat)

    lam_inv = cfg.get("lambda_inv", 1.0)
    lam_rank = cfg.get("lambda_rank", 5.0)
    lam_trip = cfg.get("lambda_triplet", 0.25)

    def update(frame):
        stats = parse_stats(stats_path)
        if not stats:
            return

        for ax in all_axes:
            ax.clear()

        steps = _get(stats, "step")
        if len(steps) < 2:
            return

        # ============================================================
        # Row 0: L_diff trend | Core vs Band | Role Embedding Norms
        # ============================================================

        # === [0,0] L_diff trend + weighted contribution overlay ===
        L_diff = _get(stats, "L_diff")
        L_inv_raw = _get(stats, "L_inv")
        L_rank_t = _get(stats, "L_rank_typed")
        ax = axes[0, 0]
        if len(L_diff) > 1:
            ax.plot(steps, ema_smooth(L_diff, SF), color="red", linewidth=0.8, alpha=0.3, label=f"EMA({SF})")
            ema_slow = ema_smooth(L_diff, SM)
            ax.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"L_diff EMA({SM})")
            if len(L_inv_raw) > 1 and L_inv_raw.any():
                inv_w = ema_smooth(lam_inv * L_inv_raw, SM)
                ax.plot(steps, inv_w, color="purple", linewidth=1.5, label=f"{lam_inv}\u00d7L_inv ({inv_w[-1]:.4f})")
            if len(L_rank_t) > 1 and L_rank_t.any():
                rk_w = ema_smooth(lam_rank * L_rank_t, SM)
                ax.plot(steps, rk_w, color="green", linewidth=1.5, label=f"{lam_rank}\u00d7L_rank ({rk_w[-1]:.4f})")
            L_trip_raw = _get(stats, "L_triplet")
            if len(L_trip_raw) > 1 and L_trip_raw.any():
                trip_w = ema_smooth(lam_trip * L_trip_raw, SM)
                ax.plot(steps, trip_w, color="orange", linewidth=1.5, label=f"{lam_trip}\u00d7L_trip ({trip_w[-1]:.4f})")
            ax.set_title(f"L_diff \u2014 step {int(steps[-1])}, trend: {ema_slow[-1]:.4f}")
        else:
            ax.set_title("L_diff (waiting...)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

        # === [0,1] Core vs Band (stacked with weighted total) ===
        core_raw = _get(stats, "L_diff_core")
        band_raw = _get(stats, "L_diff_band")
        ax = axes[0, 1]
        if len(core_raw) > 1 and core_raw.any():
            ce = ema_smooth(core_raw, SM)
            be = ema_smooth(band_raw, SM)
            ax.fill_between(steps, 0, ce, color="#2196F3", alpha=0.4)
            ax.fill_between(steps, ce, ce + be, color="#FF9800", alpha=0.4)
            ax.plot(steps, ce, color="#1565C0", linewidth=1.5, label="Core")
            ax.plot(steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
            weighted = 0.8 * ce + 0.2 * be
            ax.plot(steps, weighted, color="black", linewidth=2, linestyle="--",
                    label=f"0.8\u00d7core+0.2\u00d7band = {weighted[-1]:.4f}")
            last_x = steps[-1]
            ax.annotate(f"{ce[-1]:.3f}", xy=(last_x, ce[-1] / 2),
                        fontsize=9, fontweight="bold", color="#1565C0", ha="right")
            ax.annotate(f"{be[-1]:.3f}",
                        xy=(last_x, ce[-1] + be[-1] / 2),
                        fontsize=9, fontweight="bold", color="#E65100", ha="right")
            ax.set_title(f"Core vs Band, EMA({SM}) \u2014 weighted: {weighted[-1]:.4f}")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.set_title("Core vs Band (waiting...)")
        ax.set_xlabel("Step"); ax.set_ylabel("Per-pixel MSE")
        ax.grid(True, alpha=0.3)

        # === [0,2] Role Embedding Norms ===
        emb_g = _get(stats, "emb_global_norm")
        emb_a = _get(stats, "emb_anomaly_norm")
        emb_n = _get(stats, "emb_normal_norm")
        ax = axes[0, 2]
        has_emb = (len(emb_g) > 1 and emb_g.any()) or (len(emb_a) > 1 and emb_a.any()) or (len(emb_n) > 1 and emb_n.any())
        if has_emb:
            if len(emb_g) > 1 and emb_g.any():
                ax.plot(steps, emb_g, color="green", linewidth=1.5, label=f"Global ({emb_g[-1]:.4f})")
            if len(emb_a) > 1 and emb_a.any():
                ax.plot(steps, emb_a, color="red", linewidth=1.5, label=f"Anomaly ({emb_a[-1]:.4f})")
            if len(emb_n) > 1 and emb_n.any():
                ax.plot(steps, emb_n, color="blue", linewidth=1.5, label=f"Band ({emb_n[-1]:.4f})")
            ax.set_title("Role Embedding Norms")
        else:
            ax.set_title("Role Embedding Norms (waiting...)")
        ax.set_xlabel("Step"); ax.set_ylabel("L2 Norm")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # ============================================================
        # Row 1: Contrastive Distances | Gap Satisfaction | Normalized
        # ============================================================

        # === [1,0] Contrastive Distances (semantic + null separate) ===
        ds = _get(stats, "d_same_mean")
        dd = _get(stats, "d_diff_mean")       # semantic-only
        dd_null = _get(stats, "d_diff_null_mean")
        ax = axes[1, 0]
        if len(ds) > 1 and ds.any():
            ds_ema = ema_smooth(ds, SM)
            ax.plot(steps, ds_ema, color="blue", linewidth=2, label=f"d_same ({ds_ema[-1]:.4f})")
        if len(dd) > 1 and dd.any():
            dd_ema = ema_smooth(dd, SM)
            ax.plot(steps, dd_ema, color="red", linewidth=2, label=f"d_diff_sem ({dd_ema[-1]:.4f})")
        if len(dd_null) > 1 and dd_null.any():
            dd_null_ema = ema_smooth(dd_null, SM)
            ax.plot(steps, dd_null_ema, color="orange", linewidth=1.5, linestyle=":", label=f"d_diff_null ({dd_null_ema[-1]:.4f})")
        gap_raw = _get(stats, "gap_mean")
        if len(gap_raw) > 1 and gap_raw.any():
            gap_ema = ema_smooth(gap_raw, SM)
            ax.plot(steps, gap_ema, color="green", linewidth=1.5, linestyle="--", label=f"gap ({gap_ema[-1]:.4f})")
        _add_warmup_vline(ax, warmup_step, steps)
        ax.set_title("Contrastive Distances")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Normalized distance")
        ax.grid(True, alpha=0.3)

        # === [1,1] Gap Satisfaction Rate (same_host only) ===
        gpr_sh = _get(stats, "gap_pos_rate_same_host")
        ax = axes[1, 1]
        if len(gpr_sh) > 1 and gpr_sh.any():
            gpr_sh_ema = ema_smooth(gpr_sh, SM)
            ax.plot(steps, gpr_sh_ema, color="green", linewidth=2, label=f"gap > 0 ({gpr_sh_ema[-1]:.0%})")
        ax.set_ylim(-0.05, 1.05)
        _add_warmup_vline(ax, warmup_step, steps)
        ax.set_title("Gap Satisfaction (same_host)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)

        # === [1,2] Normalized Contrastive Diagnostics ===
        ax = axes[1, 2]
        if len(ds) > 1 and ds.any() and len(dd) > 1 and dd.any():
            ds_e = ema_smooth(ds, SM)
            dd_e = ema_smooth(dd, SM)
            gap_rel = (dd_e - ds_e) / (dd_e + ds_e + 1e-6)
            ratio = dd_e / (ds_e + 1e-6)
            ax.plot(steps, gap_rel, color="purple", linewidth=2, label=f"gap_rel ({gap_rel[-1]:.4f})")
            ax.plot(steps, ratio, color="teal", linewidth=2, label=f"d_diff/d_same ({ratio[-1]:.4f})")
        _add_warmup_vline(ax, warmup_step, steps)
        ax.set_title("Normalized Contrastive Diagnostics")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # ============================================================
        # Row 2: L_inv / L_triplet | L_rank | Rank Satisfaction
        # ============================================================

        # === [2,0] L_inv / L_triplet ===
        L_inv = _get(stats, "L_inv")
        L_trip = _get(stats, "L_triplet")
        ax = axes[2, 0]
        if len(L_inv) > 1 and L_inv.any():
            li = ema_smooth(L_inv, SM)
            ax.plot(steps, li, color="purple", linewidth=2, label=f"L_inv ({li[-1]:.4f})")
        if len(L_trip) > 1 and L_trip.any():
            lt = ema_smooth(L_trip, SM)
            ax.plot(steps, lt, color="orange", linewidth=2, label=f"L_triplet ({lt[-1]:.4f})")
        viol = _get(stats, "triplet_violation_rate")
        if len(viol) > 1 and viol.any():
            ax2 = ax.twinx()
            viol_ema = ema_smooth(viol, SM)
            ax2.plot(steps, viol_ema, color="gray", linewidth=1, linestyle=":", label=f"viol% ({viol_ema[-1]:.0%})")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("Violation rate", fontsize=8, color="gray")
            ax2.legend(loc="upper right", fontsize=7)
        _add_warmup_vline(ax, warmup_step, steps)
        ax.set_title("Regularizer Losses")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # === [2,1] L_rank typed + untyped ===
        L_rt = _get(stats, "L_rank_typed")
        L_ru = _get(stats, "L_rank_untyped")
        ax = axes[2, 1]
        if len(L_rt) > 1 and L_rt.any():
            rt = ema_smooth(L_rt, SM)
            ax.plot(steps, rt, color="green", linewidth=2, label=f"Typed ({rt[-1]:.4f})")
        if len(L_ru) > 1 and L_ru.any():
            ru = ema_smooth(L_ru, SM)
            ax.plot(steps, ru, color="brown", linewidth=2, label=f"Untyped ({ru[-1]:.4f})")
        _add_warmup_vline(ax, warmup_step, steps)
        ax.set_title("L_rank")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # === [2,2] Rank satisfaction % — split by neg source ===
        def _mask_by_count(sat_key, n_key):
            sat = _get(stats, sat_key).copy()
            n = _get(stats, n_key)
            if len(sat) > 0 and len(n) > 0:
                sat[n == 0] = np.nan
            return sat

        rs_same = _mask_by_count("rank_sat_same_host", "n_rank_same_host")
        rs_ipz = _mask_by_count("rank_sat_ip_zero", "n_rank_ip_zero")
        rs_untyped = _mask_by_count("rank_sat_untyped", "n_rank_untyped")
        ax = axes[2, 2]
        for arr, color, name in [
            (rs_same, "green", "same_host"),
            (rs_ipz, "red", "ip_zero"),
            (rs_untyped, "brown", "untyped"),
        ]:
            if len(arr) > 1 and np.any(~np.isnan(arr)):
                e = ema_smooth(arr, SM)
                ax.plot(steps, e, color=color, linewidth=2, label=f"{name} ({e[-1]:.0%})")
        ax.set_ylim(-0.05, 1.05)
        _add_warmup_vline(ax, warmup_step, steps)
        ax.set_title("Rank Satisfaction by Source")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_xlabel("Step"); ax.set_ylabel("Fraction")
        ax.grid(True, alpha=0.3)

        # ============================================================
        # Row 3: Gates | Grad Norm | L_true vs L_wrong
        # ============================================================

        # === [3,0] Gates ===
        ag = _get(stats, "attn_gate")
        fg = _get(stats, "ff_gate")
        ax = axes[3, 0]
        if len(ag) > 1:
            ax.plot(steps, ag, color="blue", linewidth=1.5, label=f"Attn ({ag[-1]:.4f})")
            ax.plot(steps, fg, color="green", linewidth=1.5, label=f"FF ({fg[-1]:.4f})")
            ax.set_title(f"Gates \u2014 attn={ag[-1]:.4f}, ff={fg[-1]:.4f}")
        else:
            ax.set_title("Gates (waiting...)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Gate value")
        ax.grid(True, alpha=0.3)

        # === [3,1] Grad norm ===
        gn = _get(stats, "grad_norm")
        ax = axes[3, 1]
        if len(gn) > 1 and gn.any():
            gn_ema = ema_smooth(gn, SM)
            ax.plot(steps, gn_ema, color="darkblue", linewidth=2.5, label=f"EMA({SM})")
            ax.set_title(f"Grad Norm \u2014 {gn_ema[-1]:.4f}")
        else:
            ax.set_title("Grad Norm (waiting...)")
        ax.set_xlabel("Step"); ax.set_ylabel("Norm")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # === [3,2] L_true vs L_wrong ===
        lt = _get(stats, "L_true_mean")
        lw = _get(stats, "L_wrong_mean")
        ax = axes[3, 2]
        if len(lt) > 1 and lt.any():
            lt_ema = ema_smooth(lt, SM)
            ax.plot(steps, lt_ema, color="blue", linewidth=2, label=f"L_true ({lt_ema[-1]:.4f})")
        if len(lw) > 1 and lw.any():
            lw_ema = ema_smooth(lw, SM)
            ax.plot(steps, lw_ema, color="red", linewidth=2, label=f"L_wrong ({lw_ema[-1]:.4f})")
        gamma_vals = _get(stats, "gamma")
        _add_warmup_vline(ax, warmup_step, steps)
        gamma_str = f" \u2014 \u03b3={gamma_vals[-1]:.4f}" if len(gamma_vals) > 1 and gamma_vals[-1] > 0 else ""
        ax.set_title(f"True vs Wrong Loss{gamma_str}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

    ani = FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
