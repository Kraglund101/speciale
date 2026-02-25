"""GPU smoke test: run 3 training steps end-to-end."""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_contrastive import train_contrastive

project_root = Path(__file__).parent.parent
save_dir = project_root / "results" / "_test_contrastive_smoke"

# Clean up any previous test
if save_dir.exists():
    shutil.rmtree(save_dir)

print("=" * 60)
print("GPU SMOKE TEST: 3 steps, Strategy A, host_batch=2, softplus+weighted")
print("=" * 60)

train_contrastive(
    data_json=project_root / "anomverse_extension/datasets/full_training_dataset/contrastive_training.json",
    save_dir=save_dir,
    captions_file=project_root / "anomverse_extension/datasets/full_training_dataset/captions_from_master.json",
    data_root=project_root / "anomverse_extension/datasets/full_training_dataset",
    n_steps=3,
    save_every=3,
    host_batch_size=2,  # Small for speed
    strategy="A",
    lambda_inv=1.0,
    lambda_rank=5.0,
    rank_gamma_scale=0.05,
    regularizer_warmup_frac=0.33,  # 1 warmup step out of 3
    multi_crop=True,
    clip_align=True,
    band_mode=2,
    noise_offset=0.05,
    no_live_viewer=True,
    pilot_subset_size=500,  # Small subset for speed
    triplet_mode="softplus",
    triplet_softplus_k=20.0,
    triplet_softplus_offset=0.0,
    typed_neg_mode="weighted",
    p_neg_same_host=0.6,
    p_neg_cross_host=0.3,
)

# Verify outputs
print("\n=== Verifying outputs ===")
assert (save_dir / "stats.csv").exists(), "stats.csv missing!"
assert (save_dir / "run_config.json").exists(), "run_config.json missing!"
assert (save_dir / "checkpoint_final").exists(), "checkpoint_final missing!"
assert (save_dir / "checkpoint_final" / "ip_adapter.pt").exists(), "ip_adapter.pt missing!"
assert (save_dir / "checkpoint_final" / "training_state.pt").exists(), "training_state.pt missing!"

# Check stats CSV has rows
with open(save_dir / "stats.csv") as f:
    lines = f.readlines()
assert len(lines) >= 4, f"Expected 4+ lines in stats.csv (header + 3 steps), got {len(lines)}"
print(f"  stats.csv: {len(lines)} lines (header + {len(lines)-1} steps)")

# Verify loss columns
header = lines[0].strip().split(",")
assert "L_diff" in header, f"L_diff column missing from stats: {header}"
assert "L_inv" in header, f"L_inv column missing from stats: {header}"
assert "L_rank_typed" in header
assert "d_same_mean" in header
assert "gamma" in header
# Verify 5 new gap metrics columns
assert "gap_positive_rate" in header, f"gap_positive_rate column missing from stats: {header}"
assert "gap_pos_rate_same_host" in header, f"gap_pos_rate_same_host column missing"
assert "gap_pos_rate_cross_host" in header, f"gap_pos_rate_cross_host column missing"
assert "gap_mean_same_host" in header, f"gap_mean_same_host column missing"
assert "gap_mean_cross_host" in header, f"gap_mean_cross_host column missing"
# Verify role embedding norm columns
assert "emb_global_norm" in header, f"emb_global_norm column missing from stats: {header}"
assert "emb_anomaly_norm" in header, f"emb_anomaly_norm column missing from stats: {header}"
assert "emb_normal_norm" in header, f"emb_normal_norm column missing from stats: {header}"
assert len(header) == 60, f"Expected 60 CSV columns, got {len(header)}: {header}"
print(f"  CSV columns: {len(header)} ({', '.join(header[:8])}...)")

# Parse last row to check values
last_row = lines[-1].strip().split(",")
col_map = {h: i for i, h in enumerate(header)}
L_diff = float(last_row[col_map["L_diff"]])
gamma = float(last_row[col_map["gamma"]])
print(f"  Last step: L_diff={L_diff:.4f}, gamma={gamma:.4f}")
assert L_diff > 0, "L_diff should be positive"

print("\n=== GPU SMOKE TEST PASSED ===")

# Clean up
shutil.rmtree(save_dir)
print("Cleaned up test directory")
