"""Unit tests for contrastive training components (CPU-only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.train_contrastive import (
    HostAnchor, VariantBatch, compute_contrastive_loss,
    _weighted_l2, _masked_mse, ContrastiveHostSampler, generate_pilot_subset,
)
import random


def test_weighted_l2():
    print("=== TEST 1: _weighted_l2 and _masked_mse ===")
    x = torch.randn(4, 64, 64)
    w = torch.ones(1, 64, 64)
    result = _weighted_l2(x, w)
    assert result.shape == (), f"Expected scalar, got {result.shape}"
    assert result > 0, f"Expected positive, got {result}"
    print(f"  _weighted_l2: {result:.4f} (OK)")

    pred = torch.randn(4, 64, 64)
    target = torch.randn(4, 64, 64)
    mse = _masked_mse(pred, target, w)
    assert mse.shape == (), f"Expected scalar, got {mse.shape}"
    assert mse > 0, f"Expected positive, got {mse}"
    print(f"  _masked_mse: {mse:.4f} (OK)")
    print("  PASS")


def _make_batch(V, anchor_ids, anchor_types, variant_roles, anchor_groups,
                gets_ldiff=None, neg_sources=None):
    """Helper: create a VariantBatch with random tensors."""
    noise = torch.randn(V, 4, 64, 64)
    weight = torch.ones(V, 1, 64, 64) * 0.5
    core = torch.zeros(V, 1, 64, 64)
    core[:, :, 10:50, 10:50] = 1.0
    band = torch.zeros(V, 1, 64, 64)
    band[:, :, 8:52, 8:52] = 1.0
    band = band - core
    # Default gets_ldiff: pos1/true/null → True, pos2/neg → False
    if gets_ldiff is None:
        gets_ldiff = [r in ("pos1", "true", "null") for r in variant_roles]
    # Default neg_sources: "cross_host" for neg roles, "" otherwise
    if neg_sources is None:
        neg_sources = ["cross_host" if r == "neg" else "" for r in variant_roles]
    return VariantBatch(
        model_input=torch.zeros(V, 9, 64, 64),
        timesteps=torch.zeros(V, dtype=torch.long),
        text_emb=torch.zeros(V, 77, 768),
        ip_image_embeds=torch.zeros(V, 16, 768),
        cross_attn_kwargs={}, t2i_kwargs={},
        noise=noise, latents=torch.zeros(V, 4, 64, 64),
        weight_map_64=weight, core_mask_64=core, band_mask_64=band,
        anchor_ids=anchor_ids,
        anchor_types=anchor_types,
        variant_roles=variant_roles,
        anchor_groups=anchor_groups,
        sample_weights=torch.ones(V),
        gets_ldiff=gets_ldiff,
        neg_sources=neg_sources,
    )


def test_strategy_A_typed():
    print("\n=== TEST 2: Strategy A (typed only) ===")
    V = 6  # 2 typed anchors x 3 variants
    eps = torch.randn(V, 4, 64, 64, requires_grad=True)
    batch = _make_batch(
        V,
        anchor_ids=[0, 0, 0, 1, 1, 1],
        anchor_types=["typed"] * 6,
        variant_roles=["pos1", "pos2", "neg", "pos1", "pos2", "neg"],
        anchor_groups=["scratch"] * 3 + ["pit"] * 3,
    )

    loss, extras = compute_contrastive_loss(
        eps, batch, strategy="A",
        lambda_inv=1.0, lambda_rank=5.0, lambda_rank_untyped=0.1,
        gamma=0.01,
    )
    print(f"  loss_total: {loss.item():.4f}")
    print(f"  L_diff: {extras['L_diff']:.4f}")
    print(f"  L_inv: {extras['L_inv']:.4f}")
    print(f"  L_rank_typed: {extras['L_rank_typed']:.4f}")
    print(f"  d_same_mean: {extras['d_same_mean']:.4f}")
    print(f"  d_diff_mean: {extras['d_diff_mean']:.4f}")
    print(f"  n_typed: {extras['n_typed']}, n_untyped: {extras['n_untyped']}")

    loss.backward()
    assert eps.grad is not None, "No gradient!"
    assert eps.grad.abs().sum() > 0, "Zero gradient!"
    print(f"  grad norm: {eps.grad.norm():.4f} (non-zero, OK)")
    print("  PASS")


def test_strategy_B_mixed():
    print("\n=== TEST 3: Strategy B (typed + untyped) ===")
    V = 8
    eps = torch.randn(V, 4, 64, 64, requires_grad=True)
    batch = _make_batch(
        V,
        anchor_ids=[0, 0, 0, 1, 1, 1, 2, 2],
        anchor_types=["typed"] * 6 + ["untyped"] * 2,
        variant_roles=["pos1", "pos2", "neg", "pos1", "pos2", "neg", "true", "null"],
        anchor_groups=["scratch"] * 3 + ["pit"] * 3 + ["other"] * 2,
    )

    loss, extras = compute_contrastive_loss(
        eps, batch, strategy="B",
        lambda_triplet=0.1, lambda_rank_untyped=0.1,
        triplet_margin_m=0.10, gamma=0.01,
    )
    print(f"  loss_total: {loss.item():.4f}")
    print(f"  L_diff: {extras['L_diff']:.4f}")
    print(f"  L_triplet: {extras['L_triplet']:.4f}")
    print(f"  L_rank_untyped: {extras['L_rank_untyped']:.4f}")
    print(f"  n_typed: {extras['n_typed']}, n_untyped: {extras['n_untyped']}")

    loss.backward()
    assert eps.grad is not None and eps.grad.abs().sum() > 0
    print(f"  grad norm: {eps.grad.norm():.4f} (OK)")
    print("  PASS")


def test_warmup_ldiff_only():
    print("\n=== TEST 4: Warmup (lambda=0) produces L_diff only ===")
    V = 6
    eps = torch.randn(V, 4, 64, 64, requires_grad=True)
    batch = _make_batch(
        V,
        anchor_ids=[0, 0, 0, 1, 1, 1],
        anchor_types=["typed"] * 6,
        variant_roles=["pos1", "pos2", "neg"] * 2,
        anchor_groups=["scratch"] * 3 + ["pit"] * 3,
    )

    loss, ex = compute_contrastive_loss(
        eps, batch, strategy="A",
        lambda_inv=0.0, lambda_rank=0.0, lambda_rank_untyped=0.0,
        lambda_triplet=0.0, gamma=0.0,
    )
    assert abs(loss.item() - ex["L_diff"]) < 1e-4, (
        f"Warmup loss should equal L_diff: {loss.item()} vs {ex['L_diff']}"
    )
    print(f"  loss_total == L_diff: {loss.item():.4f} == {ex['L_diff']:.4f} (OK)")
    print("  PASS")


def test_rank_satisfaction():
    print("\n=== TEST 5: Rank satisfaction (perfect pos1, bad neg) ===")
    V = 3
    noise = torch.randn(V, 4, 64, 64)
    eps_data = noise.clone()
    eps_data[1] = noise[1] + 0.01 * torch.randn_like(noise[1])  # pos2 near-perfect
    eps_data[2] = noise[2] + 5.0 * torch.randn_like(noise[2])   # neg very wrong
    eps = eps_data.clone().requires_grad_(True)

    batch = VariantBatch(
        model_input=torch.zeros(V, 9, 64, 64),
        timesteps=torch.zeros(V, dtype=torch.long),
        text_emb=torch.zeros(V, 77, 768),
        ip_image_embeds=torch.zeros(V, 16, 768),
        cross_attn_kwargs={}, t2i_kwargs={},
        noise=noise, latents=torch.zeros(V, 4, 64, 64),
        weight_map_64=torch.ones(V, 1, 64, 64),
        core_mask_64=torch.ones(V, 1, 64, 64),
        band_mask_64=torch.zeros(V, 1, 64, 64),
        anchor_ids=[0, 0, 0],
        anchor_types=["typed"] * 3,
        variant_roles=["pos1", "pos2", "neg"],
        anchor_groups=["scratch"] * 3,
        sample_weights=torch.ones(V),
        gets_ldiff=[True, False, False],
        neg_sources=["", "", "cross_host"],
    )

    _, ex = compute_contrastive_loss(
        eps, batch, strategy="A", lambda_inv=1.0, lambda_rank=5.0, gamma=0.001,
    )
    print(f"  L_true: {ex['L_true_mean']:.6f} (should be ~0)")
    print(f"  L_wrong: {ex['L_wrong_mean']:.4f} (should be large)")
    print(f"  rank_satisfied: {ex['pct_rank_satisfied_typed']:.0%} (should be 100%)")
    assert ex["pct_rank_satisfied_typed"] == 1.0
    assert ex["L_true_mean"] < ex["L_wrong_mean"]
    print("  PASS")


def test_untyped_ranking():
    print("\n=== TEST 6: Untyped true-vs-null ranking ===")
    V = 2
    noise = torch.randn(V, 4, 64, 64)
    eps_data = noise.clone()
    eps_data[1] = noise[1] + 3.0 * torch.randn_like(noise[1])  # null much worse
    eps = eps_data.clone().requires_grad_(True)

    batch = VariantBatch(
        model_input=torch.zeros(V, 9, 64, 64),
        timesteps=torch.zeros(V, dtype=torch.long),
        text_emb=torch.zeros(V, 77, 768),
        ip_image_embeds=torch.zeros(V, 16, 768),
        cross_attn_kwargs={}, t2i_kwargs={},
        noise=noise, latents=torch.zeros(V, 4, 64, 64),
        weight_map_64=torch.ones(V, 1, 64, 64),
        core_mask_64=torch.ones(V, 1, 64, 64),
        band_mask_64=torch.zeros(V, 1, 64, 64),
        anchor_ids=[0, 0],
        anchor_types=["untyped"] * 2,
        variant_roles=["true", "null"],
        anchor_groups=["other"] * 2,
        sample_weights=torch.ones(V),
        gets_ldiff=[True, True],  # both true and null get L_diff
        neg_sources=["", ""],
    )

    loss, ex = compute_contrastive_loss(
        eps, batch, strategy="A",
        lambda_inv=1.0, lambda_rank=5.0, lambda_rank_untyped=0.1,
        gamma=0.001,
    )
    print(f"  L_true_untyped: {ex['L_true_untyped_mean']:.6f}")
    print(f"  L_null_untyped: {ex['L_null_untyped_mean']:.4f}")
    print(f"  rank_satisfied_untyped: {ex['pct_rank_satisfied_untyped']:.0%}")
    assert ex["pct_rank_satisfied_untyped"] == 1.0
    print("  PASS")


def test_sampler():
    print("\n=== TEST 7: ContrastiveHostSampler ===")
    json_path = "anomverse_extension/datasets/full_training_dataset/contrastive_training.json"
    data_root = "anomverse_extension/datasets/full_training_dataset"
    captions = "anomverse_extension/datasets/full_training_dataset/captions_from_master.json"

    if not Path(json_path).exists():
        print("  SKIP (data not found)")
        return

    sampler = ContrastiveHostSampler(json_path, data_root, captions_file=captions)

    assert len(sampler.typed_indices) > 50000
    assert len(sampler.untyped_indices) > 4000
    assert len(sampler.groups) == 7

    random.seed(42)
    batch = sampler.sample_host_batch(host_batch_size=4)
    assert len(batch) == 4

    n_typed = sum(1 for ha in batch if ha.anchor_type == "typed")
    assert n_typed >= 1, "Must have at least 1 typed anchor"

    for ha in batch:
        if ha.anchor_type == "typed":
            assert ha.variant_roles == ["pos1", "pos2", "neg"]
            assert ha.pos2_idx is not None
            assert ha.neg_idx is not None or ha.neg_is_null
        else:
            assert ha.variant_roles == ["true", "null"]

        print(f"  {ha.anchor_type:8s} group={ha.group:15s} roles={ha.variant_roles}")

    # Pilot subset
    pilot = generate_pilot_subset(sampler, target_size=3000)
    assert len(pilot) > 1000
    assert len(pilot) < 5000
    print(f"  Pilot: {len(pilot)} indices")
    print("  PASS")


def test_loss_components_isolated():
    """Verify individual loss components behave correctly in isolation."""
    print("\n=== TEST 8: Loss components isolation ===")

    # L_inv should be 0 when pos1 and pos2 produce SAME eps output.
    # Key: all variants of the same anchor share the same noise (as in real pipeline).
    V = 3
    shared_noise = torch.randn(1, 4, 64, 64)
    noise = shared_noise.expand(V, -1, -1, -1).contiguous()  # same noise for all

    # All 3 variants produce the exact same prediction
    shared_eps = torch.randn(1, 4, 64, 64)
    eps = shared_eps.expand(V, -1, -1, -1).contiguous().clone().requires_grad_(True)

    batch = VariantBatch(
        model_input=torch.zeros(V, 9, 64, 64),
        timesteps=torch.zeros(V, dtype=torch.long),
        text_emb=torch.zeros(V, 77, 768),
        ip_image_embeds=torch.zeros(V, 16, 768),
        cross_attn_kwargs={}, t2i_kwargs={},
        noise=noise, latents=torch.zeros(V, 4, 64, 64),
        weight_map_64=torch.ones(V, 1, 64, 64),
        core_mask_64=torch.ones(V, 1, 64, 64),
        band_mask_64=torch.zeros(V, 1, 64, 64),
        anchor_ids=[0, 0, 0],
        anchor_types=["typed"] * 3,
        variant_roles=["pos1", "pos2", "neg"],
        anchor_groups=["scratch"] * 3,
        sample_weights=torch.ones(V),
        gets_ldiff=[True, False, False],
        neg_sources=["", "", "cross_host"],
    )

    # When pos1 and pos2 produce identical eps, L_inv should be ~0
    _, ex = compute_contrastive_loss(
        eps, batch, strategy="A", lambda_inv=1.0, lambda_rank=0.0, gamma=0.0,
    )
    assert ex["L_inv"] < 0.01, f"L_inv should be ~0 when predictions identical, got {ex['L_inv']}"
    assert ex["d_same_mean"] < 0.01, f"d_same should be ~0, got {ex['d_same_mean']}"
    print(f"  Identical predictions: L_inv={ex['L_inv']:.6f}, d_same={ex['d_same_mean']:.6f} (OK)")

    # When all eps identical and noise shared: L_true == L_wrong, so hinge = max(0, 0) = 0
    _, ex2 = compute_contrastive_loss(
        eps, batch, strategy="A", lambda_inv=0.0, lambda_rank=1.0, gamma=0.0,
    )
    assert ex2["L_rank_typed"] < 0.01, f"L_rank should be 0 when gamma=0 and L_true=L_wrong"
    print(f"  Equal losses, gamma=0: L_rank={ex2['L_rank_typed']:.6f} (OK)")

    # With gamma > 0 and L_true = L_wrong, hinge should = gamma
    _, ex3 = compute_contrastive_loss(
        eps, batch, strategy="A", lambda_inv=0.0, lambda_rank=1.0, gamma=0.5,
    )
    assert abs(ex3["L_rank_typed"] - 0.5) < 0.01, (
        f"L_rank should be ~gamma when L_true=L_wrong, got {ex3['L_rank_typed']}"
    )
    print(f"  Equal losses, gamma=0.5: L_rank={ex3['L_rank_typed']:.4f} (OK, ~0.5)")
    print("  PASS")


def test_strategy_B_triplet():
    """Verify triplet margin behavior."""
    print("\n=== TEST 9: Strategy B triplet margin ===")
    V = 3
    # Same anchor → same noise for all variants
    shared_noise = torch.randn(1, 4, 64, 64)
    noise = shared_noise.expand(V, -1, -1, -1).contiguous()

    # Make d_same large (pos1 != pos2) and d_diff small (pos1 ~ neg)
    eps_data = torch.randn(V, 4, 64, 64)
    base_pred = torch.randn(1, 4, 64, 64)
    eps_data[0] = base_pred[0]                                        # pos1
    eps_data[1] = base_pred[0] + 2.0 * torch.randn_like(base_pred[0])  # pos2: very different
    eps_data[2] = base_pred[0] + 0.01 * torch.randn_like(base_pred[0]) # neg: almost identical
    eps = eps_data.clone().requires_grad_(True)

    batch = VariantBatch(
        model_input=torch.zeros(V, 9, 64, 64),
        timesteps=torch.zeros(V, dtype=torch.long),
        text_emb=torch.zeros(V, 77, 768),
        ip_image_embeds=torch.zeros(V, 16, 768),
        cross_attn_kwargs={}, t2i_kwargs={},
        noise=noise, latents=torch.zeros(V, 4, 64, 64),
        weight_map_64=torch.ones(V, 1, 64, 64),
        core_mask_64=torch.ones(V, 1, 64, 64),
        band_mask_64=torch.zeros(V, 1, 64, 64),
        anchor_ids=[0, 0, 0],
        anchor_types=["typed"] * 3,
        variant_roles=["pos1", "pos2", "neg"],
        anchor_groups=["scratch"] * 3,
        sample_weights=torch.ones(V),
        gets_ldiff=[True, False, False],
        neg_sources=["", "", "cross_host"],
    )

    _, ex = compute_contrastive_loss(
        eps, batch, strategy="B", lambda_triplet=1.0,
        triplet_margin_m=0.0,  # margin=0
    )
    # d_same > d_diff → triplet should fire (positive)
    print(f"  d_same={ex['d_same_mean']:.4f}, d_diff={ex['d_diff_mean']:.4f}")
    assert ex["d_same_mean"] > ex["d_diff_mean"], "d_same should be > d_diff in this setup"
    assert ex["L_triplet"] > 0, f"Triplet should fire when d_same > d_diff"
    print(f"  L_triplet={ex['L_triplet']:.4f} > 0 (violation detected, OK)")
    print("  PASS")


def test_softplus_triplet():
    """Test 10: softplus triplet mode produces non-zero loss and gap metrics."""
    print("\n=== TEST 10: Softplus triplet mode ===")
    V = 3
    shared_noise = torch.randn(1, 4, 64, 64)
    noise = shared_noise.expand(V, -1, -1, -1).contiguous()

    # Make d_same large (pos1 != pos2) and d_diff small (pos1 ~ neg)
    eps_data = torch.randn(V, 4, 64, 64)
    base_pred = torch.randn(1, 4, 64, 64)
    eps_data[0] = base_pred[0]                                        # pos1
    eps_data[1] = base_pred[0] + 2.0 * torch.randn_like(base_pred[0])  # pos2: very different
    eps_data[2] = base_pred[0] + 0.01 * torch.randn_like(base_pred[0]) # neg: almost identical
    eps = eps_data.clone().requires_grad_(True)

    batch = VariantBatch(
        model_input=torch.zeros(V, 9, 64, 64),
        timesteps=torch.zeros(V, dtype=torch.long),
        text_emb=torch.zeros(V, 77, 768),
        ip_image_embeds=torch.zeros(V, 16, 768),
        cross_attn_kwargs={}, t2i_kwargs={},
        noise=noise, latents=torch.zeros(V, 4, 64, 64),
        weight_map_64=torch.ones(V, 1, 64, 64),
        core_mask_64=torch.ones(V, 1, 64, 64),
        band_mask_64=torch.zeros(V, 1, 64, 64),
        anchor_ids=[0, 0, 0],
        anchor_types=["typed"] * 3,
        variant_roles=["pos1", "pos2", "neg"],
        anchor_groups=["scratch"] * 3,
        sample_weights=torch.ones(V),
        gets_ldiff=[True, False, False],
        neg_sources=["", "", "same_host"],
    )

    # Softplus mode should produce positive loss when d_same > d_diff
    _, ex = compute_contrastive_loss(
        eps, batch, strategy="A", lambda_triplet=1.0, lambda_inv=0.0, lambda_rank=0.0,
        triplet_mode="softplus", triplet_softplus_k=20.0, triplet_softplus_offset=0.0,
    )
    print(f"  d_same={ex['d_same_mean']:.4f}, d_diff={ex['d_diff_mean']:.4f}")
    assert ex["d_same_mean"] > ex["d_diff_mean"], "d_same should be > d_diff in this setup"
    assert ex["L_triplet"] > 0, f"Softplus triplet should fire when d_same > d_diff, got {ex['L_triplet']}"
    print(f"  L_triplet={ex['L_triplet']:.4f} > 0 (OK)")

    # gap_positive_rate should be computed
    assert "gap_positive_rate" in ex, "gap_positive_rate missing from extras"
    print(f"  gap_positive_rate={ex['gap_positive_rate']:.2f}")
    assert "gap_pos_rate_same_host" in ex, "gap_pos_rate_same_host missing"
    print(f"  gap_pos_rate_same_host={ex['gap_pos_rate_same_host']:.2f}")
    assert "gap_mean_same_host" in ex, "gap_mean_same_host missing"
    print(f"  gap_mean_same_host={ex['gap_mean_same_host']:.4f}")

    # Softplus should always be > 0 (even when gap is positive, just small)
    _, ex2 = compute_contrastive_loss(
        eps, batch, strategy="B", lambda_triplet=1.0,
        triplet_mode="softplus", triplet_softplus_k=20.0,
    )
    assert ex2["L_triplet"] > 0, "Softplus always > 0"
    print(f"  Strategy B softplus L_triplet={ex2['L_triplet']:.4f} > 0 (OK)")

    print("  PASS")


if __name__ == "__main__":
    test_weighted_l2()
    test_strategy_A_typed()
    test_strategy_B_mixed()
    test_warmup_ldiff_only()
    test_rank_satisfaction()
    test_untyped_ranking()
    test_sampler()
    test_loss_components_isolated()
    test_strategy_B_triplet()
    test_softplus_triplet()
    print("\n" + "=" * 50)
    print("ALL CPU TESTS PASSED")
    print("=" * 50)
