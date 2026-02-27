"""Memory profiling wrapper for contrastive training — V2 FULL DIAGNOSTICS.

Logs EVERYTHING per batch: sample indices, image paths, mask sizes, image dims,
anchor types, neg sources, memory at each phase (post-CLIP, post-UNet, post-backward).

Usage:
  python scripts/debug_memory_profile.py \
    --data-json anomverse_extension/datasets/full_training_dataset/contrastive_training.json \
    --captions-file anomverse_extension/datasets/full_training_dataset/captions_from_master.json \
    --data-root anomverse_extension/datasets/full_training_dataset \
    --save-dir results/debug_mem \
    --host-batch-size 4 \
    --steps 1000
"""
import csv
import gc
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_contrastive import (
    ContrastiveHostSampler,
    build_variant_batch,
    compute_contrastive_loss,
    generate_pilot_subset,
    _load_image_mask,
    _load_clip_reference,
)
from src.utils.mask_utils import create_latent_band_mask

_SEED = 42


def _mem_gb():
    """Return (peak_alloc, curr_alloc, curr_reserved) in GB."""
    return (
        torch.cuda.max_memory_allocated() / 1e9,
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


def count_cuda_tensors() -> int:
    count = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                count += 1
        except Exception:
            pass
    return count


def run_memory_profile(
    data_json: Path,
    save_dir: Path,
    captions_file: Path,
    data_root: Path,
    n_steps: int = 1000,
    host_batch_size: int = 4,
    lr: float = 1e-4,
    lr_pretrained: float = 1e-4,
    device: str = "cuda",
    strategy: str = "A",
    lambda_inv: float = 0.25,
    lambda_rank: float = 5.0,
    lambda_rank_untyped: float = 0.1,
    lambda_triplet: float = 1.0,
    rank_gamma_scale: float = 0.05,
    triplet_margin_m: float = 0.10,
    regularizer_warmup_frac: float = 0.05,
    p_null_typed_neg: float = 0.20,
    typed_neg_mode: str = "weighted",
    use_untyped_rank_null: bool = True,
    ip_adapter_type: str = "plus",
    ip_adapter_k: int = 16,
    ip_adapter_scale: float = 1.0,
    mask_visual: bool = True,
    band_mode: int = 2,
    loss_core_ratio: float = 0.8,
    t2i_adapter_mode: str = "cascade",
    multi_crop: bool = True,
    clip_align: bool = True,
    visual_mode: int = 3,
    learnable_gates: bool = True,
    force_gates: bool = False,
    sa_num_layers: int = 3,
    sa_num_heads: int = 12,
    noise_offset: float = 0.05,
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    triplet_mode: str = "softplus",
    triplet_softplus_k: float = 20.0,
    triplet_softplus_offset: float = 0.0,
    p_neg_same_host: float = 0.9,
    p_neg_cross_host: float = 0.0,
    pilot_subset_size: int = 0,
):
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mem_csv_path = save_dir / "memory_profile.csv"
    batch_log_path = save_dir / "batch_details.jsonl"

    print("=" * 70)
    print("MEMORY PROFILING V2 — FULL DIAGNOSTICS")
    print("=" * 70)
    print(f"Steps: {n_steps}, Host batch size: {host_batch_size}")
    print(f"Dataset: {'full' if pilot_subset_size == 0 else f'{pilot_subset_size} subset'}")
    print()

    # --- Sampler ---
    sampler = ContrastiveHostSampler(
        str(data_json), str(data_root), captions_file=str(captions_file),
    )

    if pilot_subset_size > 0 and pilot_subset_size < len(sampler.all_samples):
        pilot_indices = generate_pilot_subset(sampler, target_size=pilot_subset_size)
        pilot_set = set(pilot_indices)
        sampler.typed_indices = [i for i in sampler.typed_indices if i in pilot_set]
        sampler.untyped_indices = [i for i in sampler.untyped_indices if i in pilot_set]
        for g in list(sampler.group_to_indices.keys()):
            sampler.group_to_indices[g] = [i for i in sampler.group_to_indices[g] if i in pilot_set]
        for key in list(sampler.group_product_to_indices.keys()):
            sampler.group_product_to_indices[key] = [
                i for i in sampler.group_product_to_indices[key] if i in pilot_set
            ]
        for key in list(sampler.product_group_to_indices.keys()):
            sampler.product_group_to_indices[key] = [
                i for i in sampler.product_group_to_indices[key] if i in pilot_set
            ]
        sampler.group_products = defaultdict(list)
        for (g, p), idxs in sampler.group_product_to_indices.items():
            if g != "other" and idxs and p not in sampler.group_products[g]:
                sampler.group_products[g].append(p)
        sampler.product_to_groups = defaultdict(list)
        for (p, g), idxs in sampler.product_group_to_indices.items():
            if idxs and g not in sampler.product_to_groups[p]:
                sampler.product_to_groups[p].append(g)
        sampler.groups = [g for g in sampler.groups if sampler.group_to_indices.get(g)]
        print(f"Pilot subset: {len(sampler.typed_indices)} typed, {len(sampler.untyped_indices)} untyped")

    # --- Models ---
    print("\nLoading models...")
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    ip_adapter = create_ip_adapter(
        pipeline, adapter_type=ip_adapter_type, num_tokens=ip_adapter_k,
        scale=ip_adapter_scale, load_pretrained=True, mask_visual=mask_visual,
        visual_mode=visual_mode, learnable_gates=learnable_gates,
        force_gates=force_gates, sa_num_layers=sa_num_layers,
        sa_num_heads=sa_num_heads,
    )
    ip_adapter.freeze_image_encoder()

    t2i_adapter = None
    if t2i_adapter_mode != "off":
        from src.models.t2i_adapter import T2IAdapter
        t2i_adapter = T2IAdapter(in_channels=2, injection_mode=t2i_adapter_mode).to(device)

    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # --- Optimizer ---
    from src.utils.optim_utils import build_norm_param_id_set, split_decay_no_decay
    group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
    group_b_modules = []
    if hasattr(ip_adapter, 'masked_self_attn'):
        group_b_modules.append(ip_adapter.masked_self_attn)
    if t2i_adapter is not None:
        group_b_modules.append(t2i_adapter)

    norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)
    a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

    group_c_params = []
    if hasattr(ip_adapter, 'masked_self_attn') and ip_adapter.masked_self_attn.learnable_gates:
        for layer in ip_adapter.masked_self_attn.layers:
            group_c_params.append(layer["attn_gate"].gate)
            if "ff_gate" in layer:
                group_c_params.append(layer["ff_gate"].gate)
    if t2i_adapter is not None:
        for s in t2i_adapter.scales:
            group_c_params.append(s)
    group_c_ids = {id(p) for p in group_c_params}

    b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
    b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
    b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

    param_groups = [
        {"params": a_decay,     "lr": lr_pretrained, "weight_decay": 0.0},
        {"params": a_no_decay,  "lr": lr_pretrained, "weight_decay": 0.0},
        {"params": b_decay,     "lr": lr,            "weight_decay": 1e-4},
        {"params": b_no_decay,  "lr": lr,            "weight_decay": 0.0},
        {"params": group_c_params, "lr": lr,         "weight_decay": 0.0},
    ]
    trainable_params = [p for pg in param_groups for p in pg["params"]]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    use_amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline_alloc = torch.cuda.memory_allocated() / 1e9
    baseline_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\nBaseline memory: {baseline_alloc:.3f} GB allocated, {baseline_reserved:.3f} GB reserved")

    # --- CSV + JSONL ---
    mem_fh = open(mem_csv_path, "w", newline="")
    mem_writer = csv.writer(mem_fh)
    mem_writer.writerow([
        "batch", "n_variants", "n_crops", "n_typed", "n_untyped",
        "mem_pre_alloc", "mem_pre_reserved",
        "mem_post_clip_alloc", "mem_post_clip_reserved",
        "mem_post_unet_peak", "mem_post_unet_alloc", "mem_post_unet_reserved",
        "mem_post_backward_peak", "mem_post_backward_alloc", "mem_post_backward_reserved",
        "mem_post_step_alloc", "mem_post_step_reserved",
        "cuda_tensor_count",
        "time_clip_s", "time_unet_s", "time_backward_s", "time_total_s",
        "status",
    ])
    mem_fh.flush()

    batch_log_fh = open(batch_log_path, "w")

    rows = []
    warmup_steps = int(n_steps * regularizer_warmup_frac)
    gamma = 0.0
    warmup_L_diff_accum = []

    print(f"\nRunning {n_steps} batches with FULL diagnostics...")
    print(f"Warmup: {warmup_steps} steps")
    print(f"CSV: {mem_csv_path}")
    print(f"Batch log: {batch_log_path}\n")

    pbar = tqdm(range(n_steps), desc="MemProfile")

    for step in pbar:
        t_start = time.perf_counter()

        host_anchors = sampler.sample_host_batch(
            host_batch_size=host_batch_size,
            p_null_typed_neg=p_null_typed_neg,
            typed_neg_mode=typed_neg_mode,
            p_neg_same_host=p_neg_same_host,
            p_neg_cross_host=p_neg_cross_host,
        )

        # --- Log batch details to JSONL (BEFORE forward, so it's written even if we crash) ---
        batch_info = {"step": step, "anchors": []}
        for ha in host_anchors:
            anchor_info = sampler.get_sample_info(ha.anchor_idx)
            a_entry = {
                "anchor_idx": ha.anchor_idx,
                "anchor_type": ha.anchor_type,
                "group": ha.group,
                "product": ha.product,
                "image_path": anchor_info.get("image_path", ""),
                "mask_path": anchor_info.get("mask_path", ""),
                "neg_source": ha.neg_source,
                "neg_is_null": ha.neg_is_null,
                "variant_roles": ha.variant_roles,
            }
            # Load image to get dimensions + mask coverage
            try:
                img, msk = _load_image_mask(anchor_info, sampler.data_root)
                a_entry["img_shape"] = list(img.shape)  # [C, H, W]
                a_entry["mask_coverage"] = float(msk.mean().item())
                a_entry["mask_nonzero_pixels"] = int(msk.sum().item())
            except Exception as e:
                a_entry["load_error"] = str(e)

            if ha.pos2_idx is not None:
                pos2_info = sampler.get_sample_info(ha.pos2_idx)
                a_entry["pos2_idx"] = ha.pos2_idx
                a_entry["pos2_image_path"] = pos2_info.get("image_path", "")
                a_entry["pos2_product"] = pos2_info.get("product", "")

            if ha.neg_idx is not None:
                neg_info = sampler.get_sample_info(ha.neg_idx)
                a_entry["neg_idx"] = ha.neg_idx
                a_entry["neg_image_path"] = neg_info.get("image_path", "")
                a_entry["neg_group"] = neg_info.get("group", "")
                a_entry["neg_product"] = neg_info.get("product", "")

            batch_info["anchors"].append(a_entry)

        batch_log_fh.write(json.dumps(batch_info) + "\n")
        batch_log_fh.flush()

        # --- Memory tracking ---
        optimizer.zero_grad()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_pre = (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

        n_variants = -1
        n_crops = -1
        n_typed = sum(1 for ha in host_anchors if ha.anchor_type == "typed")
        n_untyped = sum(1 for ha in host_anchors if ha.anchor_type == "untyped")

        status = "OK"

        try:
            # --- Phase 1: build_variant_batch (includes CLIP encode) ---
            t_clip_start = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                vbatch = build_variant_batch(
                    host_anchors, sampler, pipeline, ip_adapter, t2i_adapter,
                    band_mode=band_mode, loss_core_ratio=loss_core_ratio,
                    multi_crop=multi_crop, clip_align=clip_align,
                    noise_offset=noise_offset,
                    timestep_sampling=timestep_sampling,
                    logit_normal_mean=logit_normal_mean,
                    logit_normal_std=logit_normal_std,
                    device=device,
                )
            torch.cuda.synchronize()
            t_clip_end = time.perf_counter()

            n_variants = vbatch.model_input.shape[0]
            n_crops = vbatch.ip_image_embeds.shape[1]
            mem_post_clip = (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

            # Log vbatch tensor shapes
            batch_info["vbatch_shapes"] = {
                "model_input": list(vbatch.model_input.shape),
                "ip_image_embeds": list(vbatch.ip_image_embeds.shape),
                "text_emb": list(vbatch.text_emb.shape),
                "timesteps": list(vbatch.timesteps.shape),
            }
            batch_log_fh.seek(batch_log_fh.tell() - 1)  # overwrite last newline
            # Actually just write an update line
            batch_log_fh.write("\n")
            batch_log_fh.write(json.dumps({
                "step": step, "_update": "vbatch_shapes",
                "model_input": list(vbatch.model_input.shape),
                "ip_image_embeds": list(vbatch.ip_image_embeds.shape),
                "mem_post_clip_alloc": f"{mem_post_clip[0]:.4f}",
            }) + "\n")
            batch_log_fh.flush()

            # --- Phase 2: UNet forward ---
            t_unet_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                eps_pred = pipeline.unet(
                    vbatch.model_input,
                    vbatch.timesteps,
                    encoder_hidden_states=vbatch.text_emb,
                    cross_attention_kwargs=vbatch.cross_attn_kwargs,
                    **vbatch.t2i_kwargs,
                ).sample.float()
            torch.cuda.synchronize()
            t_unet_end = time.perf_counter()

            mem_post_unet = (
                torch.cuda.max_memory_allocated() / 1e9,
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.memory_reserved() / 1e9,
            )

            # --- NaN source check ---
            eps_has_nan = eps_pred.isnan().any().item()
            eps_has_inf = eps_pred.isinf().any().item()
            eps_abs_max = eps_pred.abs().max().item()
            if eps_has_nan or eps_has_inf:
                print(f"\n*** STEP {step}: eps_pred has NaN={eps_has_nan} Inf={eps_has_inf} abs_max={eps_abs_max:.2f} ***")
                # Per-variant check
                for vi in range(eps_pred.shape[0]):
                    vnan = eps_pred[vi].isnan().any().item()
                    vinf = eps_pred[vi].isinf().any().item()
                    vmax = eps_pred[vi].abs().max().item()
                    if vnan or vinf:
                        role = vbatch.variant_roles[vi] if hasattr(vbatch, 'variant_roles') else '?'
                        atype = vbatch.anchor_types[vi] if hasattr(vbatch, 'anchor_types') else '?'
                        print(f"    variant[{vi}] role={role} type={atype}: NaN={vnan} Inf={vinf} abs_max={vmax:.2f}")

            # --- Phase 3: Loss + backward ---
            # Loss computed OUTSIDE autocast → fp32 reductions (no fp16 risk)
            t_back_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            in_warmup = step < warmup_steps
            if in_warmup:
                loss_total, extras = compute_contrastive_loss(
                    eps_pred, vbatch, strategy=strategy,
                    lambda_inv=0.0, lambda_rank=0.0,
                    lambda_rank_untyped=0.0, lambda_triplet=0.0,
                    gamma=0.0, triplet_margin_m=triplet_margin_m,
                    use_untyped_rank_null=use_untyped_rank_null,
                    triplet_mode=triplet_mode,
                    triplet_softplus_k=triplet_softplus_k,
                    triplet_softplus_offset=triplet_softplus_offset,
                )
                warmup_L_diff_accum.append(extras["L_diff"])
            else:
                if step == warmup_steps and warmup_L_diff_accum:
                    L_ref = np.mean(warmup_L_diff_accum)
                    gamma = rank_gamma_scale * L_ref
                loss_total, extras = compute_contrastive_loss(
                    eps_pred, vbatch, strategy=strategy,
                    lambda_inv=lambda_inv, lambda_rank=lambda_rank,
                    lambda_rank_untyped=lambda_rank_untyped,
                    lambda_triplet=lambda_triplet,
                    gamma=gamma, triplet_margin_m=triplet_margin_m,
                    use_untyped_rank_null=use_untyped_rank_null,
                    triplet_mode=triplet_mode,
                    triplet_softplus_k=triplet_softplus_k,
                    triplet_softplus_offset=triplet_softplus_offset,
                )

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print(f"\n*** STEP {step}: loss_total NaN={loss_total.isnan().item()} Inf={loss_total.isinf().item()} val={loss_total.item()} ***")
                print(f"    eps_pred NaN={eps_has_nan} Inf={eps_has_inf} abs_max={eps_abs_max:.2f}")
                print(f"    extras: { {k: v for k, v in extras.items() if isinstance(v, (int, float))} }")
                # Free the autograd graph to prevent memory spike
                del eps_pred, loss_total, vbatch
                gc.collect()
                torch.cuda.empty_cache()
                status = "NAN"
                t_back_end = time.perf_counter()
                mem_post_backward = (0, 0, 0)
                mem_post_step = (0, 0)
            else:
                scaler.scale(loss_total).backward()
                torch.cuda.synchronize()
                t_back_end = time.perf_counter()

                mem_post_backward = (
                    torch.cuda.max_memory_allocated() / 1e9,
                    torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9,
                )

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

                mem_post_step = (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.synchronize()
            status = "OOM"
            t_back_end = t_unet_end = t_clip_end = time.perf_counter()
            p, c, r = _mem_gb()
            tc = count_cuda_tensors()

            print(f"\n\n{'='*70}")
            print(f"OOM at batch {step}!")
            print(f"  Error: {e}")
            print(f"  Variants: {n_variants}, Crops: {n_crops}")
            print(f"  Peak: {p:.3f} GB, Current: {c:.3f} GB, Reserved: {r:.3f} GB")
            print(f"  CUDA tensors: {tc}")
            print(f"\n  Batch anchors:")
            for a in batch_info["anchors"]:
                print(f"    {a['anchor_type']} | {a['group']} | {a.get('image_path','?')}")
                print(f"      mask_cov={a.get('mask_coverage','?')}, neg_src={a.get('neg_source','?')}")

            mem_post_clip = mem_post_clip if 'mem_post_clip' in dir() else (0, 0)
            mem_post_unet = mem_post_unet if 'mem_post_unet' in dir() else (0, 0, 0)
            mem_post_backward = (p, c, r)
            mem_post_step = (c, r)

        except Exception as e:
            status = f"ERROR:{type(e).__name__}"
            t_back_end = t_unet_end = t_clip_end = time.perf_counter()
            print(f"\n  Unexpected error at step {step}: {e}")
            traceback.print_exc()
            mem_post_clip = mem_post_clip if 'mem_post_clip' in dir() else (0, 0)
            mem_post_unet = mem_post_unet if 'mem_post_unet' in dir() else (0, 0, 0)
            mem_post_backward = (0, 0, 0)
            mem_post_step = (0, 0)

        t_total = time.perf_counter() - t_start

        # CUDA tensor count every 50 steps
        tc = -1
        if step % 50 == 0:
            gc.collect()
            tc = count_cuda_tensors()

        # Write CSV row
        row = [
            step, n_variants, n_crops, n_typed, n_untyped,
            f"{mem_pre[0]:.4f}", f"{mem_pre[1]:.4f}",
            f"{mem_post_clip[0]:.4f}", f"{mem_post_clip[1]:.4f}",
            f"{mem_post_unet[0]:.4f}", f"{mem_post_unet[1]:.4f}", f"{mem_post_unet[2]:.4f}" if len(mem_post_unet) > 2 else "0",
            f"{mem_post_backward[0]:.4f}", f"{mem_post_backward[1]:.4f}", f"{mem_post_backward[2]:.4f}" if len(mem_post_backward) > 2 else "0",
            f"{mem_post_step[0]:.4f}", f"{mem_post_step[1]:.4f}",
            tc,
            f"{t_clip_end - t_start:.3f}", f"{t_unet_end - t_clip_end:.3f}" if 'mem_post_unet' in dir() else "0",
            f"{t_back_end - t_unet_end:.3f}" if 'mem_post_backward' in dir() else "0",
            f"{t_total:.3f}",
            status,
        ]
        rows.append(row)
        mem_writer.writerow(row)
        mem_fh.flush()  # flush EVERY row

        pbar.set_postfix({
            "unet_peak": f"{mem_post_unet[0]:.1f}G" if isinstance(mem_post_unet, tuple) and len(mem_post_unet) > 0 else "?",
            "curr": f"{mem_post_step[0]:.1f}G" if isinstance(mem_post_step, tuple) else "?",
            "V": n_variants,
        })

        if status == "OOM":
            print(f"\nLast 20 rows:")
            for r in rows[-20:]:
                print(f"  {r}")
            break

        if status.startswith("ERROR"):
            continue

    mem_fh.close()
    batch_log_fh.close()

    # --- Summary ---
    print(f"\n\n{'='*70}")
    print("MEMORY PROFILE SUMMARY")
    print(f"{'='*70}")
    print(f"CSV: {mem_csv_path}")
    print(f"Batch log: {batch_log_path}")
    print(f"Total batches: {len(rows)}")

    ok_rows = [r for r in rows if r[-1] == "OK"]

    if len(ok_rows) > 1:
        # Post-step alloc trend (column 15)
        allocs = [float(r[15]) for r in ok_rows]
        unet_peaks = [float(r[9]) for r in ok_rows]
        back_peaks = [float(r[12]) for r in ok_rows]

        print(f"\nPost-step alloc: first={allocs[0]:.4f} GB, last={allocs[-1]:.4f} GB, delta={allocs[-1]-allocs[0]:+.4f} GB")

        # Leak check
        diffs = [allocs[i+1] - allocs[i] for i in range(len(allocs)-1)]
        n_inc = sum(1 for d in diffs if d > 0.001)
        print(f"  Steps with >1MB increase: {n_inc}/{len(diffs)} ({100*n_inc/max(len(diffs),1):.1f}%)")

        # UNet peak by variant count
        peak_by_v = defaultdict(list)
        for r in ok_rows:
            peak_by_v[int(r[1])].append(float(r[9]))
        print(f"\nUNet peak by variant count:")
        for nv in sorted(peak_by_v.keys()):
            vals = peak_by_v[nv]
            print(f"  {nv} variants: mean={np.mean(vals):.3f} GB, max={np.max(vals):.3f} GB, p99={np.percentile(vals,99):.3f} GB ({len(vals)} batches)")

        # Find outliers (>24GB peak)
        outliers = [(int(r[0]), float(r[9]), int(r[1]), int(r[2])) for r in ok_rows if float(r[9]) > 22.0]
        if outliers:
            print(f"\n*** HIGH MEMORY OUTLIERS (UNet peak > 22 GB): ***")
            for batch_i, peak, nv, nc in outliers:
                print(f"  batch {batch_i}: peak={peak:.3f} GB, variants={nv}, crops={nc}")
            print(f"  Check batch_details.jsonl for these batch indices!")
        else:
            print(f"\nNo high-memory outliers (all UNet peaks < 22 GB)")

        # Backward peak stats
        print(f"\nBackward peak: mean={np.mean(back_peaks):.3f} GB, max={np.max(back_peaks):.3f} GB")

        # Tensor count trend
        tc_rows = [(int(r[0]), int(r[17])) for r in ok_rows if int(r[17]) >= 0]
        if tc_rows:
            print(f"\nCUDA tensor count: first={tc_rows[0][1]}, last={tc_rows[-1][1]}, delta={tc_rows[-1][1]-tc_rows[0][1]:+d}")

    last_status = rows[-1][-1] if rows else "?"
    if last_status == "OOM":
        print(f"\n*** ENDED WITH OOM at batch {rows[-1][0]} ***")
    elif last_status == "OK":
        print(f"\n*** Completed {len(rows)} batches without OOM ***")

    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory profiling V2 for contrastive training")
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--captions-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="results/debug_mem")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--host-batch-size", type=int, default=4)
    parser.add_argument("--pilot-subset-size", type=int, default=0,
                        help="0 = full dataset")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_memory_profile(
        data_json=Path(args.data_json),
        save_dir=Path(args.save_dir),
        captions_file=Path(args.captions_file),
        data_root=Path(args.data_root),
        n_steps=args.steps,
        host_batch_size=args.host_batch_size,
        pilot_subset_size=args.pilot_subset_size,
        device=args.device,
    )
