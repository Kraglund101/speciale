"""Benchmark training throughput across different batch sizes.

Loads models once, then runs N training steps at each batch size.
Reports images/sec and step/sec for each.

Usage:
    python scripts/benchmark_throughput.py --batch-sizes 4 6 8 10 12 16 --steps 100
"""
import os
import sys
import time
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset
from src.utils.mask_utils import (
    dilate_mask_batch, downsample_mask_maxpool, create_latent_band_mask,
)
from src.utils.optim_utils import (
    flatten_modules, build_norm_param_id_set, split_decay_no_decay,
    L2SPRegularizer,
)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def benchmark(
    batch_sizes: list,
    n_steps: int = 100,
    warmup_steps: int = 10,
    multi_crop: bool = True,
    augment: bool = True,
    device: str = "cuda",
):
    """Benchmark training throughput for each batch size."""

    # =========================================
    # Load Data
    # =========================================
    splits_dir = Path("anomverse_extension/datasets/full_training_dataset/splits_by_type")
    data_root = Path("anomverse_extension/datasets/full_training_dataset")
    captions_file = Path("anomverse_extension/datasets/full_training_dataset/captions_from_master.json")

    dataset = AnomalyDataset(
        splits_dir,
        exclude_sources=["MANTA_TINY_256", "MulSen_AD", "mvtec3d"],
        data_root=data_root,
        captions_file=captions_file,
        return_reference=True,
        reference_mode="full",
        reference_crop_size=224,
        augment=augment,
        multi_crop=multi_crop,
    )
    print(f"Dataset: {len(dataset)} samples")

    # =========================================
    # Initialize Pipeline + IP-Adapter
    # =========================================
    print("\nLoading models (one-time cost)...")
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    ip_adapter_type = "plus"
    ip_adapter_k = 16

    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=ip_adapter_type,
        num_tokens=ip_adapter_k,
        scale=1.0,
        load_pretrained=True,
        mask_visual=True,
        visual_mode=0,
    )
    ip_adapter.freeze_image_encoder()

    from src.models.t2i_adapter import T2IAdapter
    t2i_adapter = T2IAdapter(in_channels=2, injection_mode="cascade").to(device)

    # fp32
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # Import compute_anomagic_loss from training script
    # We'll inline the import since it's in the same scripts dir
    sys.path.insert(0, str(Path(__file__).parent))
    from train_anomagic import compute_anomagic_loss

    # =========================================
    # Build optimizer (shared across runs — recreated per batch size)
    # =========================================
    def build_optimizer():
        group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
        group_b_modules = [ip_adapter.masked_self_attn, t2i_adapter]

        norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)
        a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

        group_c_params = []
        for layer in ip_adapter.masked_self_attn.layers:
            group_c_params.append(layer["attn_gate"].gate)
            if "ff_gate" in layer:
                group_c_params.append(layer["ff_gate"].gate)
        for s in t2i_adapter.scales:
            group_c_params.append(s)
        group_c_ids = {id(p) for p in group_c_params}

        b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
        b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
        b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

        param_groups = [
            {"params": a_decay,       "lr": 5e-5, "weight_decay": 0.0},
            {"params": a_no_decay,    "lr": 5e-5, "weight_decay": 0.0},
            {"params": b_decay,       "lr": 1e-4, "weight_decay": 1e-3},
            {"params": b_no_decay,    "lr": 1e-4, "weight_decay": 0.0},
            {"params": group_c_params,"lr": 1e-4, "weight_decay": 0.0},
        ]

        trainable = [p for pg in param_groups for p in pg["params"]]
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        return opt, trainable

    # =========================================
    # Benchmark each batch size
    # =========================================
    results = []
    print(f"\nBenchmarking {len(batch_sizes)} batch sizes, {n_steps} steps each ({warmup_steps} warmup)...\n")
    print(f"{'BS':>4} | {'Steps':>5} | {'Time (s)':>9} | {'step/s':>7} | {'img/s':>7} | {'VRAM Peak (GB)':>14}")
    print("-" * 68)

    for bs in batch_sizes:
        # Reset model weights to pretrained (so each test starts from the same point)
        # Not strictly necessary for throughput benchmarking, but cleaner
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        optimizer, trainable_params = build_optimizer()

        loader = DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=4, pin_memory=False, persistent_workers=True,
            drop_last=True,
        )
        data_iter = iter(infinite_loader(loader))

        # Warmup
        print(f"  BS={bs}: warming up ({warmup_steps} steps)...", end="", flush=True)
        for _ in range(warmup_steps):
            batch = next(data_iter)
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            references = batch["reference"].to(device)
            batch_types = batch["anomaly_type"]
            batch_captions = batch.get("caption", [""] * bs)
            clip_masks = batch["clip_mask"].to(device) if "clip_mask" in batch else None
            references_2 = batch["reference_2"].to(device) if "reference_2" in batch else None
            clip_masks_2 = batch["clip_mask_2"].to(device) if "clip_mask_2" in batch else None
            group_valid = batch["group_valid"].to(device) if "group_valid" in batch else None

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                loss, _ = compute_anomagic_loss(
                    pipeline, ip_adapter, images, masks, references,
                    batch_types, batch_captions,
                    reference_mode="full",
                    clip_dilation_min_r=2, clip_dilation_max_r=10,
                    band_mode=2, loss_core_ratio=0.8,
                    binary_cross_attn_mask=False,
                    t2i_adapter=t2i_adapter,
                    cond_drop_prob=0.1,
                    clip_masks=clip_masks,
                    references_2=references_2,
                    clip_masks_2=clip_masks_2,
                    group_valid=group_valid,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            del images, masks, references, clip_masks, references_2, clip_masks_2, group_valid, loss

        torch.cuda.synchronize()
        print(" done. Measuring...", flush=True)

        # Measured steps
        step_times = []
        t_total_start = time.perf_counter()

        for i in range(n_steps):
            t_step_start = time.perf_counter()

            batch = next(data_iter)
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            references = batch["reference"].to(device)
            batch_types = batch["anomaly_type"]
            batch_captions = batch.get("caption", [""] * bs)
            clip_masks = batch["clip_mask"].to(device) if "clip_mask" in batch else None
            references_2 = batch["reference_2"].to(device) if "reference_2" in batch else None
            clip_masks_2 = batch["clip_mask_2"].to(device) if "clip_mask_2" in batch else None
            group_valid = batch["group_valid"].to(device) if "group_valid" in batch else None

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                loss, _ = compute_anomagic_loss(
                    pipeline, ip_adapter, images, masks, references,
                    batch_types, batch_captions,
                    reference_mode="full",
                    clip_dilation_min_r=2, clip_dilation_max_r=10,
                    band_mode=2, loss_core_ratio=0.8,
                    binary_cross_attn_mask=False,
                    t2i_adapter=t2i_adapter,
                    cond_drop_prob=0.1,
                    clip_masks=clip_masks,
                    references_2=references_2,
                    clip_masks_2=clip_masks_2,
                    group_valid=group_valid,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            del images, masks, references, clip_masks, references_2, clip_masks_2, group_valid, loss

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t_step_start)

        t_total = time.perf_counter() - t_total_start
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        steps_per_sec = n_steps / t_total
        imgs_per_sec = (n_steps * bs) / t_total
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)

        print(f"{bs:>4} | {n_steps:>5} | {t_total:>9.1f} | {steps_per_sec:>7.2f} | {imgs_per_sec:>7.1f} | {vram_peak_gb:>14.2f}")

        results.append({
            "batch_size": bs,
            "n_steps": n_steps,
            "total_time_s": round(t_total, 2),
            "steps_per_sec": round(steps_per_sec, 3),
            "imgs_per_sec": round(imgs_per_sec, 2),
            "avg_step_time_s": round(avg_step_time, 4),
            "std_step_time_s": round(std_step_time, 4),
            "vram_peak_gb": round(vram_peak_gb, 2),
        })

        # Kill the persistent workers before changing batch size
        del loader, data_iter
        torch.cuda.empty_cache()

    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 68)
    print("RESULTS SUMMARY")
    print("=" * 68)
    print(f"{'BS':>4} | {'step/s':>7} | {'img/s':>7} | {'avg step (ms)':>13} | {'VRAM (GB)':>10}")
    print("-" * 56)

    best_imgs = max(results, key=lambda r: r["imgs_per_sec"])
    for r in results:
        marker = " <-- BEST" if r["batch_size"] == best_imgs["batch_size"] else ""
        print(f"{r['batch_size']:>4} | {r['steps_per_sec']:>7.2f} | {r['imgs_per_sec']:>7.1f} | "
              f"{r['avg_step_time_s']*1000:>10.1f} ms | {r['vram_peak_gb']:>8.2f} GB{marker}")

    print(f"\nOptimal batch size: {best_imgs['batch_size']} ({best_imgs['imgs_per_sec']:.1f} img/s)")

    # Save results
    results_file = Path("results/benchmark_throughput.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark training throughput")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12],
                        help="Batch sizes to test")
    parser.add_argument("--steps", type=int, default=100,
                        help="Measured steps per batch size (after warmup)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup steps (not timed)")
    parser.add_argument("--no-multi-crop", action="store_true",
                        help="Disable multi-crop")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation")
    args = parser.parse_args()

    benchmark(
        batch_sizes=args.batch_sizes,
        n_steps=args.steps,
        warmup_steps=args.warmup,
        multi_crop=not args.no_multi_crop,
        augment=not args.no_augment,
    )
