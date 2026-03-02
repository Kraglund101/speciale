"""
VRAM spike / segfault debug wrapper for train_anomagic.py.

Runs the same training loop but with comprehensive instrumentation:
1. VRAM + shape logger (mem_debug.csv)
2. Spike trap (memory summary dump when reserved > threshold)
3. Shape assertions (fixed H/W, ref always 224x224)
4. SDPA backend isolation (--sdp-mode: default/math_only/flash_only/mem_efficient_only)
5. No per-step empty_cache (removed to reduce allocator churn)
6. Deterministic mode (fixed seed, ordered data, batch index logging)
7. Batch index logging (exact dataset indices per step)
8. Stage fences with cuda.synchronize (--cuda-fence)
9. DataLoader isolation (--num-workers, --pin-memory, --persistent-workers)
10. Crash artifact capture (faulthandler, heartbeat, env dump)

Usage:
    # Backend isolation: flash-only with stage fences
    python scripts/train_anomagic_debug.py \
        --captions-file ... --splits-dir ... --data-root ... \
        --sdp-mode flash_only --cuda-fence --batch-size 14 \
        --steps 20000 --max-debug-steps 20000

    # DataLoader isolation: no workers
    python scripts/train_anomagic_debug.py \
        --captions-file ... --splits-dir ... --data-root ... \
        --sdp-mode flash_only --dl-num-workers 0 --no-dl-pin-memory \
        --no-dl-persistent-workers --batch-size 14 \
        --steps 20000 --max-debug-steps 20000
"""
import os
import sys
import csv
import time
import json
import random
import faulthandler
import threading
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader, Sampler
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset
from src.utils.mask_utils import (
    dilate_mask_batch, downsample_mask_maxpool, create_latent_band_mask,
)
# Import the non-loop pieces from the real training script
from scripts.train_anomagic import (
    compute_anomagic_loss,
    save_anomagic_checkpoint,
    infinite_loader,
    _launch_live_loss,
)


# =========================================================================
# SDPA backend configuration
# =========================================================================

def configure_sdp_mode(mode: str):
    """Configure SDPA backends by subtracting one at a time for isolation.

    Pure single-backend modes (flash_only, mem_efficient_only) can fail because
    the VAE attention has constraints that only some backends satisfy. Instead,
    we subtract one backend to identify which causes crashes:

    - default:          all ON (safe baseline)
    - no_math:          flash+mem_efficient only (known to crash)
    - no_flash:         mem_efficient+math only (does removing flash fix it?)
    - no_mem_efficient: flash+math only (does removing mem_efficient fix it?)
    - math_only:        only math (safe baseline, slowest)
    """
    if mode == "default":
        return "default (flash=ON, mem_efficient=ON, math=ON)"
    elif mode == "math_only":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return "math_only (flash=OFF, mem_efficient=OFF, math=ON)"
    elif mode == "no_math":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        return "no_math (flash=ON, mem_efficient=ON, math=OFF) — known crash config"
    elif mode == "no_flash":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return "no_flash (flash=OFF, mem_efficient=ON, math=ON)"
    elif mode == "no_mem_efficient":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return "no_mem_efficient (flash=ON, mem_efficient=OFF, math=ON)"
    else:
        raise ValueError(f"Unknown sdp-mode: {mode}. "
                         f"Choose from: default, math_only, no_math, no_flash, no_mem_efficient")


# =========================================================================
# Debug helpers
# =========================================================================

def heartbeat(hb_fh, step: int, stage: str, cuda_fence: bool = False):
    """Write a stage marker so we know where a hang occurred.

    If cuda_fence=True, synchronize GPU before writing — this ensures the
    heartbeat reflects actual GPU completion, not just CPU dispatch.
    """
    if cuda_fence:
        torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    hb_fh.write(f"{time.time():.1f}\tstep={step}\t{stage}\talloc={alloc:.0f}MB\treserved={reserved:.0f}MB\n")
    hb_fh.flush()


def init_mem_csv(save_dir: Path):
    """Open mem_debug.csv with header. Returns (fh, writer)."""
    path = save_dir / "mem_debug.csv"
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow([
        "step", "timestamp",
        "img_H", "img_W", "mask_H", "mask_W", "ref_H", "ref_W",
        "clip_mask_shape",
        "allocated_MB", "reserved_MB", "max_allocated_MB",
        "mask_density",
        "loss",
        "batch_indices",
    ])
    fh.flush()
    return fh, writer


def log_mem_row(writer, fh, step, shapes, mask_density, loss_val, batch_indices):
    """Write one row to mem_debug.csv."""
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    writer.writerow([
        step, f"{time.time():.1f}",
        shapes["img_H"], shapes["img_W"],
        shapes["mask_H"], shapes["mask_W"],
        shapes["ref_H"], shapes["ref_W"],
        shapes["clip_mask"],
        f"{alloc:.1f}", f"{reserved:.1f}", f"{max_alloc:.1f}",
        f"{mask_density:.6f}",
        f"{loss_val:.6f}" if loss_val is not None else "nan",
        str(batch_indices),
    ])
    fh.flush()


def spike_trap(step, save_dir, shapes, mask_density, batch_indices):
    """Dump full CUDA memory summary when spike detected.

    NOTE: No torch.cuda.synchronize() here — if GPU is in a bad state,
    synchronize would hang forever, deadlocking the debug script itself.
    """
    summary = torch.cuda.memory_summary(abbreviated=True)
    path = save_dir / f"mem_summary_step_{step}.txt"
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    with open(path, "w") as f:
        f.write(f"VRAM SPIKE at step {step}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Allocated: {alloc:.1f} MB\n")
        f.write(f"Reserved:  {reserved:.1f} MB\n")
        f.write(f"Max alloc: {max_alloc:.1f} MB\n\n")
        f.write(f"Shapes: {shapes}\n")
        f.write(f"Mask density: {mask_density:.6f}\n")
        f.write(f"Batch indices: {batch_indices}\n\n")
        f.write(summary)
    print(f"\n  *** VRAM SPIKE at step {step}! "
          f"Reserved={reserved:.0f}MB, Allocated={alloc:.0f}MB. "
          f"Saved to {path}")


def collect_shapes(images, masks, references, clip_masks):
    """Extract shape info as a flat dict."""
    return {
        "img_H": images.shape[-2], "img_W": images.shape[-1],
        "mask_H": masks.shape[-2], "mask_W": masks.shape[-1],
        "ref_H": references.shape[-2], "ref_W": references.shape[-1],
        "clip_mask": str(tuple(clip_masks.shape)) if clip_masks is not None else "None",
    }


def dump_crash_env(save_dir: Path, sdp_mode: str, cuda_fence: bool,
                   batch_size: int, dl_num_workers: int, dl_pin_memory: bool,
                   dl_persistent_workers: bool):
    """Write environment info for crash artifact analysis."""
    path = save_dir / "crash_env.txt"
    with open(path, "w") as f:
        f.write("=== Crash Environment ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        try:
            import xformers
            f.write(f"xFormers: {xformers.__version__}\n")
        except ImportError:
            f.write("xFormers: not installed\n")
        f.write(f"\nSDP mode: {sdp_mode}\n")
        f.write(f"  flash: {torch.backends.cuda.flash_sdp_enabled()}\n")
        f.write(f"  mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}\n")
        f.write(f"  math: {torch.backends.cuda.math_sdp_enabled()}\n")
        f.write(f"\ncuda_fence: {cuda_fence}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"DataLoader: workers={dl_num_workers}, pin_memory={dl_pin_memory}, "
                f"persistent_workers={dl_persistent_workers}\n")
        f.write(f"\nCUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set')}\n")
        f.write(f"TORCH_SHOW_CPP_STACKTRACES: {os.environ.get('TORCH_SHOW_CPP_STACKTRACES', 'not set')}\n")
    print(f"  Crash env written to {path}")


# =========================================================================
# IndexTrackingLoader — wraps DataLoader to expose dataset indices
# =========================================================================

class IndexTrackingDataset(torch.utils.data.Dataset):
    """Wraps AnomalyDataset to also return the index."""
    def __init__(self, dataset):
        self.dataset = dataset
        # Forward all attributes
        self.samples = dataset.samples
        self.anomaly_types = dataset.anomaly_types
        self.type_to_samples = dataset.type_to_samples
        self.reference_mode = dataset.reference_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["_index"] = idx
        return sample


# =========================================================================
# Debug training loop
# =========================================================================

def train_debug(
    splits_dir, save_dir, captions_file,
    n_steps=500, batch_size=4, lr=1e-4, device="cuda",
    save_every=1000, anomaly_types=None, exclude_sources=None,
    data_root=None, ip_adapter_type="plus", ip_adapter_k=16,
    ip_adapter_scale=1.0, reference_mode="full", mask_visual=True,
    clip_dilation_min_r=2, clip_dilation_max_r=10,
    band_mode=1, loss_core_ratio=0.8, binary_cross_attn_mask=False,
    t2i_adapter_mode="cascade", lora_rank=0, lora_alpha=16,
    cond_drop_prob=0.1, augment=False, resume_dir=None,
    # Debug-specific args
    spike_threshold_gb=18.0,
    deterministic=False,
    sdp_mode="default",
    cuda_fence=False,
    max_debug_steps=None,
    # DataLoader isolation
    dl_num_workers=4,
    dl_pin_memory=True,
    dl_persistent_workers=True,
):
    """Debug training loop — identical to train_anomagic but with instrumentation."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    spike_threshold_bytes = spike_threshold_gb * 1024**3

    print("=" * 70)
    print("ANOMAGIC DEBUG TRAINING — Segfault Investigation")
    print("=" * 70)

    # --- Environment info ---
    print(f"\nPyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    try:
        import xformers
        print(f"xFormers: {xformers.__version__}")
    except ImportError:
        print("xFormers: not installed")
    print(f"SDPA flash: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"SDPA mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"SDPA math: {torch.backends.cuda.math_sdp_enabled()}")

    # Configure SDPA backend
    sdp_desc = configure_sdp_mode(sdp_mode)
    print(f"\n>>> SDP mode: {sdp_desc}")
    print(f"    After config — flash: {torch.backends.cuda.flash_sdp_enabled()}, "
          f"mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
          f"math: {torch.backends.cuda.math_sdp_enabled()}")

    print(f"\nSpike threshold: {spike_threshold_gb:.0f} GB")
    print(f"Deterministic: {deterministic}")
    print(f"CUDA fence (synchronize per stage): {cuda_fence}")
    print(f"Batch size: {batch_size}")
    print(f"IP-Adapter: {ip_adapter_type} K={ip_adapter_k}")
    print(f"Augmentation: {augment}")
    print(f"Band mode: {band_mode}")
    print(f"DataLoader: workers={dl_num_workers}, pin_memory={dl_pin_memory}, "
          f"persistent_workers={dl_persistent_workers}")
    print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set')}")
    steps_to_run = max_debug_steps or n_steps
    print(f"Steps to run: {steps_to_run}")

    # Write crash env info upfront (so it exists even if we segfault)
    dump_crash_env(save_dir, sdp_mode, cuda_fence, batch_size,
                   dl_num_workers, dl_pin_memory, dl_persistent_workers)

    if deterministic:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # =========================================
    # Load Data (same as train_anomagic)
    # =========================================
    dataset = AnomalyDataset(
        splits_dir, anomaly_types=anomaly_types,
        exclude_sources=exclude_sources, data_root=data_root,
        captions_file=captions_file, return_reference=True,
        reference_mode=reference_mode, reference_crop_size=224,
        augment=augment,
    )
    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return

    # Wrap for index tracking
    tracked_dataset = IndexTrackingDataset(dataset)

    # =========================================
    # Initialize models (same as train_anomagic)
    # =========================================
    print("\nInitializing models...")
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    # Disable xFormers if present (we want pure SDPA control)
    if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'disable_xformers_memory_efficient_attention'):
        try:
            pipeline.unet.disable_xformers_memory_efficient_attention()
            print("  Disabled xFormers memory-efficient attention on UNet")
        except Exception:
            pass
    if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'disable_xformers_memory_efficient_attention'):
        try:
            pipeline.vae.disable_xformers_memory_efficient_attention()
            print("  Disabled xFormers memory-efficient attention on VAE")
        except Exception:
            pass

    print(f"\nLoading IP-Adapter ({ip_adapter_type}, K={ip_adapter_k})...")
    ip_adapter = create_ip_adapter(
        pipeline, adapter_type=ip_adapter_type,
        num_tokens=ip_adapter_k, scale=ip_adapter_scale,
        load_pretrained=True, mask_visual=mask_visual,
    )
    ip_adapter.freeze_image_encoder()

    t2i_adapter = None
    if t2i_adapter_mode != "off":
        from src.models.t2i_adapter import T2IAdapter
        t2i_adapter = T2IAdapter(in_channels=2, injection_mode=t2i_adapter_mode).to(device)
        print(f"T2I-Adapter ({t2i_adapter_mode}): {sum(p.numel() for p in t2i_adapter.parameters()):,} params")

    if lora_rank > 0:
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipeline.unet.add_adapter(lora_config)
        for p in pipeline.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        lora_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    else:
        lora_params = []

    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    trainable_params = ip_adapter.get_trainable_parameters()
    if t2i_adapter is not None:
        trainable_params.extend(list(t2i_adapter.parameters()))
    if lora_params:
        trainable_params.extend(lora_params)
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {n_trainable:,}")

    # Optimizer
    try:
        from prodigyopt import Prodigy
        optimizer = Prodigy(trainable_params, lr=1.0, weight_decay=0.01)
        print(f"Using Prodigy optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"Using AdamW (lr={lr})")

    use_amp = True
    scaler = torch.amp.GradScaler('cuda') if lora_rank > 0 else None

    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # Resume
    start_step = 0
    if resume_dir is not None:
        resume_dir = Path(resume_dir)
        ip_ckpt = resume_dir / "ip_adapter.pt"
        if ip_ckpt.exists():
            ip_adapter.load_finetuned(resume_dir)
            ip_adapter.masked_self_attn.float()
            ip_adapter.image_projection.float()
            for proc in ip_adapter.attn_processors.values():
                proc.float()
            print(f"  Loaded IP-Adapter from {ip_ckpt}")
        state_path = resume_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                print(f"  WARNING: optimizer state load failed: {e}")
            start_step = state.get("step", 0)
            if t2i_adapter is not None and "t2i_adapter" in state:
                t2i_adapter.load_state_dict(state["t2i_adapter"])
            del state
            torch.cuda.empty_cache()
            print(f"  Resuming from step {start_step}")

    # =========================================
    # Debug training loop
    # =========================================
    effective_steps = min(steps_to_run, n_steps - start_step)
    print(f"\n{'='*70}")
    print(f"DEBUG RUN: {effective_steps} steps from step {start_step}")
    print(f"  SDP mode: {sdp_mode}")
    print(f"  CUDA fence: {cuda_fence}")
    print(f"  DataLoader: workers={dl_num_workers}, pin_memory={dl_pin_memory}, "
          f"persistent={dl_persistent_workers}")
    print(f"{'='*70}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Baseline VRAM after model load: {baseline_reserved:.0f} MB reserved, "
          f"{torch.cuda.memory_allocated()/1024**2:.0f} MB allocated")

    # Open debug CSV
    mem_fh, mem_writer = init_mem_csv(save_dir)

    # Faulthandler: auto-dump Python stack traces if stuck for >120s
    faulthandler.enable()
    fh_dump_file = open(save_dir / "faulthandler_dump.txt", "w")
    faulthandler.dump_traceback_later(120, repeat=True, file=fh_dump_file)
    print(f"  faulthandler: will dump stack to {save_dir / 'faulthandler_dump.txt'} if stuck >120s")

    # DataLoader — configurable for isolation testing
    use_persistent = dl_persistent_workers and dl_num_workers > 0
    loader = DataLoader(
        tracked_dataset, batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=use_persistent,
        drop_last=True,
        timeout=60 if dl_num_workers > 0 else 0,
    )
    data_iter = iter(infinite_loader(loader))

    # Loss file
    loss_file = save_dir / "losses.txt"
    loss_fh = open(loss_file, "w")

    # Batch index log
    idx_log = open(save_dir / "batch_indices.txt", "w")

    # Heartbeat file — shows exactly where a hang occurs
    hb_fh = open(save_dir / "heartbeat.txt", "w")

    losses = []
    spike_count = 0
    pbar = tqdm(range(start_step, start_step + effective_steps),
                desc=f"Debug[{sdp_mode}]", initial=start_step,
                total=start_step + effective_steps)

    for step in pbar:
        heartbeat(hb_fh, step, "before_next_batch", cuda_fence)
        batch = next(data_iter)
        heartbeat(hb_fh, step, "after_next_batch", cuda_fence)

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        references = batch["reference"].to(device)
        batch_types = batch["anomaly_type"]
        batch_captions = batch.get("caption", [""] * batch_size)
        clip_masks = batch["clip_mask"].to(device) if "clip_mask" in batch else None
        batch_indices = batch["_index"].tolist()

        heartbeat(hb_fh, step, "after_to_gpu", cuda_fence)

        # --- Shape assertions ---
        assert images.shape[-2:] == masks.shape[-2:], (
            f"Step {step}: image {images.shape} vs mask {masks.shape}! "
            f"Batch indices: {batch_indices}"
        )
        assert references.shape[-2:] == (224, 224), (
            f"Step {step}: reference shape {references.shape}, expected (..., 224, 224)! "
            f"Batch indices: {batch_indices}"
        )
        if clip_masks is not None:
            assert clip_masks.shape[-2:] == (224, 224), (
                f"Step {step}: clip_mask shape {clip_masks.shape}, expected (..., 224, 224)! "
                f"Batch indices: {batch_indices}"
            )

        # Collect shapes + mask density before forward
        shapes = collect_shapes(images, masks, references, clip_masks)
        mask_density = masks.float().mean().item()

        # Log batch indices
        idx_log.write(f"{step}\t{batch_indices}\n")
        idx_log.flush()

        optimizer.zero_grad()

        heartbeat(hb_fh, step, "before_forward", cuda_fence)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss = compute_anomagic_loss(
                pipeline, ip_adapter, images, masks, references,
                batch_types, batch_captions,
                reference_mode=reference_mode,
                clip_dilation_min_r=clip_dilation_min_r,
                clip_dilation_max_r=clip_dilation_max_r,
                band_mode=band_mode,
                loss_core_ratio=loss_core_ratio,
                binary_cross_attn_mask=binary_cross_attn_mask,
                t2i_adapter=t2i_adapter,
                cond_drop_prob=cond_drop_prob,
                clip_masks=clip_masks,
            )
        heartbeat(hb_fh, step, "after_forward", cuda_fence)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  Step {step}: NaN/Inf loss! Batch indices: {batch_indices}, "
                  f"mask_density: {mask_density:.4f}")
            continue

        heartbeat(hb_fh, step, "before_backward", cuda_fence)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
        heartbeat(hb_fh, step, "after_step", cuda_fence)

        loss_val = loss.item()
        losses.append(loss_val)

        loss_fh.write(f"{loss_val}\n")
        loss_fh.flush()

        # --- VRAM logging + spike trap ---
        reserved_bytes = torch.cuda.memory_reserved()
        reserved_mb = reserved_bytes / 1024**2

        # Log every 50 steps OR on spike
        is_spike = reserved_bytes > spike_threshold_bytes
        if step % 50 == 0 or is_spike:
            log_mem_row(mem_writer, mem_fh, step, shapes, mask_density, loss_val, batch_indices)

        if is_spike:
            spike_count += 1
            spike_trap(step, save_dir, shapes, mask_density, batch_indices)

        # NO per-step empty_cache — only explicit del
        del images, masks, references, clip_masks, loss

        if step % 50 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            pbar.set_postfix(
                loss=f"{avg:.4f}",
                vram=f"{reserved_mb:.0f}MB",
                spikes=spike_count,
            )

        # Checkpoints (keep empty_cache here only)
        if (step + 1) % save_every == 0:
            torch.cuda.empty_cache()
            save_anomagic_checkpoint(
                ip_adapter, optimizer,
                save_dir / f"checkpoint_{step + 1}",
                band_mode=band_mode, t2i_adapter=t2i_adapter,
                unet=pipeline.unet if lora_rank > 0 else None,
                step=step + 1, losses=losses,
            )
            torch.cuda.empty_cache()

    # Cleanup
    loss_fh.close()
    mem_fh.close()
    idx_log.close()
    hb_fh.close()
    fh_dump_file.close()

    # =========================================
    # Summary report
    # =========================================
    print(f"\n{'='*70}")
    print("DEBUG RUN COMPLETE — SUMMARY")
    print(f"{'='*70}")
    print(f"SDP mode: {sdp_mode}")
    print(f"CUDA fence: {cuda_fence}")
    print(f"Steps run: {effective_steps}")
    print(f"Spikes (>{spike_threshold_gb:.0f}GB): {spike_count}")
    print(f"Peak reserved: {torch.cuda.max_memory_reserved()/1024**2:.0f} MB")
    print(f"Peak allocated: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    if losses:
        print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
        print(f"Final avg (last 100): {sum(losses[-100:])/max(len(losses[-100:]),1):.4f}")
    print(f"\nOutput files:")
    print(f"  {save_dir / 'mem_debug.csv'}")
    print(f"  {save_dir / 'batch_indices.txt'}")
    print(f"  {save_dir / 'heartbeat.txt'}")
    print(f"  {save_dir / 'losses.txt'}")
    print(f"  {save_dir / 'crash_env.txt'}")
    if spike_count > 0:
        print(f"  {save_dir / 'mem_summary_step_*.txt'} ({spike_count} files)")
    print(f"\nVERDICT: {'PASS — completed all steps' if len(losses) == effective_steps else f'INCOMPLETE — only {len(losses)}/{effective_steps} steps'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Segfault debug training — same as train_anomagic.py with instrumentation")

    # All standard train_anomagic args
    parser.add_argument("--splits-dir", type=str, default="data/concepts")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--captions-file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="results/debug_vram")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--ip-adapter-type", type=str, default="plus", choices=["standard", "plus"])
    parser.add_argument("--ip-adapter-k", type=int, default=16, choices=[1, 4, 16])
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--reference-mode", type=str, default="full", choices=["full", "crop"])
    parser.add_argument("--types", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-sources", type=str, nargs="+",
                        default=["MANTA_TINY_256", "MulSen_AD", "mvtec3d"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-mask-visual", action="store_true")
    parser.add_argument("--band-mode", type=int, default=1, choices=[1, 2])
    parser.add_argument("--loss-core-ratio", type=float, default=0.8)
    parser.add_argument("--binary-cross-attn-mask", action="store_true")
    parser.add_argument("--t2i-adapter-mode", type=str, default="cascade",
                        choices=["cascade", "skip_only", "off"])
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # Debug-specific args
    parser.add_argument("--spike-threshold-gb", type=float, default=18.0,
                        help="VRAM threshold in GB that triggers spike trap (default 18)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Fixed seed + ordered data for reproducibility")
    parser.add_argument("--sdp-mode", type=str, default="default",
                        choices=["default", "math_only", "no_math", "no_flash", "no_mem_efficient"],
                        help="SDPA backend isolation: subtract one backend to find the crasher")
    parser.add_argument("--cuda-fence", action="store_true",
                        help="Insert torch.cuda.synchronize() after each stage for precise localization")
    parser.add_argument("--max-debug-steps", type=int, default=None,
                        help="Stop after N steps (default: run all --steps)")

    # DataLoader isolation
    parser.add_argument("--dl-num-workers", type=int, default=4,
                        help="DataLoader num_workers (default 4, set 0 for isolation)")
    parser.add_argument("--dl-pin-memory", action="store_true", default=True,
                        help="DataLoader pin_memory (default True)")
    parser.add_argument("--no-dl-pin-memory", dest="dl_pin_memory", action="store_false",
                        help="Disable DataLoader pin_memory")
    parser.add_argument("--dl-persistent-workers", action="store_true", default=True,
                        help="DataLoader persistent_workers (default True)")
    parser.add_argument("--no-dl-persistent-workers", dest="dl_persistent_workers",
                        action="store_false", help="Disable DataLoader persistent_workers")

    # Legacy flag (kept for backwards compat, use --sdp-mode instead)
    parser.add_argument("--disable-math-sdp", action="store_true",
                        help="DEPRECATED: use --sdp-mode flash_only or mem_efficient_only instead")

    args = parser.parse_args()

    # Handle legacy flag
    if args.disable_math_sdp and args.sdp_mode == "default":
        print("WARNING: --disable-math-sdp is deprecated, treating as --sdp-mode no_math")
        args.sdp_mode = "no_math"

    project_root = Path(__file__).parent.parent

    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    captions_file = Path(args.captions_file)
    if not captions_file.is_absolute():
        captions_file = project_root / captions_file

    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir and not resume_dir.is_absolute():
        resume_dir = project_root / resume_dir

    train_debug(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        captions_file=captions_file,
        n_steps=args.steps,
        save_every=args.save_every,
        batch_size=args.batch_size,
        lr=args.lr,
        anomaly_types=args.types,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        device=args.device,
        ip_adapter_type=args.ip_adapter_type,
        ip_adapter_k=args.ip_adapter_k,
        ip_adapter_scale=args.ip_adapter_scale,
        reference_mode=args.reference_mode,
        mask_visual=not args.no_mask_visual,
        clip_dilation_min_r=2,
        clip_dilation_max_r=10,
        band_mode=args.band_mode,
        loss_core_ratio=args.loss_core_ratio,
        binary_cross_attn_mask=args.binary_cross_attn_mask,
        t2i_adapter_mode=args.t2i_adapter_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        cond_drop_prob=args.cond_drop_prob,
        augment=args.augment,
        resume_dir=resume_dir,
        spike_threshold_gb=args.spike_threshold_gb,
        deterministic=args.deterministic,
        sdp_mode=args.sdp_mode,
        cuda_fence=args.cuda_fence,
        max_debug_steps=args.max_debug_steps,
        dl_num_workers=args.dl_num_workers,
        dl_pin_memory=args.dl_pin_memory,
        dl_persistent_workers=args.dl_persistent_workers,
    )
