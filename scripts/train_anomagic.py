"""
Anomagic training script — SD inpainting + IP-Adapter + captions.

Architecture — 2 independent conditioning pathways:
1. Text (captions) → standard UNet cross-attention K/V [frozen]
2. Visual (CLIP ref) → IP-Adapter cross-attention K/V [trained]

Training loop:
1. Sample batch: (image, mask, caption, anomaly_type)
2. Prepare reference image (full downscale or anomaly crop)
3. Encode reference: ip_adapter.encode_image(ref) → [B, K, 768]
4. Encode caption: pipeline.encode_text(caption) → [B, 77, 768]
5. Create inpainting inputs: noisy_latents + mask_latents + masked_image_latents
6. UNet forward with 2 pathways via cross_attention_kwargs
7. Masked noise prediction loss
8. Backprop through IP-Adapter params only
"""
import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
from src.inference.generate import generate_anomagic_single


def train_anomagic(
    splits_dir: Path,
    save_dir: Path,
    captions_file: Path,
    n_steps: int = 10000,
    batch_size: int = 16,
    lr: float = 1e-4,
    lr_pretrained: float = 5e-5,
    lambda_sp: float = 1e-4,
    device: str = "cuda",
    save_every: int = 1000,
    anomaly_types: list = None,
    exclude_sources: list = None,
    data_root: Path = None,
    ip_adapter_type: str = "plus",
    ip_adapter_k: int = 4,
    ip_adapter_scale: float = 1.0,
    reference_mode: str = "full",
    mask_visual: bool = True,
    # CLIP dilation (transition zone around anomaly for visual tokens)
    clip_dilation_min_r: int = 2,
    clip_dilation_max_r: int = 10,
    # Latent-space band dilation
    band_mode: int = 1,
    # Loss weighting
    loss_core_ratio: float = 0.8,
    # Cross-attention mask type
    binary_cross_attn_mask: bool = False,
    # T2I-Adapter mode
    t2i_adapter_mode: str = "cascade",
    # LoRA on UNet
    lora_rank: int = 0,
    lora_alpha: int = 16,
    # Conditioning dropout for CFG (mutually exclusive per-sample)
    drop_image_prob: float = 0.10,
    drop_text_prob: float = 0.10,
    drop_both_prob: float = 0.05,
    # Data augmentation
    augment: bool = False,
    # Resume from checkpoint
    resume_dir: Path = None,
    # Visual token processing mode
    visual_mode: int = 0,
    # Gate mode for masked self-attention
    learnable_gates: bool = True,
    # Force gates to 1.0 with normal projection init (block active from step 0)
    force_gates: bool = False,
    # Number of self-attention transformer layers
    sa_num_layers: int = 1,
    # Number of attention heads
    sa_num_heads: int = 12,
    # Multi-crop (2-group CLIP cropping)
    multi_crop: bool = False,
    # CFG direction for sample generation
    cfg_mode: str = "text",
    # Headless mode (no live viewer)
    no_live_viewer: bool = False,
    # Noise offset (helps with global brightness/darkness)
    noise_offset: float = 0.05,
    # Timestep sampling strategy
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    # Optimizer
    optimizer_type: str = "adamw",
    # Context corruption: zero masked_image_latents with this probability
    corrupt_context: float = 0.0,
    # x0-prediction pseudo-diffusion ablation
    x0_objective: bool = False,
    x0_start_ratio: float = 1.0,
    x0_end_ratio: float = 1.0,
    x0_no_context: bool = False,
    x0_warmup_frac: float = 0.2,
):
    """Train Anomagic: IP-Adapter + captions (2 pathways)."""
    # Seed everything for reproducibility across ablation runs
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANOMAGIC TRAINING — IP-Adapter + Captions")
    print("=" * 70)
    print(f"Learning rate (scratch): {lr}")
    print(f"Learning rate (pretrained): {lr_pretrained}")
    print(f"L2-SP lambda: {lambda_sp}")
    print(f"Batch size: {batch_size}")
    print(f"Training steps: {n_steps}")
    print(f"IP-Adapter type: {ip_adapter_type} (K={ip_adapter_k})")
    print(f"IP-Adapter scale: {ip_adapter_scale}")
    print(f"Reference mode: {reference_mode}")
    print(f"Captions file: {captions_file}")
    print(f"Masked visual cross-attention: {mask_visual}")
    print(f"CLIP dilation: min_r={clip_dilation_min_r}, max_r={clip_dilation_max_r}")
    print(f"Band mode: {band_mode} (Chebyshev latent-space dilation)")
    print(f"Loss ratio: {loss_core_ratio:.0%} core / {1-loss_core_ratio:.0%} band")
    print(f"Cross-attn mask: {'binary' if binary_cross_attn_mask else 'soft alpha'}")
    print(f"T2I-Adapter: {t2i_adapter_mode}")
    print(f"LoRA: {'off' if lora_rank == 0 else f'rank={lora_rank}, alpha={lora_alpha}'}")
    print(f"Conditioning dropout: image={drop_image_prob:.0%}, text={drop_text_prob:.0%}, both={drop_both_prob:.0%}")
    print(f"Data augmentation: {augment}")
    print(f"Visual mode: {visual_mode}")
    print(f"Learnable gates: {learnable_gates} (SA layers: {sa_num_layers}, heads: {sa_num_heads})")
    print(f"Multi-crop: {multi_crop}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Noise offset: {noise_offset}")
    print(f"Timestep sampling: {timestep_sampling}" +
          (f" (mean={logit_normal_mean}, std={logit_normal_std})" if timestep_sampling == "logit_normal" else ""))
    if x0_objective:
        warmup_steps = int(n_steps * x0_warmup_frac)
        print(f"x0-objective: ratio {x0_start_ratio:.0%}->{x0_end_ratio:.0%}, "
              f"warmup={warmup_steps} steps ({x0_warmup_frac:.0%}), no_context={x0_no_context}")
    if resume_dir:
        print(f"Resume from: {resume_dir}")
    print()

    # =========================================
    # Load Data
    # =========================================
    dataset = AnomalyDataset(
        splits_dir,
        anomaly_types=anomaly_types,
        exclude_sources=exclude_sources,
        data_root=data_root,
        captions_file=captions_file,
        return_reference=True,
        reference_mode=reference_mode,
        reference_crop_size=224,
        augment=augment,
        multi_crop=multi_crop,
    )

    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return

    n_captions = sum(1 for s in dataset.samples if s.get("caption"))
    print(f"Samples with captions: {n_captions}/{len(dataset.samples)}")
    if n_captions == 0:
        print("WARNING: No captions found! Check --captions-file path.")

    # =========================================
    # Initialize Pipeline
    # =========================================
    print("\nInitializing models...")

    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()

    # Freeze everything on the pipeline (UNet, VAE, text encoder + embeddings)
    pipeline.freeze_all()

    # =========================================
    # Initialize IP-Adapter
    # =========================================
    print(f"\nLoading IP-Adapter ({ip_adapter_type}, K={ip_adapter_k}, visual_mode={visual_mode})...")
    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=ip_adapter_type,
        num_tokens=ip_adapter_k,
        scale=ip_adapter_scale,
        load_pretrained=True,
        mask_visual=mask_visual,
        visual_mode=visual_mode,
        learnable_gates=learnable_gates,
        force_gates=force_gates,
        sa_num_layers=sa_num_layers,
        sa_num_heads=sa_num_heads,
    )
    ip_adapter.freeze_image_encoder()

    # =========================================
    # Initialize T2I-Adapter (optional spatial conditioning)
    # =========================================
    t2i_adapter = None
    if t2i_adapter_mode != "off":
        from src.models.t2i_adapter import T2IAdapter
        t2i_adapter = T2IAdapter(in_channels=2, injection_mode=t2i_adapter_mode).to(device)
        n_t2i = sum(p.numel() for p in t2i_adapter.parameters())
        print(f"\nT2I-Adapter ({t2i_adapter_mode}): {n_t2i:,} params")

    # =========================================
    # Initialize LoRA on UNet (optional)
    # =========================================
    if lora_rank > 0:
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipeline.unet.add_adapter(lora_config)
        # Cast LoRA params to fp32 (they inherit fp16 from UNet base weights)
        for p in pipeline.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        # Exclude IP-Adapter attn_processor params (already in Group A)
        ip_proc_ids = {id(p) for proc in ip_adapter.attn_processors.values() for p in proc.parameters()}
        lora_params = [p for p in pipeline.unet.parameters() if p.requires_grad and id(p) not in ip_proc_ids]
        n_lora = sum(p.numel() for p in lora_params)
        print(f"\nLoRA (rank={lora_rank}, alpha={lora_alpha}): {n_lora:,} params")
    else:
        lora_params = []

    # Ensure all trainable modules are fp32 (pretrained weights may load as fp16)
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # =========================================
    # Build Optimizer Parameter Groups
    # =========================================
    # Group A (pretrained): image_projection + attn_processors  → low LR, L2-SP
    # Group B (scratch):    masked_self_attn + t2i_adapter       → higher LR, weight decay
    # Group C (gates/scales): gate scalars + t2i scales          → higher LR, no weight decay

    group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
    group_b_modules = []
    if hasattr(ip_adapter, 'masked_self_attn'):
        group_b_modules.append(ip_adapter.masked_self_attn)
    if t2i_adapter is not None:
        group_b_modules.append(t2i_adapter)

    norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)

    # --- Group A: pretrained IP-Adapter ---
    a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

    # --- Group C: gate/scale scalars (extract BEFORE splitting B) ---
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

    # --- Group B: from-scratch modules (minus Group C params) ---
    b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
    b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
    b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

    # Build param groups
    param_groups = [
        {"params": a_decay,     "lr": lr_pretrained, "weight_decay": 0.0, "label": "A_decay"},
        {"params": a_no_decay,  "lr": lr_pretrained, "weight_decay": 0.0, "label": "A_no_decay"},
        {"params": b_decay,     "lr": lr,            "weight_decay": 1e-3, "label": "B_decay"},
        {"params": b_no_decay,  "lr": lr,            "weight_decay": 0.0, "label": "B_no_decay"},
        {"params": group_c_params, "lr": lr,         "weight_decay": 0.0, "label": "C_gates"},
    ]

    # Optional Group D: LoRA
    if lora_params:
        param_groups.append(
            {"params": lora_params, "lr": lr, "weight_decay": 1e-3, "label": "D_lora"}
        )

    # --- Verify: no duplicate params across groups ---
    all_group_ids = []
    for pg in param_groups:
        for p in pg["params"]:
            pid = id(p)
            assert pid not in all_group_ids, f"Duplicate param in group {pg['label']}"
            all_group_ids.append(pid)

    # --- Verify: total param count matches expected ---
    # Skip when LoRA is active — peft restructures UNet module hierarchy,
    # making independent param counting unreliable. Duplicate check above
    # is the primary safety net.
    if not lora_params:
        group_numel = sum(p.numel() for pg in param_groups for p in pg["params"])
        expected_params = ip_adapter.get_trainable_parameters()
        if t2i_adapter is not None:
            expected_params.extend(list(t2i_adapter.parameters()))
        expected_numel = sum(p.numel() for p in expected_params)
        assert group_numel == expected_numel, (
            f"Param count mismatch: groups={group_numel:,} vs expected={expected_numel:,}"
        )

    trainable_params = [p for pg in param_groups for p in pg["params"]]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n  Trainable parameters: {n_trainable:,}")

    # Per-group diagnostic
    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"    {pg['label']:12s}: {n:>10,} params  (lr={pg['lr']:.1e}, wd={pg['weight_decay']:.1e})")

    if optimizer_type == "prodigy":
        from prodigyopt import Prodigy
        # Prodigy auto-tunes LR: set lr=1.0 for all groups.
        # Zero out all per-group weight_decay — Prodigy handles its own regularization.
        for pg in param_groups:
            pg["lr"] = 1.0
            pg["weight_decay"] = 0.0
        optimizer = Prodigy(
            param_groups,
            lr=1.0,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            safeguard_warmup=True,
            use_bias_correction=True,
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # =========================================
    # L2-SP Regularizer (anchored to pretrained weights, before resume)
    # =========================================
    l2sp_reg = None
    if lambda_sp > 0 and len(a_decay) > 0:
        l2sp_reg = L2SPRegularizer(a_decay, lambda_sp=lambda_sp)
        print(f"\n  L2-SP: {l2sp_reg.total_elements:,} elements, lambda={lambda_sp:.1e}")

    # AMP autocast handles fp16 forward pass; trainable params stay fp32.
    # GradScaler needed when LoRA is active: LoRA params are fp32 but their
    # gradients flow through fp16 UNet intermediates, which can overflow.
    # Without LoRA, all trainable params (IP-Adapter) receive gradients through
    # fp32 projections, so scaler is unnecessary.
    use_amp = True
    scaler = torch.amp.GradScaler('cuda') if lora_rank > 0 else None

    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # =========================================
    # Resume from checkpoint (if requested)
    # =========================================
    start_step = 0
    resumed_losses = []

    if resume_dir is not None:
        resume_dir = Path(resume_dir)
        print(f"\nResuming from checkpoint: {resume_dir}")

        # Load IP-Adapter weights
        ip_ckpt = resume_dir / "ip_adapter.pt"
        if ip_ckpt.exists():
            ip_adapter.load_finetuned(resume_dir)
            # Re-cast to fp32 after loading (weights may have been saved as fp16)
            ip_adapter.masked_self_attn.float()
            ip_adapter.image_projection.float()
            for proc in ip_adapter.attn_processors.values():
                proc.float()
            print(f"  Loaded IP-Adapter weights from {ip_ckpt}")
        else:
            print(f"  WARNING: {ip_ckpt} not found, starting with pretrained weights")

        # Load training state (optimizer, step, T2I, LoRA)
        state_path = resume_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)

            # Optimizer
            try:
                optimizer.load_state_dict(state["optimizer"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  WARNING: Could not load optimizer state ({e}), using fresh optimizer")

            # Step counter
            start_step = state.get("step", 0)
            print(f"  Resuming from step {start_step}")

            # Previous losses (for continuity in loss plot)
            resumed_losses = state.get("losses", [])
            if resumed_losses:
                print(f"  Loaded {len(resumed_losses)} previous loss values")

            # T2I-Adapter
            if t2i_adapter is not None and "t2i_adapter" in state:
                t2i_adapter.load_state_dict(state["t2i_adapter"])
                print(f"  Loaded T2I-Adapter weights")

            # LoRA
            if lora_rank > 0 and "lora_weights" in state:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(pipeline.unet, state["lora_weights"])
                # Re-cast LoRA params to fp32
                for p in pipeline.unet.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
                print(f"  Loaded LoRA weights")

            del state
            torch.cuda.empty_cache()
        else:
            print(f"  WARNING: {state_path} not found, starting fresh (IP-Adapter weights loaded)")

    remaining_steps = n_steps - start_step
    if remaining_steps <= 0:
        print(f"\nAlready completed {start_step}/{n_steps} steps. Nothing to do.")
        return

    # =========================================
    # Step-0 baseline samples (pretrained, before any training)
    # =========================================
    if start_step == 0:
        print("\nGenerating step-0 baseline samples (pretrained IP-Adapter, no training)...")
        torch.cuda.empty_cache()
        try:
            generate_anomagic_samples(
                pipeline, ip_adapter, dataset,
                save_dir / "samples_0_pretrained.png", device,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                cfg_mode=cfg_mode,
            )
            print(f"  Saved to {save_dir / 'samples_0_pretrained.png'}")
        except RuntimeError as e:
            print(f"  Warning: baseline sample generation failed ({e})")
        torch.cuda.empty_cache()

    # =========================================
    # Training Loop
    # =========================================
    print(f"\nStarting training for {remaining_steps} remaining steps ({start_step}/{n_steps} done)...")

    # Save run config for live viewer title
    import json
    config_file = save_dir / "run_config.json"
    run_config = {
        "visual_mode": visual_mode,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
        "lr_pretrained": lr_pretrained,
        "lr": lr,
        "drop_image_prob": drop_image_prob,
        "drop_text_prob": drop_text_prob,
        "drop_both_prob": drop_both_prob,
        "band_mode": band_mode,
        "noise_offset": noise_offset,
        "timestep_sampling": timestep_sampling,
        "sa_num_layers": sa_num_layers,
        "sa_num_heads": sa_num_heads,
        "learnable_gates": learnable_gates,
        "force_gates": force_gates,
        "multi_crop": multi_crop,
        "ip_adapter_type": ip_adapter_type,
        "ip_adapter_k": ip_adapter_k,
        "optimizer": optimizer_type,
        "corrupt_context": corrupt_context,
        "x0_objective": x0_objective,
        "x0_start_ratio": x0_start_ratio,
        "x0_end_ratio": x0_end_ratio,
        "x0_no_context": x0_no_context,
        "x0_warmup_frac": x0_warmup_frac,
    }
    with open(config_file, "w") as cf:
        json.dump(run_config, cf, indent=2)

    # Launch live loss viewer
    loss_file = save_dir / "losses.txt"
    # Pre-populate loss file with resumed losses so live viewer shows full history
    if resumed_losses:
        with open(loss_file, "w") as lf:
            lf.writelines(f"{v}\n" for v in resumed_losses)
    if not no_live_viewer:
        _launch_live_loss(loss_file)

    losses = resumed_losses
    type_losses = defaultdict(list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=False, persistent_workers=True,
                        drop_last=True)
    data_iter = iter(infinite_loader(loader))

    # Keep loss file open for the entire run (avoids Windows file locking issues)
    loss_fh = open(loss_file, "a")

    # Stats file: gate values + LR per step (CSV)
    stats_file = save_dir / "stats.csv"
    stats_fh = open(stats_file, "a")
    if stats_fh.tell() == 0:
        stats_fh.write("step,loss,lr_pretrained,lr_scratch,attn_gate,ff_gate,l2sp,core_loss,band_loss,x0_ratio,x0_loss,eps_loss,grad_norm\n")

    pbar = tqdm(range(start_step, n_steps), desc="Training", initial=start_step, total=n_steps)
    skipped_nan = 0

    for step in pbar:
        batch = next(data_iter)

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        references = batch["reference"].to(device)
        batch_types = batch["anomaly_type"]
        batch_captions = batch.get("caption", [""] * batch_size)
        # Augmented CLIP mask (from crop_utils) — None when augment=False
        clip_masks = batch["clip_mask"].to(device) if "clip_mask" in batch else None
        # Multi-crop data (None when multi_crop=False)
        references_2 = batch["reference_2"].to(device) if "reference_2" in batch else None
        clip_masks_2 = batch["clip_mask_2"].to(device) if "clip_mask_2" in batch else None
        group_valid = batch["group_valid"].to(device) if "group_valid" in batch else None

        optimizer.zero_grad()

        # x0-objective annealing (with warmup)
        if x0_objective:
            warmup_steps = int(n_steps * x0_warmup_frac)
            if step < warmup_steps:
                x0_ratio = x0_start_ratio
            else:
                anneal_progress = (step - warmup_steps) / max(n_steps - warmup_steps - 1, 1)
                x0_ratio = x0_start_ratio + (x0_end_ratio - x0_start_ratio) * anneal_progress
        else:
            x0_ratio = 0.0

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss_diff, loss_extras = compute_anomagic_loss(
                pipeline, ip_adapter, images, masks, references,
                batch_types, batch_captions,
                reference_mode=reference_mode,
                clip_dilation_min_r=clip_dilation_min_r,
                clip_dilation_max_r=clip_dilation_max_r,
                band_mode=band_mode,
                loss_core_ratio=loss_core_ratio,
                binary_cross_attn_mask=binary_cross_attn_mask,
                t2i_adapter=t2i_adapter,
                drop_image_prob=drop_image_prob,
                drop_text_prob=drop_text_prob,
                drop_both_prob=drop_both_prob,
                clip_masks=clip_masks,
                references_2=references_2,
                clip_masks_2=clip_masks_2,
                group_valid=group_valid,
                noise_offset=noise_offset,
                timestep_sampling=timestep_sampling,
                logit_normal_mean=logit_normal_mean,
                logit_normal_std=logit_normal_std,
                corrupt_context=corrupt_context,
                x0_ratio=x0_ratio,
                x0_no_context=x0_no_context,
            )

        # L2-SP regularization (fp32, outside autocast)
        l2sp_val = 0.0
        if l2sp_reg is not None:
            l2sp_loss = l2sp_reg.compute()
            l2sp_val = l2sp_loss.item()
            loss = loss_diff + l2sp_loss
        else:
            loss = loss_diff

        if torch.isnan(loss) or torch.isinf(loss):
            skipped_nan += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()

        loss_val = loss_diff.item()
        losses.append(loss_val)

        # Append to loss file for live viewer
        loss_fh.write(f"{loss_val}\n")
        loss_fh.flush()

        # Log stats every step
        attn_gate = ff_gate = 0.0
        if hasattr(ip_adapter, 'masked_self_attn'):
            sa = ip_adapter.masked_self_attn
            if sa.learnable_gates:
                for layer in sa.layers:
                    attn_gate = layer["attn_gate"].gate.item()
                    if "ff_gate" in layer:
                        ff_gate = layer["ff_gate"].gate.item()
                    break  # first layer only
            else:
                attn_gate = ff_gate = 1.0  # fixed gates
        lr_pre = optimizer.param_groups[0]["lr"]
        lr_scr = optimizer.param_groups[2]["lr"]  # B_decay group
        core_l = loss_extras.get("core_loss", 0.0)
        band_l = loss_extras.get("band_loss", 0.0)
        x0_l = loss_extras.get("x0_loss", 0.0)
        eps_l = loss_extras.get("eps_loss", 0.0)
        stats_fh.write(f"{step},{loss_val},{lr_pre},{lr_scr},{attn_gate},{ff_gate},{l2sp_val},{core_l},{band_l},{x0_ratio},{x0_l},{eps_l},{grad_norm}\n")
        stats_fh.flush()

        for t in batch_types:
            type_losses[t].append(loss_val)

        if step % 50 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            pbar.set_postfix(loss=f"{avg:.4f}")

        # Free unreferenced CUDA tensors every step to prevent VRAM creep
        del images, masks, references, clip_masks, references_2, clip_masks_2, group_valid, loss, loss_diff, loss_extras
        #torch.cuda.empty_cache()

        # Save checkpoints + samples
        if (step + 1) % save_every == 0:
            print(f"\n  Checkpoint at step {step + 1}...", flush=True)
            save_anomagic_checkpoint(
                ip_adapter, optimizer,
                save_dir / f"checkpoint_{step + 1}",
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                unet=pipeline.unet if lora_rank > 0 else None,
                step=step + 1,
                losses=losses,
            )
            torch.cuda.empty_cache()
            try:
                generate_anomagic_samples(
                    pipeline, ip_adapter, dataset,
                    save_dir / f"samples_{step + 1}.png", device,
                    band_mode=band_mode,
                    t2i_adapter=t2i_adapter,
                    cfg_mode=cfg_mode,
                )
            except RuntimeError as e:
                print(f"  Warning: sample generation failed ({e}), continuing training...")
            torch.cuda.empty_cache()
            save_loss_plot(losses, save_dir / "stats.csv",
                           save_dir / f"loss_{step + 1}.png")

    loss_fh.close()

    # =========================================
    # Final Outputs
    # =========================================
    print("\nTraining complete!")
    print(f"  Total steps: {n_steps} (resumed from {start_step})" if start_step > 0 else f"  Total steps: {n_steps}")
    print(f"  Skipped (NaN/Inf): {skipped_nan}")
    print(f"  Valid losses recorded: {len(losses)}")

    # Final loss plot (same 2x3 layout as checkpoints)
    save_loss_plot(losses, save_dir / "stats.csv", save_dir / "training_loss.png")

    # Final checkpoint
    save_anomagic_checkpoint(
        ip_adapter, optimizer,
        save_dir / "checkpoint_final",
        band_mode=band_mode,
        t2i_adapter=t2i_adapter,
        unet=pipeline.unet if lora_rank > 0 else None,
        step=n_steps,
        losses=losses,
    )

    # Final sample grid
    torch.cuda.empty_cache()
    try:
        generate_anomagic_samples(
            pipeline, ip_adapter, dataset,
            save_dir / "final_samples.png", device,
            band_mode=band_mode,
            t2i_adapter=t2i_adapter,
            cfg_mode=cfg_mode,
        )
    except RuntimeError as e:
        print(f"  Warning: final samples failed ({e})")

    print(f"\nResults saved to: {save_dir}")


def compute_anomagic_loss(
    pipeline, ip_adapter, images, masks, references,
    batch_types, captions,
    reference_mode: str = "full",
    clip_dilation_min_r: int = 2,
    clip_dilation_max_r: int = 10,
    band_mode: int = 1,
    loss_core_ratio: float = 0.8,
    binary_cross_attn_mask: bool = False,
    t2i_adapter=None,
    drop_image_prob: float = 0.10,
    drop_text_prob: float = 0.10,
    drop_both_prob: float = 0.05,
    clip_masks=None,
    # Multi-crop (Mode 4)
    references_2=None,
    clip_masks_2=None,
    group_valid=None,
    # Noise offset
    noise_offset: float = 0.0,
    # Timestep sampling
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    # Context corruption
    corrupt_context: float = 0.0,
    # x0-prediction pseudo-diffusion ablation
    x0_ratio: float = 0.0,
    x0_no_context: bool = False,
):
    """
    Compute masked diffusion loss with 2 conditioning pathways.

    Pathway 1 (text):    caption → frozen UNet cross-attn K/V
    Pathway 2 (visual):  reference image → CLIP → IP-Adapter K/V [trained]

    Multi-crop: when references_2 is provided, encode both crops independently
    through the SAME IP-Adapter, concatenate embeddings → [B, 2K, dim], and
    build null_token_mask to suppress invalid group tokens.

    Mask strategy:
    - CLIP mask: tight image-space dilation for anomaly-focused feature extraction
    - UNet masks: latent-space band dilation (Chebyshev, 1-2 pixels at 64×64)
    - Cross-attn: soft alpha_map (default) or binary dilated mask
    - Loss weight: flat alpha + ratio scaling (no scipy)
    """
    batch_size = images.shape[0]
    device = images.device

    images = images.float()
    masks_fp32 = masks.float()

    # --- CLIP mask ---
    # When augmented: clip_masks come from crop_utils (already cropped to 224×224)
    # When non-augmented: dilate the full-res mask for masked self-attention
    if clip_masks is None:
        clip_dilated = dilate_mask_batch(masks_fp32, clip_dilation_min_r, clip_dilation_max_r)
    else:
        clip_dilated = None  # not needed — clip_masks used directly

    # Encode image to latent space
    latents = pipeline.encode_image(images)

    # Timestep sampling
    if timestep_sampling == "logit_normal":
        # SD3-style: sample from logit-normal distribution (bell-shaped, mid-range focus)
        u = torch.sigmoid(logit_normal_mean + logit_normal_std * torch.randn(batch_size, device=device))
        timesteps = (u * 1000).long().clamp(0, 999)
    else:
        # Uniform (standard DDPM)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)

    noise = torch.randn_like(latents)
    if noise_offset > 0:
        # Per-channel constant offset: helps model handle global brightness shifts
        noise += noise_offset * torch.randn(
            latents.shape[0], latents.shape[1], 1, 1, device=device, dtype=latents.dtype
        )
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

    # --- UNet masks: maxpool to 64×64, then latent-space band dilation ---
    latent_h = latents.shape[-1]
    kernel = masks_fp32.shape[-1] // latent_h
    core_mask_64 = F.max_pool2d(masks_fp32, kernel_size=kernel)
    core_mask_64 = (core_mask_64 > 0.5).float()

    dilated_binary_64, alpha_map_64, weight_map_64, band_mask_64 = create_latent_band_mask(
        core_mask_64, band_mode, core_ratio=loss_core_ratio,
    )

    # --- Pathway 1: Text conditioning (captions only, no TI tokens) ---
    prompts = []
    for i, atype in enumerate(batch_types):
        cap = captions[i] if i < len(captions) else ""
        if cap:
            prompts.append(cap)
        else:
            type_word = atype.replace("_", " ")
            prompts.append(f"a photo of a {type_word} defect")

    text_emb = pipeline.encode_text(prompts, enable_grad=False)

    # --- Pathway 2: Visual conditioning via IP-Adapter ---
    ref_01 = (references + 1.0) / 2.0
    if clip_masks is not None:
        # Augmented: clip_masks are [B, 1, 224, 224] from crop_utils, already aligned to reference
        clip_mask = clip_masks
    else:
        clip_mask = clip_dilated if reference_mode == "full" else None
    ip_image_embeds = ip_adapter.encode_image(ref_01, mask=clip_mask)  # [B, K, cross_attn_dim]

    # --- Multi-crop: encode second crop and concatenate ---
    null_token_mask = None
    if references_2 is not None:
        ref_2_01 = (references_2 + 1.0) / 2.0
        clip_mask_2 = clip_masks_2 if clip_masks_2 is not None else None
        ip_image_embeds_2 = ip_adapter.encode_image(ref_2_01, mask=clip_mask_2)  # [B, K, dim]

        # Concatenate: [B, K, dim] + [B, K, dim] → [B, 2K, dim]
        K = ip_image_embeds.shape[1]
        ip_image_embeds = torch.cat([ip_image_embeds, ip_image_embeds_2], dim=1)  # [B, 2K, dim]

        # Build null_token_mask [B, 2K]: 1=valid, 0=null (from invalid groups)
        if group_valid is not None:
            # group_valid [B, 2]: validity per group
            # Expand: group 1 valid → first K tokens valid, group 2 valid → last K tokens valid
            mask_1 = group_valid[:, 0:1].expand(-1, K)  # [B, K]
            mask_2 = group_valid[:, 1:2].expand(-1, K)  # [B, K]
            null_token_mask = torch.cat([mask_1, mask_2], dim=1)  # [B, 2K]

    # --- Conditioning dropout for CFG (mutually exclusive, per-sample) ---
    # Matches IP-Adapter training: single rand(), cumulative thresholds.
    # Image drop = zeros on ip_image_embeds. Text drop = empty-string encoding.
    if (drop_image_prob + drop_text_prob + drop_both_prob) > 0 and batch_size > 0:
        uncond_text = pipeline.encode_text([""] * batch_size, enable_grad=False)
        for i in range(batch_size):
            r = random.random()
            if r < drop_image_prob:
                ip_image_embeds[i] = 0.0
            elif r < drop_image_prob + drop_text_prob:
                text_emb[i] = uncond_text[i]
            elif r < drop_image_prob + drop_text_prob + drop_both_prob:
                ip_image_embeds[i] = 0.0
                text_emb[i] = uncond_text[i]

    # --- Inpainting inputs — latent-space dilated mask defines regeneration region ---
    mask_latents = dilated_binary_64  # binary [B, 1, 64, 64]
    unet_mask_512 = F.interpolate(dilated_binary_64, size=images.shape[-2:], mode='nearest')
    masked_image = images * (1 - unet_mask_512)
    masked_image_latents = pipeline.encode_image(masked_image)

    # --- x0-objective: replace noisy input for pseudo-path samples ---
    x0_samples = None
    if x0_ratio > 0:
        x0_samples = torch.rand(batch_size, device=device) < x0_ratio  # [B] bool
        if x0_samples.any():
            noisy_pseudo = pipeline.scheduler.add_noise(masked_image_latents, noise, timesteps)
            noisy_latents = torch.where(
                x0_samples[:, None, None, None].expand_as(noisy_latents),
                noisy_pseudo, noisy_latents,
            )
            # Optionally zero inpainting context for x0-path samples only
            if x0_no_context:
                masked_image_latents = torch.where(
                    x0_samples[:, None, None, None].expand_as(masked_image_latents),
                    torch.zeros_like(masked_image_latents),
                    masked_image_latents,
                )
        else:
            x0_samples = None  # no samples selected, skip x0 logic in loss

    # Context corruption: zero out masked_image_latents to force cross-attention usage
    if corrupt_context > 0.0 and random.random() < corrupt_context:
        masked_image_latents = torch.zeros_like(masked_image_latents)

    model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

    # --- T2I-Adapter spatial features (AMP autocast handles dtype) ---
    t2i_kwargs = {}
    if t2i_adapter is not None:
        t2i_input = torch.cat([core_mask_64, band_mask_64], dim=1)  # [B, 2, 64, 64]
        t2i_features = t2i_adapter(t2i_input, mask=dilated_binary_64)
        t2i_kwargs = t2i_adapter.prepare_unet_kwargs(t2i_features)

    # --- UNet forward with both pathways via cross_attention_kwargs ---
    cross_attn_kwargs = {"ip_adapter_image_embeds": ip_image_embeds}
    # Cross-attn mask: soft alpha (default) or binary
    if binary_cross_attn_mask:
        cross_attn_kwargs["ip_adapter_mask"] = dilated_binary_64
    else:
        cross_attn_kwargs["ip_adapter_mask"] = alpha_map_64
    # Multi-crop null masking
    if null_token_mask is not None:
        cross_attn_kwargs["null_token_mask"] = null_token_mask

    noise_pred = pipeline.unet(
        model_input,
        timesteps,
        encoder_hidden_states=text_emb,
        cross_attention_kwargs=cross_attn_kwargs,
        **t2i_kwargs,
    ).sample.float()

    # --- Ratio-based loss weighting (flat alpha, no scipy) ---
    weight_expanded = weight_map_64.expand_as(noise_pred)

    if x0_samples is not None and x0_samples.any():
        # Mixed batch: x0-space loss for pseudo-path, ε-space for standard
        alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=device)
        alpha_t = alphas_cumprod[timesteps]  # [B]
        sqrt_alpha = alpha_t.sqrt()[:, None, None, None]
        sqrt_one_minus_alpha = (1 - alpha_t).sqrt()[:, None, None, None]

        # x0 prediction from ε prediction
        x0_pred = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        x0_diff = (x0_pred - latents.float()) ** 2
        eps_diff = (noise_pred - noise.float()) ** 2

        # Select per sample
        x0_sel = x0_samples[:, None, None, None].expand_as(noise_pred)
        diff = torch.where(x0_sel, x0_diff, eps_diff)
    else:
        diff = (noise_pred - noise.float()) ** 2

    weighted_diff = diff * weight_expanded

    weight_sum = weight_expanded.sum()
    if weight_sum < 1e-8:
        return torch.tensor(0.0, device=device, requires_grad=True), {}
    loss = weighted_diff.sum() / weight_sum

    # --- Diagnostics: core vs band loss (detached, no grad impact) ---
    with torch.no_grad():
        core_expanded = core_mask_64.expand_as(noise_pred)
        band_expanded = band_mask_64.expand_as(noise_pred)
        core_n = core_expanded.sum().clamp(min=1)
        band_n = band_expanded.sum().clamp(min=1)
        core_loss_val = (diff * core_expanded).sum() / core_n
        band_loss_val = (diff * band_expanded).sum() / band_n

        # x0 vs eps per-objective loss (only when mixed batch)
        x0_loss_val = 0.0
        eps_loss_val = 0.0
        if x0_samples is not None and x0_samples.any():
            x0_sel_4d = x0_samples[:, None, None, None].expand_as(noise_pred)
            eps_sel_4d = ~x0_sel_4d
            x0_w = (weight_expanded * x0_sel_4d.float())
            eps_w = (weight_expanded * eps_sel_4d.float())
            x0_w_sum = x0_w.sum().clamp(min=1e-8)
            eps_w_sum = eps_w.sum().clamp(min=1e-8)
            x0_loss_val = (x0_diff * x0_w).sum() / x0_w_sum
            eps_loss_val = (eps_diff * eps_w).sum() / eps_w_sum
            x0_loss_val = x0_loss_val.item()
            eps_loss_val = eps_loss_val.item()

    return loss, {
        "core_loss": core_loss_val.item(),
        "band_loss": band_loss_val.item(),
        "x0_loss": x0_loss_val,
        "eps_loss": eps_loss_val,
    }


def _ema_smooth(values, smoothing: float = 0.99):
    """Exponential moving average."""
    arr = np.array(values, dtype=np.float64)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def save_loss_plot(losses: list, stats_file: Path, save_path: Path):
    """Render 2x3 training diagnostics plot. Two layouts:
    - Default: trend, core/band, band/core ratio, gates, grad norm, (empty)
    - x0 ablation: trend, core/band, x0 vs eps + annealing, gates, grad norm, band/core ratio
    """
    if len(losses) < 2:
        return

    # --- Parse stats.csv ---
    stats_steps, diff_losses = [], []
    attn_gates, ff_gates, l2sp_vals = [], [], []
    core_losses, band_losses = [], []
    x0_losses, eps_losses, x0_ratios, grad_norms = [], [], [], []
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            cols = header.split(",")
            n_cols = len(cols)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                stats_steps.append(int(parts[0]))
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
    except FileNotFoundError:
        pass

    s_steps = np.array(stats_steps)
    s_attn = np.array(attn_gates)
    s_ff = np.array(ff_gates)
    s_core = np.array(core_losses)
    s_band = np.array(band_losses)
    s_x0 = np.array(x0_losses)
    s_eps = np.array(eps_losses)
    s_x0r = np.array(x0_ratios)
    s_gn = np.array(grad_norms)

    SM = 0.99
    SF = 0.6

    steps = np.arange(len(losses))
    arr = np.array(losses)
    ema_fast = _ema_smooth(arr, SF)
    ema_slow = _ema_smooth(arr, SM)

    has_cb = len(s_steps) > 0 and len(s_core) > 0 and s_core.any()
    has_x0 = len(s_steps) > 0 and s_x0.any()
    has_gn = len(s_steps) > 0 and s_gn.any()

    # --- Precompute core/band decomposition (shared by both layouts) ---
    ce = be = core_ps = band_ps = None
    if has_cb:
        w_core_s = 0.8 * s_core
        w_band_s = 0.2 * s_band
        w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
        core_share = w_core_s / w_total_s
        core_share_ps = np.interp(steps, s_steps, core_share)
        core_ps = arr * core_share_ps / 0.8
        band_ps = arr * (1.0 - core_share_ps) / 0.2
        ce = _ema_smooth(core_ps, SM)
        be = _ema_smooth(band_ps, SM)

    # =====================================================
    # Choose layout based on x0 data presence
    # =====================================================
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    if has_x0:
        # x0 ABLATION LAYOUT:
        # [0,0] Smooth Trend    [0,1] Core vs Band    [0,2] x0 vs eps + annealing
        # [1,0] Gates           [1,1] Band/Core Ratio  [1,2] Grad Norm
        ax_trend  = axes[0, 0]
        ax_cb     = axes[0, 1]
        ax_x0eps  = axes[0, 2]
        ax_gates  = axes[1, 0]
        ax_ratio  = axes[1, 1]
        ax_gn     = axes[1, 2]
    else:
        # DEFAULT LAYOUT:
        # [0,0] Smooth Trend    [0,1] Core vs Band    [0,2] Band/Core Ratio
        # [1,0] Gates           [1,1] Grad Norm        [1,2] (empty)
        ax_trend  = axes[0, 0]
        ax_cb     = axes[0, 1]
        ax_ratio  = axes[0, 2]
        ax_gates  = axes[1, 0]
        ax_gn     = axes[1, 1]
        ax_x0eps  = None
        axes[1, 2].set_visible(False)

    # --- Smooth Trend ---
    ax_trend.plot(steps, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SF})")
    ax_trend.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
    ax_trend.set_xlabel("Step"); ax_trend.set_ylabel("Loss")
    ax_trend.set_title(f"Smooth Trend — step {len(losses)}, trend: {ema_slow[-1]:.4f}")
    ax_trend.legend(loc="upper right", fontsize=8)
    ax_trend.grid(True, alpha=0.3)

    # --- Core vs Band (stacked area) ---
    if has_cb:
        ax_cb.fill_between(steps, 0, ce, color="#2196F3", alpha=0.4)
        ax_cb.fill_between(steps, ce, ce + be, color="#FF9800", alpha=0.4)
        ax_cb.plot(steps, ce, color="#1565C0", linewidth=1.5, label="Core")
        ax_cb.plot(steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
        ax_cb.plot(steps, ema_slow, color="black", linewidth=2, linestyle="--",
                   label=f"0.8\u00d7core+0.2\u00d7band = {ema_slow[-1]:.4f}")
        ax_cb.annotate(f"{ce[-1]:.3f}", xy=(steps[-1], ce[-1] / 2),
                       fontsize=9, fontweight="bold", color="#1565C0", ha="right")
        ax_cb.annotate(f"{be[-1]:.3f}", xy=(steps[-1], ce[-1] + be[-1] / 2),
                       fontsize=9, fontweight="bold", color="#E65100", ha="right")
        ax_cb.legend(loc="upper right", fontsize=8)
        ax_cb.set_title(f"Core vs Band, EMA({SM}) — weighted: {ema_slow[-1]:.4f}")
    else:
        ax_cb.set_title("Core vs Band (no data yet)")
    ax_cb.set_xlabel("Step"); ax_cb.set_ylabel("Per-pixel MSE")
    ax_cb.grid(True, alpha=0.3)

    # --- Band/Core Ratio ---
    if has_cb:
        safe_c_ps = np.maximum(core_ps, 1e-10)
        cbr = band_ps / safe_c_ps
        cbre = _ema_smooth(cbr, SM)
        ax_ratio.plot(steps, cbr, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
        ax_ratio.plot(steps, cbre, color="purple", linewidth=2, label=f"EMA ({SM})")
        ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_ratio.set_title(f"Band/Core Ratio, EMA({SM}) — {cbre[-1]:.2f}x")
        ax_ratio.legend(loc="upper left", fontsize=8)
    else:
        ax_ratio.set_title("Band/Core Ratio (no data yet)")
    ax_ratio.set_xlabel("Step"); ax_ratio.set_ylabel("Band / Core")
    ax_ratio.grid(True, alpha=0.3)

    # --- Gates ---
    if len(s_steps) > 0:
        ax_gates.plot(s_steps, s_attn, color="blue", linewidth=1.5, label="Attn gate")
        ax_gates.plot(s_steps, s_ff, color="green", linewidth=1.5, label="FF gate")
        ax_gates.set_title(f"Gates — attn={s_attn[-1]:.4f}, ff={s_ff[-1]:.4f}")
        ax_gates.legend(loc="upper left", fontsize=8)
    else:
        ax_gates.set_title("Gates (no data yet)")
    ax_gates.set_xlabel("Step"); ax_gates.set_ylabel("Gate value")
    ax_gates.grid(True, alpha=0.3)

    # --- Grad Norm ---
    if has_gn:
        gn_ema_fast = _ema_smooth(s_gn, SF)
        gn_ema_slow = _ema_smooth(s_gn, SM)
        ax_gn.plot(s_steps, gn_ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SF})")
        ax_gn.plot(s_steps, gn_ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax_gn.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
        ax_gn.set_title(f"Grad Norm — trend: {gn_ema_slow[-1]:.4f}")
        ax_gn.legend(loc="upper right", fontsize=8)
    else:
        ax_gn.set_title("Grad Norm (no data yet)")
    ax_gn.set_xlabel("Step"); ax_gn.set_ylabel("Norm")
    ax_gn.grid(True, alpha=0.3)

    # --- x0 vs eps + annealing (x0 layout only) ---
    if ax_x0eps is not None and has_x0:
        x0_mask = s_x0 > 0
        x0_steps_f, x0_vals_f = s_steps[x0_mask], s_x0[x0_mask]
        eps_mask = s_eps > 0
        eps_steps_f, eps_vals_f = s_steps[eps_mask], s_eps[eps_mask]

        lines = []
        if len(x0_vals_f) > 1:
            ax_x0eps.plot(x0_steps_f, _ema_smooth(x0_vals_f, SF), color="#CE93D8", linewidth=0.8, alpha=0.3)
            x0_slow = _ema_smooth(x0_vals_f, SM)
            l1, = ax_x0eps.plot(x0_steps_f, x0_slow, color="#6A1B9A", linewidth=2.5,
                                label=f"x0 = {x0_slow[-1]:.4f}")
            lines.append(l1)
        if len(eps_vals_f) > 1:
            ax_x0eps.plot(eps_steps_f, _ema_smooth(eps_vals_f, SF), color="#80CBC4", linewidth=0.8, alpha=0.3)
            eps_slow = _ema_smooth(eps_vals_f, SM)
            l2, = ax_x0eps.plot(eps_steps_f, eps_slow, color="#00695C", linewidth=2.5,
                                label=f"\u03b5 = {eps_slow[-1]:.4f}")
            lines.append(l2)

        # Annealing ratio on second y-axis
        ax_r2 = ax_x0eps.twinx()
        l3, = ax_r2.plot(s_steps, s_x0r, color="gray", linewidth=1.5, linestyle="--",
                         alpha=0.6, label="x0 ratio")
        ax_r2.set_ylim(-0.05, 1.1)
        ax_r2.set_ylabel("x0 ratio", color="gray")
        ax_r2.tick_params(axis="y", labelcolor="gray")
        lines.append(l3)

        ax_x0eps.legend(handles=lines, loc="upper right", fontsize=7)
        title_parts = []
        if len(x0_vals_f) > 1:
            title_parts.append(f"x0={x0_slow[-1]:.4f}")
        if len(eps_vals_f) > 1:
            title_parts.append(f"\u03b5={eps_slow[-1]:.4f}")
        ax_x0eps.set_title(f"x0 vs \u03b5 Loss, EMA({SM}) — {', '.join(title_parts)}")
        ax_x0eps.set_xlabel("Step"); ax_x0eps.set_ylabel("Loss")
        ax_x0eps.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_anomagic_checkpoint(
    ip_adapter, optimizer, save_dir,
    band_mode: int = 1, t2i_adapter=None, unet=None,
    step: int = 0, losses: list = None,
):
    """Save IP-Adapter + T2I-Adapter + LoRA checkpoint."""
    save_dir = Path(save_dir)
    ip_adapter.save_ip_adapter(save_dir)

    state = {
        "optimizer": optimizer.state_dict(),
        "band_mode": band_mode,
        "step": step,
        "losses": losses or [],
    }
    if t2i_adapter is not None:
        state["t2i_adapter"] = t2i_adapter.state_dict()
    if unet is not None:
        from peft import get_peft_model_state_dict
        state["lora_weights"] = get_peft_model_state_dict(unet)
    torch.save(state, save_dir / "training_state.pt")


def generate_anomagic_samples(
    pipeline, ip_adapter, dataset,
    save_path, device,
    band_mode: int = 1,
    t2i_adapter=None,
    cfg_mode: str = "text",
):
    """Generate samples showing: real | mask | reference | generated.

    Uses a fixed seed so the same samples are shown at every checkpoint,
    making it easy to track visual progress across training.
    """
    sample_rng = random.Random(42)
    # Fixed (type, image_path) pairs for consistent visualization across ALL runs
    _fixed_samples = [
        ("cut_lead", "AnomVerse_data_filtered/mvtec/mvtec/transistor/test/cut_lead/005.png"),
        ("broken_large", "AnomVerse_data_filtered/mvtec/mvtec/bottle/test/broken_large/011.png"),
        ("fold", "AnomVerse_data_filtered/mvtec/mvtec/leather/test/fold/016.png"),
        ("cut", "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/test/cut/011.png"),
        ("thread", "AnomVerse_data_filtered/mvtec/mvtec/carpet/test/thread/011.png"),
        ("faulty_imprint", "AnomVerse_data_filtered/mvtec/mvtec/pill/test/faulty_imprint/011.png"),
        ("contamination", "realiad_1024/mint/NG/ZW/S0070/mint_0070_NG_ZW_C5_20230910095530.jpg"),
    ]
    # Build lookup: image_path → dataset index (normalize to forward-slash suffix)
    _path_to_idx = {}
    for i, s in enumerate(dataset.samples):
        p = s["image_path"].replace("\\", "/")
        _path_to_idx[p] = i
        # Also index by the relative tail so fixed paths match regardless of data_root
        for prefix in ("AnomVerse_data_filtered/", "realiad_1024/"):
            idx = p.find(prefix)
            if idx >= 0:
                _path_to_idx[p[idx:]] = i
                break
    types_to_show = []
    fixed_indices = {}
    for atype, img_path in _fixed_samples:
        if atype in dataset.type_to_samples and img_path in _path_to_idx:
            types_to_show.append(atype)
            fixed_indices[atype] = _path_to_idx[img_path]
    # Fallback: fill remaining slots from available types
    if len(types_to_show) < 6:
        remaining = [t for t in dataset.anomaly_types if t not in types_to_show]
        for t in sample_rng.sample(remaining, min(6 - len(types_to_show), len(remaining))):
            types_to_show.append(t)

    n_cols = len(types_to_show)
    fig, axes = plt.subplots(6, n_cols, figsize=(3.5 * n_cols, 19), squeeze=False)

    for i, atype in enumerate(types_to_show):
        if atype in fixed_indices:
            idx = fixed_indices[atype]
        else:
            indices = dataset.type_to_samples[atype]
            idx = sample_rng.choice(indices)
        sample = dataset[idx]

        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        reference = sample["reference"].unsqueeze(0).to(device)

        img_path = sample.get("image_path", "unknown")
        path_parts = Path(img_path).parts
        short_path = "/".join(path_parts[-3:]) if len(path_parts) >= 3 else img_path

        with torch.no_grad():
            gen_text_cfg = generate_anomagic_single(
                pipeline, ip_adapter, image, mask, reference,
                atype, sample.get("caption", ""),
                num_steps=30,
                reference_mode=dataset.reference_mode,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                seed=42 + i,
                cfg_mode="text",
            )
            gen_visual_cfg = generate_anomagic_single(
                pipeline, ip_adapter, image, mask, reference,
                atype, sample.get("caption", ""),
                num_steps=30,
                reference_mode=dataset.reference_mode,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                seed=42 + i,
                cfg_mode="visual",
            )
            gen_both_cfg = generate_anomagic_single(
                pipeline, ip_adapter, image, mask, reference,
                atype, sample.get("caption", ""),
                num_steps=30,
                reference_mode=dataset.reference_mode,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                seed=42 + i,
                cfg_mode="both",
            )

        img_np = ((image[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        gen_text_np = ((gen_text_cfg[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        gen_vis_np = ((gen_visual_cfg[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        gen_both_np = ((gen_both_cfg[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        mask_np = mask[0, 0].cpu().numpy()
        ref_np = ((reference[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Real ({atype})\n{short_path}", fontsize=6)
        axes[0, i].axis("off")

        axes[1, i].imshow(mask_np, cmap="gray")
        axes[1, i].set_title(f"Mask ({mask_np.mean() * 100:.1f}%)", fontsize=7)
        axes[1, i].axis("off")

        axes[2, i].imshow(ref_np)
        ref_label = "clip_crop" if dataset.augment else dataset.reference_mode
        axes[2, i].set_title(f"Reference ({ref_label})", fontsize=7)
        axes[2, i].axis("off")

        axes[3, i].imshow(gen_text_np)
        axes[3, i].set_title("Gen (CFG=text)", fontsize=8)
        axes[3, i].axis("off")

        axes[4, i].imshow(gen_vis_np)
        axes[4, i].set_title("Gen (CFG=visual)", fontsize=8)
        axes[4, i].axis("off")

        axes[5, i].imshow(gen_both_np)
        axes[5, i].set_title("Gen (CFG=both)", fontsize=8)
        axes[5, i].axis("off")

    plt.suptitle("Anomagic Generation Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _launch_live_loss(loss_file: Path):
    """Launch live loss viewer as a detached subprocess."""
    import subprocess
    script = Path(__file__).parent / "live_loss.py"
    loss_file.write_text("")
    try:
        subprocess.Popen(
            [sys.executable, str(script), str(loss_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  Live loss viewer launched (reading {loss_file})")
    except Exception as e:
        print(f"  Warning: could not launch live loss viewer: {e}")


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomagic training: IP-Adapter + captions")
    parser.add_argument("--splits-dir", type=str, default="data/concepts",
                        help="Directory with anomaly type JSONs")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for resolving relative image paths")
    parser.add_argument("--captions-file", type=str, required=True,
                        help="Path to captions.json")
    parser.add_argument("--save-dir", type=str, default="results/anomagic_training",
                        help="Where to save results")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. results/anomagic_training/checkpoint_8000)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Training steps")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint + samples every N steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for IP-Adapter")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--ip-adapter-type", type=str, default="plus",
                        choices=["standard", "plus"],
                        help="standard: ViT-L/14 CLS→linear→4 tokens. plus: ViT-H/14 patches→perceiver→K tokens")
    parser.add_argument("--ip-adapter-k", type=int, default=16, choices=[1, 4, 16],
                        help="Visual tokens after projection: 1=no upproject, 4=standard, 16=pretrained Plus default")
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0,
                        help="Visual pathway strength")
    parser.add_argument("--reference-mode", type=str, default="full", choices=["full", "crop"],
                        help="Reference image mode: 'full'=downscale entire image, 'crop'=anomaly bbox crop")
    parser.add_argument("--types", type=str, nargs="+", default=None,
                        help="Specific anomaly types to train on (default: all)")
    parser.add_argument("--exclude-sources", type=str, nargs="+",
                        default=None,
                        help="Dataset sources to exclude (default: none)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-mask-visual", action="store_true",
                        help="Disable masked cross-attention for visual pathway "
                             "(ablation — all positions receive visual conditioning. "
                             "Default: masked, only anomaly positions get visual signal)")
    # Visual token processing mode
    parser.add_argument("--visual-mode", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Visual token processing: 0=all 256 tokens (default), "
                             "1=selective residual, 2=anomaly-only+attn-only, "
                             "3=anomaly-only+full transformer (padding dead everywhere)")
    parser.add_argument("--no-learnable-gates", action="store_true",
                        help="Replace scalar gates with zero-init output projections. "
                             "Same identity-at-init, but per-dimension gradient signal.")
    parser.add_argument("--force-gates", action="store_true",
                        help="Fixed gate=1.0 with normal projection init. "
                             "Block contributes fully from step 0 — forces reliance.")
    parser.add_argument("--sa-num-layers", type=int, default=1,
                        help="Number of transformer layers in masked self-attention (default 1)")
    parser.add_argument("--sa-num-heads", type=int, default=12,
                        help="Number of attention heads in masked self-attention (default 12, matching Resampler)")
    parser.add_argument("--cfg-mode", type=str, default="text", choices=["text", "visual", "both"],
                        help="CFG direction for sample generation: 'text' (default) amplifies text "
                             "on visual baseline, 'visual' amplifies IP-Adapter on text baseline, "
                             "'both' amplifies both pathways on unconditional baseline")
    # Multi-crop (orthogonal to visual-mode)
    parser.add_argument("--multi-crop", action="store_true",
                        help="Enable 2-group CLIP cropping: encode 2 independent anomaly "
                             "crops → 2K tokens in UNet cross-attention. Combinable with any --visual-mode")
    # CLIP dilation settings — currently unused (CLIP path uses cropping instead).
    # Kept for potential future use with reference_mode="full".
    # parser.add_argument("--clip-dilation-min-r", type=int, default=2,
    #                     help="CLIP mask dilation min radius (tight, for anomaly-focused features)")
    # parser.add_argument("--clip-dilation-max-r", type=int, default=10,
    #                     help="CLIP mask dilation max radius")
    # Latent-space band dilation
    parser.add_argument("--band-mode", type=int, default=1, choices=[1, 2],
                        help="Band mode: 1=single 1px band (alpha=0.5), 2=inner+outer (alpha=2/3, 1/3)")
    # Loss settings
    parser.add_argument("--loss-core-ratio", type=float, default=0.8,
                        help="Fraction of total gradient to core (e.g. 0.8 = 80%% core, 20%% band)")
    # Cross-attention mask type
    parser.add_argument("--binary-cross-attn-mask", action="store_true",
                        help="Use binary dilated mask for cross-attn instead of soft alpha_map")
    # T2I-Adapter
    parser.add_argument("--t2i-adapter-mode", type=str, default="cascade",
                        choices=["cascade", "skip_only", "off"],
                        help="T2I-Adapter injection mode (cascade=encoder+decoder, skip_only=decoder, off=disabled)")
    # Conditioning dropout for CFG (3-category, mutually exclusive per-sample)
    parser.add_argument("--drop-image-prob", type=float, default=0.10,
                        help="Probability of zeroing IP-Adapter image embeddings per sample for CFG")
    parser.add_argument("--drop-text-prob", type=float, default=0.10,
                        help="Probability of replacing text with empty-string encoding per sample for CFG")
    parser.add_argument("--drop-both-prob", type=float, default=0.05,
                        help="Probability of dropping both image and text per sample for CFG")
    # LoRA
    # Data augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (UNet: color jitter only, CLIP: flip/continuous rotate/jitter/crop_utils)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank for UNet attention layers (0=disabled, 16=matches IP-Adapter Plus K)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lr-pretrained", type=float, default=1e-4,
                        help="Learning rate for pretrained IP-Adapter params (Group A)")
    parser.add_argument("--lambda-sp", type=float, default=0.0,
                        help="L2-SP regularization strength (default 0=disabled)")
    parser.add_argument("--no-live-viewer", action="store_true",
                        help="Disable live loss viewer (for headless servers)")
    parser.add_argument("--noise-offset", type=float, default=0.05,
                        help="Noise offset for global brightness/darkness (default 0.05, 0=disabled)")
    parser.add_argument("--timestep-sampling", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal"],
                        help="Timestep sampling: 'logit_normal' (default, SD3-style bell curve) "
                             "or 'uniform' (standard DDPM)")
    parser.add_argument("--logit-normal-mean", type=float, default=0.0,
                        help="Mean for logit-normal timestep sampling (default 0.0, centers on t=500)")
    parser.add_argument("--logit-normal-std", type=float, default=1.0,
                        help="Std for logit-normal timestep sampling (default 1.0, lower=tighter peak)")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "prodigy"],
                        help="Optimizer: 'adamw' (default) or 'prodigy' (auto-tuned LR)")
    parser.add_argument("--corrupt-context", type=float, default=0.0,
                        help="Probability of zeroing masked_image_latents per step (0.0-1.0). "
                             "Forces UNet to rely on cross-attention instead of inpainting prior.")
    parser.add_argument("--x0-objective", action="store_true",
                        help="Enable pseudo-diffusion x0-prediction: input=noised(masked_image), "
                             "loss=||x0_pred - x0||². Forces cross-attention usage for anomaly generation.")
    parser.add_argument("--x0-start-ratio", type=float, default=1.0,
                        help="Fraction of samples using x0 path at step 0 (default 1.0)")
    parser.add_argument("--x0-end-ratio", type=float, default=1.0,
                        help="Fraction of samples using x0 path at final step (default 1.0, set to 0.1 to anneal)")
    parser.add_argument("--x0-no-context", action="store_true",
                        help="Zero masked_image_latents inpainting channel for x0-path samples "
                             "(removes all shortcuts, forces pure cross-attention)")
    parser.add_argument("--x0-warmup-frac", type=float, default=0.2,
                        help="Fraction of total steps to hold x0_ratio at start value before annealing "
                             "(default 0.2 = first 1/5 of training)")

    args = parser.parse_args()

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

    train_anomagic(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        captions_file=captions_file,
        n_steps=args.steps,
        save_every=args.save_every,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_pretrained=args.lr_pretrained,
        lambda_sp=args.lambda_sp,
        anomaly_types=args.types,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        device=args.device,
        ip_adapter_type=args.ip_adapter_type,
        ip_adapter_k=args.ip_adapter_k,
        ip_adapter_scale=args.ip_adapter_scale,
        reference_mode=args.reference_mode,
        mask_visual=not args.no_mask_visual,
        clip_dilation_min_r=getattr(args, 'clip_dilation_min_r', 2),
        clip_dilation_max_r=getattr(args, 'clip_dilation_max_r', 10),
        band_mode=args.band_mode,
        loss_core_ratio=args.loss_core_ratio,
        binary_cross_attn_mask=args.binary_cross_attn_mask,
        t2i_adapter_mode=args.t2i_adapter_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        drop_image_prob=args.drop_image_prob,
        drop_text_prob=args.drop_text_prob,
        drop_both_prob=args.drop_both_prob,
        augment=args.augment,
        resume_dir=resume_dir,
        visual_mode=args.visual_mode,
        learnable_gates=not args.no_learnable_gates,
        force_gates=args.force_gates,
        sa_num_layers=args.sa_num_layers,
        sa_num_heads=args.sa_num_heads,
        multi_crop=args.multi_crop,
        cfg_mode=args.cfg_mode,
        no_live_viewer=args.no_live_viewer,
        noise_offset=args.noise_offset,
        timestep_sampling=args.timestep_sampling.replace("-", "_"),
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        optimizer_type=args.optimizer,
        corrupt_context=args.corrupt_context,
        x0_objective=args.x0_objective,
        x0_start_ratio=args.x0_start_ratio,
        x0_end_ratio=args.x0_end_ratio,
        x0_no_context=args.x0_no_context,
        x0_warmup_frac=args.x0_warmup_frac,
    )
