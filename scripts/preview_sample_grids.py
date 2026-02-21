"""Quick preview of the 3 sample grids using an existing checkpoint.

Usage:
    python scripts/preview_sample_grids.py --checkpoint results/run_50k_fixed/checkpoint_10000
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="results/preview_grids")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    adapter_type = config["adapter_type"]
    num_tokens = config["num_tokens"]
    scale = config.get("scale", 1.0)
    mask_visual = config.get("mask_visual", True)
    visual_mode = config.get("visual_mode", 0)

    print(f"Config: type={adapter_type}, K={num_tokens}, scale={scale}, "
          f"mask_visual={mask_visual}, visual_mode={visual_mode}")

    # Pipeline
    from src.models.base import create_pipeline
    pipeline = create_pipeline("sd_1.5", device=args.device)
    pipeline.load_pipeline()
    pipeline.freeze_all()
    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # IP-Adapter
    from src.models.ip_adapter import create_ip_adapter
    ip_adapter = create_ip_adapter(
        pipeline, adapter_type=adapter_type, num_tokens=num_tokens,
        scale=scale, load_pretrained=True, mask_visual=mask_visual,
        visual_mode=visual_mode,
    )
    ip_adapter.freeze_image_encoder()
    ip_adapter.load_finetuned(checkpoint_dir)
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()
    ip_adapter.masked_self_attn.eval()
    ip_adapter.image_projection.eval()
    for proc in ip_adapter.attn_processors.values():
        proc.eval()
    print(f"Loaded IP-Adapter from {checkpoint_dir}")

    # T2I-Adapter
    t2i_adapter = None
    state_path = checkpoint_dir / "training_state.pt"
    band_mode = 2
    if state_path.exists():
        state = torch.load(state_path, map_location=args.device)
        band_mode = state.get("band_mode", 2)
        if "t2i_adapter" in state:
            from src.models.t2i_adapter import T2IAdapter
            t2i_adapter = T2IAdapter(in_channels=2).to(args.device)
            t2i_adapter.load_state_dict(state["t2i_adapter"])
            t2i_adapter.eval()
            print("Loaded T2I-Adapter")
        del state
        torch.cuda.empty_cache()

    # Dataset (minimal — just enough for sample generation)
    from src.data.anomaly_dataset import AnomalyDataset
    data_root = Path("anomverse_extension/datasets/full_training_dataset")
    captions_file = data_root / "captions_from_master.json"
    splits_dir = data_root / "splits_by_type"

    dataset = AnomalyDataset(
        splits_dir=str(splits_dir),
        captions_file=str(captions_file),
        data_root=str(data_root),
        augment=True,
        reference_mode="full",
        return_reference=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataset.anomaly_types)} types")

    # Generate
    from scripts.train_anomagic import generate_anomagic_samples
    save_path = output_dir / "preview_samples.png"
    print(f"\nGenerating grids -> {output_dir}/")
    print("  This will produce: preview_samples.png, preview_samples_swap.png, preview_samples_ood.png")

    generate_anomagic_samples(
        pipeline, ip_adapter, dataset,
        save_path=save_path,
        device=args.device,
        band_mode=band_mode,
        t2i_adapter=t2i_adapter,
    )
    print("\nDone!")
    print(f"  Main grid:  {save_path}")
    print(f"  Swap grid:  {save_path.parent / (save_path.stem + '_swap' + save_path.suffix)}")
    print(f"  OOD grid:   {save_path.parent / (save_path.stem + '_ood' + save_path.suffix)}")


if __name__ == "__main__":
    main()
