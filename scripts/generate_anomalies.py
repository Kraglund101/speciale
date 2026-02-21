#!/usr/bin/env python
"""
CLI script for generating synthetic anomalies.

Examples:
    # Generate a single image
    python scripts/generate_anomalies.py \
        --concept scratch \
        --reference path/to/reference.png \
        --background path/to/normal.png \
        --mask path/to/mask.png \
        --output results/generated.png

    # Generate from a dataset split
    python scripts/generate_anomalies.py \
        --concept scratch \
        --from_split data/concepts/scratch.json \
        --num_samples 10 \
        --output_dir results/scratch/

    # Generate with different IP-Adapter scales for comparison
    python scripts/generate_anomalies.py \
        --concept scratch \
        --reference path/to/reference.png \
        --background path/to/normal.png \
        --mask path/to/mask.png \
        --ip_scales 0.5 0.75 1.0 1.25 \
        --output_dir results/scale_comparison/
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from PIL import Image


def generate_single(
    concept: str,
    reference: Path,
    background: Path,
    mask: Path,
    output: Path,
    sd_version: str = "sd_1.5",
    phase1_dir: Path = Path("checkpoints/phase1"),
    phase2_dir: Optional[Path] = Path("checkpoints/phase2/final"),
    num_tokens: int = 4,
    ip_adapter_scale: float = 1.0,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
) -> None:
    """Generate a single synthetic anomaly image."""
    from src.inference.generate import create_generator

    generator = create_generator(
        sd_version=sd_version,
        phase1_dir=str(phase1_dir),
        phase2_dir=str(phase2_dir) if phase2_dir else None,
        num_tokens=num_tokens,
        ip_adapter_scale=ip_adapter_scale,
    )

    result = generator.generate(
        concept_name=concept,
        reference_image=reference,
        background_image=background,
        mask=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        seed=seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(f"Saved: {output}")


def generate_from_split(
    concept: str,
    split_json: Path,
    output_dir: Path,
    num_samples: int = 10,
    sd_version: str = "sd_1.5",
    phase1_dir: Path = Path("checkpoints/phase1"),
    phase2_dir: Optional[Path] = Path("checkpoints/phase2/final"),
    num_tokens: int = 4,
    ip_adapter_scale: float = 1.0,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
) -> None:
    """Generate samples using images from a dataset split."""
    from src.inference.generate import create_generator

    # Load split
    with open(split_json, 'r') as f:
        data = json.load(f)

    images = data['images']
    if num_samples < len(images):
        if seed is not None:
            random.seed(seed)
        images = random.sample(images, num_samples)

    generator = create_generator(
        sd_version=sd_version,
        phase1_dir=str(phase1_dir),
        phase2_dir=str(phase2_dir) if phase2_dir else None,
        num_tokens=num_tokens,
        ip_adapter_scale=ip_adapter_scale,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_data in enumerate(images):
        # Use image as both reference and background (self-reconstruction test)
        # In practice, you'd use different images
        image_path = img_data.get('image_path_full') or img_data.get('image_path_abs', img_data['image_path'])
        mask_path = img_data.get('mask_path_full') or img_data.get('mask_path_abs', img_data.get('mask_path'))

        if not mask_path or not Path(mask_path).exists():
            print(f"Skipping {i}: no mask")
            continue

        img_seed = seed + i if seed is not None else None

        result = generator.generate(
            concept_name=concept,
            reference_image=image_path,
            background_image=image_path,  # Using same image for demo
            mask=mask_path,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            seed=img_seed,
        )

        output_path = output_dir / f"{concept}_{i:04d}.png"
        result.save(output_path)
        print(f"Saved: {output_path}")

        # Also save comparison grid
        original = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        # Create side-by-side comparison
        width = original.width
        height = original.height
        comparison = Image.new("RGB", (width * 3, height))
        comparison.paste(original.resize((width, height)), (0, 0))
        comparison.paste(mask_img.convert("RGB").resize((width, height)), (width, 0))
        comparison.paste(result.resize((width, height)), (width * 2, 0))

        comparison_path = output_dir / f"{concept}_{i:04d}_comparison.png"
        comparison.save(comparison_path)


def generate_scale_comparison(
    concept: str,
    reference: Path,
    background: Path,
    mask: Path,
    output_dir: Path,
    ip_scales: List[float],
    sd_version: str = "sd_1.5",
    phase1_dir: Path = Path("checkpoints/phase1"),
    phase2_dir: Optional[Path] = Path("checkpoints/phase2/final"),
    num_tokens: int = 4,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
) -> None:
    """Generate images with different IP-Adapter scales for comparison."""
    from src.inference.generate import create_generator

    generator = create_generator(
        sd_version=sd_version,
        phase1_dir=str(phase1_dir),
        phase2_dir=str(phase2_dir) if phase2_dir else None,
        num_tokens=num_tokens,
        ip_adapter_scale=1.0,  # Will override per-generation
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for scale in ip_scales:
        result = generator.generate(
            concept_name=concept,
            reference_image=reference,
            background_image=background,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            ip_adapter_scale=scale,
            seed=seed,
        )

        output_path = output_dir / f"scale_{scale:.2f}.png"
        result.save(output_path)
        results.append((scale, result))
        print(f"Saved: {output_path}")

    # Create comparison grid
    if results:
        width = results[0][1].width
        height = results[0][1].height
        grid = Image.new("RGB", (width * len(results), height))

        for i, (scale, img) in enumerate(results):
            grid.paste(img, (i * width, 0))

        grid_path = output_dir / "scale_comparison.png"
        grid.save(grid_path)
        print(f"Saved comparison grid: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic anomalies")

    # Required
    parser.add_argument("--concept", type=str, required=True, help="Anomaly concept name")

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--reference", type=Path, help="Reference anomaly image")
    input_group.add_argument("--from_split", type=Path, help="Generate from split JSON")

    # Single image inputs
    parser.add_argument("--background", type=Path, help="Background/normal image")
    parser.add_argument("--mask", type=Path, help="Inpainting mask")

    # Output
    parser.add_argument("--output", type=Path, help="Output image path (single image)")
    parser.add_argument("--output_dir", type=Path, help="Output directory (batch)")

    # Model settings
    parser.add_argument("--sd_version", type=str, default="sd_1.5",
                        choices=["sd_1.5", "sdxl", "sd_3.0"])
    parser.add_argument("--phase1_dir", type=Path, default=Path("checkpoints/phase1"))
    parser.add_argument("--phase2_dir", type=Path, default=Path("checkpoints/phase2/final"))
    parser.add_argument("--num_tokens", type=int, default=4, choices=[1, 4, 16])

    # Generation settings
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)

    # Batch settings
    parser.add_argument("--num_samples", type=int, default=10)

    # Scale comparison
    parser.add_argument("--ip_scales", type=float, nargs="+",
                        help="Multiple IP-Adapter scales for comparison")

    args = parser.parse_args()

    # Validate arguments
    if args.reference and not args.from_split:
        if not args.background or not args.mask:
            parser.error("--background and --mask required with --reference")
        if not args.output and not args.output_dir:
            parser.error("--output or --output_dir required")

    if args.from_split and not args.output_dir:
        parser.error("--output_dir required with --from_split")

    # Execute
    if args.ip_scales:
        # Scale comparison mode
        generate_scale_comparison(
            concept=args.concept,
            reference=args.reference,
            background=args.background,
            mask=args.mask,
            output_dir=args.output_dir or Path("results/scale_comparison"),
            ip_scales=args.ip_scales,
            sd_version=args.sd_version,
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir if args.phase2_dir.exists() else None,
            num_tokens=args.num_tokens,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            seed=args.seed,
        )
    elif args.from_split:
        # Batch from split
        generate_from_split(
            concept=args.concept,
            split_json=args.from_split,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            sd_version=args.sd_version,
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir if args.phase2_dir.exists() else None,
            num_tokens=args.num_tokens,
            ip_adapter_scale=args.ip_adapter_scale,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            seed=args.seed,
        )
    else:
        # Single image
        generate_single(
            concept=args.concept,
            reference=args.reference,
            background=args.background,
            mask=args.mask,
            output=args.output or args.output_dir / "generated.png",
            sd_version=args.sd_version,
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir if args.phase2_dir.exists() else None,
            num_tokens=args.num_tokens,
            ip_adapter_scale=args.ip_adapter_scale,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
