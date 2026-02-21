#!/usr/bin/env python
"""
CLI script for evaluating generated synthetic anomalies.

Examples:
    # Evaluate a directory of generated images
    python scripts/evaluate_generation.py \
        --generated results/generated/ \
        --reference data/reference/ \
        --background data/backgrounds/ \
        --masks data/masks/ \
        --output results/evaluation/

    # Evaluate from split JSON (auto-finds paths)
    python scripts/evaluate_generation.py \
        --concept scratch \
        --generated results/scratch/ \
        --from_split data/concepts/scratch.json \
        --output results/evaluation/scratch/

    # Quick evaluation (skip FID, faster)
    python scripts/evaluate_generation.py \
        --generated results/generated/ \
        --reference data/reference/ \
        --background data/backgrounds/ \
        --masks data/masks/ \
        --output results/evaluation/ \
        --no_fid
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional


def evaluate_from_dirs(
    generated_dir: Path,
    reference_dir: Path,
    background_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    compute_fid: bool = True,
    compute_diversity: bool = True,
    device: str = "cuda",
) -> None:
    """Evaluate generated images from directory structure."""
    from src.inference.evaluate import AnomalyEvaluator

    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching files
    generated_files = sorted(generated_dir.glob("*.png")) + sorted(generated_dir.glob("*.jpg"))

    generated_images = []
    reference_images = []
    background_images = []
    masks = []

    for gen_path in generated_files:
        name = gen_path.stem
        # Try different extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            ref_path = reference_dir / f"{name}{ext}"
            bg_path = background_dir / f"{name}{ext}"
            mask_path = mask_dir / f"{name}{ext}"

            if ref_path.exists():
                reference_images.append(ref_path)
                generated_images.append(gen_path)
                break
            elif (reference_dir / f"{name}.png").exists():
                reference_images.append(reference_dir / f"{name}.png")
                generated_images.append(gen_path)
                break

        # Background and mask
        for ext in [".png", ".jpg", ".jpeg"]:
            bg_path = background_dir / f"{name}{ext}"
            if bg_path.exists():
                background_images.append(bg_path)
                break
        else:
            # Use generated as background fallback
            background_images.append(gen_path)

        for ext in [".png", ".jpg", ".jpeg"]:
            mask_path = mask_dir / f"{name}{ext}"
            if mask_path.exists():
                masks.append(mask_path)
                break

    # Ensure equal lengths
    min_len = min(len(generated_images), len(reference_images), len(background_images), len(masks))
    generated_images = generated_images[:min_len]
    reference_images = reference_images[:min_len]
    background_images = background_images[:min_len]
    masks = masks[:min_len]

    print(f"Found {len(generated_images)} matching image sets")

    if len(generated_images) == 0:
        print("No matching images found!")
        return

    # Evaluate
    evaluator = AnomalyEvaluator(device=device)
    metrics = evaluator.evaluate_batch(
        generated_images=generated_images,
        reference_images=reference_images,
        background_images=background_images,
        masks=masks,
        compute_fid=compute_fid,
        compute_diversity=compute_diversity,
    )

    # Generate report
    report = evaluator.generate_report(metrics, output_dir / "evaluation_report.txt")
    print(report)

    # Create comparison grids
    print("Creating comparison grids...")
    for i in range(min(10, len(generated_images))):
        evaluator.create_comparison_grid(
            generated=generated_images[i],
            reference=reference_images[i],
            background=background_images[i],
            mask=masks[i],
            output_path=output_dir / f"comparison_{i:03d}.png",
        )


def evaluate_from_split(
    concept: str,
    generated_dir: Path,
    split_json: Path,
    output_dir: Path,
    compute_fid: bool = True,
    compute_diversity: bool = True,
    device: str = "cuda",
) -> None:
    """Evaluate using paths from split JSON."""
    from src.inference.evaluate import AnomalyEvaluator

    with open(split_json, 'r') as f:
        data = json.load(f)

    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = sorted(generated_dir.glob("*.png"))

    generated_images = []
    reference_images = []
    background_images = []
    masks = []

    # Match generated files to split entries
    split_images = {Path(img.get('image_path_full') or img.get('image_path_abs', img['image_path'])).stem: img
                    for img in data['images']}

    for gen_path in generated_files:
        # Try to find matching entry
        name = gen_path.stem
        # Remove concept prefix if present
        clean_name = name.replace(f"{concept}_", "")

        # Try different matching strategies
        matched = None
        for key in split_images:
            if clean_name in key or key in clean_name:
                matched = split_images[key]
                break

        if matched:
            ref_path = matched.get('image_path_full') or matched.get('image_path_abs', matched['image_path'])
            mask_path = matched.get('mask_path_full') or matched.get('mask_path_abs', matched.get('mask_path'))

            if Path(ref_path).exists() and mask_path and Path(mask_path).exists():
                generated_images.append(gen_path)
                reference_images.append(ref_path)
                background_images.append(ref_path)  # Same as reference for now
                masks.append(mask_path)

    print(f"Matched {len(generated_images)} images from split")

    if len(generated_images) == 0:
        print("No matching images found! Using sequential matching...")
        # Fallback: sequential matching
        for i, gen_path in enumerate(generated_files):
            if i >= len(data['images']):
                break
            img_data = data['images'][i]
            ref_path = img_data.get('image_path_full') or img_data.get('image_path_abs', img_data['image_path'])
            mask_path = img_data.get('mask_path_full') or img_data.get('mask_path_abs', img_data.get('mask_path'))

            if Path(ref_path).exists() and mask_path and Path(mask_path).exists():
                generated_images.append(gen_path)
                reference_images.append(ref_path)
                background_images.append(ref_path)
                masks.append(mask_path)

        print(f"Sequential match: {len(generated_images)} images")

    if len(generated_images) == 0:
        print("Still no matches found!")
        return

    # Evaluate
    evaluator = AnomalyEvaluator(device=device)
    metrics = evaluator.evaluate_batch(
        generated_images=generated_images,
        reference_images=reference_images,
        background_images=background_images,
        masks=masks,
        compute_fid=compute_fid,
        compute_diversity=compute_diversity,
    )

    # Generate report
    report = evaluator.generate_report(metrics, output_dir / "evaluation_report.txt")
    print(report)

    # Create comparison grids
    print("Creating comparison grids...")
    for i in range(min(10, len(generated_images))):
        evaluator.create_comparison_grid(
            generated=generated_images[i],
            reference=reference_images[i],
            background=background_images[i],
            mask=masks[i],
            output_path=output_dir / f"comparison_{i:03d}.png",
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated synthetic anomalies")

    # Required
    parser.add_argument("--generated", type=Path, required=True,
                        help="Directory with generated images")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for evaluation results")

    # Input modes
    parser.add_argument("--reference", type=Path, help="Directory with reference images")
    parser.add_argument("--background", type=Path, help="Directory with background images")
    parser.add_argument("--masks", type=Path, help="Directory with masks")
    parser.add_argument("--from_split", type=Path, help="Split JSON for path lookup")
    parser.add_argument("--concept", type=str, help="Concept name (for split matching)")

    # Options
    parser.add_argument("--no_fid", action="store_true", help="Skip FID computation")
    parser.add_argument("--no_diversity", action="store_true", help="Skip diversity computation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    if args.from_split:
        if not args.concept:
            # Infer concept from split filename
            args.concept = args.from_split.stem

        evaluate_from_split(
            concept=args.concept,
            generated_dir=args.generated,
            split_json=args.from_split,
            output_dir=args.output,
            compute_fid=not args.no_fid,
            compute_diversity=not args.no_diversity,
            device=args.device,
        )
    else:
        if not args.reference or not args.masks:
            parser.error("--reference and --masks required without --from_split")

        background_dir = args.background or args.reference

        evaluate_from_dirs(
            generated_dir=args.generated,
            reference_dir=args.reference,
            background_dir=background_dir,
            mask_dir=args.masks,
            output_dir=args.output,
            compute_fid=not args.no_fid,
            compute_diversity=not args.no_diversity,
            device=args.device,
        )


if __name__ == "__main__":
    main()
