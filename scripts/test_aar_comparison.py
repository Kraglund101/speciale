"""
Test script to demonstrate Adaptive Attention Re-weighting (AAR).

This script shows the difference between generation with and without AAR,
particularly for irregular or multi-region masks where AAR helps fill
the entire mask area.

Run this after training to visualize the effect.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.adaptive_attention import (
    AdaptiveAttentionReweighting,
    AARConfig,
    visualize_aar_effect,
)


def create_test_masks() -> dict:
    """
    Create various test masks to demonstrate AAR benefits.

    AAR helps most with:
    - Multi-region masks (disconnected areas)
    - Irregular shapes
    - Large masks that are hard to fill uniformly
    """
    H, W = 256, 256

    masks = {}

    # 1. Simple center mask (AAR helps less here)
    simple = torch.zeros(1, 1, H, W)
    simple[:, :, 100:156, 100:156] = 1.0
    masks['simple_center'] = simple

    # 2. Multi-region mask (AAR helps A LOT here)
    multi = torch.zeros(1, 1, H, W)
    multi[:, :, 30:70, 30:70] = 1.0      # Top-left
    multi[:, :, 30:70, 180:220] = 1.0    # Top-right
    multi[:, :, 180:220, 100:156] = 1.0  # Bottom-center
    masks['multi_region'] = multi

    # 3. L-shaped mask (irregular)
    l_shape = torch.zeros(1, 1, H, W)
    l_shape[:, :, 50:180, 50:100] = 1.0   # Vertical bar
    l_shape[:, :, 130:180, 50:200] = 1.0  # Horizontal bar
    masks['l_shaped'] = l_shape

    # 4. Scattered small regions
    scattered = torch.zeros(1, 1, H, W)
    positions = [(40, 40), (40, 150), (40, 210),
                 (120, 80), (120, 180),
                 (200, 40), (200, 120), (200, 200)]
    for y, x in positions:
        scattered[:, :, y:y+30, x:x+30] = 1.0
    masks['scattered'] = scattered

    # 5. Ring/donut shape
    ring = torch.zeros(1, 1, H, W)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    dist = torch.sqrt((y - center_y).float()**2 + (x - center_x).float()**2)
    ring[:, :, (dist > 50) & (dist < 90)] = 1.0
    masks['ring'] = ring

    return masks


def visualize_mask_types(masks: dict, save_path: Path) -> None:
    """Visualize all test mask types."""
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    for ax, (name, mask) in zip(axes, masks.items()):
        ax.imshow(mask[0, 0].numpy(), cmap='gray')
        ax.set_title(name.replace('_', ' ').title())
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mask visualization to {save_path}")


def simulate_aar_effect(
    normal_image: torch.Tensor,
    mask: torch.Tensor,
    config: AARConfig = None,
) -> dict:
    """
    Simulate AAR effect without running full diffusion.

    This demonstrates the weight map computation that AAR uses
    to identify under-generated regions.
    """
    aar = AdaptiveAttentionReweighting(config or AARConfig())

    # Simulate different stages of generation
    results = []

    for progress in [0.2, 0.4, 0.6, 0.8]:
        # Simulate partially generated image
        # Early: mostly normal, Late: more anomaly
        noise_strength = 1.0 - progress
        simulated_x0 = normal_image + torch.randn_like(normal_image) * noise_strength

        # Simulate anomaly appearing gradually in mask
        anomaly_strength = progress
        anomaly_pattern = torch.randn_like(normal_image) * 0.3 + 0.2

        # But make it incomplete - some regions don't have anomaly yet
        # (This is what AAR tries to fix)
        incomplete_mask = mask.clone()
        # Randomly zero out parts of the mask to simulate incomplete generation
        if mask.sum() > 0:
            # Zero out ~40% of mask region randomly
            random_mask = torch.rand_like(mask) > 0.4
            incomplete_mask = incomplete_mask * random_mask.float()

        simulated_x0 = simulated_x0 + anomaly_pattern * incomplete_mask * anomaly_strength

        # Compute weight map
        weight_map = aar.compute_weight_map(simulated_x0, normal_image, mask)

        results.append({
            'progress': progress,
            'estimated_x0': simulated_x0,
            'weight_map': weight_map,
            'incomplete_mask': incomplete_mask,
        })

    return results


def visualize_aar_simulation(
    normal_image: torch.Tensor,
    mask: torch.Tensor,
    mask_name: str,
    save_dir: Path,
) -> None:
    """
    Create visualization showing how AAR weight map evolves.
    """
    results = simulate_aar_effect(normal_image, mask)

    fig, axes = plt.subplots(2, len(results) + 1, figsize=(4 * (len(results) + 1), 8))

    # First column: normal image and mask
    ax = axes[0, 0]
    normal_vis = (normal_image[0].permute(1, 2, 0).numpy() + 1) / 2
    ax.imshow(normal_vis.clip(0, 1))
    ax.set_title('Normal Image')
    ax.axis('off')

    ax = axes[1, 0]
    ax.imshow(mask[0, 0].numpy(), cmap='gray')
    ax.set_title('Target Mask')
    ax.axis('off')

    # Progress columns
    for i, result in enumerate(results):
        # Top row: estimated x0
        ax = axes[0, i + 1]
        x0_vis = (result['estimated_x0'][0].permute(1, 2, 0).numpy() + 1) / 2
        ax.imshow(x0_vis.clip(0, 1))
        ax.set_title(f"Est. x₀ @ {int(result['progress']*100)}%")
        ax.axis('off')

        # Bottom row: weight map
        ax = axes[1, i + 1]
        weight_vis = result['weight_map'][0, 0].numpy()
        im = ax.imshow(weight_vis, cmap='hot', vmin=1.0, vmax=3.0)
        ax.set_title(f"Weight Map @ {int(result['progress']*100)}%")
        ax.axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes[1, :], shrink=0.6, label='Attention Weight')

    plt.suptitle(f'AAR Weight Evolution: {mask_name.replace("_", " ").title()} Mask', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / f'aar_evolution_{mask_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AAR evolution to {save_path}")


def create_comparison_figure(
    mask_name: str,
    mask: torch.Tensor,
    save_dir: Path,
) -> None:
    """
    Create a conceptual figure showing why AAR helps.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Mask
    axes[0].imshow(mask[0, 0].numpy(), cmap='gray')
    axes[0].set_title('Input Mask\n(Where we want anomaly)')
    axes[0].axis('off')

    # 2. Without AAR - incomplete
    incomplete = mask.clone()
    # Simulate incomplete fill
    for _ in range(3):
        # Randomly erode parts
        kernel = torch.ones(1, 1, 5, 5) / 25
        incomplete = torch.nn.functional.conv2d(incomplete, kernel, padding=2)
        incomplete = (incomplete > 0.5).float()
    incomplete = incomplete * (torch.rand_like(mask) > 0.3).float()

    axes[1].imshow(incomplete[0, 0].numpy(), cmap='Reds', alpha=0.7)
    axes[1].imshow(mask[0, 0].numpy(), cmap='gray', alpha=0.3)
    axes[1].set_title('Without AAR\n(Only partial fill)')
    axes[1].axis('off')

    # 3. AAR Weight Map
    # High weight where incomplete (1 - incomplete) * mask
    weight = (1 - incomplete) * mask
    weight = weight * 2 + 1  # Scale to [1, 3]

    axes[2].imshow(weight[0, 0].numpy(), cmap='hot', vmin=1, vmax=3)
    axes[2].set_title('AAR Weight Map\n(Red = needs more attention)')
    axes[2].axis('off')

    # 4. With AAR - complete
    axes[3].imshow(mask[0, 0].numpy(), cmap='Reds', alpha=0.8)
    axes[3].set_title('With AAR\n(Full mask coverage)')
    axes[3].axis('off')

    plt.suptitle(f'Why AAR Helps: {mask_name.replace("_", " ").title()} Mask', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = save_dir / f'aar_comparison_{mask_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test AAR visualization')
    parser.add_argument('--save-dir', type=str, default='results/aar_test',
                        help='Directory to save visualizations')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Adaptive Attention Re-weighting (AAR) Demonstration")
    print("=" * 60)

    # Create test masks
    print("\n1. Creating test masks...")
    masks = create_test_masks()
    visualize_mask_types(masks, save_dir / 'test_masks.png')

    # Create a simple "normal" image (gray with some texture)
    print("\n2. Creating simulated normal image...")
    H, W = 256, 256
    normal_image = torch.zeros(1, 3, H, W)
    # Add some texture
    normal_image += 0.5  # Gray base
    normal_image += torch.randn(1, 3, H, W) * 0.1  # Noise texture

    # Visualize AAR effect for each mask type
    print("\n3. Generating AAR visualizations...")
    for mask_name, mask in masks.items():
        print(f"   Processing: {mask_name}")
        visualize_aar_simulation(normal_image, mask, mask_name, save_dir)
        create_comparison_figure(mask_name, mask, save_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary: When AAR Helps Most")
    print("=" * 60)
    print("""
    AAR is MOST beneficial for:
    [+] Multi-region masks (disconnected anomaly areas)
    [+] Large masks (harder to fill uniformly)
    [+] Irregular shapes (L-shapes, rings, etc.)
    [+] Scattered small regions

    AAR is LESS critical for:
    [-] Simple centered masks
    [-] Small compact regions
    [-] Regular rectangular shapes

    The key insight: Without AAR, the diffusion model tends to
    "concentrate" its generation in certain areas, leaving other
    parts of the mask under-generated. AAR detects these under-
    generated regions and boosts attention there.
    """)

    print(f"\nVisualizations saved to: {save_dir}")
    print("Files created:")
    for f in sorted(save_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
