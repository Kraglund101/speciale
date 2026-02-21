"""
Evaluation metrics for generated synthetic anomalies.

Metrics included:
- FID (Fréchet Inception Distance): Distribution similarity to real anomalies
- LPIPS (Learned Perceptual Image Patch Similarity): Perceptual quality
- SSIM (Structural Similarity): Structural preservation outside mask
- Mask Consistency: Anomaly localization within mask region
- Diversity: Variation among generated samples
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class ImageMetrics:
    """
    Compute image quality and similarity metrics.

    Lightweight metrics that don't require large pretrained models.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Peak Signal-to-Noise Ratio.

        Higher = more similar. Typical range: 20-40 dB.
        """
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        C1: float = 0.01**2,
        C2: float = 0.03**2,
    ) -> float:
        """
        Structural Similarity Index.

        Range: [-1, 1], higher = more similar.
        """
        # Ensure 4D tensors [B, C, H, W]
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        # Create Gaussian window
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g = g / g.sum()
            return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

        window = gaussian_window(window_size).to(img1.device)
        window = window.expand(img1.shape[1], 1, window_size, window_size)

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    @staticmethod
    def mask_consistency(
        generated: torch.Tensor,
        background: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Measure how well the generation respects the mask boundary.

        Returns:
            - outside_change: How much the non-mask region changed (lower = better)
            - inside_change: How much the mask region changed (higher = expected)
        """
        # Ensure same shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)

        # Compute per-pixel difference
        diff = torch.abs(generated - background)

        # Outside mask (should be unchanged)
        outside_mask = 1 - mask
        outside_diff = (diff * outside_mask).sum() / (outside_mask.sum() + 1e-8)

        # Inside mask (should be changed)
        inside_diff = (diff * mask).sum() / (mask.sum() + 1e-8)

        return {
            "outside_change": outside_diff.item(),
            "inside_change": inside_diff.item(),
            "mask_respect_score": 1.0 - outside_diff.item(),  # Higher = better
        }


class LPIPSMetric:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.

    Uses pretrained VGG or AlexNet features.
    Lower LPIPS = more perceptually similar.
    """

    def __init__(self, net: str = "vgg", device: str = "cuda"):
        self.device = device
        self.net_type = net
        self.model = None

    def load(self) -> None:
        """Load LPIPS model."""
        try:
            import lpips
            self.model = lpips.LPIPS(net=self.net_type).to(self.device)
            self.model.eval()
        except ImportError:
            print("LPIPS not installed. Install with: pip install lpips")
            self.model = None

    @torch.no_grad()
    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute LPIPS between two images.

        Args:
            img1, img2: Tensors in [-1, 1] range, shape [B, 3, H, W] or [3, H, W]

        Returns:
            LPIPS distance (lower = more similar)
        """
        if self.model is None:
            self.load()
            if self.model is None:
                return float('nan')

        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        return self.model(img1, img2).mean().item()


class FIDMetric:
    """
    Fréchet Inception Distance metric.

    Measures distribution similarity between real and generated images.
    Lower FID = more similar distributions.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.inception = None
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def load(self) -> None:
        """Load Inception model for feature extraction."""
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.inception.fc = torch.nn.Identity()  # Remove classification head
            self.inception = self.inception.to(self.device)
            self.inception.eval()
        except Exception as e:
            print(f"Could not load Inception model: {e}")
            self.inception = None

    @torch.no_grad()
    def extract_features(self, images: List[Union[Image.Image, torch.Tensor]]) -> np.ndarray:
        """Extract Inception features from images."""
        if self.inception is None:
            self.load()
            if self.inception is None:
                return np.array([])

        features = []
        for img in tqdm(images, desc="Extracting features"):
            if isinstance(img, Image.Image):
                img = self.transform(img)
            if img.dim() == 3:
                img = img.unsqueeze(0)

            img = img.to(self.device)
            feat = self.inception(img)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def compute(
        self,
        real_images: List[Union[Image.Image, torch.Tensor]],
        generated_images: List[Union[Image.Image, torch.Tensor]],
    ) -> float:
        """
        Compute FID between real and generated image sets.

        Args:
            real_images: List of real anomaly images
            generated_images: List of generated anomaly images

        Returns:
            FID score (lower = better)
        """
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)

        if len(real_features) == 0 or len(gen_features) == 0:
            return float('nan')

        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(gen_features, rowvar=False)

        # Compute FID
        diff = mu_real - mu_gen

        # Matrix square root
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)

        return float(fid)


class DiversityMetric:
    """
    Measure diversity among generated samples.

    Uses pairwise LPIPS distances - higher average = more diverse.
    """

    def __init__(self, device: str = "cuda"):
        self.lpips = LPIPSMetric(device=device)

    @torch.no_grad()
    def compute(self, images: List[torch.Tensor], max_pairs: int = 100) -> Dict[str, float]:
        """
        Compute diversity metrics for a set of generated images.

        Returns:
            - mean_lpips: Average pairwise LPIPS (higher = more diverse)
            - std_lpips: Standard deviation of pairwise LPIPS
        """
        self.lpips.load()

        n = len(images)
        if n < 2:
            return {"mean_lpips": 0.0, "std_lpips": 0.0}

        # Compute pairwise distances
        distances = []
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

        if len(pairs) > max_pairs:
            import random
            pairs = random.sample(pairs, max_pairs)

        for i, j in pairs:
            dist = self.lpips.compute(images[i], images[j])
            if not np.isnan(dist):
                distances.append(dist)

        if not distances:
            return {"mean_lpips": float('nan'), "std_lpips": float('nan')}

        return {
            "mean_lpips": float(np.mean(distances)),
            "std_lpips": float(np.std(distances)),
        }


class AnomalyEvaluator:
    """
    Comprehensive evaluator for synthetic anomaly generation.

    Combines multiple metrics for thorough assessment.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.image_metrics = ImageMetrics(device)
        self.lpips = LPIPSMetric(device=device)
        self.fid = FIDMetric(device=device)
        self.diversity = DiversityMetric(device=device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _load_image(self, path: Union[str, Path, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Load and preprocess image."""
        if isinstance(path, torch.Tensor):
            return path
        if isinstance(path, (str, Path)):
            path = Image.open(path).convert("RGB")
        return self.transform(path)

    def evaluate_single(
        self,
        generated: Union[str, Path, Image.Image, torch.Tensor],
        reference: Union[str, Path, Image.Image, torch.Tensor],
        background: Union[str, Path, Image.Image, torch.Tensor],
        mask: Union[str, Path, Image.Image, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Evaluate a single generated image.

        Args:
            generated: Generated anomaly image
            reference: Real reference anomaly image
            background: Original background/normal image
            mask: Anomaly mask

        Returns:
            Dictionary of metrics
        """
        gen = self._load_image(generated).to(self.device)
        ref = self._load_image(reference).to(self.device)
        bg = self._load_image(background).to(self.device)

        # Load mask
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask).convert("L")
        if isinstance(mask, Image.Image):
            mask = transforms.ToTensor()(mask)
        mask = mask.to(self.device)

        results = {}

        # Perceptual similarity to reference
        results["lpips_to_reference"] = self.lpips.compute(gen, ref)

        # Structural metrics
        results["ssim_to_reference"] = self.image_metrics.ssim(gen, ref)
        results["psnr_to_reference"] = self.image_metrics.psnr(gen, ref)

        # Mask consistency
        mask_metrics = self.image_metrics.mask_consistency(gen, bg, mask)
        results.update(mask_metrics)

        return results

    def evaluate_batch(
        self,
        generated_images: List[Union[str, Path, Image.Image]],
        reference_images: List[Union[str, Path, Image.Image]],
        background_images: List[Union[str, Path, Image.Image]],
        masks: List[Union[str, Path, Image.Image]],
        compute_fid: bool = True,
        compute_diversity: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of generated images.

        Returns aggregated metrics across all samples.
        """
        all_metrics = []
        gen_tensors = []

        print("Evaluating individual samples...")
        for gen, ref, bg, mask in tqdm(
            zip(generated_images, reference_images, background_images, masks),
            total=len(generated_images)
        ):
            metrics = self.evaluate_single(gen, ref, bg, mask)
            all_metrics.append(metrics)
            gen_tensors.append(self._load_image(gen))

        # Aggregate metrics
        results = {}
        metric_names = all_metrics[0].keys()

        for name in metric_names:
            values = [m[name] for m in all_metrics if not np.isnan(m[name])]
            if values:
                results[f"{name}_mean"] = float(np.mean(values))
                results[f"{name}_std"] = float(np.std(values))

        # FID (distribution metric)
        if compute_fid and len(generated_images) >= 10:
            print("Computing FID...")
            ref_pil = [Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p
                       for p in reference_images]
            gen_pil = [Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p
                       for p in generated_images]
            results["fid"] = self.fid.compute(ref_pil, gen_pil)

        # Diversity
        if compute_diversity and len(gen_tensors) >= 2:
            print("Computing diversity...")
            diversity_metrics = self.diversity.compute(gen_tensors)
            results["diversity_mean_lpips"] = diversity_metrics["mean_lpips"]
            results["diversity_std_lpips"] = diversity_metrics["std_lpips"]

        return results

    def create_comparison_grid(
        self,
        generated: Union[str, Path, Image.Image],
        reference: Union[str, Path, Image.Image],
        background: Union[str, Path, Image.Image],
        mask: Union[str, Path, Image.Image],
        output_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Create a visual comparison grid.

        Layout: [Background | Mask | Reference | Generated]
        """
        # Load images
        if isinstance(background, (str, Path)):
            background = Image.open(background).convert("RGB")
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask).convert("L")
        if isinstance(reference, (str, Path)):
            reference = Image.open(reference).convert("RGB")
        if isinstance(generated, (str, Path)):
            generated = Image.open(generated).convert("RGB")

        # Resize to common size
        size = (256, 256)
        background = background.resize(size)
        mask = mask.resize(size)
        reference = reference.resize(size)
        generated = generated.resize(size)

        # Create grid
        width, height = size
        grid = Image.new("RGB", (width * 4, height))

        grid.paste(background, (0, 0))
        grid.paste(mask.convert("RGB"), (width, 0))
        grid.paste(reference, (width * 2, 0))
        grid.paste(generated, (width * 3, 0))

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(output_path)

        return grid

    def generate_report(
        self,
        metrics: Dict[str, float],
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate a text report of evaluation metrics."""
        lines = [
            "=" * 60,
            "Synthetic Anomaly Evaluation Report",
            "=" * 60,
            "",
            "Perceptual Quality:",
            f"  LPIPS to reference (mean): {metrics.get('lpips_to_reference_mean', 'N/A'):.4f}",
            f"  SSIM to reference (mean):  {metrics.get('ssim_to_reference_mean', 'N/A'):.4f}",
            f"  PSNR to reference (mean):  {metrics.get('psnr_to_reference_mean', 'N/A'):.2f} dB",
            "",
            "Mask Consistency:",
            f"  Outside change (mean):     {metrics.get('outside_change_mean', 'N/A'):.4f} (lower=better)",
            f"  Inside change (mean):      {metrics.get('inside_change_mean', 'N/A'):.4f}",
            f"  Mask respect score (mean): {metrics.get('mask_respect_score_mean', 'N/A'):.4f}",
            "",
        ]

        if "fid" in metrics:
            lines.extend([
                "Distribution Similarity:",
                f"  FID: {metrics['fid']:.2f} (lower=better)",
                "",
            ])

        if "diversity_mean_lpips" in metrics:
            lines.extend([
                "Diversity:",
                f"  Mean pairwise LPIPS: {metrics['diversity_mean_lpips']:.4f} (higher=more diverse)",
                "",
            ])

        lines.extend([
            "=" * 60,
            "",
            "Interpretation:",
            "  - LPIPS < 0.3: High similarity to reference",
            "  - SSIM > 0.8: Good structural preservation",
            "  - Mask respect > 0.95: Anomaly well-localized",
            "  - FID < 50: Good distribution match",
            "",
        ])

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

            # Also save JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2)

        return report


def evaluate_generation(
    generated_dir: Path,
    reference_dir: Path,
    background_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Convenience function to evaluate a directory of generated images.

    Expects matching filenames in each directory.
    """
    generated_dir = Path(generated_dir)
    reference_dir = Path(reference_dir)
    background_dir = Path(background_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching files
    generated_files = sorted(generated_dir.glob("*.png"))

    generated_images = []
    reference_images = []
    background_images = []
    masks = []

    for gen_path in generated_files:
        name = gen_path.stem
        ref_path = reference_dir / f"{name}.png"
        bg_path = background_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"

        if ref_path.exists() and bg_path.exists() and mask_path.exists():
            generated_images.append(gen_path)
            reference_images.append(ref_path)
            background_images.append(bg_path)
            masks.append(mask_path)

    print(f"Found {len(generated_images)} matching image sets")

    # Evaluate
    evaluator = AnomalyEvaluator(device=device)
    metrics = evaluator.evaluate_batch(
        generated_images=generated_images,
        reference_images=reference_images,
        background_images=background_images,
        masks=masks,
    )

    # Generate report
    report = evaluator.generate_report(metrics, output_dir / "evaluation_report.txt")
    print(report)

    # Create comparison grids for first few samples
    print("Creating comparison grids...")
    for i, (gen, ref, bg, mask) in enumerate(
        zip(generated_images[:10], reference_images[:10], background_images[:10], masks[:10])
    ):
        evaluator.create_comparison_grid(
            generated=gen,
            reference=ref,
            background=bg,
            mask=mask,
            output_path=output_dir / f"comparison_{i:03d}.png",
        )

    return metrics
