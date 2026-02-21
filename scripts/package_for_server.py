"""Package project for server deployment.

Creates a zip with all code + metadata (JSON configs, captions, splits).
The image data (~218GB) must be transferred separately via rsync/scp.

Usage:
    python scripts/package_for_server.py [--include-data] [--output package.zip]

Without --include-data: creates a small (~1MB) code-only zip.
With --include-data: includes the JSON metadata files (~100MB) but NOT images.
"""
import argparse
import zipfile
from pathlib import Path


def package(output: str = "server_package.zip", include_data: bool = False):
    project_root = Path(__file__).parent.parent
    output_path = project_root / output

    # Core source code
    code_patterns = [
        "src/**/*.py",
        "scripts/train_anomagic.py",
        "scripts/live_loss.py",
        "scripts/ablation_viewer.py",
        "scripts/run_ablation_vm3.sh",
        "scripts/run_ablation_server.sh",
        "scripts/benchmark_throughput.py",
        "scripts/generate_cashew.py",
        "scripts/package_for_server.py",
        "CLAUDE.md",
        "requirements.txt",
    ]

    # Data metadata (JSONs only — no images)
    data_patterns = [
        "anomverse_extension/datasets/full_training_dataset/splits_by_type/*.json",
        "anomverse_extension/datasets/full_training_dataset/captions_from_master.json",
        "anomverse_extension/datasets/full_training_dataset/master_training.json",
    ]

    patterns = code_patterns + (data_patterns if include_data else [])

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in patterns:
            for p in project_root.glob(pattern):
                if p.is_file():
                    arcname = p.relative_to(project_root)
                    zf.write(p, arcname)
                    print(f"  + {arcname}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {output_path} ({size_mb:.1f} MB)")

    print("\n--- Server Setup Instructions ---")
    print("1. Upload this zip to the server")
    print("2. unzip server_package.zip -d anomagic/")
    if not include_data:
        print("3. Transfer data JSONs (splits + captions):")
        print("   rsync -avz anomverse_extension/datasets/full_training_dataset/splits_by_type/ server:anomagic/anomverse_extension/datasets/full_training_dataset/splits_by_type/")
        print("   scp anomverse_extension/datasets/full_training_dataset/captions_from_master.json server:anomagic/anomverse_extension/datasets/full_training_dataset/")
    print(f"{'3' if include_data else '4'}. Transfer image data (~218GB):")
    print("   rsync -avz --progress anomverse_extension/datasets/full_training_dataset/AnomVerse_data_filtered/ server:anomagic/anomverse_extension/datasets/full_training_dataset/AnomVerse_data_filtered/")
    print(f"{'4' if include_data else '5'}. Install dependencies: pip install torch torchvision diffusers transformers accelerate safetensors Pillow tqdm matplotlib prodigyopt scipy")
    print(f"{'5' if include_data else '6'}. Run ablation:")
    print("   bash scripts/run_ablation_server.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package project for server")
    parser.add_argument("--output", type=str, default="server_package.zip")
    parser.add_argument("--include-data", action="store_true",
                        help="Include JSON metadata (splits, captions, master)")
    args = parser.parse_args()
    package(output=args.output, include_data=args.include_data)
