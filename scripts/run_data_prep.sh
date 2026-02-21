#!/bin/bash
# Step 0: Data Preparation
# Splits raw JSON annotations by anomaly type and preprocesses images

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Data Preparation Pipeline ==="
echo "Project root: $PROJECT_ROOT"

# Step 1: Split JSON annotations by anomaly type
echo ""
echo "--- Splitting annotations by anomaly type ---"
python -m src.data.split_by_type \
    --input_dir data/AnomVerse_data \
    --output_dir data/concepts

# Step 2: Preprocess images
echo ""
echo "--- Preprocessing images ---"
python -m src.data.preprocess \
    --splits_dir data/concepts \
    --output_dir data/processed \
    --image_size 512

echo ""
echo "=== Data preparation complete ==="
