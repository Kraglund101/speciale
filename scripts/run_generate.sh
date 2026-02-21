#!/bin/bash
# Step 3: Synthetic Anomaly Generation
# Generates synthetic anomalies using finetuned IP-Adapter and concept tokens

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Synthetic Anomaly Generation ==="

# Parse arguments
K_VARIANT="${1:-k4_plus}"  # Default to k4_plus
ANOMALY_TYPE="${2:-}"      # Optional: specific anomaly type

if [ -z "$ANOMALY_TYPE" ]; then
    echo "Generating for all anomaly types with variant: $K_VARIANT"
    python -m src.inference.generate \
        --config "configs/phase2/${K_VARIANT}.yaml" \
        --ip_adapter_ckpt "checkpoints/phase2/${K_VARIANT}" \
        --concept_tokens_dir "checkpoints/phase1" \
        --output_dir "results/generated/${K_VARIANT}"
else
    echo "Generating for anomaly type: $ANOMALY_TYPE with variant: $K_VARIANT"
    python -m src.inference.generate \
        --config "configs/phase2/${K_VARIANT}.yaml" \
        --ip_adapter_ckpt "checkpoints/phase2/${K_VARIANT}" \
        --concept_tokens_dir "checkpoints/phase1" \
        --anomaly_type "$ANOMALY_TYPE" \
        --output_dir "results/generated/${K_VARIANT}"
fi

echo ""
echo "=== Generation complete ==="
