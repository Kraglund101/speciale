#!/bin/bash
# Server ablation: Visual Mode 3 — AdamW only
#
# 3 LoRA settings: none, rank=64, rank=128
# 3 corrupt levels: 0.0, 0.2, 0.5
# 2 gate modes: learnable (init=0), forced (=1, normal init)
# = 18 runs total x 5k steps each
#
# Prodigy excluded: unstable with corruption + no-LoRA (adaptive LR spirals).
#
# Adjust --batch-size based on your GPU VRAM:
#   24GB (4090/A5000): BS=12
#   40GB (A100-40):    BS=16
#   80GB (A100-80/H100): BS=24
#
# Usage:
#   PYTORCH_ALLOC_CONF=expandable_segments:True BS=12 bash scripts/run_ablation_server.sh

set -e

BS=${BS:-12}
ABLATION_DIR="results/ablation_server"

BASE_CMD="python scripts/train_anomagic.py \
  --captions-file anomverse_extension/datasets/full_training_dataset/captions_from_master.json \
  --splits-dir anomverse_extension/datasets/full_training_dataset/splits_by_type \
  --data-root anomverse_extension/datasets/full_training_dataset \
  --ip-adapter-type plus --ip-adapter-k 16 \
  --batch-size $BS --steps 5000 --save-every 1000 \
  --augment --band-mode 2 --multi-crop \
  --visual-mode 3 \
  --sa-num-layers 3 --sa-num-heads 12 \
  --cond-drop-prob 0.2 \
  --no-live-viewer"

LORA_CONFIGS=(
  "nolora 0 16"
  "lora64 64 64"
  "lora128 128 128"
)

CORRUPTIONS=("0.0" "0.2" "0.5")
CORRUPT_LABELS=("c00" "c02" "c05")
GATE_MODES=("learned" "forced")

total=18
count=0

echo "=============================================="
echo "  SERVER ABLATION (AdamW) — $total runs x 5k steps (BS=$BS)"
echo "=============================================="

# Start background comparison image saver (every 60s)
python scripts/ablation_viewer.py "$ABLATION_DIR" --save comparison.png --every 60 &
VIEWER_PID=$!
echo "Background image saver started (PID=$VIEWER_PID) — saves comparison.png every 60s"

# Ensure saver is killed on exit
trap "kill $VIEWER_PID 2>/dev/null; echo 'Stopped background image saver'" EXIT

for i in "${!LORA_CONFIGS[@]}"; do
  read -r lora_label lora_rank lora_alpha <<< "${LORA_CONFIGS[$i]}"
  for j in "${!CORRUPTIONS[@]}"; do
    corrupt="${CORRUPTIONS[$j]}"
    c_label="${CORRUPT_LABELS[$j]}"
    for gate_mode in "${GATE_MODES[@]}"; do
      count=$((count + 1))

      # Build flags
      LORA_FLAGS=""
      if [ "$lora_rank" -gt 0 ]; then
        LORA_FLAGS="--lora-rank $lora_rank --lora-alpha $lora_alpha"
      fi

      GATE_FLAG=""
      if [ "$gate_mode" = "forced" ]; then
        GATE_FLAG="--force-gates"
      fi

      run_name="adamw_${lora_label}_${c_label}_${gate_mode}"
      save_dir="${ABLATION_DIR}/${run_name}"

      echo ""
      echo "[$count/$total] adamw | $lora_label | corrupt=$corrupt | $gate_mode"
      $BASE_CMD --optimizer adamw \
        --corrupt-context "$corrupt" \
        $LORA_FLAGS \
        $GATE_FLAG \
        --save-dir "$save_dir"

      # Save comparison snapshot after each run completes
      python scripts/ablation_viewer.py "$ABLATION_DIR" --save comparison.png
      echo "  -> Saved comparison snapshot ($count/$total done)"
    done
  done
done

echo ""
echo "=============================================="
echo "  ALL $total RUNS COMPLETE"
echo "=============================================="
echo "Final comparison: ${ABLATION_DIR}/comparison.png"
