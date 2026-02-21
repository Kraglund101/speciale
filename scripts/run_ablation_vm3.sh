#!/bin/bash
# Ablation: Visual Mode 3 — AdamW only
# 6 runs: 3 corrupt levels x 2 gate modes
# All: VM3, LoRA=16, BS=8, 5k steps, multi-crop, augment, band-mode 2
#
# Gate modes:
#   learnable: scalar gates init=0, block starts as identity, learns to open
#   forced:    gate=1.0 fixed, normal projection init, block active from step 0

set -e

BASE_CMD="python scripts/train_anomagic.py \
  --captions-file anomverse_extension/datasets/full_training_dataset/captions_from_master.json \
  --splits-dir anomverse_extension/datasets/full_training_dataset/splits_by_type \
  --data-root anomverse_extension/datasets/full_training_dataset \
  --ip-adapter-type plus --ip-adapter-k 16 \
  --batch-size 8 --steps 5000 --save-every 1000 \
  --augment --band-mode 2 --multi-crop \
  --visual-mode 3 --lora-rank 16 --lora-alpha 16 \
  --sa-num-layers 3 --sa-num-heads 12 \
  --cond-drop-prob 0.2 \
  --no-live-viewer"

echo "=============================================="
echo "  VM3 ABLATION (AdamW) — 6 runs x 5k steps"
echo "=============================================="

echo ""
echo "[1/6] AdamW | corrupt=0.0 | learnable gates"
$BASE_CMD --optimizer adamw --corrupt-context 0.0 \
  --save-dir results/ablation_vm3/adamw_c00_learned

echo ""
echo "[2/6] AdamW | corrupt=0.0 | forced gates"
$BASE_CMD --optimizer adamw --corrupt-context 0.0 --force-gates \
  --save-dir results/ablation_vm3/adamw_c00_forced

echo ""
echo "[3/6] AdamW | corrupt=0.2 | learnable gates"
$BASE_CMD --optimizer adamw --corrupt-context 0.2 \
  --save-dir results/ablation_vm3/adamw_c02_learned

echo ""
echo "[4/6] AdamW | corrupt=0.2 | forced gates"
$BASE_CMD --optimizer adamw --corrupt-context 0.2 --force-gates \
  --save-dir results/ablation_vm3/adamw_c02_forced

echo ""
echo "[5/6] AdamW | corrupt=0.5 | learnable gates"
$BASE_CMD --optimizer adamw --corrupt-context 0.5 \
  --save-dir results/ablation_vm3/adamw_c05_learned

echo ""
echo "[6/6] AdamW | corrupt=0.5 | forced gates"
$BASE_CMD --optimizer adamw --corrupt-context 0.5 --force-gates \
  --save-dir results/ablation_vm3/adamw_c05_forced

echo ""
echo "=============================================="
echo "  ALL 6 RUNS COMPLETE"
echo "=============================================="
