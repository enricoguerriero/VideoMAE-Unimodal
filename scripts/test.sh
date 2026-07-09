#!/usr/bin/env bash
# Evaluate a trained checkpoint on the held-out 14-case test split.
set -euo pipefail

MODEL="${1:-VideoMAE}"
CKPT="${2:?path to checkpoint .pt required}"
GPU="${3:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.test \
    --model "${MODEL}" \
    --model_path "${CKPT}" \
    --test_data data/test.csv \
    --results_dir results/
