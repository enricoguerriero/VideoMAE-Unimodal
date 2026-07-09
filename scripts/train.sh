#!/usr/bin/env bash
# Train VideoMAE (or VideoMAEGiant) on the combined 4-class dataset.
# Run from the repo root. Edit configs/config.yaml first (paths, LR, epochs).
set -euo pipefail

MODEL="${1:-VideoMAE}"          # VideoMAE | VideoMAEGiant
GPU="${2:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.training --model "${MODEL}"
