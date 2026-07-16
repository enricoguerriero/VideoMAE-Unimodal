#!/usr/bin/env bash
# Run a trained checkpoint over an ENTIRE episode video and open the synced
# viewer (video + predicted label per second).
#
# Pick a case interactively from the test set (no paths to type):
#   bash scripts/infer_video.sh <MODEL> <CKPT> [GPU] [PORT] [CASE]
#
# e.g.
#   bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro.pt
#   bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro.pt 0 8000 11848523
#
# The full-episode video AND its annotation are auto-resolved from data/test.csv.
# To run on an arbitrary video instead, call the module directly with --video.
set -euo pipefail

MODEL="${1:-VideoMAE}"
CKPT="${2:?path to checkpoint .pt required}"
GPU="${3:-0}"
PORT="${4:-8000}"
CASE="${5:-}"

ARGS=(--model "${MODEL}" --model_path "${CKPT}" --test-csv data/test.csv --port "${PORT}" --serve)
if [[ -n "${CASE}" ]]; then
    ARGS+=(--case "${CASE}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.infer_video "${ARGS[@]}"
