#!/usr/bin/env bash
# Run a trained checkpoint over an ENTIRE episode video and open the synced
# viewer (video + predicted label per second).
#
#   bash scripts/infer_video.sh <MODEL> <CKPT> <VIDEO> [GPU] [PORT] [ANNOTATION]
#
# e.g.
#   bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro.pt \
#        /.../Unprocessed_data/videos/11848523.mp4 0 8000 \
#        /.../Unprocessed_data/anot_files/11848523.txt
set -euo pipefail

MODEL="${1:-VideoMAE}"
CKPT="${2:?path to checkpoint .pt required}"
VIDEO="${3:?path to full episode video required}"
GPU="${4:-0}"
PORT="${5:-8000}"
ANNOTATION="${6:-}"

ARGS=(--model "${MODEL}" --model_path "${CKPT}" --video "${VIDEO}" --port "${PORT}" --serve)
if [[ -n "${ANNOTATION}" ]]; then
    ARGS+=(--annotation "${ANNOTATION}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.infer_video "${ARGS[@]}"
