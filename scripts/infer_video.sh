#!/usr/bin/env bash
# Run a trained checkpoint over an ENTIRE episode video and produce a viewer of
# the video + predicted label per second.
#
# Pick a case interactively from the test set (no paths to type):
#   bash scripts/infer_video.sh <MODEL> <CKPT> [GPU] [PORT] [CASE]
#
# By default it writes a STANDALONE annotated.mp4 (offline-friendly: no server or
# browser — just copy the file off the VM and play it in VLC). This is the right
# mode for a headless / offline VM.
#
#   bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro.pt
#   bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro.pt 0 8000 11848523
#
# Set SERVE=1 to instead serve the interactive HTML viewer over HTTP (needs a
# browser that can reach the VM, e.g. via SSH / VS Code port forwarding):
#   SERVE=1 bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro.pt 0 8000
#
# The full-episode video AND its annotation are auto-resolved from data/test.csv.
# To run on an arbitrary video instead, call the module directly with --video.
set -euo pipefail

MODEL="${1:-VideoMAE}"
CKPT="${2:?path to checkpoint .pt required}"
GPU="${3:-0}"
PORT="${4:-8000}"
CASE="${5:-}"
SERVE="${SERVE:-0}"

ARGS=(--model "${MODEL}" --model_path "${CKPT}" --test-csv data/test.csv)
if [[ -n "${CASE}" ]]; then
    ARGS+=(--case "${CASE}")
fi
if [[ "${SERVE}" == "1" ]]; then
    ARGS+=(--serve --port "${PORT}")
else
    ARGS+=(--render-video)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.infer_video "${ARGS[@]}"
