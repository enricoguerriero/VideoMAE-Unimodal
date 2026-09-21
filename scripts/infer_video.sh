#!/usr/bin/env bash
# Run a trained checkpoint over an ENTIRE episode video and produce a viewer of
# the video + predictions per second.
#
# The task is read back from the checkpoint. A multiclass model draws ONE
# colour-coded timeline; a multilabel model draws ONE TIMELINE PER ACTIVITY, so
# seconds where two activities fire show two lit bars at once.
#
# Pick a case interactively from the test set (no paths to type):
#   bash scripts/infer_video.sh <MODEL> <CKPT> [GPU] [PORT] [CASE]
#
# The menu lists each episode's ground-truth clip count PER ACTIVITY. Press
# Enter to take the suggestion: a RANDOM episode whose ground truth contains
# EVERY activity — worth looking at, because an overlay with two flat tracks
# says nothing about those two. Re-running suggests a different one, so you are
# not always judging the model on the same baby, camera and operator.
# Only episodes that are actually runnable are listed: they need both a video
# and an annotation, since an overlay with no ground-truth reference is the
# least useful thing this produces. Each row shows its hospital.
#   AUTO=1   skip the prompt and take that random pick (nohup / batch runs)
#   SEED=42  make the pick reproducible, for regenerating a specific figure
#   ALL=1    list every episode, including those with no GT / no video
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
#
# RONALD=1 picks the episode from Ronald Paleczny's test manifest and resolves it
# in HIS tree (videos_corrected/ + annotations_corrected/) instead of ours. Use
# it with a checkpoint trained by `train.sh --ronald`:
#   RONALD=1 bash scripts/infer_video.sh VideoMAE checkpoints/<ckpt>.pt
set -euo pipefail

MODEL="${1:-VideoMAE}"
CKPT="${2:?path to checkpoint .pt required}"
GPU="${3:-0}"
PORT="${4:-8000}"
CASE="${5:-}"
DATA_CONFIG="${6:-}"
SERVE="${SERVE:-0}"
RONALD="${RONALD:-0}"
AUTO="${AUTO:-0}"
SEED="${SEED:-}"
ALL="${ALL:-0}"

# No --test-csv here: src.infer_video picks the right default on its own
# (data/test.csv, or his manifest under --ronald). Hardcoding it would override
# that and send a Ronald-trained checkpoint looking for OUR episodes.
ARGS=(--model "${MODEL}" --model_path "${CKPT}")
if [[ "${RONALD}" == "1" ]]; then
    ARGS+=(--ronald)
fi
if [[ "${AUTO}" == "1" ]]; then
    ARGS+=(--auto-case)
fi
if [[ -n "${SEED}" ]]; then
    ARGS+=(--seed "${SEED}")
fi
if [[ "${ALL}" == "1" ]]; then
    ARGS+=(--all-cases)
fi
if [[ -n "${CASE}" ]]; then
    ARGS+=(--case "${CASE}")
fi
if [[ -n "${DATA_CONFIG}" ]]; then
    ARGS+=(--data-config "${DATA_CONFIG}")
fi
if [[ "${SERVE}" == "1" ]]; then
    ARGS+=(--serve --port "${PORT}")
else
    ARGS+=(--render-video)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.infer_video "${ARGS[@]}"
