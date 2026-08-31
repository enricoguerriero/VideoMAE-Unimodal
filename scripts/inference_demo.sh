#!/usr/bin/env bash
# Qualitative demo: one episode from EACH hospital, rendered as a video with the
# model's per-second activity probabilities plotted underneath.
#
# Usage: bash scripts/inference_demo.sh [MODEL] <CKPT> [GPU] [EXTRA...]
#
#   bash scripts/inference_demo.sh VideoMAE checkpoints/<ckpt>.pt
#   bash scripts/inference_demo.sh VideoMAE <ckpt>.pt 0 --seed 7
#   bash scripts/inference_demo.sh VideoMAE <ckpt>.pt --haydom-video /path/a.mp4 \
#                                                     --drc-video /path/b.mp4
#
# With no --*-video the script picks a random case from data/test_haydom.csv and
# data/test_drc.csv and resolves the full episode from the sibling
# Unprocessed_data tree. Output goes to inference_output/.
#
# Positionals stop at the first flag, so GPU may be omitted even with EXTRA args.
set -euo pipefail

POS=()
while [[ $# -gt 0 && "$1" != -* ]]; do POS+=("$1"); shift; done
if [[ ${#POS[@]} -gt 3 ]]; then
    echo "error: too many positional arguments (${POS[*]})." >&2
    echo "usage: inference_demo.sh [MODEL] <CKPT> [GPU] [EXTRA...]" >&2
    exit 2
fi

MODEL="${POS[0]:-VideoMAE}"
CKPT="${POS[1]:?path to checkpoint .pt required}"
GPU="${POS[2]:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.inference_demo \
    --model "${MODEL}" --model_path "${CKPT}" "$@"
