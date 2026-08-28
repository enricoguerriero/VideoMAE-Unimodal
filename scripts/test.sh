#!/usr/bin/env bash
# Evaluate a trained checkpoint on BOTH hospitals' held-out test sets.
#
# Usage: bash scripts/test.sh [MODEL] <CKPT> [GPU] [DATA_CONFIG] [EXTRA...]
#
# With no EXTRA args it evaluates every test set listed under `test_data:` in the
# checkpoint's config — data/test_haydom.csv and data/test_drc.csv — writing one
# results_*.csv / scores_*.npz per site and printing a side-by-side comparison.
# A gap between the two IS the cross-site generalisation result; do not average
# it away.
#
# The task (multiclass/multilabel, class names, decision thresholds) is read back
# from the checkpoint, so nothing extra is needed for a multilabel model. Pass
# DATA_CONFIG only to deliberately override it — e.g. to re-score the same
# checkpoint at different `decision_thresholds` (use "" to skip the argument).
#
# Useful EXTRA args:
#   --thesis-only                 score only the thesis' 14 frozen cases (the
#                                 like-for-like multimodal comparison)
#   --test_data data/test.csv     one pooled score over both sites instead
#   --test_data haydom=data/test_haydom.csv   an explicit, named subset
set -euo pipefail

MODEL="${1:-VideoMAE}"
CKPT="${2:?path to checkpoint .pt required}"
GPU="${3:-0}"
DATA_CONFIG="${4:-}"
shift $(( $# > 4 ? 4 : $# ))

ARGS=(--model "${MODEL}" --model_path "${CKPT}" --results_dir results/)
if [[ -n "${DATA_CONFIG}" ]]; then
    ARGS+=(--data-config "${DATA_CONFIG}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.test "${ARGS[@]}" "$@"
