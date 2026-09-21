#!/usr/bin/env bash
# Evaluate a trained checkpoint on BOTH hospitals' held-out test sets.
#
# Usage: bash scripts/test.sh [MODEL] <CKPT> [GPU] [DATA_CONFIG] [EXTRA...]
#
# With no EXTRA args it evaluates the test sets listed under `test_data:` in the
# checkpoint's config — data/test_haydom.csv and data/test_drc.csv — writing one
# results_*.csv / scores_*.npz per site and printing a side-by-side comparison.
# A gap between the two IS the cross-site generalisation result; do not average
# it away.
#
# ONE EXCEPTION, and it is the one you want after `train.sh --sites Haydom`: a
# checkpoint that records which hospitals it was TRAINED on is scored on those
# only. Scoring a Haydom-only model on DRC by default answers a question nobody
# asked and buries the number that matters in a two-row table. Pass
# `--sites all` to score everything anyway — that IS the cross-site experiment,
# and it is worth running on purpose.
#
# The task (multiclass/multilabel, class names, decision thresholds) is read back
# from the checkpoint, so nothing extra is needed for a multilabel model. Pass
# DATA_CONFIG only to deliberately override it — e.g. to re-score the same
# checkpoint at different `decision_thresholds` (use "" to skip the argument).
#
# Useful EXTRA args:
#   --ronald                      score on Ronald Paleczny's test manifest
#                                 (implied by a checkpoint trained with --ronald)
#   --sites Haydom                score one hospital only (default: whichever
#                                 the checkpoint was trained on)
#   --sites all                   score every test set, including hospitals this
#                                 model never saw — the generalisation number
#   --thesis-only                 score only the thesis' frozen cases (the
#                                 like-for-like multimodal comparison)
#   --test_data data/test.csv     one pooled score over both sites instead
#   --test_data val=data/validation.csv   score validation, to tune thresholds on
#                                 (see src/tune_thresholds.py)
#   --legacy-pooling on|off|auto  pooling for checkpoints trained before the
#                                 2026-08-31 fix (head fitted on one patch token
#                                 instead of the mean over all of them). `auto`,
#                                 the default, detects them and does the right
#                                 thing; you should not need to pass this.
#
# GPU and DATA_CONFIG may be omitted even when passing EXTRA flags:
#   bash scripts/test.sh VideoMAE <ckpt>.pt --test_data val=data/validation.csv
set -euo pipefail

# Positional args are collected only until the first flag, so a passthrough like
# `--test_data ...` can never be mistaken for the GPU or DATA_CONFIG slot. Both
# of these therefore work:
#   test.sh VideoMAE ckpt.pt --test_data val=data/validation.csv
#   test.sh VideoMAE ckpt.pt 0 "" --test_data val=data/validation.csv
POS=()
while [[ $# -gt 0 && "$1" != -* ]]; do POS+=("$1"); shift; done
if [[ ${#POS[@]} -gt 4 ]]; then
    echo "error: too many positional arguments (${POS[*]})." >&2
    echo "usage: test.sh [MODEL] <CKPT> [GPU] [DATA_CONFIG] [EXTRA...]" >&2
    exit 2
fi

MODEL="${POS[0]:-VideoMAE}"
CKPT="${POS[1]:?path to checkpoint .pt required}"
GPU="${POS[2]:-0}"
DATA_CONFIG="${POS[3]:-}"

ARGS=(--model "${MODEL}" --model_path "${CKPT}" --results_dir results/)
if [[ -n "${DATA_CONFIG}" ]]; then
    ARGS+=(--data-config "${DATA_CONFIG}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.test "${ARGS[@]}" "$@"
