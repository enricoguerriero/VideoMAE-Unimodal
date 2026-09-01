#!/usr/bin/env bash
# Train VideoMAE (or VideoMAEGiant). Run from the repo root.
# Edit configs/config.yaml first (paths, LR, epochs).
#
# Usage: bash scripts/train.sh [MODEL] [GPU] [DATA_CONFIG] [EXTRA...]
#
#   bash scripts/train.sh VideoMAE 0                                 # multilabel (config default)
#   bash scripts/train.sh VideoMAE 0 configs/data.yaml               # 4-class, thesis-comparable
#   bash scripts/train.sh VideoMAE 0 configs/data.yaml --sites Haydom  # one hospital only
#
# DATA_CONFIG decides the task, the thresholds and the bucket keep/drop list, and
# with it the head width, the output activation and the loss. Omit it to use
# `data_config:` from configs/config.yaml.
#
# Useful EXTRA args (passed straight through to src.training):
#   --sites Haydom              train + validate on one hospital only (repeatable,
#                               case-insensitive; test sets are already per-site)
#   --attention_pooling         learned pooling instead of the pretrained fc_norm
#   --only_train                skip validation entirely
#
# GPU and DATA_CONFIG may be omitted even when passing EXTRA flags, because
# positionals are collected only until the first flag:
#   bash scripts/train.sh VideoMAE --sites Haydom
set -euo pipefail

POS=()
while [[ $# -gt 0 && "$1" != -* ]]; do POS+=("$1"); shift; done
if [[ ${#POS[@]} -gt 3 ]]; then
    echo "error: too many positional arguments (${POS[*]})." >&2
    echo "usage: train.sh [MODEL] [GPU] [DATA_CONFIG] [EXTRA...]" >&2
    exit 2
fi

MODEL="${POS[0]:-VideoMAE}"          # VideoMAE | VideoMAEGiant
GPU="${POS[1]:-0}"
DATA_CONFIG="${POS[2]:-}"

ARGS=(--model "${MODEL}")
if [[ -n "${DATA_CONFIG}" ]]; then
    ARGS+=(--data-config "${DATA_CONFIG}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.training "${ARGS[@]}" "$@"
