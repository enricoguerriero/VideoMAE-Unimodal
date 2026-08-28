#!/usr/bin/env bash
# Train VideoMAE (or VideoMAEGiant). Run from the repo root.
# Edit configs/config.yaml first (paths, LR, epochs).
#
# Usage: bash scripts/train.sh [MODEL] [GPU] [DATA_CONFIG]
#
#   bash scripts/train.sh VideoMAE 0                                # multilabel (config default)
#   bash scripts/train.sh VideoMAE 0 configs/data.yaml               # 4-class, thesis-comparable
#
# DATA_CONFIG decides the task, the thresholds and the bucket keep/drop list, and
# with it the head width, the output activation and the loss. Omit it to use
# `data_config:` from configs/config.yaml.
set -euo pipefail

MODEL="${1:-VideoMAE}"          # VideoMAE | VideoMAEGiant
GPU="${2:-0}"
DATA_CONFIG="${3:-}"

ARGS=(--model "${MODEL}")
if [[ -n "${DATA_CONFIG}" ]]; then
    ARGS+=(--data-config "${DATA_CONFIG}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.training "${ARGS[@]}"
