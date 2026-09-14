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
# GROUND TRUTH is found automatically. The annotation directories come from
# scripts/site_paths.sh, so you never pass them: DRC has a cleaned anot_files/
# beside its videos, and Haydom does not, so the known Haydom export directories
# are handed over as a fallback and matched by case id. Override with
# GT_DIRS="/a /b" if you need a specific one, or pass your own --gt-dir.
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

# shellcheck source=scripts/site_paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/site_paths.sh"

ARGS=(--model "${MODEL}" --model_path "${CKPT}")

# Ground-truth directories are supplied from site_paths.sh, NOT asked for. Only
# when the caller names a --gt-dir itself is the default list left out, so an
# explicit one replaces it rather than being appended to it.
if [[ " $* " != *" --gt-dir "* ]]; then
    GT_CANDIDATES=()
    if [[ -n "${GT_DIRS:-}" ]]; then
        read -r -a GT_CANDIDATES <<< "$GT_DIRS"
    else
        GT_CANDIDATES=(${ALL_ANNOTATION_DIRS[@]+"${ALL_ANNOTATION_DIRS[@]}"})
    fi
    n_gt=0
    for d in ${GT_CANDIDATES[@]+"${GT_CANDIDATES[@]}"}; do
        if [[ -d "$d" ]]; then ARGS+=(--gt-dir "$d"); n_gt=$((n_gt + 1))
        else echo "[warn] annotation dir absent, skipping: $d" >&2; fi
    done
    if [[ $n_gt -eq 0 ]]; then
        echo "[warn] no annotation directory found — the panels will show the" >&2
        echo "       prediction curves with no ground-truth shading." >&2
    else
        echo "[info] ground truth from $n_gt annotation director(y/ies)"
    fi
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.inference_demo "${ARGS[@]}" "$@"
