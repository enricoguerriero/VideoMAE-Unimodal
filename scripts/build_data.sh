#!/usr/bin/env bash
# Build the combined manifest from existing processed clips (both sites), then
# split into train / validation / ONE TEST SET PER HOSPITAL at the whole-case
# level. Each site's test set is ~TEST_RATIO of that site's own clips, seeded
# with the thesis' 14 frozen cases (which stay flagged by the `thesis_test`
# column, so the thesis-comparable evaluation is still one filter away).
#
# Both outputs are TASK-AGNOSTIC: they record each clip's label bucket and its
# per-activity window fractions, not a resolved label. Switching multiclass <->
# multilabel, moving a threshold or admitting a bucket is a configs/data.yaml
# edit — you do NOT need to re-run this script for any of that.
#
# EDIT the two clip roots below to point at the thesis' processed video clips
# on the VM (the `.../videos` directory that contains the per-class subfolders).
#
# Usage: bash scripts/build_data.sh [DATA_CONFIG] [TEST_RATIO] [TRAIN_RATIO]
#   DATA_CONFIG only affects the reported label distribution (and `tag_keys`) —
#   the split itself is deliberately independent of it.
#
# Audit the result (before or after) with:
#   python -m src.data.explore_data --manifest data/clips_all.csv --splits-dir data
set -euo pipefail

# Defaults to whatever `data_config:` in configs/config.yaml says, so the census
# you read here is the one training will actually use. Override as argument 1.
DATA_CONFIG="${1:-$(python -c "import yaml;print(yaml.safe_load(open('configs/config.yaml')).get('data_config','configs/data.yaml'))")}"
TEST_RATIO="${2:-0.20}"    # share of EACH site's clips held out as its test set
TRAIN_RATIO="${3:-0.80}"   # share of the remainder used for training

HAYDOM_VIDEOS="/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Processed_data_stratified_BIG_update_strict_label/videos"
DRC_VIDEOS="/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing/Processed_data_new_dataset_no_suction_merge_bulp_new_anot_chestmov/videos"

python -m src.data.build_manifest \
    --root "Haydom=${HAYDOM_VIDEOS}" \
    --root "DRC=${DRC_VIDEOS}" \
    --data-config "${DATA_CONFIG}" \
    --out data/clips_all.csv

python -m src.data.split_cases \
    --manifest data/clips_all.csv \
    --out-dir data \
    --data-config "${DATA_CONFIG}" \
    --test-ratio "${TEST_RATIO}" \
    --train-ratio "${TRAIN_RATIO}" \
    --seed 2025

python -m src.data.explore_data \
    --manifest data/clips_all.csv \
    --splits-dir data \
    --data-config "${DATA_CONFIG}" \
    --target-test-ratio "${TEST_RATIO}" \
    --out-dir results/data_report
