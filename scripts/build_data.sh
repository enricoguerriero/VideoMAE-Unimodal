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
#        BACKFILL=0 bash scripts/build_data.sh     # skip the Haydom fraction backfill
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

# ---------------------------------------------------------------- backfill
# The Haydom tree was cut before data_process.py wrote `_stim0.67` fraction
# tags, so its labels were frozen at the cut the processor used and moving a
# threshold in configs/data.yaml changed DRC only — the two sites silently
# stopped meaning the same thing. --annotations rebuilds those fractions from
# the annotation files (see src/data/annotations.py) and unfreezes them.
#
# It refuses to run unless it first reproduces tags that already exist, which is
# what HAYDOM_VERIFY is for: a small tagged vintage of the same site, used as
# ground truth and never indexed into the manifest. scripts/audit_source_data.py
# section 8c is the same check, run standalone.
#
# Set BACKFILL=0 to build the manifest the old way (Haydom stays untagged and
# keeps its bucket+directory labels).
BACKFILL="${BACKFILL:-1}"
# All five directories the audit locates Haydom cases in. Two of them is not
# enough: a case ambiguous in one export (two files, different content) can be
# clean in another, and passing only 2 left 4 cases unresolved instead of 3 —
# 1,222 clips rather than 595. Order does not matter, AnnotationIndex ranks by
# size and prefers an exact filename match over a digit-run one.
HAYDOM_ANNOTATION_DIRS=(
    "/spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations"
    "/spo/LS-Haydom/Data/FullDataset/2025-2026/March2026Sync/annotations"
    "/spo/LS-Haydom/ProcessedData/Athavan_Frida/FullDataset_Combined/Annotations"
    "/spo/LS-Haydom/ProcessedData/Ronald/data/Tanzania/annotations_corrected"
    "/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Unprocessed_data/temp_folder/unique_data/videos/annotations"
)
HAYDOM_VERIFY="/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Processed_data_stratified_BIG_update_strict_label_test/videos"
DRC_ANNOTATIONS="/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing/Unprocessed_data/anot_files"

BACKFILL_ARGS=()
if [[ "$BACKFILL" == "1" ]]; then
    for d in "${HAYDOM_ANNOTATION_DIRS[@]}"; do
        [[ -d "$d" ]] && BACKFILL_ARGS+=(--annotations "Haydom=$d")
    done
    [[ -d "$HAYDOM_VERIFY" ]] && BACKFILL_ARGS+=(--verify-root "Haydom=$HAYDOM_VERIFY")
    [[ -d "$DRC_ANNOTATIONS" ]] && BACKFILL_ARGS+=(--annotations "DRC=$DRC_ANNOTATIONS")
    if [[ ${#BACKFILL_ARGS[@]} -eq 0 ]]; then
        echo "[WARN] BACKFILL=1 but no annotation directory exists — building untagged."
    fi
fi

python -m src.data.build_manifest \
    --root "Haydom=${HAYDOM_VIDEOS}" \
    --root "DRC=${DRC_VIDEOS}" \
    --data-config "${DATA_CONFIG}" \
    ${BACKFILL_ARGS[@]+"${BACKFILL_ARGS[@]}"} \
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
