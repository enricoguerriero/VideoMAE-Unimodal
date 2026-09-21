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

# RE-CUT TREE (2026-09: 461 cases / ~138.9k clips, cut from Ronald's corrected
# annotations by scripts/recut_haydom.sh). The old tree is still on disk at
# Processed_data_stratified_BIG_update_strict_label/videos — swap the two lines
# back to rebuild the pre-recut manifest.
HAYDOM_VIDEOS="/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Processed_data_recut/videos"
DRC_VIDEOS="/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing/Processed_data_new_dataset_no_suction_merge_bulp_new_anot_chestmov/videos"

# ---------------------------------------------------------------- backfill
# Since the re-cut, Haydom's clips DO carry `_stim0.67` fraction tags, so the
# backfill is no longer about unfreezing an untagged site. It is still required
# for a PER-DEVICE config: filenames only ever carried `_stim/_vent/_suct`, so
# penguin/bulb/tube can only come from the annotation files' column 5.
#
# The backfill refuses to run unless the same computation reproduces tags that
# already exist; the re-cut tree's own tagged rows are that reference now.
#
# Set BACKFILL=0 to build the manifest the old way (Haydom stays untagged and
# keeps its bucket+directory labels).
BACKFILL="${BACKFILL:-1}"
# ONE directory now: the anot_files recut_haydom.sh STAGED. They are the exact
# annotations the re-cut clips were produced from, keyed by the same case ids,
# and they keep the original event string in column 5 — which is what lets the
# per-device backfill resolve penguin/bulb/tube. Mixing the other exports back in
# would reintroduce the vintage confound the re-cut exists to remove.
HAYDOM_ANNOTATION_DIRS=(
    "/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Data_processing_recut/Unprocessed_data/anot_files"
)
# EMPTY since the re-cut. --verify-root is only needed when a site's own clips
# carry no fraction tags; the re-cut tree is tagged (data_process.py:286), so
# the manifest's own Haydom rows are the reference. The old vintage below was
# cut from DIFFERENT annotations, so verifying against it would compare two
# pipeline runs and fail the >= min-agreement gate, silently skipping the
# backfill and leaving no per-device suction fractions.
#   old: .../Processed_data_stratified_BIG_update_strict_label_test/videos
HAYDOM_VERIFY=""
DRC_ANNOTATIONS="/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing/Unprocessed_data/anot_files"

# --rebackfill-all is not optional once a data config slices an activity more
# finely than the filenames do. Filenames only ever carried `_stim/_vent/_suct`,
# so a bucket-1 clip looks fully tagged while knowing nothing about tube: without
# this flag it keeps a false frac_suction_tube=0.00. --extra-frac suction stores
# the legacy aggregate next to the per-device columns, so ONE manifest and ONE
# case split serve both configs/data_multilabel.yaml (control) and
# configs/data_suction3.yaml (per-device) — the only way the split's effect is
# attributable to the split rather than to a reshuffled test set.
BACKFILL_ARGS=(--rebackfill-all --extra-frac suction)
if [[ "$BACKFILL" == "1" ]]; then
    for d in "${HAYDOM_ANNOTATION_DIRS[@]}"; do
        [[ -d "$d" ]] && BACKFILL_ARGS+=(--annotations "Haydom=$d")
    done
    [[ -d "$HAYDOM_VERIFY" ]] && BACKFILL_ARGS+=(--verify-root "Haydom=$HAYDOM_VERIFY")
    [[ -d "$DRC_ANNOTATIONS" ]] && BACKFILL_ARGS+=(--annotations "DRC=$DRC_ANNOTATIONS")
    if [[ ${#BACKFILL_ARGS[@]} -eq 2 ]]; then
        echo "[WARN] BACKFILL=1 but no annotation directory exists — building untagged."
        echo "       A per-device config CANNOT work from an untagged manifest: no"
        echo "       directory names suction_penguin/bulb/tube, so those clips drop."
    fi
else
    BACKFILL_ARGS=()
fi

python -m src.data.build_manifest \
    --root "Haydom=${HAYDOM_VIDEOS}" \
    --root "DRC=${DRC_VIDEOS}" \
    --data-config "${DATA_CONFIG}" \
    ${BACKFILL_ARGS[@]+"${BACKFILL_ARGS[@]}"} \
    --out data/clips_all.csv

# Extra flags for the case selector, e.g. reproducing the pre-2026-08-28 split:
#   SPLIT_ARGS="--freeze-test --no-balanced-val --train-ratio 0.7" bash scripts/build_data.sh
# Appended last, so they win over the ratios above (argparse keeps the final
# occurrence). See configs/config_thesis.yaml.
SPLIT_EXTRA=()
[[ -n "${SPLIT_ARGS:-}" ]] && read -r -a SPLIT_EXTRA <<< "$SPLIT_ARGS"

python -m src.data.split_cases \
    --manifest data/clips_all.csv \
    --out-dir data \
    --data-config "${DATA_CONFIG}" \
    --test-ratio "${TEST_RATIO}" \
    --train-ratio "${TRAIN_RATIO}" \
    --seed 2025 \
    ${SPLIT_EXTRA[@]+"${SPLIT_EXTRA[@]}"}

# --out-dir writes per_case.csv AND report.txt (the whole printed audit), so the
# build leaves a pasteable record of what it produced.
python -m src.data.explore_data \
    --manifest data/clips_all.csv \
    --splits-dir data \
    --data-config "${DATA_CONFIG}" \
    --target-test-ratio "${TEST_RATIO}" \
    --out-dir results/data_report
