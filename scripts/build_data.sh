#!/usr/bin/env bash
# Build the combined manifest from existing processed clips (both sites), then
# split into train/validation/test at the whole-case level (14 fixed test cases).
#
# EDIT the two clip roots below to point at the thesis' processed video clips
# on the VM (the `.../videos` directory that contains the per-class subfolders).
set -euo pipefail

HAYDOM_VIDEOS="/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Processed_data_stratified_BIG_update_strict_label/videos"
DRC_VIDEOS="/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing//Processed_data_new_dataset_no_suction_merge_bulp_new_anot_chestmov/videos"

python -m src.data.build_manifest \
    --root "Haydom=${HAYDOM_VIDEOS}" \
    --root "DRC=${DRC_VIDEOS}" \
    --out data/clips_all.csv

python -m src.data.split_cases \
    --manifest data/clips_all.csv \
    --out-dir data \
    --train-ratio 0.7 \
    --seed 2025
