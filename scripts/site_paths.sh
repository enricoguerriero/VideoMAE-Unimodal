# =============================================================================
# site_paths.sh — where each hospital's data lives on the VM. SOURCE, don't run.
# =============================================================================
#   source "$(dirname "${BASH_SOURCE[0]}")/site_paths.sh"
#
# One place for the absolute paths, so a script that needs annotations does not
# ask the caller for something the repo already knows. Every value is
# `${VAR:-default}`, so exporting VAR before the call still overrides it.
#
# scripts/build_data.sh predates this file and still carries its own copy of the
# same lists. It is the pipeline that produces data/, so it was left alone
# rather than refactored under a working run; consolidating it is a one-line
# change (drop its definitions, source this) whenever that is convenient.
# =============================================================================

# ---------------------------------------------------------------- clip roots
HAYDOM_VIDEOS="${HAYDOM_VIDEOS:-/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Processed_data_stratified_BIG_update_strict_label/videos}"
DRC_VIDEOS="${DRC_VIDEOS:-/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing/Processed_data_new_dataset_no_suction_merge_bulp_new_anot_chestmov/videos}"

# A small TAGGED vintage of Haydom, used only to verify the fraction backfill.
HAYDOM_VERIFY="${HAYDOM_VERIFY:-/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing/Processed_data_stratified_BIG_update_strict_label_test/videos}"

# ---------------------------------------------------------------- site bases
HAYDOM_BASE="${HAYDOM_BASE:-/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing}"
DRC_BASE="${DRC_BASE:-/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing}"

# ---------------------------------------------------------------- annotations
# DRC keeps a cleaned anot_files/ next to its videos, so anything walking up
# from a clip path finds it on its own. Haydom has NO anot_files/ and its files
# are not named after the case, which is why the list below exists at all:
# these are the directories the audit located, ordered by the coverage it
# measured of the 246 cases in the clip tree.
#
#   .../FullDataset/2023-2025/Annotations                  242/246  (98%)
#   .../FullDataset/2025-2026/March2026Sync/annotations    241/246  (98%)
#   .../FullDataset_Combined/Annotations                   240/246  (98%)
#   .../Ronald/data/Tanzania/annotations_corrected         240/246  (98%)  no conflicts
#   .../temp_folder/unique_data/videos/annotations         123/246  (50%)
#
# Callers that only need to LOOK something up (an inference overlay) should take
# all of them: matching is by case key, coverage is what matters, and a
# qualitative reference is not a training label. Callers that produce LABELS
# should prefer one directory — see the note in scripts/recut_haydom.sh about
# mixing export vintages.
HAYDOM_ANNOTATION_DIRS=(
    "/spo/LS-Haydom/ProcessedData/Ronald/data/Tanzania/annotations_corrected"
    "/spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations"
    "/spo/LS-Haydom/Data/FullDataset/2025-2026/March2026Sync/annotations"
    "/spo/LS-Haydom/ProcessedData/Athavan_Frida/FullDataset_Combined/Annotations"
    "$HAYDOM_BASE/Unprocessed_data/temp_folder/unique_data/videos/annotations"
)
DRC_ANNOTATIONS="${DRC_ANNOTATIONS:-$DRC_BASE/Unprocessed_data/anot_files}"

# Every annotation directory for either site, existing ones only. This is what a
# lookup-only caller wants.
ALL_ANNOTATION_DIRS=()
for _d in "${HAYDOM_ANNOTATION_DIRS[@]}" "$DRC_ANNOTATIONS"; do
    [[ -d "$_d" ]] && ALL_ANNOTATION_DIRS+=("$_d")
done
unset _d
