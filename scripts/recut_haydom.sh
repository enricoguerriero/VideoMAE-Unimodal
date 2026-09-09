#!/usr/bin/env bash
# =============================================================================
# recut_haydom.sh — recover the Haydom episodes the current clip tree is missing
# =============================================================================
# The clip tree build_data.sh indexes covers 246 Haydom cases.
# `Unprocessed_data/videos` holds 400 videos, and the annotation directories
# cover 242-243 of the 246 cases each — so the material for the other ~154 is
# already on the VM. Per-case clip density is identical to Ronald Paleczny's
# Haydom pipeline (290 s of clips per case vs his 288), which is what says the
# gap is missing EPISODES, not a different extraction rule.
#
# Cause (scripts/audit_source_data.py section 8b): Haydom's annotation files are
# not named after the case, `Unprocessed_data/anot_files/` is ABSENT for this
# site, and the original driver pairs videos to annotations on the exact
# filename stem. src/data/recut_site.py pairs by case KEY instead — exact stem
# OR any run of >= 5 digits, the same rule build_manifest.py's backfill already
# uses successfully here.
#
# Usage:
#   bash scripts/recut_haydom.sh              # DRY RUN: project, write nothing
#   bash scripts/recut_haydom.sh --yes        # stage + cut (hours; tens of GB)
#   WORKERS=6 bash scripts/recut_haydom.sh --yes
#   EXTRA_ANNOTATIONS=1 bash scripts/recut_haydom.sh   # add the fallback dirs
#
# Anything after the script name is passed straight through to recut_site.py,
# so --limit 3 / --only 11848523 / --skip-existing all work.
set -euo pipefail

HAYDOM_BASE="${HAYDOM_BASE:-/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing}"
VIDEOS="${VIDEOS:-$HAYDOM_BASE/Unprocessed_data/videos}"

# ---------------------------------------------------------------- annotations
# ONE primary source by default, deliberately. The audit found the four Haydom
# annotation directories cover 240-242 of 246 cases EACH, so a second directory
# buys ~2 cases while mixing two export vintages into one label set — the exact
# confound findings [7]-[9] are about. 2023-2025/Annotations is the primary
# because it has the highest coverage (242/246) and its suction vocabulary is
# intact ('Suction using Penguine Device' and friends).
#
# Set EXTRA_ANNOTATIONS=1 to add the rest. AnnotationIndex ranks directories by
# size and tries the exact case id everywhere before any digit-run match, so a
# case that is CONFLICTED inside one directory can still resolve from another —
# which is the only reason to want them. Ronald's annotations_corrected is
# listed first among the fallbacks because the audit found it has zero
# conflicting keys (his conflict rule already resolved them).
ANNOTATIONS=("/spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations")
if [[ "${EXTRA_ANNOTATIONS:-0}" == "1" ]]; then
    ANNOTATIONS+=(
        "/spo/LS-Haydom/ProcessedData/Ronald/data/Tanzania/annotations_corrected"
        "/spo/LS-Haydom/Data/FullDataset/2025-2026/March2026Sync/annotations"
        "/spo/LS-Haydom/ProcessedData/Athavan_Frida/FullDataset_Combined/Annotations"
    )
fi

# ---------------------------------------------------------------- outputs
# Siblings of the existing trees, with _recut in the name, so nothing the
# current build_data.sh reads is touched. The old tree stays exactly where it
# is — this is additive, and reverting means pointing build_data.sh back.
STAGE_DIR="${STAGE_DIR:-$HAYDOM_BASE/Data_processing_recut}"
OUT_DIR="${OUT_DIR:-$HAYDOM_BASE/Processed_data_recut}"
WORKERS="${WORKERS:-1}"
COMPARE="${COMPARE:-data/clips_all.csv}"

ARGS=(--site Haydom --videos "$VIDEOS" --stage-dir "$STAGE_DIR" --out "$OUT_DIR"
      --workers "$WORKERS")
for a in "${ANNOTATIONS[@]}"; do
    if [[ -d "$a" ]]; then ARGS+=(--annotations "$a")
    else echo "[warn] annotation dir absent, skipping: $a"; fi
done
[[ -f "$COMPARE" ]] && ARGS+=(--compare "$COMPARE") \
    || echo "[info] $COMPARE not found — the projection will have nothing to diff against"

# No --yes and no --dry-run given => recut_site.py projects and stops, which is
# what we want the bare invocation to do. Pass --dry-run explicitly to make that
# unmistakable in the log.
if [[ $# -eq 0 ]]; then
    ARGS+=(--dry-run)
    echo "[info] no arguments: DRY RUN. Re-run with --yes to stage and cut."
fi

set -x
python -m src.data.recut_site "${ARGS[@]}" "$@"
set +x

cat <<'NOTE'

-----------------------------------------------------------------------------
After a real (--yes) run, rebuild the manifest against the NEW tree. Edit
scripts/build_data.sh so HAYDOM_VIDEOS points at the _recut tree's videos/ and
the Haydom --annotations points at the staged anot_files (they keep the original
event string in column 5, which is what lets the per-device suction backfill
resolve penguin/bulb/tube from them), then:

    bash scripts/build_data.sh configs/data_suction3.yaml

Re-splitting is unavoidable: new cases mean new train/val/test membership, so
every number from before the re-cut is measured on a different test set. The 14
thesis seed cases stay pinned inside their site's test set, so the
`--thesis-only` comparison survives.
NOTE
