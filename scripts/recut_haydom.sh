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

# shellcheck source=scripts/site_paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/site_paths.sh"
VIDEOS="${VIDEOS:-$HAYDOM_BASE/Unprocessed_data/videos}"

# ---------------------------------------------------------------- annotations
# ONE primary source by default, deliberately. The four Haydom annotation
# directories each cover 240-242 of the 246 cases, so a second one buys ~2 cases
# while mixing two export vintages into one LABEL set — the confound audit
# findings [7]-[9] are about. HAYDOM_ANNOTATION_DIRS (site_paths.sh) is ordered
# by the coverage the audit measured, so element 0 is the primary: 2023-2025/
# Annotations, 242/246, suction vocabulary intact.
#
# EXTRA_ANNOTATIONS=1 admits the rest. The one reason to want them is that
# AnnotationIndex tries the exact case id across every directory before any
# digit-run match, so a case CONFLICTED inside the primary can still resolve
# from another — Ronald's annotations_corrected has no conflicting keys at all.
if [[ "${EXTRA_ANNOTATIONS:-0}" == "1" ]]; then
    ANNOTATIONS=("${HAYDOM_ANNOTATION_DIRS[@]}")
else
    ANNOTATIONS=("${HAYDOM_ANNOTATION_DIRS[0]}")
fi

# ---------------------------------------------------------------- outputs
# Siblings of the existing trees, with _recut in the name, so nothing the
# current build_data.sh reads is touched. The old tree stays exactly where it
# is — this is additive, and reverting means pointing build_data.sh back.
STAGE_DIR="${STAGE_DIR:-$HAYDOM_BASE/Data_processing_recut}"
OUT_DIR="${OUT_DIR:-$HAYDOM_BASE/Processed_data_recut}"
WORKERS="${WORKERS:-1}"
COMPARE="${COMPARE:-data/clips_all.csv}"

# ---------------------------------------------------------------- resolution
# EMPTY = write clips at the SOURCE resolution and let VideoMAEImageProcessor do
# the one resize it was going to do anyway. This is the default since 2026-09 and
# it is a deliberate break from the thesis, which wrote 256x192.
#
# Why: 256x192 is 4:3 and the Haydom source is 16:9, so it is an aspect SQUASH,
# not a downscale — and the processor then resizes the 192 px short edge back up
# to 224, so the model is fed upsampled detail that no longer exists. Ventilation
# survives that (a bag mask covers the whole face); suction does not (a penguin
# device at the nose is a handful of pixels at 192 px). Ronald Paleczny's
# pipeline applies no scale filter at all, which is the difference this undoes.
#
# COST: clips are several times larger. Check free space on the OUT_DIR volume
# before a full run — the 256x192 Haydom tree is the size reference.
#
#   CLIP_SIZE=256x192 bash scripts/recut_haydom.sh --yes    # reproduce the old tree
CLIP_SIZE="${CLIP_SIZE:-}"

ARGS=(--site Haydom --videos "$VIDEOS" --stage-dir "$STAGE_DIR" --out "$OUT_DIR"
      --workers "$WORKERS")
[[ -n "$CLIP_SIZE" ]] && ARGS+=(--clip-size "$CLIP_SIZE")
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

Clips are now written at the SOURCE resolution (see CLIP_SIZE above), so the new
tree is NOT byte-comparable with the old one and is much larger on disk. Labels,
buckets, filenames and clip boundaries are unchanged — only the pixels differ —
so a before/after comparison on the same split is still a clean single-variable
test of the resolution.

Re-splitting is unavoidable: new cases mean new train/val/test membership, so
every number from before the re-cut is measured on a different test set. The 14
thesis seed cases stay pinned inside their site's test set, so the
`--thesis-only` comparison survives.
NOTE
