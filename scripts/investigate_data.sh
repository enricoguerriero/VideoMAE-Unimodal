#!/usr/bin/env bash
# =============================================================================
# investigate_data.sh — one-shot, READ-ONLY investigation of the source data
# =============================================================================
# Runs the whole diagnostic sequence in one go and saves a timestamped report:
#
#   1  environment + repo state
#   2  path probe        — which of the notebooks' paths actually exist
#   3  annotation hunt   — where the annotation .txt files really live
#                          (incl. Ronald's Haydom-only ProcessedData/Ronald tree)
#   4  clip vintages     — which Processed_* trees exist, and which you train on
#   5  full audit        — scripts/audit_source_data.py against the right trees
#
# It exists because Haydom's annotations are NOT under the path the thesis
# notebook writes them to, which blocks recovering its fraction tags. This finds
# them, or proves they are gone.
#
# WRITES NOTHING except its own timestamped report under $OUT_DIR (default
# ./audit_reports/). No file under /spo is opened for writing.
#
# Needs only a python with the standard library — no numpy/pandas/torch, so it
# runs regardless of the state of your conda env.
#
# Usage:
#   bash scripts/investigate_data.sh
#   HAYDOM_BASE=/some/other/path bash scripts/investigate_data.sh
#   DEEP=1 bash scripts/investigate_data.sh      # slower, wider filesystem search
#
# Env overrides: PYTHON, HAYDOM_BASE, DRC_BASE, RONALD_BASE, OUT_DIR, DEEP,
#                FIND_TIMEOUT
# =============================================================================
set -uo pipefail          # not -e: a missing path must not abort the report

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-audit_reports}"
DEEP="${DEEP:-0}"
FIND_TIMEOUT="${FIND_TIMEOUT:-180}"     # seconds per filesystem search
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HAYDOM_BASE="${HAYDOM_BASE:-/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing}"
DRC_BASE="${DRC_BASE:-}"

# The DRC prefix is spelled two ways across the notebooks; take whichever exists.
if [[ -z "$DRC_BASE" ]]; then
    for cand in /spo/LS-DRC.marta2/ProcessedData/Athavan_Frida/Data_processing \
                /spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing; do
        [[ -d "$cand" ]] && { DRC_BASE="$cand"; break; }
    done
    DRC_BASE="${DRC_BASE:-/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing}"
fi

# Every root the two notebooks mention, per site. Searched for annotations.
#
# The Ronald/ entries are NOT from the Athavan & Frida notebooks. They are the
# Haydom-only master project (Ronald Paleczny), whose data_preprocessing.py
# stage produces exactly what Haydom is missing here: annotations renamed to a
# canonical numeric id and converted to integer milliseconds. It is the best
# candidate source for a Haydom fraction backfill, and the earlier runs of this
# script never looked at it.
RONALD_BASE="${RONALD_BASE:-/spo/LS-Haydom/ProcessedData/Ronald}"
HAYDOM_ROOTS=(
    "$HAYDOM_BASE"
    /spo/LS-Haydom/ProcessedData/Athavan_Frida/FullDataset_Combined
    /spo/LS-Haydom/ProcessedData/Athavan_Frida/Models
    /spo/LS-Haydom/Data/FullDataset/2023-2025
    /spo/LS-Haydom/Data/FullDataset/2025-2026/March2026Sync
    "$RONALD_BASE/data/Tanzania/annotations_corrected"
    "$RONALD_BASE/data/Tanzania/annotations_temp"
    "$RONALD_BASE/data/Tanzania/annotations"
    "$RONALD_BASE/data/Tanzania"
)
DRC_ROOTS=(
    "$DRC_BASE"
    /spo/LS-DRC.marta2/ProcessedData/Athavan_Frida/FullDataset_Combined_DRC
    /spo/LS-DRC/ProcessedData/Athavan_Frida/FullDataset_Combined_DRC
    /spo/LS-DRC.marta2/2023-2025/DRC_LivebornStation_Data
    /spo/LS-DRC.marta2/2025-2026/March2026Sync
)

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/investigate_$(date +%Y%m%d_%H%M%S).txt"

# Everything from here on is tee'd into the report.
exec > >(tee "$REPORT") 2>&1

banner() { echo; echo "=============================================================================="; echo "$1"; echo "=============================================================================="; }
sub()    { echo; echo "-- $1 ------------------------------------------------------------"; }

banner "investigate_data.sh — READ-ONLY.  $(date -Iseconds)"
echo "report -> $REPORT"

# ---------------------------------------------------------------- 1. environment
banner "1. ENVIRONMENT"
echo "host       : $(hostname)"
echo "cwd        : $(pwd)"
echo "python     : $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])' 2>&1 | head -1)"
echo "repo commit: $(git -C "$HERE" log --oneline -1 2>/dev/null || echo 'not a git repo')"
echo "repo dirty : $(git -C "$HERE" status --short 2>/dev/null | wc -l) modified file(s)"
echo "HAYDOM_BASE: $HAYDOM_BASE"
echo "DRC_BASE   : $DRC_BASE"

# ---------------------------------------------------------------- 2. path probe
banner "2. PATH PROBE — what exists, from the paths the notebooks name"
probe() {
    local label="$1" path="$2"
    if [[ -d "$path" ]]; then
        local n
        n=$(ls -1 "$path" 2>/dev/null | wc -l)
        printf '  %-14s %7s entries  %s\n' "$label" "$n" "$path"
    elif [[ -e "$path" ]]; then
        printf '  %-14s %7s          %s\n' "$label" "file" "$path"
    else
        printf '  %-14s %7s          %s\n' "$label" "ABSENT" "$path"
    fi
}

for site in Haydom DRC; do
    base_var="${site^^}_BASE"; [[ $site == Haydom ]] && base="$HAYDOM_BASE" || base="$DRC_BASE"
    sub "$site  ($base)"
    probe "Unprocessed"  "$base/Unprocessed_data"
    probe "anot_files"   "$base/Unprocessed_data/anot_files"
    probe "temp_folder"  "$base/Unprocessed_data/temp_folder"
    probe "raw_annot"    "$base/Unprocessed_data/temp_folder/raw_annotations"
    probe "temp_corr"    "$base/Unprocessed_data/temp_folder/temp_corrected_annotations"
    probe "unique_data"  "$base/Unprocessed_data/temp_folder/unique_data"
    probe "videos"       "$base/Unprocessed_data/videos"
    probe "acc_data"     "$base/Unprocessed_data/acceleration_data"
    echo "   what IS inside temp_folder/ (if any):"
    ls -1 "$base/Unprocessed_data/temp_folder" 2>/dev/null | sed 's/^/       /' | head -20 \
        || echo "       (absent)"
    echo "   other roots for this site:"
    if [[ $site == Haydom ]]; then roots=("${HAYDOM_ROOTS[@]}"); else roots=("${DRC_ROOTS[@]}"); fi
    for r in "${roots[@]:1}"; do probe "  " "$r"; done
done

# ---------------------------------------------------------------- 3. annotation hunt
banner "3. ANNOTATION HUNT — where do annotation-shaped .txt files live?"
echo "Searching each existing root for *.txt, grouped by directory. This is how"
echo "a site's missing anot_files/ gets located. Depth-limited unless DEEP=1."
MAXDEPTH=$([[ "$DEEP" == "1" ]] && echo "" || echo "-maxdepth 6")

for site in Haydom DRC; do
    if [[ $site == Haydom ]]; then roots=("${HAYDOM_ROOTS[@]}"); else roots=("${DRC_ROOTS[@]}"); fi
    sub "$site"
    for r in "${roots[@]}"; do
        [[ -d "$r" ]] || { echo "   [skip] $r"; continue; }
        echo "   searching $r ..."
        # shellcheck disable=SC2086
        timeout "$FIND_TIMEOUT" find "$r" $MAXDEPTH -type f -name '*.txt' 2>/dev/null \
            | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -12 \
            | sed 's/^/       /'
        rc=$?
        [[ $rc -eq 124 ]] && echo "       [timed out after ${FIND_TIMEOUT}s — re-run with FIND_TIMEOUT=600]"
    done
done

echo
echo "   A directory holding hundreds of .txt is a candidate annotation store."
echo "   The filenames do NOT have to be case ids: the audit matches on the exact"
echo "   stem OR any run of 5+ digits inside it (Ronald's canonical-id rule), so"
echo "   'LS_11848523_reviewed.txt' still resolves to case 11848523."

# ---------------------------------------------------------------- 4. clip vintages
banner "4. CLIP VINTAGES — which Processed_* trees exist per site"
echo "build_data.sh currently points at:"
grep -oE "Processed_data[^/\"]*" "$HERE/scripts/build_data.sh" 2>/dev/null | sed 's/^/   /'
for site in Haydom DRC; do
    [[ $site == Haydom ]] && base="$HAYDOM_BASE" || base="$DRC_BASE"
    sub "$site"
    found=0
    for v in "$base"/Processed_*; do
        [[ -d "$v/videos" ]] || continue
        found=1
        n=$(timeout "$FIND_TIMEOUT" find "$v/videos" -name '*.mp4' 2>/dev/null | wc -l)
        tagged=$(timeout "$FIND_TIMEOUT" find "$v/videos" -name '*_suct*.mp4' -o -name '*_stim*.mp4' \
                 -o -name '*_vent*.mp4' 2>/dev/null | wc -l)
        printf '   %-62s %8s mp4  %8s tagged\n' "$(basename "$v")" "$n" "$tagged"
    done
    [[ $found -eq 0 ]] && echo "   (no Processed_*/videos under $base)"
done

# ---------------------------------------------------------------- 5. the audit
banner "5. FULL AUDIT — scripts/audit_source_data.py"
AUDIT_ARGS=(--site "Haydom=$HAYDOM_BASE" --site "DRC=$DRC_BASE")
for r in "${HAYDOM_ROOTS[@]}" "${DRC_ROOTS[@]}"; do
    [[ -d "$r" ]] && AUDIT_ARGS+=(--find-annotations "$r")
done
echo "running: $PYTHON scripts/audit_source_data.py ${AUDIT_ARGS[*]}"
echo
"$PYTHON" "$HERE/scripts/audit_source_data.py" "${AUDIT_ARGS[@]}"
AUDIT_RC=$?

banner "DONE"
echo "audit exit code : $AUDIT_RC"
echo "full report     : $REPORT"
echo
echo "Nothing under /spo was modified. Paste the report back, or read section 9"
echo "of the audit (FINDINGS) first."
