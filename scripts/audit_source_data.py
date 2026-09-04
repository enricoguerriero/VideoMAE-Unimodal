#!/usr/bin/env python3
"""
audit_source_data.py — READ-ONLY audit of the two sites' SOURCE data.

Answers one question: are Haydom and DRC actually processed the same way?
Everything downstream (thresholds, class balance, per-site scores) assumes they
are, and the folder names on the VM suggest they may not be:

    Haydom  Processed_data_stratified_BIG_update_strict_label
    DRC     Processed_data_new_dataset_no_suction_merge_bulp_new_anot_chestmov

Those describe different pipeline configurations. This script checks the data
itself rather than the names.

WRITES NOTHING. Opens every file read-only and prints to stdout. Redirect if you
want to keep the report:  python scripts/audit_source_data.py > audit.txt

STDLIB ONLY — no numpy, pandas, torch or av. It runs in a broken environment,
which is when you most want it.

--------------------------------------------------------------------------
What it checks
--------------------------------------------------------------------------
 0  LAYOUT          which annotation stages and clip vintages exist per site
 1  FILE FORMAT     column counts of anot_files (4-col = older vintage)
 2  VOCABULARY      original Event string -> mapped category, PER SITE, diffed.
                    This is where a bulb/penguin/tube suction difference shows up.
                    A site with no anot_files/ is read from its raw export
                    instead, so the diff still runs.
 3  INTERVALS       per-class interval counts and durations per site
 4  RAW vs CLEANED  interval counts before/after cleaning — detects whether
                    merge_close_intervals() was applied asymmetrically
 5  CLIPS           bucket census, tagged/untagged, directory names, time units
 6  TAG RECOMPUTE   for clips that HAVE fraction tags, recompute the fraction
                    from the annotations and compare. This is the load-bearing
                    check: if it reproduces the tags, the same computation can
                    backfill fractions for the untagged site and give it the
                    same threshold-tuning freedom.
 7  IMPLIED CUT     the thresholds actually applied, measured from tagged clips
 8  BACKFILL        can every untagged clip be matched to an annotation file?
 9  FINDINGS        the cross-site discrepancies, collected

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    python scripts/audit_source_data.py                      # discovered defaults
    python scripts/audit_source_data.py --sample 0           # recompute EVERY clip
    python scripts/audit_source_data.py \
        --site Haydom=/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing \
        --site DRC=/spo/LS-DRC.marta2/ProcessedData/Athavan_Frida/Data_processing \
        --clips Haydom=/…/Processed_data_stratified_BIG_update_strict_label/videos

`--site NAME=BASEPATH` points at the dir CONTAINING `Unprocessed_data/`.
Clip roots are auto-discovered as `<BASEPATH>/Processed_*/videos` unless given.

--------------------------------------------------------------------------
Reading HAYDOM: three conventions borrowed from Ronald Paleczny's pipeline
--------------------------------------------------------------------------
This audit was first written against the DRC exports, which are tidy. Haydom's
are not, and each difference produced a silent false negative that read as
"the annotations are gone":

  FILENAMES    Haydom files are not named `<case_id>.txt`. Matching on the
               exact stem found 61 of 246 cases in a 489-file directory.
               Files are now indexed by case KEY — the exact stem plus any
               run of >= 5 digits — which is Ronald's canonical-id rule
               (`data_preprocessing.extract_digits`) widened to a lookup.
  TIME UNITS   Haydom rows mix HH:MM:SS(.mmm) tokens with decimal SECONDS.
               Reading "83.400" as 83 shrinks every interval ~1000x, which is
               indistinguishable from "this case has no suction". Colon tokens
               are dropped and decimals scaled by 1000
               (`data_preprocessing.to_milliseconds`); section 1 reports the
               counts so a wrong guess is visible.
  SPELLING     Every Haydom suction string on disk is the misspelled
               "penguine" form. The notebooks fix typos in a separate
               `corrections` pass before `relevant_patterns`; both are folded
               into `classify_event()`. Applying `relevant_patterns` alone
               mapped every Haydom suction interval to "Ignored label".

Ronald's own CATEGORY choices are deliberately NOT copied — he tracks T-piece
ventilation and ignores "Bag-mask squeezed - sound", the thesis does the
reverse. The thesis rules are authoritative here because they are what cut the
clip trees on disk.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------
# What the thesis pipeline does, mirrored here so we can check the data against it
# ---------------------------------------------------------------------------
#: data_process.py load_annotation_data(): Event string -> category code.
MAP_LABELS = {"Ignored label": 4, "Suction": 3, "Ventilation": 2,
              "Stimulation": 1, "Non-target": 0}
CODE_NAME = {0: "Non-target", 1: "Stimulation", 2: "Ventilation", 3: "Suction",
             4: "Ignored label"}
#: `relevant_patterns` from the thesis notebooks — the CANONICAL annotator
#: string -> category map. Identical in the Haydom and DRC notebooks in the
#: final snapshot. Anything absent maps to "Ignored label", which is how
#: "Suction using tube" and "T-piece ventilation" get discarded.
#:
#: Kept for reference only: the notebooks run a `corrections` typo pass BEFORE
#: this one, so on its own it does not match real annotator text. Use
#: `classify_event()`, which folds both passes together — see EVENT_CATEGORY.
#: Likewise the visibility filter, which is `is_visibility()` and not an exact
#: compare against one spelling.
RELEVANT_PATTERNS = {
    "Bag-mask ventilation": "Ventilation",
    "Bag-mask squeezed - sound": "Ventilation",
    "Stimulation of trunk": "Stimulation",
    "Suction using penguin device": "Suction",
    "Suction using bulb device": "Suction",
    "Crying": "Non-target",
    "Chest/abdomen movement": "Non-target",
}
#: filename tag abbreviation -> category code
TAG_CODE = {"stim": 1, "vent": 2, "suct": 3}
SEGMENT_MS = 3000          # segment_size 3 s
FRAC_TOL = 0.0051          # tags are written with %.2f, so ±0.005 plus slack
#: buckets whose clips have, by construction, a non-zero overlap — so
#: _overlap_suffix MUST have written a tag for them. Buckets 0/4/5 have no
#: activity overlap at all, so their clips are CORRECTLY untagged and must not
#: be counted as evidence of a missing-tag vintage.
TAG_BEARING = frozenset({1, 2, 3, 6, 7, 8})
#: the cuts data_process.py applies (STRONG_THRESHOLD / suction_threshold /
#: weak_threshold). The audit measures the data against THESE, not against
#: configs/data.yaml, because these are what actually cut the clips.
PROCESSOR_CUT = {1: 0.50, 2: 0.50, 3: 0.25}
PROCESSOR_WEAK = 0.20

DEFAULT_SITES = {
    "Haydom": "/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing",
    "DRC": "/spo/LS-DRC.marta2/ProcessedData/Athavan_Frida/Data_processing",
}
#: checked as fallbacks when a default BasePath is absent (the LS-DRC vs
#: LS-DRC.marta2 spelling differs between the two notebooks)
ALT_BASES = {
    "DRC": ["/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing"],
    "Haydom": [],
}

#: the clip trees scripts/build_data.sh points at. When several vintages sit
#: side by side, auditing the alphabetically-first one describes a tree the
#: pipeline never reads — so prefer these, and say so.
PIPELINE_VINTAGE = {
    "Haydom": "Processed_data_stratified_BIG_update_strict_label",
    "DRC": "Processed_data_new_dataset_no_suction_merge_bulp_new_anot_chestmov",
}

STAGES = {
    "raw_annotations": "Unprocessed_data/temp_folder/raw_annotations",
    "temp_corrected": "Unprocessed_data/temp_folder/temp_corrected_annotations",
    "anot_files": "Unprocessed_data/anot_files",
    "videos": "Unprocessed_data/videos",
}

_TAG_RE = re.compile(r"_(stim|vent|suct)(\d+\.\d+)")
_WINDOW_RE = re.compile(r"_start_([\d.]+)_end_([\d.]+)")

# ---------------------------------------------------------------------------
# Haydom's raw conventions, taken from Ronald Paleczny's Haydom-only pipeline
# (Master-project/src/data/data_preprocessing.py + clips_and_video_stats.py)
# ---------------------------------------------------------------------------
# The DRC exports are tidy: one file per case named `<case_id>.txt`, timestamps
# as bare integer milliseconds, event strings already spelled canonically. This
# audit was written against those, and every one of those assumptions is FALSE
# at Haydom. Ronald's preprocessing stage exists purely to absorb the
# difference, so his rules are reused verbatim here:
#
#   * FILE NAMING. Haydom annotation files are not named after the case id — in
#     the March2026Sync export 489 files matched only 61 of 246 cases on an
#     exact stem compare, which read as "the annotations are gone" when they
#     are merely named differently. Ronald renames every file to the first run
#     of digits in its stem; `case_keys` indexes both spellings so either finds
#     the file.
#   * TIME UNITS. Haydom rows carry HH:MM:SS(.mmm) tokens AND decimal-SECOND
#     timestamps in the same line. Reading "83.400" as 83 makes every interval
#     ~1000x too short, which is indistinguishable from "this case has no
#     suction". Ronald drops the colon tokens and multiplies decimals by 1000;
#     bare integers are already milliseconds.
#   * SPELLING. Every Haydom suction string on disk is the misspelled
#     "penguine" form. RELEVANT_PATTERNS lists only the canonical "penguin",
#     because the notebook fixes typos in a SEPARATE `corrections` pass first.
#     Applying RELEVANT_PATTERNS alone to a raw export maps every Haydom
#     suction interval to "Ignored label" and yields a site with zero suction.
#     EVENT_CATEGORY below folds both notebook passes into one lookup.
_DIGITS_RE = re.compile(r"(\d+)")
_HHMMSS_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d+)?$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_DEC_RE = re.compile(r"^[+-]?\d+\.\d+$")
_SPLIT_RE = re.compile(r"\t+| {2,}")

#: minimum length of a digit run that may stand in for a case id. Haydom ids
#: run 5-8 digits ("30097" .. "11848523"); a 4-digit floor would let a leading
#: year collide every file in a directory onto one key.
_CASE_KEY_MIN_DIGITS = 5

#: how timestamps were read, so a wrong unit assumption shows up in section 1
#: instead of silently shrinking every interval.
UNIT_STATS = Counter()


def normalize_tokens(line: str) -> list:
    """One raw annotation line -> tokens, timestamps normalised to milliseconds.

    Ronald's `process_annotation_line` / `to_milliseconds` policy: split on tabs
    or runs of 2+ spaces, DROP any HH:MM:SS(.mmm) token, keep a bare integer as
    milliseconds, multiply a decimal (seconds) by 1000. Non-numeric tokens pass
    through unchanged, so a multi-word event name survives intact.
    """
    out = []
    for part in _SPLIT_RE.split(line.rstrip("\n")):
        token = part.strip()
        if not token:
            continue
        if _HHMMSS_RE.match(token):
            UNIT_STATS["hhmmss_dropped"] += 1
            continue
        if _INT_RE.match(token):
            UNIT_STATS["integer_ms"] += 1
            out.append(str(int(token)))
        elif _DEC_RE.match(token):
            UNIT_STATS["decimal_seconds_scaled"] += 1
            out.append(str(int(Decimal(token) * 1000)))
        else:
            out.append(token)
    return out


def case_keys(stem: str) -> set:
    """Every id a file or clip may be addressed by: the exact stem, plus each
    run of >= 5 digits in it (Ronald's canonical-id rule, widened from "first
    run" to "any run" because we only need to LOOK UP a file, not rename it)."""
    keys = {stem}
    keys.update(d for d in _DIGITS_RE.findall(stem)
                if len(d) >= _CASE_KEY_MIN_DIGITS)
    return keys


def _norm_event(text) -> str:
    """Ronald's `normalize_label`: lowercase, punctuation -> space, collapse."""
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


#: The thesis notebooks map an annotator string to a category in TWO passes: a
#: `corrections` dict of typo fixes, then `relevant_patterns`. Merged here into
#: one table keyed on the NORMALISED string, so both passes happen at once.
#:
#: The spellings come from the notebooks' `corrections` dict; Ronald's variant
#: sets were cross-checked against it and agree exactly on suction (the same
#: five "penguine" misspellings) and on stimulation. They deliberately DIVERGE
#: on ventilation and that divergence is NOT copied: Ronald tracks `T-piece
#: ventilation` and ignores `Bag-mask squeezed - sound`, whereas the thesis does
#: the reverse. The thesis reading is authoritative here, because these are the
#: rules the clip trees on disk were cut with.
_EVENT_VARIANTS = {
    "Ventilation": [
        "Bag-mask ventilation", "Bag mask ventilation", "Bag-mask ventilatio",
        "Bagmask ventilation", "BMV", "Bag-mask ventiltation",
        "Badmask ventilation", "Bag-mask ventiation", "Bag-mask ventilatioon",
        "Bag-mask vintilation",
        "Bag-mask squeezed - sound", "Bag mask squeezed-sound",
        "Bagmask squeezed -sound", "BagMask squeezed sound",
        "BagMask Squeezed sound", "Bagmask squeezed-sound",
        "Bag-mask squeezed sound", "Bag-mask squeezed- -sound",
        "Bag-mask squeezed- sound", "BagMask squeeed sound",
        "Bagmask Squeezed sound", "Bagmask squeezed sound",
        "Bag mask squeezed -sound", "Bagmak squeezed - sound",
        "Bagmask squeezed- sound", "Bagmask squezeed-sound",
        "Bagmsk squeezed-sound", "Bag-mask squeezed-sound",
    ],
    "Stimulation": [
        "Stimulation of trunk", "Stimulation of Trunk", "Stimulation of tunk",
        "Stiomulation of trunk", "Stumulation of trunk",
        "Stimulation of the Trunk", "Stimualtion of trunk",
        "Stimuklation of trunk", "Stimulation of  trunk",
        "Stimulationof trunk", "Stiomulation of Trunk",
        "Stimulation of the trunk",
    ],
    "Suction": [
        "Suction using penguin device", "Suction using penguine device",
        "Suction using Penguine Device", "Suction using Penguine device",
        "Suction using penguine devece", "Sunction using penguine device",
        "Suction using bulb device",
    ],
    "Non-target": [
        "Crying",
        "Chest/abdomen movement", "Chest/Abdomen movement",
        "Chest/adomen movement", "Chest/abdomen device",
        "Chest/abdomen ventilation",
    ],
}
EVENT_CATEGORY = {_norm_event(v): cat
                  for cat, variants in _EVENT_VARIANTS.items() for v in variants}

#: `Newborn visible in video frame` and its 18 known misspellings. These rows
#: are dropped before anything else (data_process.py line 92). Matching only the
#: canonical spelling leaves a typo'd one to become an "Ignored label" interval
#: spanning most of the episode, which suppresses the ventilation and non-target
#: branches for the whole case (both require `other == 0`).
VISIBILITY_VARIANTS = {_norm_event(v) for v in [
    "Newborn visible in video frame", "New born visible in video frame",
    "Newborn in video frame", "Newborn visible in vedeo frame",
    "Newborn visible in video fgrame", "Newborn visible in video Frame",
    "Newborn visible on video frame", "Newboen visible in video frame",
    "Newborn Visible in video Frame", "Newborn Visible in vodeo frame",
    "Newborn visible in visible frame", "Newborn visisble in video frame",
    "Neborn visible in video frame", "New-born visible in video frame",
    "Newborn visible in frame", "Newborn visible in the video frame",
    "Newborn visible in video  frame", "Newborn visible in videon frame",
    "Newborn visivle in video frame", "Newborn Visible in video frame",
]}


def classify_event(text):
    """Original annotator string -> thesis category, or None if untracked."""
    return EVENT_CATEGORY.get(_norm_event(text))


def is_visibility(text) -> bool:
    return _norm_event(text) in VISIBILITY_VARIANTS

FINDINGS: list[str] = []      # things that are WRONG or inconsistent
NOTES: list[str] = []         # things that are USEFUL — capabilities, not faults
HUNT_RESULT: dict = {}


def finding(msg: str) -> None:
    """A problem: something inconsistent, missing, or unsafe to rely on."""
    FINDINGS.append(msg)


def note(msg: str) -> None:
    """Good news: a capability the data turns out to have. Kept out of FINDINGS
    so a clean corpus reports zero problems rather than a pile of positives."""
    NOTES.append(msg)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def h(title: str) -> None:
    print(f"\n-- {title} " + "-" * max(0, 74 - len(title)))


# ---------------------------------------------------------------------------
# Annotation reading — mirrors data_process.load_annotation_data + merge_intervals
# ---------------------------------------------------------------------------
def read_annotation(path: Path):
    """-> (rows, ncols_seen) where each row is (event, start, end, original).

    Tolerant on purpose: a short or unparseable row is reported, not fatal, so a
    single bad file cannot hide the rest of the audit.

    Timestamps go through `normalize_tokens`, so a Haydom row written as
    HH:MM:SS plus decimal seconds yields the same milliseconds a DRC row states
    outright. `ncols_seen` still counts RAW tab-separated fields, so section 1's
    "4-col = older vintage" reading is unchanged.
    """
    rows, ncols = [], Counter()
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return None, f"unreadable: {exc}"
    for line in text.splitlines():
        if not line.strip():
            continue
        ncols[len(line.rstrip("\n").split("\t"))] += 1
        parts = normalize_tokens(line)
        if len(parts) == 1 and " " in parts[0]:
            # Single-space-separated export. Ronald's `parse_annotation_line`
            # rule: the last three tokens are start/end/duration, everything
            # before them is the event name (which may contain spaces).
            toks = parts[0].split()
            if len(toks) >= 4:
                tail = normalize_tokens("\t".join(toks[-3:]))
                if len(tail) == 3 and all(_INT_RE.match(t) for t in tail):
                    parts = [" ".join(toks[:-3])] + tail
        if len(parts) < 3:
            continue
        # Column layout varies by export generation (4, 5 and 6 columns all
        # occur) and so does the position of the timestamps, so nothing is
        # assumed from the index. Prefer the first ADJACENT TRIPLE that is
        # self-consistent (end - start == duration): that identifies the real
        # start/end pair even in a 6-column row that also carries a redundant
        # copy of the times. Fall back to the first ascending adjacent pair.
        nums = [(i, int(t)) for i, t in enumerate(parts) if _INT_RE.match(t)]
        start = end = None
        for j in range(len(nums) - 2):
            (i0, a), (i1, b), (i2, d) = nums[j], nums[j + 1], nums[j + 2]
            if i1 == i0 + 1 and i2 == i1 + 1 and b >= a and abs((b - a) - d) <= 2:
                start, end = a, b
                break
        if start is None:
            for j in range(len(nums) - 1):
                (i0, a), (i1, b) = nums[j], nums[j + 1]
                if i1 == i0 + 1 and b >= a:
                    start, end = a, b
                    break
        if start is None:
            continue
        original = parts[4] if len(parts) >= 5 else ""
        rows.append((parts[0], start, end, original))
    return (rows, ncols), None


def merge_intervals(intervals):
    """Verbatim from data_process.merge_intervals: touching intervals merge."""
    if not intervals:
        return []
    out = [tuple(sorted(intervals)[0])]
    for start, end in sorted(intervals)[1:]:
        if start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def annotation_kind(rows):
    """'cleaned' if column 1 already holds categories, else 'raw'.

    A cleaned anot_files row reads  Suction \t start \t end \t dur \t Suction using bulb device
    A raw export reads              Suction using bulb device \t start \t end \t dur
    Telling them apart decides whether the event mapping still has to be applied.
    """
    if not rows:
        return "empty"
    known = sum(1 for e, _, _, _ in rows if e in MAP_LABELS)
    return "cleaned" if known >= 0.5 * len(rows) else "raw"


def intervals_by_code(rows, kind=None):
    """visibility filter -> category -> merge, for either annotation stage.

    For a raw export the category comes from `classify_event`, which folds the
    notebook's `corrections` and `relevant_patterns` passes into one lookup on
    the normalised string; for a cleaned file column 1 is already the category.
    Visibility rows are matched typo-tolerantly, because a missed one becomes an
    "Ignored label" interval spanning the episode and suppresses the ventilation
    and non-target branches for the whole case.
    """
    kind = kind or annotation_kind(rows)
    buckets = defaultdict(list)
    for event, start, end, original in rows:
        if is_visibility(original) or is_visibility(event):
            continue
        cat = event if kind == "cleaned" else (classify_event(event) or "Ignored label")
        code = MAP_LABELS.get(cat)
        if code is None:
            continue
        buckets[code].append((start, end))
    return {c: merge_intervals(v) for c, v in buckets.items()}


def overlap_ms(a0, a1, intervals):
    """Verbatim from data_process.overlap_ms."""
    return sum(min(a1, e) - max(a0, s) for s, e in intervals if a0 < e and a1 > s)


# ---------------------------------------------------------------------------
# Clip filename parsing
# ---------------------------------------------------------------------------
def parse_clip(stem: str):
    """-> dict | None. Mirrors DataSpec.parse_stem plus the time window."""
    try:
        bucket = int(stem.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None
    tags = {TAG_CODE[k]: float(v) for k, v in _TAG_RE.findall(stem)}
    w = _WINDOW_RE.search(stem)
    start = end = None
    if w:
        try:
            start, end = float(w.group(1)), float(w.group(2))
        except ValueError:
            pass
    # Normalise the window to MILLISECONDS. The live saver writes raw ms
    # (`start_time = clip_start`), but a commented-out variant in data_process.py
    # wrote `clip_start/1000`, so a tree cut by that vintage would be in seconds.
    # Comparing a seconds window against ms intervals silently yields 0 overlap
    # and would look like a total recomputation failure.
    start_ms = end_ms = None
    if start is not None and end is not None:
        span = end - start
        scale = 1.0 if span > 100 else 1000.0
        start_ms, end_ms = start * scale, end * scale
    return {"case_id": stem.split("_interval_")[0], "bucket": bucket,
            "tags": tags, "tagged": bool(tags), "start": start, "end": end,
            "start_ms": start_ms, "end_ms": end_ms}


def discover(base: Path):
    """Which pipeline stages and which clip vintages exist under a BasePath."""
    present = {k: (base / v) for k, v in STAGES.items() if (base / v).is_dir()}
    vintages = sorted(p for p in base.glob("Processed_*") if (p / "videos").is_dir())
    return present, vintages


# ---------------------------------------------------------------------------
# Locating a site's annotations — by case KEY, not by exact filename
# ---------------------------------------------------------------------------
def index_annotation_dir(d: Path):
    """One directory -> ({case key: file}, {case key: [conflicting files]}).

    A file is reachable under its exact stem AND under any >= 5-digit run in it,
    so Haydom's `LS_11848523_reviewed.txt` answers to `11848523`. When two files
    claim the same key, Ronald's rule applies: byte-identical content keeps the
    first, differing content drops BOTH and records a conflict — silently
    picking one of two disagreeing versions of a case's annotations is exactly
    what his conflicts/ folder exists to prevent.
    """
    index, claims = {}, defaultdict(list)
    try:
        files = sorted(d.glob("*.txt"))
    except OSError:
        return {}, {}
    for f in files:
        for k in case_keys(f.stem):
            claims[k].append(f)
    conflicts = {}
    for k, fs in claims.items():
        if len(fs) == 1:
            index[k] = fs[0]
            continue
        blobs = set()
        for f in fs:
            try:
                blobs.add(f.read_bytes())
            except OSError:
                blobs.add(None)
        if len(blobs) == 1:
            index[k] = fs[0]          # duplicates of the same content
        else:
            conflicts[k] = fs         # genuinely different versions — refuse
    return index, conflicts


def locate_annotations(site, roots):
    """Every annotation directory reachable for a site, best source first.

    The site's own `anot_files/` leads (it is the CLEANED stage the clips were
    actually cut from); every other directory follows in descending size, which
    is the only quality signal available before the clip inventory is read.
    Each directory keeps its own index so section 8b can still report coverage
    per directory, and the merged view resolves a case to its best source.
    """
    out, seen = [], set()
    own = site["present"].get("anot_files")
    for root in ([own] if own is not None else []) + [Path(r).expanduser() for r in roots]:
        if not root.is_dir():
            continue
        try:
            dirs = sorted({f.parent for f in root.rglob("*.txt")})
        except OSError:
            continue
        for d in dirs:
            if d in seen:
                continue
            seen.add(d)
            idx, conf = index_annotation_dir(d)
            if idx:
                out.append({"dir": d, "index": idx, "conflicts": conf,
                            "own": root is own})
    # Biggest store wins after the site's own cleaned stage. Argument order is
    # not a quality ranking: rglob over the first root happened to surface
    # `unique_data/acceleration/annotations` (2 files, 2 cases), and sections
    # 1-3 then described a 246-case site from two of its cases.
    out.sort(key=lambda e: (not e["own"], -len(e["index"])))
    return out


def ambiguous_cases(site, cases):
    """Of `cases`, the ones that HAVE annotation files but cannot be used.

    Two files with different content claim the same case key, so Ronald's
    conflict rule refuses both. That is a very different problem from "this
    case was never annotated": the data exists and someone has to decide which
    copy is authoritative. Reporting them as absent hides a fixable conflict.
    """
    out = set()
    for c in cases:
        keys = [c] + sorted(case_keys(c) - {c})
        for entry in site.get("annot_dirs", []):
            if any(k in entry["conflicts"] for k in keys):
                out.add(c)
                break
    return out


def annotation_files_for(site):
    """(files, raw_source_dir) — the best annotation stage available for a site.

    The cleaned `anot_files/` when the site still has one (raw_source_dir None),
    otherwise the highest-priority LOCATED directory. Sections 1, 2 and 3 all go
    through this, so a site whose cleaned stage was deleted reports a real
    column shape, vocabulary and interval census instead of a row of dashes —
    which previously read as "this hospital annotates nothing at all".
    """
    d = site["present"].get("anot_files")
    if d is not None:
        return sorted(d.glob("*.txt")), None
    if site.get("annot_dirs"):
        e = site["annot_dirs"][0]
        return sorted(set(e["index"].values())), e["dir"]
    return [], None


def lookup_annotation(site, case_id):
    """Best annotation file for one case id, or None.

    KEY-MAJOR, not directory-major: every directory is tried on the EXACT case
    id before any of them is tried on a digit run. A digit key is a fallback for
    oddly-named files, and it must never outrank a file actually named after the
    case — DRC's `2-33998-1` reduces to the digit key `33998`, which a Haydom
    filename could also carry, and a directory-major search would let that
    Haydom file win purely by being listed first.
    """
    for k in [case_id] + sorted(case_keys(case_id) - {case_id}):
        for entry in site.get("annot_dirs", []):
            f = entry["index"].get(k)
            if f is not None:
                return f
    return None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_layout(sites):
    rule("0. LAYOUT — what exists on disk, per site")
    for name, s in sites.items():
        h(name)
        print(f"   BasePath : {s['base']}" + ("" if s["base"].is_dir() else "   [MISSING]"))
        if not s["base"].is_dir():
            finding(f"{name}: BasePath does not exist: {s['base']}")
            continue
        for stage in STAGES:
            p = s["present"].get(stage)
            if p is None:
                print(f"   {stage:<16} ABSENT")
                if stage in ("raw_annotations", "temp_corrected"):
                    finding(f"{name}: stage `{stage}` is absent — cannot rebuild "
                            f"annotations from before the cleaning step")
            else:
                n = sum(1 for _ in p.iterdir()) if p.is_dir() else 0
                print(f"   {stage:<16} {n:>7,} entries   {p}")
        print(f"   clip vintages found under BasePath ({len(s['vintages'])}):")
        for v in s["vintages"]:
            mark = ""
            if s.get("clips") and v / "videos" == s["clips"]:
                mark = f"  <- AUDITED (chosen by {s.get('picked_by')})"
            elif v.name == PIPELINE_VINTAGE.get(name):
                mark = "  <- build_data.sh trains on THIS one"
            print(f"       {v.name}{mark}")
        if s.get("picked_by") == "FIRST ALPHABETICALLY":
            finding(f"{name}: build_data.sh's vintage "
                    f"'{PIPELINE_VINTAGE.get(name)}' was NOT found; audited "
                    f"'{s['clips'].parent.name}' instead — sections 5-8 may not "
                    f"describe the tree you train on. Pass --clips {name}=<path>.")
        elif len(s["vintages"]) > 1:
            print(f"   ({len(s['vintages'])} vintages present; audited the one "
                  f"build_data.sh uses)")


def section_format(sites):
    rule("1. ANNOTATION FILE FORMAT — column counts and time units")
    print(f"{'site':<10}{'files':>8}{'rows':>10}   column-count histogram")
    UNIT_STATS.clear()
    for name, s in sites.items():
        files, raw_src = annotation_files_for(s)
        if not files:
            print(f"{name:<10}{'-':>8}{'-':>10}   (no annotation file located)")
            continue
        ncols, nrows, bad = Counter(), 0, []
        for f in files:
            got, err = read_annotation(f)
            if err:
                bad.append((f.name, err)); continue
            rows, cols = got
            ncols.update(cols); nrows += len(rows)
        hist = ", ".join(f"{c} cols x{n:,}" for c, n in sorted(ncols.items()))
        print(f"{name:<10}{len(files):>8,}{nrows:>10,}   {hist or '(empty)'}"
              + (f"   [RAW: {raw_src}]" if raw_src else ""))
        s["anot_dir"] = raw_src or s["present"].get("anot_files")
        s["anot_files"] = files
        s["anot_raw_src"] = raw_src
        # <5 columns only matters for the CLEANED stage, where the 5th column is
        # the preserved original label. A raw export legitimately has 4 or 6.
        if raw_src is None and any(c < 5 for c in ncols):
            finding(f"{name}: some anot_files rows have <5 columns — the 5th column "
                    f"(Corrected_Original_Event) is what makes re-mapping possible")
        for fn, err in bad[:3]:
            print(f"           [WARN] {fn}: {err}")

    # How the timestamps were read, across everything parsed so far. A file that
    # states seconds ("83.400") and one that states milliseconds ("83400") look
    # identical to a positional parser; Ronald's rule scales the first and not
    # the second, and this line is where a wrong guess becomes visible instead
    # of silently shrinking every interval by ~1000x.
    if UNIT_STATS:
        print(f"\n  time tokens: {UNIT_STATS['integer_ms']:,} bare integers "
              f"(already ms), {UNIT_STATS['decimal_seconds_scaled']:,} decimals "
              f"scaled x1000 (seconds -> ms), "
              f"{UNIT_STATS['hhmmss_dropped']:,} HH:MM:SS tokens dropped")
        if UNIT_STATS["decimal_seconds_scaled"]:
            print("  (decimal-second rows follow Ronald's data_preprocessing rule; if a "
                  "site\n   writes fractional MILLISECONDS instead, this scaling is wrong "
                  "— check\n   section 3's max_ms against the real episode length)")


def section_vocabulary(sites):
    rule("2. EVENT VOCABULARY — original annotator string -> mapped category")
    print("The 5th column preserves the ORIGINAL label; the 1st is the mapped")
    print("category. A string mapped differently between sites is a labelling")
    print("difference between hospitals, not a data difference.\n")

    per_site = {}
    for name, s in sites.items():
        pairs = Counter()
        # A site with no anot_files/ is not a site with no vocabulary: fall back
        # to the best LOCATED directory, mapping its raw strings the way the
        # notebooks do. Without this the cross-site diff below — the check that
        # actually answers the suction question — could never run for Haydom.
        files, raw_src = annotation_files_for(s)
        if raw_src is not None:
            print(f"   [{name}: no anot_files/ — reading {len(files)} raw file(s) from")
            print(f"    {raw_src}, mapped with the notebooks' corrections + patterns]")
        for f in files:
            got, err = read_annotation(f)
            if err:
                continue
            for event, _, _, original in got[0]:
                if raw_src is None:
                    pairs[(original or "<no 5th col>", event)] += 1
                elif is_visibility(event):
                    pairs[(event, "(visibility — dropped)")] += 1
                else:
                    pairs[(event, classify_event(event) or "Ignored label")] += 1
        per_site[name] = pairs
        h(f"{name}: original -> mapped  (interval counts)")
        if not pairs:
            print("   (nothing read)")
            continue
        for (orig, mapped), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
            star = "  *" if "suction" in orig.lower() or mapped == "Suction" else ""
            print(f"   {n:>7,}  {orig[:46]:<46} -> {mapped}{star}")
        s["vocab"] = pairs

    names = list(sites)
    if len(names) < 2:
        return
    a, b = names[0], names[1]
    h(f"CROSS-SITE DIFF  ({a} vs {b})")
    empty = [n for n in (a, b) if not per_site[n]]
    if empty:
        print(f"   NOT COMPARABLE — no annotations were read for {', '.join(empty)}.")
        print("   Everything below would just re-list the other site's vocabulary,")
        print("   which says nothing about whether the two agree.")
        for n in empty:
            finding(f"{n}: no annotations readable, so the event vocabulary CANNOT be "
                    f"compared against the other site — the suction question stays open")
        return
    map_a = defaultdict(set)
    map_b = defaultdict(set)
    for (o, m) in per_site[a]:
        map_a[o].add(m)
    for (o, m) in per_site[b]:
        map_b[o].add(m)

    # A visibility row is dropped at both sites (data_process.py line 92). It
    # only LOOKS different because a cleaned file stores it as the literal
    # "Ignored label" while a raw one still carries the original string, so
    # comparing the two spellings reports a conflict that does not exist.
    conflicts = [o for o in set(map_a) & set(map_b)
                 if map_a[o] != map_b[o] and not is_visibility(o)]
    only_a = sorted(set(map_a) - set(map_b))
    only_b = sorted(set(map_b) - set(map_a))

    if conflicts:
        print("   [CONFLICT] the same original string maps differently:")
        for o in sorted(conflicts):
            print(f"       {o[:44]:<44} {a}->{sorted(map_a[o])}  {b}->{sorted(map_b[o])}")
            finding(f"VOCABULARY CONFLICT: '{o}' maps to {sorted(map_a[o])} at {a} "
                    f"but {sorted(map_b[o])} at {b}")
    else:
        print("   no string maps to different categories between the two sites")

    # A string that appears at ONE site only is not automatically a fault. Two
    # hospitals genuinely use different equipment — Haydom suctions with a
    # penguin device, DRC also with a bulb — and a spelling can simply be one
    # annotator's habit. What matters is whether the missing string would have
    # landed in a category the other site DOES populate: only then is the class
    # itself built from different evidence at the two sites. Anything that maps
    # to "Ignored label" at both is noise either way.
    for label, only, site, mp in ((f"only in {a}", only_a, a, map_a),
                                  (f"only in {b}", only_b, b, map_b)):
        if not only:
            continue
        print(f"   [{label}] {len(only)} original string(s):")
        for o in only[:12]:
            print(f"       {o[:50]:<50} -> {sorted(mp[o])}")
        tracked = sorted({c for o in only for c in mp[o]} - {"Ignored label",
                                                             "(visibility — dropped)"})
        if tracked:
            for cat in tracked:
                strings = sorted(o for o in only if cat in mp[o])
                print(f"       -> {cat} at {site} draws on {len(strings)} string(s) the "
                      f"other site never uses")
                note(f"{cat}: {site} annotates it with {strings[:4]}"
                     f"{' ...' if len(strings) > 4 else ''}, which the other site never "
                     f"uses — expected when the two hospitals differ in equipment or "
                     f"annotator habit, but it does mean this class is built from "
                     f"different evidence per site")


def section_intervals(sites):
    rule("3. INTERVAL STATISTICS — per category, per site (after merging)")
    print(f"{'site':<10}{'category':<15}{'cases':>7}{'intervals':>11}"
          f"{'total_s':>11}{'median_ms':>11}{'max_ms':>10}")
    for name, s in sites.items():
        per_code = defaultdict(list)
        cases_with = defaultdict(set)
        files, raw_src = annotation_files_for(s)
        if raw_src is not None:
            print(f"{name:<10}(from raw export {raw_src})")
        for f in files:
            got, err = read_annotation(f)
            if err:
                continue
            for code, ivs in intervals_by_code(got[0]).items():
                per_code[code].extend(e - st for st, e in ivs)
                if ivs:
                    cases_with[code].add(f.stem)
        s["interval_stats"] = {}
        for code in sorted(CODE_NAME):
            d = sorted(per_code.get(code, []))
            if not d:
                print(f"{name:<10}{CODE_NAME[code]:<15}{0:>7}{0:>11}{0:>11}{'-':>11}{'-':>10}")
                continue
            med = d[len(d) // 2]
            print(f"{name:<10}{CODE_NAME[code]:<15}{len(cases_with[code]):>7}"
                  f"{len(d):>11,}{sum(d)/1000:>11,.0f}{med:>11,}{d[-1]:>10,}")
            s["interval_stats"][code] = (len(d), sum(d), med, d[-1])
    print("\n  A category with 0 intervals at one site and many at the other is the")
    print("  clearest possible cross-site labelling difference.")


def section_raw_vs_clean(sites):
    rule("4. RAW vs CLEANED — was interval merging applied asymmetrically?")
    print("merge_close_intervals(max_gap_ms=6000, events_to_merge=...) runs BEFORE")
    print("the category mapping. If it was enabled for one site only, that site's")
    print("intervals are fewer and longer. Comparing stage 1 to stage 3 shows it.\n")
    print(f"{'site':<10}{'stage':<18}{'files':>7}{'rows':>10}   note")
    for name, s in sites.items():
        for stage in ("raw_annotations", "temp_corrected", "anot_files"):
            d = s["present"].get(stage)
            if d is None:
                print(f"{name:<10}{stage:<18}{'-':>7}{'-':>10}   ABSENT")
                continue
            files = sorted(d.glob("*.txt"))
            rows = 0
            for f in files:
                got, err = read_annotation(f)
                if not err:
                    rows += len(got[0])
            print(f"{name:<10}{stage:<18}{len(files):>7,}{rows:>10,}")
            s.setdefault("stage_rows", {})[stage] = rows
    for name, s in sites.items():
        sr = s.get("stage_rows", {})
        if "raw_annotations" in sr and "anot_files" in sr and sr["raw_annotations"]:
            drop = 100 * (1 - sr["anot_files"] / sr["raw_annotations"])
            print(f"\n  {name}: cleaning removed/merged {drop:.1f}% of raw rows")
            s["clean_drop_pct"] = drop
    drops = {n: s["clean_drop_pct"] for n, s in sites.items() if "clean_drop_pct" in s}
    if len(drops) == 2:
        a, b = list(drops)
        if abs(drops[a] - drops[b]) > 10:
            finding(f"CLEANING ASYMMETRY: raw->cleaned row reduction is "
                    f"{drops[a]:.1f}% at {a} vs {drops[b]:.1f}% at {b}")


def section_clips(sites):
    rule("5. CLIP INVENTORY — buckets, fraction tags, directories, time units")
    for name, s in sites.items():
        root = s.get("clips")
        h(f"{name}: {root or '(no clip root)'}")
        if not root or not root.is_dir():
            print("   not found — pass --clips NAME=PATH")
            finding(f"{name}: clip root not found; sections 5-8 skipped for this site")
            continue
        clips = sorted(root.rglob("*.mp4"))
        buckets, tagged, untagged, dirs, unparsed = Counter(), 0, 0, Counter(), 0
        no_tag_due = 0
        units = Counter()
        parsed = []
        for c in clips:
            info = parse_clip(c.stem)
            if info is None:
                unparsed += 1
                continue
            info["path"] = c
            info["rel_dir"] = c.parent.relative_to(root).as_posix()
            buckets[info["bucket"]] += 1
            dirs[(info["rel_dir"], info["bucket"])] += 1
            if info["tagged"]:
                tagged += 1
            elif info["bucket"] in TAG_BEARING:
                untagged += 1          # a tag was due here and is missing
            else:
                no_tag_due += 1        # bucket 0/4/5: zero overlap, correctly bare
            if info["start"] is not None and info["end"] is not None:
                span = info["end"] - info["start"]
                units["milliseconds" if span > 100 else
                      "SECONDS" if 0 < span <= 100 else "odd"] += 1
            else:
                units["no window in filename"] += 1
            parsed.append(info)
        s["clip_info"] = parsed
        print(f"   {len(clips):,} mp4 ({unparsed:,} unparseable)")
        print(f"   tag-bearing buckets {sorted(TAG_BEARING)}: {tagged:,} tagged, "
              f"{untagged:,} UNTAGGED")
        print(f"   buckets 0/4/5 (no overlap, correctly bare): {no_tag_due:,}")
        print(f"   filename time units: {dict(units)}")
        if units.get("SECONDS"):
            finding(f"{name}: {units['SECONDS']:,} clips encode the window in SECONDS, "
                    f"not ms — mixed filename conventions")
        if tagged and untagged:
            finding(f"{name}: MIXED tagging within tag-bearing buckets "
                    f"({tagged:,} tagged, {untagged:,} untagged) — more than one "
                    f"processor vintage in this tree")
        print("   bucket census: " + ", ".join(f"{b}:{n:,}" for b, n in sorted(buckets.items())))
        print(f"   directories ({len(dirs)}):")
        for (rel, b), n in sorted(dirs.items(), key=lambda kv: -kv[1])[:14]:
            print(f"       {(rel or '<root>'):<44} bucket {b}  {n:>8,}")
        s["tagged_n"], s["untagged_n"] = tagged, untagged


def _load_case_intervals(site, case_id, cache):
    """Merged intervals for one case, from whichever located source has it.

    Goes through `lookup_annotation` rather than `anot_files/<case>.txt`, so a
    site whose cleaned stage was deleted can still be checked against a raw
    export whose filenames do not equal the case ids.
    """
    if case_id in cache:
        return cache[case_id]
    ivs = None
    f = lookup_annotation(site, case_id)
    if f is not None:
        got, err = read_annotation(f)
        if not err:
            ivs = intervals_by_code(got[0])
    cache[case_id] = ivs
    return ivs


def section_recompute(sites, sample):
    rule("6. TAG RECOMPUTATION — can fractions be rebuilt from the annotations?")
    print("For every clip that already HAS a tag, recompute frac from")
    print("anot_files + the window in the filename, and compare. If this")
    print("reproduces the tags, the same computation can backfill the UNTAGGED")
    print("site and give it identical threshold-tuning freedom.\n")
    for name, s in sites.items():
        h(name)
        tagged = [c for c in s.get("clip_info", []) if c["tagged"]
                  and c["start_ms"] is not None]
        if not tagged:
            print("   no tagged clips with a parseable window — nothing to verify here")
            print("   (this is expected for the untagged site; see section 8)")
            continue
        subset = tagged if not sample else tagged[:: max(1, len(tagged) // sample)]
        cache, checked, ok, missing_anot = {}, 0, 0, 0
        mism = []
        for c in subset:
            ivs = _load_case_intervals(s, c["case_id"], cache)
            if ivs is None:
                missing_anot += 1
                continue
            # BOTH directions. A tag that is absent means the overlap was zero
            # (_overlap_suffix writes nothing for 0), so `want` is 0.0 there —
            # otherwise a source that ADDS activity the clip never recorded would
            # pass unnoticed, which is exactly the bulb-suction question.
            for code in (1, 2, 3):
                want = c["tags"].get(code, 0.0)
                got = overlap_ms(int(c["start_ms"]), int(c["end_ms"]),
                                 ivs.get(code, [])) / SEGMENT_MS
                checked += 1
                if abs(got - want) <= FRAC_TOL:
                    ok += 1
                elif len(mism) < 6:
                    kind = "tag absent, so expected 0" if code not in c["tags"] else ""
                    mism.append((c["path"].name, CODE_NAME[code] + (" *" if kind else ""),
                                 want, got))
        pct = 100 * ok / checked if checked else 0.0
        print(f"   {len(subset):,} clips sampled ({len(tagged):,} tagged total), "
              f"{checked:,} tag values checked")
        print(f"   reproduced within +/-{FRAC_TOL}: {ok:,}/{checked:,}  ({pct:.1f}%)")
        if missing_anot:
            print(f"   {missing_anot:,} clips had no matching anot_files/<case>.txt")
        for fn, cat, want, got in mism:
            print(f"       MISMATCH {cat:<12} filename={want:.2f}  recomputed={got:.2f}"
                  f"   {fn[:56]}")
        if checked and pct < 99.0:
            finding(f"{name}: only {pct:.1f}% of fraction tags reproduce from the "
                    f"annotations — the clips and the annotations on disk are not "
                    f"from the same pipeline run")
        elif checked:
            print("   => the annotations on disk MATCH the clips; backfilling the")
            print("      untagged site with this computation is sound.")


def section_implied_cut(sites):
    rule("7. IMPLIED CUT — the thresholds actually applied, measured from tags")
    print("Bucket N (1=stim, 2=vent, 3=suct) means 'this activity cleared its cut")
    print("and the others stayed under weak_threshold'.")
    print()
    print("The smallest own-fraction observed is a LOWER BOUND on the cut, not the")
    print("cut itself — with few clips the minimum sits above the true threshold by")
    print("chance. So only one direction is sound evidence: a clip BELOW the")
    print("processor's cut proves a different (lower) cut was used. A minimum above")
    print("it proves nothing, and is reported as 'consistent', not as a match.\n")
    print(f"reference: stim>={PROCESSOR_CUT[1]:.2f} vent>={PROCESSOR_CUT[2]:.2f} "
          f"suct>={PROCESSOR_CUT[3]:.2f}, weak<={PROCESSOR_WEAK:.2f} "
          f"(data_process.py)\n")
    print(f"{'site':<10}{'bucket':<7}{'activity':<13}{'clips':>8}{'min own':>9}"
          f"{'p01':>7}{'cut':>7}{'max other':>10}   verdict")
    verdicts = defaultdict(dict)
    for name, s_ in sites.items():
        any_row = False
        for code in (1, 2, 3):
            own, other = [], []
            for c in s_.get("clip_info", []):
                if c["bucket"] != code or not c["tagged"]:
                    continue
                if code in c["tags"]:
                    own.append(c["tags"][code])
                other.extend(v for k, v in c["tags"].items() if k != code)
            if not own:
                continue
            any_row = True
            own.sort()
            lo = own[0]
            p01 = own[max(0, int(0.01 * len(own)) - 1)] if len(own) >= 100 else own[0]
            cut = PROCESSOR_CUT[code]
            mo = max(other) if other else 0.0
            if lo < cut - 0.011:
                verdict = "** LOWER CUT USED **"
            elif mo > PROCESSOR_WEAK + 0.011:
                verdict = "** purity guard breached **"
            else:
                verdict = "consistent"
            verdicts[code][name] = (verdict, lo, len(own))
            print(f"{name:<10}{code:<7}{CODE_NAME[code]:<13}{len(own):>8,}"
                  f"{lo:>9.2f}{p01:>7.2f}{cut:>7.2f}{mo:>10.2f}   {verdict}")
            if verdict.startswith("**"):
                finding(f"{name}: bucket {code} ({CODE_NAME[code]}) — {verdict.strip('* ')}; "
                        f"observed min own-fraction {lo:.2f} vs processor cut "
                        f"{cut:.2f}, max other {mo:.2f} vs weak {PROCESSOR_WEAK:.2f}")
        if not any_row:
            print(f"{name:<10}(no tagged activity clips — this site's cut is not "
                  f"measurable, and not changeable)")

    # Cross-site: only a DISAGREEMENT of verdicts is evidence. Two different
    # sample minima are not, which is why the raw numbers are not compared.
    for code, per in verdicts.items():
        if len(per) == 2:
            (na, (va, la, ca)), (nb, (vb, lb, cb)) = per.items()
            if va != vb:
                finding(f"CUT DISAGREEMENT for {CODE_NAME[code]}: {na} is '{va}' "
                        f"(min {la:.2f}, n={ca:,}) but {nb} is '{vb}' "
                        f"(min {lb:.2f}, n={cb:,})")
    print("\n  Sample minima are NOT compared across sites directly: with different")
    print("  clip counts they differ by chance even when the cut is identical.")


def section_backfill(sites):
    rule("8. BACKFILL FEASIBILITY — can the untagged site be given fractions?")
    for name, s in sites.items():
        h(name)
        clips = s.get("clip_info", [])
        # Only tag-bearing buckets need backfilling: a bucket 0/4/5 clip has no
        # activity overlap, so its fractions are a genuine all-zero, not unknown.
        untagged = [c for c in clips if not c["tagged"] and c["bucket"] in TAG_BEARING]
        bare_ok = sum(1 for c in clips if not c["tagged"] and c["bucket"] not in TAG_BEARING)
        if not untagged:
            print(f"   every tag-bearing clip already carries fractions "
                  f"({bare_ok:,} bucket-0/4/5 clips are correctly all-zero) — "
                  f"nothing to backfill")
            continue
        have_window = [c for c in untagged if c["start_ms"] is not None]
        cases = {c["case_id"] for c in untagged}
        # Every LOCATED source counts, not just anot_files/ — a site whose
        # cleaned stage was deleted is not a site whose annotations are gone.
        # Checking anot_files/ alone reported 0% recoverable for Haydom while
        # section 8b was finding the same cases in four other directories.
        found = {c for c in cases if lookup_annotation(s, c) is not None}
        print(f"   {len(untagged):,} untagged clips across {len(cases)} cases")
        print(f"   with a parseable time window in the filename : "
              f"{len(have_window):,}/{len(untagged):,} "
              f"({100*len(have_window)/max(len(untagged),1):.1f}%)")
        print(f"   cases with an annotation file anywhere located: "
              f"{len(found)}/{len(cases)} "
              f"({100*len(found)/max(len(cases),1):.1f}%)")
        if not s.get("annot_dirs"):
            print("   (no annotation directory located — pass --find-annotations DIR)")
        covered = sum(1 for c in untagged
                      if c["start_ms"] is not None and c["case_id"] in found)
        print(f"   => RECOVERABLE fractions: {covered:,}/{len(untagged):,} clips "
              f"({100*covered/max(len(untagged),1):.1f}%)")
        miss = cases - found
        if miss:
            amb = ambiguous_cases(s, miss)
            absent = sorted(miss - amb)
            if amb:
                print(f"   AMBIGUOUS ({len(amb)}): {sorted(amb)[:8]}"
                      f"{' ...' if len(amb) > 8 else ''}")
                print("      — annotated, but two files disagree; pick the authoritative")
                print("        copy and these become recoverable too")
                finding(f"{name}: {len(amb)} case(s) are annotated but UNUSABLE — two "
                        f"files with different content claim the same case; choosing "
                        f"one would recover them")
            if absent:
                print(f"   ABSENT ({len(absent)}): {absent[:8]}"
                      f"{' ...' if len(absent) > 8 else ''}")
                finding(f"{name}: {len(absent)} case(s) have clips but no annotation "
                        f"file in any located directory")
        lost = len(untagged) - covered
        if lost:
            finding(f"{name}: {lost:,} of {len(untagged):,} untagged clips "
                    f"({100*lost/len(untagged):.1f}%) cannot have their fractions "
                    f"recovered — the other {covered:,} can")


def section_hunt(sites, roots):
    """Locate, then FINGERPRINT, the annotation files a site is missing.

    Finding the files is only half of it: what matters is whether they are raw
    or cleaned, how much of the corpus they cover, and — the whole point — which
    original event strings they contain. That last one is what finally makes the
    suction question answerable for a site with no anot_files/ of its own.

    Returns {site: [directories that matched]} for the validation section.

    Directories and their case-key indexes are built once in main()
    (`locate_annotations`), so this section, section 8 and section 8c all agree
    on what was found instead of each re-deriving it with different rules.
    """
    rule("8b. ANNOTATION SOURCES — locate and fingerprint")
    found_dirs = {}
    for name, s_ in sites.items():
        cases = {c["case_id"] for c in s_.get("clip_info", [])}
        found_dirs[name] = []
        if not cases:
            continue
        h(f"{name}: {len(cases)} case ids to account for")
        hits = Counter()
        if s_["present"].get("anot_files") is not None:
            print("   anot_files/ present; fingerprinting it too for comparison")
        # NB: do NOT pre-seed anot_files here — the rglob below walks it too, and
        # counting it twice reported 200% coverage.
        # Matching is by case KEY (exact stem OR a >= 5-digit run), not by exact
        # filename: Haydom's exports are not named after the case id, and an
        # exact-stem compare found 61 of 246 cases in a 489-file directory.
        by_dir = {}
        for entry in s_.get("annot_dirs", []):
            matched = {c for c in cases if any(k in entry["index"] for k in
                                               ([c] + sorted(case_keys(c) - {c})))}
            if matched:
                hits[entry["dir"]] = len(matched)
                by_dir[entry["dir"]] = (matched, entry)
        if not hits:
            print("   no directory anywhere in the searched roots holds a .txt named")
            print("   after one of this site's cases — the annotations are genuinely gone")
            finding(f"{name}: no annotation files found for ANY of its {len(cases)} "
                    f"cases in the searched roots")
            continue

        union, conflict_keys, suction_noted = set(), set(), False
        for d, n in sorted(hits.items(), key=lambda kv: -kv[1]):
            matched, entry = by_dir[d]
            files = sorted({entry["index"][k] for c in matched
                            for k in ([c] + sorted(case_keys(c) - {c}))
                            if k in entry["index"]})
            union |= matched
            found_dirs[name].append(d)
            ncols, kinds, vocab, rows_total = Counter(), Counter(), Counter(), 0
            for f in files[:400]:                       # fingerprint a slice; enough
                got, err = read_annotation(f)
                if err:
                    continue
                rws, cols = got
                ncols.update(cols)
                k = annotation_kind(rws)
                kinds[k] += 1
                rows_total += len(rws)
                for event, _, _, original in rws:
                    vocab[original if (k == "cleaned" and original) else event] += 1
            print(f"\n   {d}")
            print(f"       {n:>5,} of this site's cases   "
                  f"({100*n/len(cases):.0f}% coverage)   "
                  f"stage={'/'.join(kinds) or '?'}   "
                  f"cols={dict(ncols) or '?'}")
            if entry["conflicts"]:
                bad = sorted(entry["conflicts"])[:4]
                print(f"       {len(entry['conflicts'])} ambiguous case key(s) — two "
                      f"files, different content, e.g. {bad}")
                conflict_keys.update(entry["conflicts"])
            sucty = {k: v for k, v in vocab.items() if "suction" in k.lower()}
            if vocab:
                print("       original event strings (top 8):")
                for ev, cnt in vocab.most_common(8):
                    mark = "  <== SUCTION" if "suction" in ev.lower() else ""
                    print(f"           {cnt:>6,}  {ev[:52]}{mark}")
            if sucty:
                print(f"       SUCTION VARIANTS HERE: {sorted(sucty)}")
                # One note per SITE, not per directory: four copies of the same
                # vocabulary buried the findings list under duplicates.
                if not suction_noted:
                    suction_noted = True
                    note(f"{name}: suction vocabulary recoverable from {d.name}/ — "
                         f"{sorted(sucty)}")
            else:
                print("       no suction-like string in this directory")
        print(f"\n   UNION over all directories above: {len(union)}/{len(cases)} cases "
              f"({100*len(union)/len(cases):.0f}%)")
        if conflict_keys:
            finding(f"{name}: {len(conflict_keys)} case key(s) across the located "
                    f"directories map to two annotation files with DIFFERENT content "
                    f"— neither is used (Ronald's conflict rule)")
        missing = cases - union
        if missing:
            amb = ambiguous_cases(s_, missing)
            if amb:
                print(f"   of those, {len(amb)} are ANNOTATED BUT AMBIGUOUS "
                      f"(two files disagree): {sorted(amb)[:5]}")
            absent = sorted(missing - amb)
            if absent:
                finding(f"{name}: {len(absent)} case(s) have clips but no annotation "
                        f"anywhere in the searched roots, e.g. {absent[:5]}")
        else:
            note(f"{name}: annotations for ALL {len(cases)} cases are locatable — "
                 f"a full fraction backfill is possible")
    return found_dirs


def section_validate(sites, found_dirs):
    """Validate a backfill against a vintage that DOES carry tags.

    A site whose training tree is untagged may still have a small tagged tree
    lying around from another processor run. Those tags are ground truth for the
    recomputation: reproduce them from the recovered annotations and the backfill
    is proven for this site, exactly as DRC's tags prove it for DRC.
    """
    rule("8c. BACKFILL VALIDATION — recompute a tagged tree from recovered annotations")
    for name, s_ in sites.items():
        h(name)
        if not any(not c["tagged"] and c["bucket"] in TAG_BEARING
                   for c in s_.get("clip_info", [])):
            print("   the audited tree is already fully tagged — nothing to validate")
            continue
        # look for ANY other vintage of this site carrying tags
        ref = None
        for v in s_.get("vintages", []):
            vroot = v / "videos"
            if s_.get("clips") and vroot.resolve() == s_["clips"].resolve():
                continue
            try:
                sample = [c for c in list(vroot.rglob("*.mp4"))[:4000]]
            except OSError:
                continue
            tagged = [t for t in (parse_clip(c.stem) for c in sample)
                      if t and t["tagged"] and t["start_ms"] is not None]
            if tagged:
                ref = (v, tagged)
                break
        if ref is None:
            print("   no other vintage of this site carries fraction tags, so there is")
            print("   no ground truth here to validate a backfill against.")
            print("   (Validate on the OTHER site instead — same code path.)")
            continue
        v, tagged = ref
        print(f"   reference tree: {v.name}  ({len(tagged):,} tagged clips sampled)")
        dirs = found_dirs.get(name) or []
        if not dirs:
            print("   no annotation directory located for this site — cannot validate")
            continue
        validated = False
        for d in dirs:
            cache, checked, ok = {}, 0, 0
            mism = []
            idx = next((e["index"] for e in s_.get("annot_dirs", [])
                        if e["dir"] == d), {})
            for c in tagged:
                if c["case_id"] not in cache:
                    cid = c["case_id"]
                    f = next((idx[k] for k in [cid] + sorted(case_keys(cid) - {cid})
                              if k in idx), None)
                    ivs = None
                    if f is not None:
                        got, err = read_annotation(f)
                        if not err:
                            ivs = intervals_by_code(got[0])
                    cache[c["case_id"]] = ivs
                ivs = cache[c["case_id"]]
                if ivs is None:
                    continue
                for code in (1, 2, 3):
                    want = c["tags"].get(code, 0.0)   # no tag == zero overlap
                    got_f = overlap_ms(int(c["start_ms"]), int(c["end_ms"]),
                                       ivs.get(code, [])) / SEGMENT_MS
                    checked += 1
                    if abs(got_f - want) <= FRAC_TOL:
                        ok += 1
                    elif len(mism) < 4:
                        mism.append((CODE_NAME[code] +
                                     ("" if code in c["tags"] else " (no tag)"),
                                     want, got_f))
            if not checked:
                print(f"   {d.name:<38} no overlapping cases")
                continue
            pct = 100 * ok / checked
            verdict = "MATCHES — backfill is sound from here" if pct >= 99 else \
                      "does NOT reproduce the tags"
            print(f"   {d.name:<38} {ok:,}/{checked:,} ({pct:.1f}%)  {verdict}")
            for cat, w, g in mism:
                print(f"       e.g. {cat:<22} tag={w:.2f} recomputed={g:.2f}")
            if pct >= 99 and not validated:
                validated = True   # one note per site; the rest are the same fact
                note(f"{name}: fraction backfill VALIDATED against {v.name} using "
                     f"{d.name}/ ({ok:,}/{checked:,} tags reproduced)")


def section_vintage_diff(sites):
    """What changed between the clip vintages sitting side by side.

    Two trees of the same corpus with different bucket distributions mean a
    labelling-policy change, not a data change. Keying on (case, start, end)
    ignores the tag and the bucket, so the same 3 s window can be followed from
    one vintage to the next.
    """
    rule("10. VINTAGE DIFF — what a re-cut actually changed")
    for name, s_ in sites.items():
        vints = s_.get("vintages", [])
        if len(vints) < 2:
            continue
        h(f"{name}: {len(vints)} vintages")
        audited = s_.get("clips")
        maps = {}
        for v in vints:
            m = {}
            try:
                for c in (v / "videos").rglob("*.mp4"):
                    info = parse_clip(c.stem)
                    if info and info["start_ms"] is not None:
                        m[(info["case_id"], int(info["start_ms"]))] = info["bucket"]
            except OSError as exc:
                print(f"   [error reading {v.name}: {exc}]")
                continue
            maps[v.name] = m
            print(f"   {v.name:<58} {len(m):>8,} windows")
        base_name = audited.parent.name if audited else None
        if base_name not in maps:
            continue
        base = maps[base_name]
        for other, m in maps.items():
            if other == base_name:
                continue
            common = base.keys() & m.keys()
            moved = [(m[k], base[k]) for k in common if m[k] != base[k]]
            print(f"\n   {other}  ->  {base_name}  (the audited one)")
            print(f"       {len(common):,} shared windows, {len(moved):,} changed bucket "
                  f"({100*len(moved)/max(len(common),1):.1f}%)")
            for (frm, to), n in Counter(moved).most_common(6):
                print(f"           bucket {frm} -> {to}   {n:>8,}")
            if len(moved) > 0.05 * max(len(common), 1):
                finding(f"{name}: '{base_name}' reclassifies "
                        f"{100*len(moved)/len(common):.0f}% of windows vs "
                        f"'{other}' — these vintages use different labelling policies")


def _dir_activities(rel_dir):
    """Codes named by a clip's directory (1=stim, 2=vent, 3=suct)."""
    low = rel_dir.lower()
    return {c for c, w in ((1, "stimulation"), (2, "ventilation"), (3, "suction"))
            if w in low}


def _policy_evidence(site):
    """Stream every tagged clip of one vintage, keeping only the extremes.

    Returns (vintage_name, n_tagged, evidence) where evidence[code] is
    {"strong_min", "strong_n", "partial_max", "partial_n"} and
    evidence["weak_max"] is the largest OTHER-activity fraction inside a pure
    bucket. Streaming means no sample cap, so no directory can be missed —
    an earlier capped version silently sampled only the first 20,000 paths and
    reported `partial n = 0` for a site that has 1,220 such clips.

    Prefers the vintage the pipeline actually trains on; falls back to whichever
    tagged vintage is largest.
    """
    audited = site.get("clips")
    candidates = []
    for v in site.get("vintages", []):
        vroot = v / "videos"
        ev = {c: {"strong_min": None, "strong_n": 0,
                  "partial_max": None, "partial_n": 0} for c in (1, 2, 3)}
        weak_max, n_tag = None, 0
        try:
            for c in vroot.rglob("*.mp4"):
                info = parse_clip(c.stem)
                if not info or not info["tagged"]:
                    continue
                n_tag += 1
                b, tags = info["bucket"], info["tags"]
                rel = c.parent.relative_to(vroot).as_posix()
                if b in (1, 2, 3) and b in tags:
                    e = ev[b]
                    e["strong_n"] += 1
                    e["strong_min"] = tags[b] if e["strong_min"] is None \
                        else min(e["strong_min"], tags[b])
                    for k, val in tags.items():
                        if k != b:
                            weak_max = val if weak_max is None else max(weak_max, val)
                elif b == 6:
                    named = _dir_activities(rel)
                    if len(named) == 1:
                        code = next(iter(named))
                        if code in tags:
                            e = ev[code]
                            e["partial_n"] += 1
                            e["partial_max"] = tags[code] if e["partial_max"] is None \
                                else max(e["partial_max"], tags[code])
        except OSError:
            continue
        if n_tag:
            ev["weak_max"] = weak_max
            candidates.append((v.name, n_tag, ev,
                               audited is not None
                               and vroot.resolve() == audited.resolve()))
    if not candidates:
        return None, 0, None
    preferred = [c for c in candidates if c[3]]
    name, n, ev, _ = (preferred or sorted(candidates, key=lambda c: -c[1]))[0]
    return name, n, ev


def section_policy(sites):
    """Bracket each site's ACTUAL thresholds, and decide whether they agree.

    An untagged site's labels are frozen at whatever cut the processor used; a
    tagged site's are recomputed from the config every run. They describe the
    same policy ONLY if the config equals the frozen constants.

    Tagged clips make that decidable, because a bucket is a THRESHOLD DECISION
    and the tag is the number it was taken on:

        bucket a (pure "strong")      => frac_a >= T_a   -> UPPER bound on T_a
        partial/a (bucket 6, only a)  => frac_a <  T_a   -> LOWER bound on T_a

    IMPORTANT — the lower bound is valid for stimulation and suction ONLY.
    data_process.py's ventilation branch carries an extra clause:

        elif vent_strong and not stim_weak and not suct_weak and other == 0:

    so a fully-ventilated clip touched by an "Ignored label" interval is demoted
    to partial/ventilation despite clearing the threshold. Reading those as
    "below the cut" produced an inverted bracket like (1.00, 0.50]. Ventilation
    therefore gets an upper bound only.
    """
    rule("11. POLICY EQUIVALENCE — is the FROZEN cut the same as the LIVE one?")
    print("An untagged site cannot be re-thresholded, so its labels match the tagged")
    print("site's only if the configured cut equals the one baked into its buckets.\n")
    print("Bounds come from tagged clips. For ventilation only an UPPER bound is")
    print("sound: data_process.py demotes vent-strong clips to partial/ventilation")
    print("when an 'Ignored label' interval touches them (`and other == 0`), so")
    print("partial/ventilation is NOT 'below the cut'.\n")

    brackets = {}
    for name, s_ in sites.items():
        vname, n_tag, ev = _policy_evidence(s_)
        h(name)
        if ev is None:
            print("   no tagged clip in ANY vintage — this site's cut is UNKNOWABLE from")
            print("   the clips alone.")
            finding(f"{name}: no tagged clip in any vintage — its labelling cut cannot "
                    f"be verified against the other site's")
            continue
        print(f"   {n_tag:,} tagged clips from {vname}")
        if s_.get("clips") and vname != s_["clips"].parent.name:
            print(f"   NOTE: not the audited tree ({s_['clips'].parent.name}); section 10's")
            print("   vintage diff is what justifies carrying the cut across.")

        site_b = {}
        print(f"\n   {'activity':<13}{'strong n':>9}{'min own':>9}"
              f"{'partial n':>11}{'max own':>9}   inferred bound on T")
        for code in (1, 2, 3):
            e = ev[code]
            up = e["strong_min"]
            # The lower bound comes from a tag written with %.2f, so a clip that
            # was really at 0.4995 reads as 0.50 and appears to sit ON the cut it
            # is by construction below. Give the bound back that rounding, or
            # every activity whose partial clips crowd the threshold is reported
            # as "CONTRADICTORY" when the data is merely rounded.
            lo = (e["partial_max"] - FRAC_TOL) if (code != 2 and e["partial_max"]
                                                   is not None) else None
            if up is None and lo is None:
                print(f"   {CODE_NAME[code]:<13}{'-':>9}{'-':>9}{'-':>11}{'-':>9}   "
                      f"no tagged evidence")
                continue
            if lo is not None and up is not None and lo >= up:
                print(f"   {CODE_NAME[code]:<13}{e['strong_n']:>9,}{up:>9.2f}"
                      f"{e['partial_n']:>11,}{lo:>9.2f}   ** CONTRADICTORY **")
                finding(f"{name}: {CODE_NAME[code]} bounds are contradictory "
                        f"(partial max {lo:.2f} >= strong min {up:.2f}) — the bucket "
                        f"rule for this activity is not a plain threshold")
                continue
            site_b[code] = (lo, up)
            desc = (f"({lo:.2f}, {up:.2f}]" if lo is not None and up is not None
                    else f"<= {up:.2f}" if up is not None else f"> {lo:.2f}")
            pn = f"{e['partial_n']:,}" if code != 2 else "n/a"
            pm = f"{lo:.2f}" if lo is not None else "n/a"
            print(f"   {CODE_NAME[code]:<13}{e['strong_n']:>9,}{up:>9.2f}"
                  f"{pn:>11}{pm:>9}   {desc}")
        if ev.get("weak_max") is not None:
            print(f"   {'weak_thresh':<13}{'':>9}{'':>9}{'':>11}{ev['weak_max']:>9.2f}"
                  f"   > {ev['weak_max']:.2f}")
            if ev["weak_max"] >= PROCESSOR_WEAK + 0.011:
                finding(f"{name}: an activity reached {ev['weak_max']:.2f} inside a pure "
                        f"bucket, above weak_threshold {PROCESSOR_WEAK:.2f}")
        brackets[name] = site_b

        print(f"\n   against the processor constants (stim {PROCESSOR_CUT[1]:.2f}, "
              f"vent {PROCESSOR_CUT[2]:.2f}, suct {PROCESSOR_CUT[3]:.2f}):")
        for code in (1, 2, 3):
            if code not in site_b:
                print(f"       {CODE_NAME[code]:<13} no verdict (insufficient evidence)")
                continue
            lo, up = site_b[code]
            cut = PROCESSOR_CUT[code]
            inside = ((lo is None or cut > lo - 1e-9) and (up is None or cut <= up + 1e-9))
            print(f"       {CODE_NAME[code]:<13} "
                  f"{'CONSISTENT' if inside else '** OUTSIDE THE BOUND **'}")
            if not inside:
                finding(f"{name}: the cut used for {CODE_NAME[code]} is bounded by "
                        f"lo={lo} up={up}, which EXCLUDES the configured {cut:.2f}")

    h("VERDICT")
    names = [n for n in brackets if brackets[n]]
    if len(names) < 2:
        print("   fewer than two sites have measurable thresholds — equivalence cannot")
        print("   be demonstrated. Treat the labels as NOT known to agree.")
        return
    a, b = names[0], names[1]
    for code in (1, 2, 3):
        if code not in brackets[a] or code not in brackets[b]:
            print(f"   {CODE_NAME[code]:<13} not measurable at both sites — UNRESOLVED")
            finding(f"{CODE_NAME[code]}: cut not measurable at both sites, so the two "
                    f"hospitals cannot be shown to use the same rule")
            continue
        (la, ua), (lb, ub) = brackets[a][code], brackets[b][code]
        los = [x for x in (la, lb) if x is not None]
        ups = [x for x in (ua, ub) if x is not None]
        lo, up = (max(los) if los else None), (min(ups) if ups else None)
        overlap = lo is None or up is None or lo < up + 1e-9
        fa = f"({la}, {ua}]" if la is not None else f"<= {ua}"
        fb = f"({lb}, {ub}]" if lb is not None else f"<= {ub}"
        print(f"   {CODE_NAME[code]:<13} {a} {fa}   {b} {fb}   "
              f"-> {'COMPATIBLE' if overlap else '** DISJOINT — different cuts **'}")
        if not overlap:
            finding(f"{CODE_NAME[code]}: {a} and {b} were cut with DIFFERENT thresholds "
                    f"— {fa} and {fb} do not overlap")
    print("\n   COMPATIBLE means the evidence admits a single shared cut, not that one")
    print("   is proven. A one-sided bound (<= x) is weak evidence; a two-sided")
    print("   bracket is strong. DISJOINT is proof of a difference.")


def section_findings():
    rule("9. FINDINGS")
    if not FINDINGS:
        print("  PROBLEMS: none — the two sites look consistently processed on every")
        print("  check above.")
    else:
        print("  PROBLEMS")
        for i, f in enumerate(FINDINGS, 1):
            print(f"  [{i:>2}] {f}")
        print(f"\n  {len(FINDINGS)} problem(s). Anything labelled CONFLICT, MISMATCH or")
        print("  ASYMMETRY means the hospitals were not processed identically, and a")
        print("  per-site score comparison is measuring that as well as the model.")
    if NOTES:
        print("\n  WHAT IS POSSIBLE  (capabilities the data has, not faults)")
        for i, n in enumerate(NOTES, 1):
            print(f"  ({i:>2}) {n}")


# ---------------------------------------------------------------------------
def parse_kv(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), Path(v).expanduser()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", action="append", type=parse_kv, metavar="NAME=BASEPATH",
                   help="Dir containing Unprocessed_data/. Repeatable. Defaults to "
                        "the two paths found in the thesis notebooks.")
    p.add_argument("--clips", action="append", type=parse_kv, default=None,
                   metavar="NAME=PATH", help="Clip `videos/` root for a site. "
                        "Auto-discovered as <BASEPATH>/Processed_*/videos if omitted.")
    p.add_argument("--sample", type=int, default=4000,
                   help="Clips per site to use for the section-6 recomputation "
                        "(0 = all; default 4000, which is plenty).")
    p.add_argument("--skip", default="", help="Comma-separated section numbers to skip.")
    p.add_argument("--find-annotations", action="append", default=None, metavar="DIR",
                   help="Search DIR recursively for annotation .txt files. Matching "
                        "is by case KEY (exact stem, or any >= 5-digit run in it), so "
                        "files not named after the case id are still found. Use when "
                        "a site's anot_files/ is missing. Repeatable.")
    args = p.parse_args()

    raw_sites = dict(args.site) if args.site else {k: Path(v) for k, v in DEFAULT_SITES.items()}
    clip_over = dict(args.clips) if args.clips else {}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print("audit_source_data.py — READ-ONLY. Nothing on disk is modified.")
    print(f"python {sys.version.split()[0]}   (stdlib only)")

    sites = {}
    for name, base in raw_sites.items():
        if not base.is_dir():
            for alt in ALT_BASES.get(name, []):
                if Path(alt).is_dir():
                    print(f"[info] {name}: {base} absent, using {alt}")
                    base = Path(alt)
                    break
        present, vintages = discover(base) if base.is_dir() else ({}, [])
        clips = clip_over.get(name)
        picked_by = "--clips"
        if clips is None and vintages:
            want = PIPELINE_VINTAGE.get(name)
            match = next((v for v in vintages if v.name == want), None)
            if match is not None:
                clips, picked_by = match / "videos", "build_data.sh"
            elif len(vintages) == 1:
                # only one tree here, so there is nothing to get wrong
                clips, picked_by = vintages[0] / "videos", "the only vintage present"
            else:
                clips, picked_by = vintages[0] / "videos", "FIRST ALPHABETICALLY"
        sites[name] = {"base": base, "present": present, "vintages": vintages,
                       "clips": clips, "picked_by": picked_by}
        # Index every reachable annotation directory ONCE, by case key. Sections
        # 2, 6, 8, 8b and 8c all read this, so they can no longer disagree about
        # which annotations exist.
        sites[name]["annot_dirs"] = locate_annotations(
            sites[name], args.find_annotations or [])
        ndirs = len(sites[name]["annot_dirs"])
        nkeys = len({k for e in sites[name]["annot_dirs"] for k in e["index"]})
        print(f"[info] {name}: {ndirs} annotation director{'y' if ndirs == 1 else 'ies'} "
              f"indexed, {nkeys:,} case key(s)")

    def _hunt(sites_, a):
        global HUNT_RESULT
        HUNT_RESULT = section_hunt(sites_, a.find_annotations or [])

    for num, fn in [("0", lambda: section_layout(sites)),
                    ("1", lambda: section_format(sites)),
                    ("2", lambda: section_vocabulary(sites)),
                    ("3", lambda: section_intervals(sites)),
                    ("4", lambda: section_raw_vs_clean(sites)),
                    ("5", lambda: section_clips(sites)),
                    ("6", lambda: section_recompute(sites, args.sample)),
                    ("7", lambda: section_implied_cut(sites)),
                    ("8", lambda: section_backfill(sites)),
                    ("8b", lambda: _hunt(sites, args)),
                    ("8c", lambda: section_validate(sites, HUNT_RESULT)),
                    ("10", lambda: section_vintage_diff(sites)),
                    ("11", lambda: section_policy(sites))]:
        if num in skip:
            print(f"\n[skipped section {num}]")
            continue
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001 - report, continue
            import traceback
            print(f"\n[section {num} FAILED: {type(exc).__name__}: {exc}]")
            traceback.print_exc()
            finding(f"section {num} crashed ({type(exc).__name__}: {exc}) — its checks "
                    f"did not run")
    section_findings()


if __name__ == "__main__":
    main()
