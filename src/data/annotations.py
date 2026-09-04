"""
annotations.py

Reading the Haydom / DRC annotation exports, and recomputing the per-activity
window fractions that data_process.py wrote into clip filenames.

Why this module exists
----------------------
`_overlap_suffix` in data_process.py is not extra knowledge the processor had
and threw away. It is a pure function of the clip window and the annotation
intervals:

    frac[a] = overlap_ms(start, end, merge_intervals(intervals[a])) / 3000

Nothing else feeds it — in particular NOT the per-case offsets, which only ever
touched the (removed) accelerometer branch. So any clip whose filename still
carries `_start_{ms}_end_{ms}` can have its fractions rebuilt from the
annotations alone, with no video decoding and no re-cutting. That is what makes
the untagged Haydom tree re-thresholdable; see build_manifest.py --annotations.

Reading Haydom is the hard part
-------------------------------
The DRC exports are tidy: one file per case named `<case_id>.txt`, timestamps as
bare integer milliseconds, event strings spelled canonically. Every one of those
assumptions is false at Haydom, and each violation fails SILENTLY — it produces
a site that looks like it was never annotated rather than an error. The three
conventions below are taken from Ronald Paleczny's Haydom-only pipeline
(Master-project/src/data/data_preprocessing.py), which exists to absorb exactly
these differences:

  FILENAMES   Haydom files are not named after the case. Matching on the exact
              stem found 61 of 246 cases in a 489-file directory. `case_keys`
              indexes the exact stem PLUS any run of >= 5 digits, which is
              Ronald's canonical-id rule (`extract_digits`) widened from a
              rename to a lookup.
  TIME UNITS  Haydom rows mix HH:MM:SS(.mmm) tokens with decimal SECONDS.
              Reading "83.400" as 83 shrinks every interval ~1000x, which is
              indistinguishable from "this case has no suction".
              `normalize_tokens` drops the colon tokens and scales decimals by
              1000 (`to_milliseconds`).
  SPELLING    Every Haydom suction string on disk is the misspelled "penguine"
              form. The thesis notebooks fix typos in a `corrections` pass
              BEFORE applying `relevant_patterns`, so `relevant_patterns` alone
              maps every Haydom suction interval to "Ignored label".
              `EVENT_CATEGORY` folds both passes into one lookup.

Ronald's own CATEGORY choices are deliberately not copied: he tracks `T-piece
ventilation` and ignores `Bag-mask squeezed - sound`, the thesis does the
reverse. The thesis rules are authoritative here, because they are the rules the
clip trees on disk were cut with.

STDLIB ONLY (no yaml, no numpy) — scripts/audit_source_data.py loads this file
directly, by path, so it keeps running in an environment with nothing installed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

#: data_process.py load_annotation_data(): Event string -> category code.
MAP_LABELS = {"Ignored label": 4, "Suction": 3, "Ventilation": 2,
              "Stimulation": 1, "Non-target": 0}
CODE_NAME = {v: k for k, v in MAP_LABELS.items()}

SEGMENT_MS = 3000          # data_process.py segment_size = 3 s
#: filename fractions are written with %.2f, so a recomputed value may sit up to
#: half a step away from the printed one. Anything inside this is agreement.
FRAC_TOL = 0.0051

_DIGITS_RE = re.compile(r"(\d+)")
_HHMMSS_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d+)?$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_DEC_RE = re.compile(r"^[+-]?\d+\.\d+$")
_SPLIT_RE = re.compile(r"\t+| {2,}")
_WINDOW_RE = re.compile(r"_start_([\d.]+)_end_([\d.]+)")

#: minimum length of a digit run that may stand in for a case id. Haydom ids run
#: 5-8 digits ("30097" .. "11848523"); a 4-digit floor would let a leading year
#: collapse every file in a directory onto one key.
CASE_KEY_MIN_DIGITS = 5

#: how timestamps were read, so a wrong unit assumption is reportable rather
#: than silent. Callers may `.clear()` it to scope a count.
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
    run of >= 5 digits in it."""
    keys = {stem}
    keys.update(d for d in _DIGITS_RE.findall(stem) if len(d) >= CASE_KEY_MIN_DIGITS)
    return keys


def normalize_event(text) -> str:
    """Ronald's `normalize_label`: lowercase, punctuation -> space, collapse."""
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


#: The notebooks' `corrections` (typo fixes) merged with `relevant_patterns`
#: (category map) into one lookup on the NORMALISED string. Verified against
#: Ronald's variant sets: they agree exactly on suction (the same five
#: "penguine" misspellings) and on stimulation.
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
EVENT_CATEGORY = {normalize_event(v): cat
                  for cat, variants in _EVENT_VARIANTS.items() for v in variants}

#: `Newborn visible in video frame` and its 19 known misspellings. These rows are
#: dropped before anything else (data_process.py line 92). Matching only the
#: canonical spelling leaves a typo'd one to become an "Ignored label" interval
#: spanning most of the episode, which suppresses the ventilation and non-target
#: branches for the whole case (both require `other == 0`).
VISIBILITY_VARIANTS = {normalize_event(v) for v in [
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
    return EVENT_CATEGORY.get(normalize_event(text))


def is_visibility(text) -> bool:
    return normalize_event(text) in VISIBILITY_VARIANTS


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_annotation(path: Path):
    """-> ((rows, ncols_seen), None) | (None, error).

    Each row is (event, start_ms, end_ms, original). Tolerant on purpose: a
    short or unparseable row is skipped, not fatal, so one bad file cannot hide
    the rest of a corpus.

    The (start, end) pair is chosen by SELF-CONSISTENCY with the duration column
    — the first adjacent numeric triple where `end - start == duration` — and
    only falls back to the first ascending adjacent pair when no triple fits.
    Positional guessing was wrong for the 6-column FullDataset exports, which
    carry two copies of the times in different units.

    `ncols_seen` counts RAW tab-separated fields, so "4-col = older vintage"
    still reads the way it always did.
    """
    rows, ncols = [], Counter()
    try:
        text = Path(path).read_text(errors="replace")
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
        rows.append((parts[0], start, end, parts[4] if len(parts) >= 5 else ""))
    return (rows, ncols), None


def annotation_kind(rows) -> str:
    """'cleaned' if column 1 already holds categories, else 'raw'.

    A cleaned anot_files row reads  Suction \\t start \\t end \\t dur \\t Suction using bulb device
    A raw export reads              Suction using bulb device \\t start \\t end \\t dur
    """
    if not rows:
        return "empty"
    known = sum(1 for e, _, _, _ in rows if e in MAP_LABELS)
    return "cleaned" if known >= 0.5 * len(rows) else "raw"


def merge_intervals(intervals):
    """Verbatim from data_process.merge_intervals: touching intervals merge."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [tuple(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def overlap_ms(a0, a1, intervals) -> int:
    """Verbatim from data_process.overlap_ms."""
    return sum(min(a1, e) - max(a0, s) for s, e in intervals if a0 < e and a1 > s)


def intervals_by_category(rows, kind=None) -> dict:
    """{category name -> merged intervals}, for either annotation stage.

    Visibility rows are dropped first (typo-tolerantly), then a raw export's
    strings go through `classify_event` while a cleaned file's column 1 is
    already the category. Merging before measuring is what data_process.py does,
    so an event annotated twice over the same span is not counted twice.
    """
    kind = kind or annotation_kind(rows)
    buckets = defaultdict(list)
    for event, start, end, original in rows:
        if is_visibility(original) or is_visibility(event):
            continue
        cat = event if kind == "cleaned" else (classify_event(event) or "Ignored label")
        if cat not in MAP_LABELS:
            continue
        buckets[cat].append((start, end))
    return {c: merge_intervals(v) for c, v in buckets.items()}


def window_from_stem(stem: str):
    """(start_ms, end_ms) from a clip filename stem, or None.

    data_process.py writes `_start_{ms}_end_{ms}`. A commented-out variant in
    that file wrote seconds instead, so a window narrower than one segment is
    rescaled rather than silently measured against millisecond intervals.
    """
    m = _WINDOW_RE.search(stem)
    if not m:
        return None
    try:
        start, end = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None
    if 0 < end - start <= SEGMENT_MS / 1000.0 + 1e-6:
        start, end = start * 1000.0, end * 1000.0
    return int(round(start)), int(round(end))


# ---------------------------------------------------------------------------
# Locating a case's annotation file
# ---------------------------------------------------------------------------
class AnnotationIndex:
    """Case id -> annotation file, over one or more directories.

    Files are reachable under their exact stem AND any >= 5-digit run in it.
    When two files in one directory claim the same key, Ronald's rule applies:
    byte-identical content keeps the first, differing content drops BOTH and
    records a conflict. Silently picking between two disagreeing versions of a
    case's annotations is what his conflicts/ folder exists to prevent.

    Lookup is KEY-MAJOR: every directory is tried on the EXACT case id before
    any of them is tried on a digit run, so DRC's `2-33998-1.txt` can never lose
    to a Haydom file that merely contains `33998`.
    """

    def __init__(self, dirs=()):
        self.entries = []          # [{dir, index, conflicts}]
        for d in dirs:
            self.add_dir(Path(d))

    def add_dir(self, d: Path) -> bool:
        index, claims, conflicts = {}, defaultdict(list), {}
        try:
            files = sorted(Path(d).glob("*.txt"))
        except OSError:
            return False
        for f in files:
            for k in case_keys(f.stem):
                claims[k].append(f)
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
                index[k] = fs[0]
            else:
                conflicts[k] = fs
        if not index and not conflicts:
            return False
        self.entries.append({"dir": Path(d), "index": index, "conflicts": conflicts})
        return True

    @classmethod
    def from_roots(cls, roots):
        """Index every directory under `roots` that holds .txt files, largest
        first — argument order is not a quality ranking, and the first directory
        an rglob surfaces can easily be a two-file scratch folder."""
        self = cls()
        seen = set()
        for root in roots:
            root = Path(root).expanduser()
            if not root.is_dir():
                continue
            try:
                dirs = sorted({f.parent for f in root.rglob("*.txt")})
            except OSError:
                continue
            for d in dirs:
                if d not in seen:
                    seen.add(d)
                    self.add_dir(d)
        self.entries.sort(key=lambda e: -len(e["index"]))
        return self

    def lookup(self, case_id: str):
        for k in [case_id] + sorted(case_keys(case_id) - {case_id}):
            for e in self.entries:
                f = e["index"].get(k)
                if f is not None:
                    return f
        return None

    def is_ambiguous(self, case_id: str) -> bool:
        """True when the case is annotated but unusable: two files disagree.
        A very different problem from 'never annotated' — the data exists and
        someone has to say which copy is authoritative."""
        keys = [case_id] + sorted(case_keys(case_id) - {case_id})
        return any(k in e["conflicts"] for e in self.entries for k in keys)

    @property
    def dirs(self):
        return [e["dir"] for e in self.entries]

    def __len__(self):
        return len({k for e in self.entries for k in e["index"]})


class FractionSource:
    """Per-activity window fractions for a site, recomputed from annotations.

    Wraps an AnnotationIndex with a per-case interval cache, because a case
    contributes hundreds of clips and its annotation file should be read once.
    """

    def __init__(self, index: AnnotationIndex, categories: dict, segment_ms=SEGMENT_MS):
        """`categories` maps activity name -> annotation Event string, i.e.
        `{a: spec.event_name(a) for a in spec.activities}`."""
        self.index = index
        self.categories = dict(categories)
        self.segment_ms = segment_ms
        self._cache = {}
        self.misses = Counter()

    def intervals(self, case_id: str):
        if case_id not in self._cache:
            ivs = None
            f = self.index.lookup(case_id)
            if f is None:
                self.misses["ambiguous" if self.index.is_ambiguous(case_id)
                            else "no_annotation_file"] += 1
            else:
                got, err = read_annotation(f)
                if err:
                    self.misses["unreadable"] += 1
                else:
                    ivs = intervals_by_category(got[0])
            self._cache[case_id] = ivs
        return self._cache[case_id]

    def fractions(self, case_id: str, stem: str):
        """{activity: fraction} for one clip, or None when unrecoverable.

        Rounded to 2 decimals ON PURPOSE: a tagged site's fractions come from
        `f"{frac:.2f}"` in the filename, and giving one site more precision than
        the other would move clips across a threshold at one site only — the
        exact asymmetry this backfill exists to remove.
        """
        window = window_from_stem(stem)
        if window is None:
            self.misses["no_window_in_filename"] += 1
            return None
        ivs = self.intervals(case_id)
        if ivs is None:
            return None
        start, end = window
        return {a: round(overlap_ms(start, end, ivs.get(cat, [])) / self.segment_ms, 2)
                for a, cat in self.categories.items()}
