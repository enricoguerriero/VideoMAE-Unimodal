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
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# What the thesis pipeline does, mirrored here so we can check the data against it
# ---------------------------------------------------------------------------
#: data_process.py load_annotation_data(): Event string -> category code.
MAP_LABELS = {"Ignored label": 4, "Suction": 3, "Ventilation": 2,
              "Stimulation": 1, "Non-target": 0}
CODE_NAME = {0: "Non-target", 1: "Stimulation", 2: "Ventilation", 3: "Suction",
             4: "Ignored label"}
#: rows with this in the last column are dropped before anything else
DROP_ORIGINAL = "Newborn visible in video frame"
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

FINDINGS: list[str] = []


def finding(msg: str) -> None:
    FINDINGS.append(msg)


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
    """
    rows, ncols = [], Counter()
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return None, f"unreadable: {exc}"
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        ncols[len(parts)] += 1
        if len(parts) < 3:
            continue
        original = parts[4] if len(parts) >= 5 else ""
        try:
            start, end = int(float(parts[1])), int(float(parts[2]))
        except ValueError:
            continue
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


def intervals_by_code(rows):
    """Apply the DROP_ORIGINAL filter + MAP_LABELS, then merge per category."""
    buckets = defaultdict(list)
    for event, start, end, original in rows:
        if original == DROP_ORIGINAL:
            continue
        code = MAP_LABELS.get(event)
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
    rule("1. ANNOTATION FILE FORMAT — column counts in anot_files/")
    print(f"{'site':<10}{'files':>8}{'rows':>10}   column-count histogram")
    for name, s in sites.items():
        d = s["present"].get("anot_files")
        if d is None:
            print(f"{name:<10}{'-':>8}{'-':>10}   (no anot_files/)")
            continue
        files = sorted(d.glob("*.txt"))
        ncols, nrows, bad = Counter(), 0, []
        for f in files:
            got, err = read_annotation(f)
            if err:
                bad.append((f.name, err)); continue
            rows, cols = got
            ncols.update(cols); nrows += len(rows)
        hist = ", ".join(f"{c} cols x{n:,}" for c, n in sorted(ncols.items()))
        print(f"{name:<10}{len(files):>8,}{nrows:>10,}   {hist or '(empty)'}")
        s["anot_dir"] = d
        s["anot_files"] = files
        if any(c < 5 for c in ncols):
            finding(f"{name}: some anot_files rows have <5 columns — the 5th column "
                    f"(Corrected_Original_Event) is what makes re-mapping possible")
        for fn, err in bad[:3]:
            print(f"           [WARN] {fn}: {err}")


def section_vocabulary(sites):
    rule("2. EVENT VOCABULARY — original annotator string -> mapped category")
    print("The 5th column preserves the ORIGINAL label; the 1st is the mapped")
    print("category. A string mapped differently between sites is a labelling")
    print("difference between hospitals, not a data difference.\n")

    per_site = {}
    for name, s in sites.items():
        pairs = Counter()
        for f in s.get("anot_files", []):
            got, err = read_annotation(f)
            if err:
                continue
            for event, _, _, original in got[0]:
                pairs[(original or "<no 5th col>", event)] += 1
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

    conflicts = [o for o in set(map_a) & set(map_b) if map_a[o] != map_b[o]]
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

    for label, only, site in ((f"only in {a}", only_a, a), (f"only in {b}", only_b, b)):
        if only:
            print(f"   [{label}] {len(only)} original string(s):")
            for o in only[:12]:
                print(f"       {o[:50]:<50} -> {sorted(map_a[o] if site == a else map_b[o])}")
            sucty = [o for o in only if "suction" in o.lower()]
            if sucty:
                finding(f"SUCTION VOCABULARY: {sucty} present only at {site}")


def section_intervals(sites):
    rule("3. INTERVAL STATISTICS — per category, per site (after merging)")
    print(f"{'site':<10}{'category':<15}{'cases':>7}{'intervals':>11}"
          f"{'total_s':>11}{'median_ms':>11}{'max_ms':>10}")
    for name, s in sites.items():
        per_code = defaultdict(list)
        cases_with = defaultdict(set)
        for f in s.get("anot_files", []):
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
    if case_id in cache:
        return cache[case_id]
    d = site["present"].get("anot_files")
    ivs = None
    if d is not None:
        f = d / f"{case_id}.txt"
        if f.is_file():
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
            for code, want in c["tags"].items():
                # the tag is always overlap_ms / (segment_size * 1000)
                got = overlap_ms(int(c["start_ms"]), int(c["end_ms"]),
                                 ivs.get(code, [])) / SEGMENT_MS
                checked += 1
                if abs(got - want) <= FRAC_TOL:
                    ok += 1
                elif len(mism) < 6:
                    mism.append((c["path"].name, CODE_NAME[code], want, got))
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
        d = s["present"].get("anot_files")
        found = {c for c in cases if d is not None and (d / f"{c}.txt").is_file()}
        print(f"   {len(untagged):,} untagged clips across {len(cases)} cases")
        print(f"   with a parseable time window in the filename : "
              f"{len(have_window):,}/{len(untagged):,} "
              f"({100*len(have_window)/max(len(untagged),1):.1f}%)")
        print(f"   cases with a matching anot_files/<case>.txt   : "
              f"{len(found)}/{len(cases)} "
              f"({100*len(found)/max(len(cases),1):.1f}%)")
        covered = sum(1 for c in untagged
                      if c["start_ms"] is not None and c["case_id"] in found)
        print(f"   => RECOVERABLE fractions: {covered:,}/{len(untagged):,} clips "
              f"({100*covered/max(len(untagged),1):.1f}%)")
        if cases - found:
            miss = sorted(cases - found)
            print(f"   cases with NO annotation file ({len(miss)}): {miss[:8]}"
                  f"{' ...' if len(miss) > 8 else ''}")
            finding(f"{name}: {len(miss)} case(s) have clips but no anot_files entry")
        if covered < len(untagged):
            finding(f"{name}: only {100*covered/max(len(untagged),1):.1f}% of untagged "
                    f"clips can have their fractions recovered")


def section_hunt(sites, roots):
    """Search given roots for files named after the clip case ids.

    Runs only with --find-annotations. Useful when a site's anot_files/ is not
    where the notebook says it should be: the clips name their cases, so the
    annotations can be located by matching filenames anywhere on the volume.
    """
    rule("8b. ANNOTATION HUNT — looking for the missing annotation files")
    for name, s in sites.items():
        cases = {c["case_id"] for c in s.get("clip_info", [])}
        if not cases:
            continue
        if s["present"].get("anot_files") is not None:
            print(f"   {name}: anot_files/ already present, skipping")
            continue
        h(f"{name}: {len(cases)} case ids to locate")
        for root in roots:
            root = Path(root).expanduser()
            if not root.is_dir():
                print(f"   {root}  [not a directory]")
                continue
            hits, where = 0, Counter()
            try:
                for f in root.rglob("*.txt"):
                    if f.stem in cases:
                        hits += 1
                        where[f.parent] += 1
            except OSError as exc:
                print(f"   {root}  [error: {exc}]")
                continue
            print(f"   {root}")
            if not hits:
                print("       no .txt matching any case id")
            for d, n in where.most_common(5):
                print(f"       {n:>6,} matches in {d}")
            if hits:
                finding(f"{name}: {hits:,} case-matching .txt files found under "
                        f"{root} — annotations may be recoverable from there")


def section_findings():
    rule("9. FINDINGS")
    if not FINDINGS:
        print("  none — the two sites look consistently processed on every check above.")
        return
    for i, f in enumerate(FINDINGS, 1):
        print(f"  [{i:>2}] {f}")
    print(f"\n  {len(FINDINGS)} finding(s). Anything labelled CONFLICT, MISMATCH or")
    print("  ASYMMETRY means the hospitals were not processed identically, and a")
    print("  per-site score comparison is measuring that as well as the model.")


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
                   help="Search DIR recursively for .txt files named after the clip "
                        "case ids. Use when a site's anot_files/ is missing. "
                        "Repeatable.")
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

    for num, fn in [("0", lambda: section_layout(sites)),
                    ("1", lambda: section_format(sites)),
                    ("2", lambda: section_vocabulary(sites)),
                    ("3", lambda: section_intervals(sites)),
                    ("4", lambda: section_raw_vs_clean(sites)),
                    ("5", lambda: section_clips(sites)),
                    ("6", lambda: section_recompute(sites, args.sample)),
                    ("7", lambda: section_implied_cut(sites)),
                    ("8", lambda: section_backfill(sites)),
                    ("8b", lambda: section_hunt(sites, args.find_annotations)
                     if args.find_annotations else None)]:
        if num in skip:
            print(f"\n[skipped section {num}]")
            continue
        fn()
    section_findings()


if __name__ == "__main__":
    main()
