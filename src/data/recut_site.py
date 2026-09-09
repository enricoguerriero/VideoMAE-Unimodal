#!/usr/bin/env python3
"""
recut_site.py

Re-cut a site's 3-second clips from the raw videos, pairing each video to its
annotation file by CASE KEY instead of by exact filename.

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
`process_dataset.py` discovers cases as `videos & annots` — an intersection of
raw filename STEMS. That works at DRC, where every annotation file is named
`<case_id>.txt`, and fails at Haydom, where they are not: the audit
(`scripts/audit_source_data.py`, section 8b) finds 489 files matching only 61 of
246 cases on an exact stem, while the same directories cover 242 of 246 once the
canonical-id rule is applied.

The consequence is measurable. `Unprocessed_data/videos` holds 400 Haydom
videos; the clip tree `build_data.sh` indexes covers 246 cases. The per-case
clip density is identical to Ronald Paleczny's Haydom-only pipeline (290 s of
clips per case vs his 288), so the missing data is not a labelling or extraction
difference — it is 154 staged videos that never produced a clip, and 62 more
that were never staged at all.

This script closes that gap. It pairs by `AnnotationIndex` (exact stem OR any
run of >= 5 digits — Ronald's rule, already used by build_manifest.py's
backfill), stages the annotations into the CLEANED 5-column format
`data_process.py` expects, and cuts with the unchanged `VideoDataProcessor`, so
the new clips are labelled by exactly the same rule as the existing ones.

--------------------------------------------------------------------------
--dry-run projects the result WITHOUT encoding anything
--------------------------------------------------------------------------
Cutting 400 episodes is hours of work and tens of gigabytes, so the projection
is not an estimate: `data_process.py` now exposes the two pure functions that
decide the outcome — `windows()` (how many clips) and `label_window()` /
`bucket_for_label()` (which bucket each lands in) — and this script calls the
same ones the cut will. Given the annotation intervals and the video duration,
the projected bucket census is what the cut produces, modulo clips whose frames
fail to decode.

    # what would I get?
    python -m src.data.recut_site --site Haydom \\
        --videos   /spo/LS-Haydom/.../Unprocessed_data/videos \\
        --annotations /spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations \\
        --stage-dir /spo/.../Data_processing_recut \\
        --out       /spo/.../Processed_data_recut \\
        --compare   data/clips_all.csv \\
        --dry-run

    # do it (the projection prints first either way; --yes is required to cut)
    python -m src.data.recut_site ... --yes --workers 4

--------------------------------------------------------------------------
What the staged annotations look like, and why column 5 matters
--------------------------------------------------------------------------
Each paired case is written to
`<stage-dir>/Unprocessed_data/anot_files/<case_id>.txt` as

    <legacy category>\\t<start_ms>\\t<end_ms>\\t<duration_ms>\\t<original event>

Column 1 is the category `data_process.py` maps through `map_labels`, derived
with `annotations.classify_event` + `legacy_category` — so Haydom's misspelled
"penguine" strings become `Suction` (the notebooks' `corrections` pass, which
`relevant_patterns` alone does not do) and `Suction using tube` becomes
`Ignored label`, exactly as the processor treated them.

Column 5 keeps the ORIGINAL annotator string. That is deliberate and load
bearing: it makes the staged file read as `kind == "cleaned"` by
`annotations.annotation_kind`, which is the one path where
`intervals_by_category` recovers the per-DEVICE suction categories. Drop it and
`configs/data_suction3.yaml` loses penguin/bulb/tube on the re-cut tree the same
way it would on any device-less export.

Visibility rows are removed here rather than left for the processor's
exact-string filter, using the typo-tolerant `annotations.is_visibility`. They
are not discarded: `<stage-dir>/visibility.csv` records every one, so the
"newborn actually on camera" gate that Ronald's pipeline applies (and that this
corpus currently ignores) can be added later without re-reading a single
annotation file.

READ-ONLY on the source trees. It writes only under --stage-dir and --out.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

from .annotations import (CASE_KEY_MIN_DIGITS, MAP_LABELS, AnnotationIndex,
                          annotation_kind, classify_event, intervals_by_category,
                          is_visibility, legacy_category, overlap_ms,
                          read_annotation)
from .data_process import bucket_for_label, label_window, windows

SEGMENT_MS = 3000
SHIFT_MS = 1000
BUCKET_NAMES = {0: "non_target", 1: "stimulation", 2: "ventilation", 3: "suction",
                4: "no_overlap", 5: "no_label", 6: "partial", 7: "target_overlap",
                8: "partial_overlap"}
#: the five interval lists data_process.load_annotation_data builds, in the order
#: label_window takes them
CATEGORY_ORDER = ("Stimulation", "Ventilation", "Suction", "Non-target", "Ignored label")

_DIGITS = re.compile(r"(\d+)")


def canonical_case_id(stem: str) -> str:
    """Video filename stem -> the case id to file its clips under.

    Ronald's `extract_digits` rule, floored at CASE_KEY_MIN_DIGITS: the first run
    of >= 5 digits, else the stem unchanged. Using it here is what makes the
    re-cut tree's `case_id` column line up with the existing manifest and with
    `split_cases.DEFAULT_TEST_CASES` — a video staged as
    `Newborn_2023_37103123_cam1.mp4` must become case `37103123`, not a new id
    that looks like a different episode to the splitter.
    """
    for run in _DIGITS.findall(stem):
        if len(run) >= CASE_KEY_MIN_DIGITS:
            return run
    return stem


def probe_duration_ms(path: Path) -> int:
    """Video duration in ms, by the SAME arithmetic as `load_video_data`.

    `int((frame_count / fps) * 1000)` — matched deliberately, because the clip
    count depends on it and a projection computed a different way would drift
    from the cut by a window or two per episode. Only the header is read.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return int((frames / fps) * 1000) if fps else 0


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def pair_videos(video_dir: Path, index: AnnotationIndex):
    """-> (paired, report) where paired is [{case_id, video, annotation}].

    One entry per case. A canonical id claimed by two videos keeps the first and
    reports the rest: two files for one episode is a staging accident, and
    silently cutting both would put the same baby in the manifest twice under
    one case id, which the whole-case split cannot then separate.
    """
    rep = {"videos": 0, "no_annotation": [], "duplicate_id": [], "ambiguous": []}
    seen: dict[str, Path] = {}
    paired = []
    for mp4 in sorted(video_dir.glob("*.mp4")):
        rep["videos"] += 1
        case_id = canonical_case_id(mp4.stem)
        if case_id in seen:
            rep["duplicate_id"].append((case_id, mp4.name, seen[case_id].name))
            continue
        annot = index.lookup(case_id)
        if annot is None and case_id != mp4.stem:
            annot = index.lookup(mp4.stem)      # a file named after the raw stem
        if annot is None:
            if index.is_ambiguous(case_id):
                rep["ambiguous"].append(case_id)
            else:
                rep["no_annotation"].append(case_id)
            continue
        seen[case_id] = mp4
        paired.append({"case_id": case_id, "video": mp4, "annotation": annot})
    return paired, rep


# ---------------------------------------------------------------------------
# Reading one case: rows -> (intervals, visibility rows, cleaned lines)
# ---------------------------------------------------------------------------
def prepare_case(annot: Path):
    """-> (intervals, cleaned_rows, visibility_rows, error).

    `intervals` is `intervals_by_category`'s output, i.e. the merged interval
    lists the processor would build. `cleaned_rows` are the 5-column tuples to
    stage. Both come from ONE classification pass, so the file written and the
    census projected cannot disagree.
    """
    got, err = read_annotation(annot)
    if err:
        return None, None, None, err
    rows = got[0]
    kind = annotation_kind(rows)
    cleaned, visibility = [], []
    for event, start, end, original in rows:
        if is_visibility(original) or is_visibility(event):
            visibility.append((start, end, (original or event)))
            continue
        if kind == "cleaned":
            cat = classify_event(original) or event
            keep = original or event
        else:
            cat = classify_event(event) or "Ignored label"
            keep = event
        leg = legacy_category(cat)
        if leg not in MAP_LABELS:
            continue
        # tabs/newlines in an annotator's free text would shift the column the
        # processor reads, so they are flattened rather than escaped.
        keep = re.sub(r"\s+", " ", str(keep)).strip()
        cleaned.append((leg, int(start), int(end), int(end) - int(start), keep))
    if not cleaned:
        return None, None, visibility, "no usable annotation row"
    return intervals_by_category(rows), cleaned, visibility, None


def project_case(intervals, cleaned, video_duration_ms, segment_ms, shift_ms):
    """Bucket census this case would produce. Same functions as the cut."""
    last_end = max(r[2] for r in cleaned)
    effective = min(video_duration_ms, last_end) if video_duration_ms else last_end
    iv = [intervals.get(c, []) for c in CATEGORY_ORDER]
    census = Counter()
    for start, end in windows(video_duration_ms or last_end, effective,
                              segment_ms, shift_ms):
        stim, vent, suct, nt, other = (overlap_ms(start, end, v) for v in iv)
        bucket, _ = bucket_for_label(
            label_window(stim, vent, suct, nt, other, segment_ms))
        census[bucket] += 1
    return census


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------
def stage_case(entry, cleaned, stage_dir: Path, copy_videos: bool) -> None:
    """Write the cleaned annotation and link the video into the layout
    `VideoDataProcessor` reads: <stage>/Unprocessed_data/{anot_files,videos}."""
    unp = stage_dir / "Unprocessed_data"
    (unp / "anot_files").mkdir(parents=True, exist_ok=True)
    (unp / "videos").mkdir(parents=True, exist_ok=True)
    txt = unp / "anot_files" / f"{entry['case_id']}.txt"
    with txt.open("w", newline="") as f:
        for cat, start, end, dur, original in cleaned:
            f.write(f"{cat}\t{start}\t{end}\t{dur}\t{original}\n")
    dst = unp / "videos" / f"{entry['case_id']}.mp4"
    if dst.exists() or dst.is_symlink():
        return
    if copy_videos:
        import shutil
        shutil.copy2(entry["video"], dst)
    else:
        # A symlink keeps this cheap and non-destructive; cv2 follows it.
        os.symlink(os.path.realpath(entry["video"]), dst)


def cut_case(case_id: str, stage_dir: str, out_dir: str, segment_size: int, shift: int):
    """One case through the unchanged VideoDataProcessor. Top-level so it can be
    handed to a ProcessPoolExecutor."""
    from .data_process import VideoDataProcessor
    proc = VideoDataProcessor(
        video_file=f"{case_id}.mp4", annotation_file=f"{case_id}.txt",
        segment_size=segment_size, shift=shift, date_of_recording=case_id,
        folder_name=out_dir, for_predict=False, base_dir=stage_dir)
    proc.run_video_only()
    return case_id


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def read_manifest_census(path: Path, site: str):
    """Bucket census + case count for `site` in an existing manifest CSV."""
    census, cases = Counter(), set()
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if site and row.get("site") != site:
                    continue
                census[int(row["bucket"])] += 1
                cases.add(row["case_id"])
    except (OSError, KeyError, ValueError) as exc:
        print(f"[warn] could not read {path} for comparison: {exc}")
        return None, None
    return census, cases


def report_pairing(rep, index: AnnotationIndex, n_paired: int) -> None:
    print("\n--- pairing -------------------------------------------------------")
    print(f"  videos found                 : {rep['videos']:,}")
    print(f"  annotation directories       : {len(index.dirs)} "
          f"({len(index):,} case key(s))")
    print(f"  PAIRED                       : {n_paired:,}")
    for label, key in (("no annotation file", "no_annotation"),
                       ("annotated but AMBIGUOUS", "ambiguous")):
        ids = rep[key]
        if ids:
            print(f"  {label:<29}: {len(ids):,}  {sorted(ids)[:8]}"
                  f"{' ...' if len(ids) > 8 else ''}")
    if rep["duplicate_id"]:
        print(f"  {'two videos, one case id':<29}: {len(rep['duplicate_id']):,} "
              f"(kept the first)")
        for cid, dropped, kept in rep["duplicate_id"][:5]:
            print(f"      {cid}: kept {kept}, skipped {dropped}")
    if rep["ambiguous"]:
        print("  AMBIGUOUS cases are annotated twice with different content. Pick the"
              "\n  authoritative copy (or point --annotations at only one directory) and"
              "\n  they pair on the next run.")


def report_projection(census, per_case, unusable, compare, site) -> None:
    total = sum(census.values())
    print("\n--- projected clips ----------------------------------------------")
    if unusable:
        print(f"  {len(unusable)} paired case(s) produced nothing: "
              f"{[c for c, _ in unusable[:5]]}"
              f"{' ...' if len(unusable) > 5 else ''}")
        for why, n in Counter(w for _, w in unusable).most_common():
            print(f"      {n:>4} x {why}")
    if not total:
        print("  nothing to cut.")
        return
    ncases = len(per_case)
    print(f"  {total:,} clips from {ncases:,} case(s) "
          f"({total / max(ncases, 1):.0f} per case)")

    old, old_cases = (compare if compare else (None, None))
    head = f"  {'bucket':<26}{'projected':>12}"
    if old is not None:
        head += f"{'existing':>12}{'change':>12}"
    print("\n" + head)
    for b in sorted(BUCKET_NAMES):
        line = f"  {b} {BUCKET_NAMES[b]:<24}{census.get(b, 0):>12,}"
        if old is not None:
            was = old.get(b, 0)
            delta = census.get(b, 0) - was
            line += f"{was:>12,}{delta:>+12,}"
        print(line)
    line = f"  {'TOTAL':<26}{total:>12,}"
    if old is not None:
        was = sum(old.values())
        line += f"{was:>12,}{total - was:>+12,}"
        print(line)
        print(f"  {'cases':<26}{ncases:>12,}{len(old_cases):>12,}"
              f"{ncases - len(old_cases):>+12,}")
        gain = 100 * (total - was) / max(was, 1)
        print(f"\n  -> {gain:+.0f}% clips, "
              f"{100 * (ncases - len(old_cases)) / max(len(old_cases), 1):+.0f}% cases "
              f"vs the {site or 'existing'} rows of the manifest")
        s_new = census.get(3, 0) + census.get(7, 0)
        s_old = old.get(3, 0) + old.get(7, 0)
        print(f"  -> suction-positive buckets (3 + 7): {s_old:,} -> {s_new:,} "
              f"({100 * (s_new - s_old) / max(s_old, 1):+.0f}%)")
    else:
        print(line)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, type=Path,
                   help="Directory of source *.mp4 (e.g. <base>/Unprocessed_data/videos).")
    p.add_argument("--annotations", action="append", required=True, type=Path,
                   metavar="DIR", help="Annotation directory, searched recursively. "
                        "Repeatable; every one is indexed by case key and the exact "
                        "case id wins over a digit-run match. Prefer a directory the "
                        "audit reports at high coverage with its suction vocabulary "
                        "intact.")
    p.add_argument("--stage-dir", required=True, type=Path,
                   help="Where to write the cleaned Unprocessed_data/{anot_files,videos} "
                        "layout. Must NOT be an existing site tree — this writes into it.")
    p.add_argument("--out", required=True, type=Path,
                   help="Clip dataset root; a videos/<class>/ tree is created under it. "
                        "Point build_manifest.py at <out>/videos afterwards.")
    p.add_argument("--site", default=None,
                   help="Site name, used only to filter --compare. e.g. Haydom")
    p.add_argument("--compare", type=Path, default=None, metavar="CSV",
                   help="An existing manifest (data/clips_all.csv) to diff the "
                        "projection against, per bucket.")
    p.add_argument("--dry-run", action="store_true",
                   help="Project and stop. Writes nothing at all.")
    p.add_argument("--yes", action="store_true",
                   help="Required to actually stage and cut. Without it the projection "
                        "prints and the run stops, which is the safe default for a job "
                        "that encodes tens of GB.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel cutting processes (default 1). Each case is "
                        "independent; 4-8 is usually I/O bound.")
    p.add_argument("--copy-videos", action="store_true",
                   help="Copy source videos into the staging dir instead of "
                        "symlinking them (symlinks are the default and are free).")
    p.add_argument("--no-probe", action="store_true",
                   help="Do not open video headers for the projection; assume each "
                        "episode is exactly as long as its last annotation. Lets the "
                        "dry run work without cv2, at the cost of a slightly optimistic "
                        "clip count for episodes whose video ends early.")
    p.add_argument("--segment-size", type=int, default=SEGMENT_MS // 1000,
                   help="Clip length in seconds (default 3 — the thesis' value).")
    p.add_argument("--shift", type=int, default=SHIFT_MS // 1000,
                   help="Stride in seconds (default 1 — the thesis' value).")
    p.add_argument("--limit", type=int, default=0,
                   help="Only handle the first N paired cases (smoke test).")
    p.add_argument("--only", action="append", default=None, metavar="CASE_ID",
                   help="Only this case id. Repeatable.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cases that already have clips under --out, so an "
                        "interrupted run can be resumed.")
    args = p.parse_args()

    if not args.videos.is_dir():
        raise SystemExit(f"--videos is not a directory: {args.videos}")
    seg_ms, shift_ms = args.segment_size * 1000, args.shift * 1000

    print("recut_site.py — the source trees are opened read-only.")
    print(f"  videos      : {args.videos}")
    print(f"  annotations : {[str(a) for a in args.annotations]}")
    print(f"  stage-dir   : {args.stage_dir}")
    print(f"  out         : {args.out}")
    print(f"  geometry    : {args.segment_size}s clip / {args.shift}s stride")

    index = AnnotationIndex.from_roots(args.annotations)
    if not len(index):
        raise SystemExit("no annotation file found under --annotations")
    paired, rep = pair_videos(args.videos, index)
    report_pairing(rep, index, len(paired))

    if args.only:
        want = set(args.only)
        paired = [e for e in paired if e["case_id"] in want]
    if args.limit:
        paired = paired[: args.limit]
    if args.skip_existing and args.out.is_dir():
        done = {n.split("_interval_")[0] for n in
                (q.name for q in args.out.rglob("*.mp4"))}
        before = len(paired)
        paired = [e for e in paired if e["case_id"] not in done]
        if before != len(paired):
            print(f"\n  --skip-existing: {before - len(paired):,} case(s) already have "
                  f"clips under {args.out}")
    if not paired:
        raise SystemExit("no case left to process")

    # ---- read every annotation once; project, and keep the cleaned rows -----
    census, per_case, unusable, prepared = Counter(), {}, [], {}
    for i, entry in enumerate(paired, 1):
        intervals, cleaned, visibility, err = prepare_case(entry["annotation"])
        if err:
            unusable.append((entry["case_id"], err))
            continue
        dur = 0 if args.no_probe else probe_duration_ms(entry["video"])
        if not args.no_probe and dur <= 0:
            unusable.append((entry["case_id"], "video header unreadable (0 frames/fps)"))
            continue
        c = project_case(intervals, cleaned, dur, seg_ms, shift_ms)
        if not sum(c.values()):
            unusable.append((entry["case_id"], "no window fits inside the video"))
            continue
        census.update(c)
        per_case[entry["case_id"]] = sum(c.values())
        prepared[entry["case_id"]] = (cleaned, visibility)
        if i % 50 == 0:
            print(f"  ...projected {i:,}/{len(paired):,} case(s)", flush=True)

    compare = None
    if args.compare:
        old, old_cases = read_manifest_census(args.compare, args.site)
        compare = (old, old_cases) if old is not None else None
    report_projection(census, per_case, unusable, compare, args.site)

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return
    if not args.yes:
        print("\nStopping: pass --yes to stage and cut. Nothing was written.")
        return

    # ---- stage --------------------------------------------------------------
    print(f"\n--- staging into {args.stage_dir} ---------------------------------")
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    vis_path = args.stage_dir / "visibility.csv"
    with vis_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "start_ms", "end_ms", "original_event"])
        n_vis = 0
        for entry in paired:
            got = prepared.get(entry["case_id"])
            if got is None:
                continue
            cleaned, visibility = got
            stage_case(entry, cleaned, args.stage_dir, args.copy_videos)
            for start, end, original in visibility:
                w.writerow([entry["case_id"], start, end, original])
                n_vis += 1
    print(f"  {len(prepared):,} case(s) staged "
          f"({'copied' if args.copy_videos else 'symlinked'} videos)")
    print(f"  {n_vis:,} visibility interval(s) -> {vis_path}")
    print("  (that file is the newborn-on-camera gate this corpus does not yet "
          "apply;\n   nothing reads it today)")

    # ---- cut ----------------------------------------------------------------
    todo = [c for c in per_case if c in prepared]
    print(f"\n--- cutting {len(todo):,} case(s) -> {args.out} "
          f"({args.workers} worker(s)) ---")
    args.out.mkdir(parents=True, exist_ok=True)
    stage_s, out_s = str(args.stage_dir), str(args.out)
    ok, failed = 0, []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(cut_case, c, stage_s, out_s,
                                 args.segment_size, args.shift): c for c in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                case_id = futures[fut]
                try:
                    fut.result()
                    ok += 1
                    status = "ok"
                except Exception as exc:               # noqa: BLE001 — keep going
                    failed.append((case_id, f"{type(exc).__name__}: {exc}"))
                    status = "FAILED"
                print(f"  [{n:,}/{len(todo):,}] {case_id} {status}", flush=True)
    else:
        for n, case_id in enumerate(todo, 1):
            try:
                cut_case(case_id, stage_s, out_s, args.segment_size, args.shift)
                ok += 1
                status = "ok"
            except Exception as exc:                   # noqa: BLE001 — keep going
                failed.append((case_id, f"{type(exc).__name__}: {exc}"))
                status = "FAILED"
            print(f"  [{n:,}/{len(todo):,}] {case_id} {status}", flush=True)

    print(f"\n--- done ---------------------------------------------------------")
    print(f"  cut     : {ok:,} case(s)")
    if failed:
        print(f"  FAILED  : {len(failed):,} case(s)")
        for case_id, why in failed[:10]:
            print(f"      {case_id}: {why}")
        print("  Re-run with --skip-existing to retry only these.")
    actual = sum(1 for _ in args.out.rglob("*.mp4"))
    print(f"  clips on disk under {args.out}: {actual:,} "
          f"(projected {sum(census.values()):,})")
    print(f"\nNext: index the new tree and re-split.\n"
          f"  python -m src.data.build_manifest \\\n"
          f"      --root {args.site or 'SITE'}={args.out}/videos \\\n"
          f"      --root DRC=<unchanged DRC videos root> \\\n"
          f"      --data-config configs/data_suction3.yaml \\\n"
          f"      --rebackfill-all --extra-frac suction \\\n"
          f"      --annotations {args.site or 'SITE'}={args.stage_dir}/Unprocessed_data/anot_files \\\n"
          f"      --out data/clips_all.csv\n"
          f"  (the staged anot_files keep the original event string in column 5, so "
          f"the\n   per-device suction backfill resolves from them directly)")


if __name__ == "__main__":
    main()
