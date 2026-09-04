#!/usr/bin/env python3
"""
build_manifest.py

Scan the processed 3-second video clips produced by the multimodal thesis'
DataProcessor (for BOTH the Haydom and DRC sites) and emit ONE combined,
TASK-AGNOSTIC clip-level manifest CSV.

This is the PRIMARY data path for this repo: the processed clips already exist
on the VM (they are data, not code), so we simply index them — no re-cutting,
which guarantees the VideoMAE model trains on exactly the same clips as the
multimodal MoViNet video base model.

--------------------------------------------------------------------------
The manifest records EVIDENCE, not labels
--------------------------------------------------------------------------
Every clip filename encodes its original label bucket, and — if it was cut by a
processor generation that wrote them — the per-activity share of the 3 s window
that activity covered:

    {case}_interval_{n}_start_{ms}_end_{ms}[_stim0.67][_vent0.55]_{bucket}.mp4

All of it is copied verbatim into the manifest:

    video_path, case_id, site, bucket, clip_dir, tagged, frac_<activity>...

--------------------------------------------------------------------------
Two generations of clips, and why `clip_dir` matters
--------------------------------------------------------------------------
The Haydom and DRC trees were cut by different versions of data_process.py. DRC
filenames carry the `_stim0.67` fraction tags; Haydom filenames do NOT — for
those clips the fractions are UNKNOWN, and reading them as zero silently labels
every Haydom activity clip as "no activity".

The identity survives anyway, because data_process.py files each clip under a
directory that names the activities involved — `ventilation/`,
`partial/stimulation/`, `target_overlap/stimulation+ventilation/`. So the
manifest records `clip_dir` (the path relative to the site root) and `tagged`
(0 when the bucket implies a tag that is absent), and the DataSpec labels each
kind on its own terms. The only thing genuinely lost for an untagged site is the
ability to MOVE a threshold: its labels are frozen at the cut the processor
applied. `report_tag_coverage` prints how much of each site is affected.

--------------------------------------------------------------------------
Backfilling an untagged site (--annotations)
--------------------------------------------------------------------------
A tag is not information the processor had and discarded. It is

    frac[a] = overlap_ms(start, end, merge_intervals(intervals[a])) / 3000

and every clip filename still carries `_start_{ms}_end_{ms}`, so the fractions
can be rebuilt from the annotations alone — no video decoding, no re-cutting,
and no per-case offsets (those only ever touched the removed accelerometer
branch). `--annotations SITE=DIR` does exactly that for a site whose clips have
no tags, which turns `thresholds` from inert into live for it.

It REFUSES unless it first reproduces tags that already exist. Recomputing from
an annotation export of a different vintage than the clips yields plausible,
quietly wrong numbers, and the only defence is ground truth: clips that carry a
tag. Point `--verify-root SITE=PATH` at any tagged vintage of the same site
(it is used for the check only, never indexed). A clip whose fractions cannot
be recovered is LEFT untagged and keeps its bucket+directory label, so a
partial backfill is safe — the two mechanisms coexist row by row.

NO thresholding, NO bucket filtering and NO task decision happens here. All of
that lives in configs/data.yaml and is applied by DataSpec at Dataset load time
(src/data/spec.py). Consequence: switching multiclass <-> multilabel, moving a
threshold, or admitting a previously-dropped bucket needs NO rescan and NO
re-split — just edit the YAML and re-run training.

The data config is still read here, but only for `tag_keys` (which filename
abbreviation belongs to which activity) and to REPORT what the current settings
would yield.

Usage:
    python -m src.data.build_manifest \
        --root Haydom=/path/to/Haydom/Processed_.../videos \
        --root DRC=/path/to/DRC/Processed_.../videos \
        --out data/clips_all.csv [--data-config configs/data.yaml]

    # ... and, for a site whose clips carry no fraction tags:
        --annotations Haydom=/path/to/Haydom/Annotations \
        --verify-root Haydom=/path/to/Haydom/Processed_..._test/videos

Each --root is SITE=PATH; PATH is a `videos/` directory containing the per-class
subfolders (or any tree of *.mp4 clips following the naming convention).
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from .annotations import FRAC_TOL, SEGMENT_MS, AnnotationIndex, FractionSource
from .spec import BUCKET_NAMES, DataSpec, TAG_BEARING_BUCKETS


def scan_root(site: str, root: Path, spec: DataSpec):
    """Yield one row per clip under `root`, plus counters for the report.

    Two things are recorded that the filename alone cannot always give:

    `clip_dir` — the clip's directory relative to the root. data_process.py
    files every clip under a directory that NAMES the activities involved
    (`ventilation/`, `partial/stimulation/`,
    `target_overlap/stimulation+ventilation/`). That is the only surviving
    record of activity identity for a tree cut before fraction tags were
    written, so it is preserved verbatim rather than resolved here.

    `tagged` — 0 when the filename carries no fraction tag although its bucket
    requires one, i.e. the fractions are UNKNOWN rather than zero. Mixing a
    tagged and an untagged site in one manifest is fine and expected; the
    DataSpec labels each kind on its own terms.
    """
    rows, buckets, unparsed = [], Counter(), []
    dir_census, structural = Counter(), []
    for mp4 in sorted(root.rglob("*.mp4")):
        bucket, fracs, tagged = spec.parse_stem(mp4.stem)
        if bucket is None:
            if len(unparsed) < 5:
                unparsed.append(str(mp4))
            continue
        buckets[bucket] += 1
        rel_dir = mp4.parent.relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dir_acts = spec.activities_from_path(rel_dir)
        dir_census[(rel_dir, bucket, dir_acts)] += 1

        # Structural check: a bucket that names an activity must be filed under a
        # directory that names it too, or the identity is unrecoverable for an
        # untagged clip. Reported, never silently repaired.
        if 1 <= bucket <= len(spec.activities):
            expected = spec.activities[bucket - 1]
            if expected not in dir_acts and len(structural) < 5:
                structural.append(f"{mp4} — bucket {bucket} implies {expected}, "
                                  f"directory says {dir_acts or '()'}")
        elif bucket in (6, 7, 8) and not dir_acts:
            if len(structural) < 5:
                structural.append(f"{mp4} — bucket {bucket} needs an activity in its "
                                  f"directory, found none")

        row = {
            "video_path": str(mp4.resolve()),
            "case_id": spec.case_id_from_stem(mp4.stem),
            "site": site,
            "bucket": bucket,
            "clip_dir": rel_dir,
            "tagged": int(tagged),
        }
        for a in spec.activities:
            row[f"frac_{a}"] = f"{fracs.get(a, 0.0):.2f}"
        rows.append(row)
    return rows, buckets, unparsed, dir_census, structural


# ---------------------------------------------------------------------------
# Backfill: give an untagged site the fractions its filenames never recorded
# ---------------------------------------------------------------------------
def verify_recompute(rows, source: FractionSource, spec: DataSpec, limit=4000):
    """Recompute the fractions of clips that ALREADY have tags, and compare.

    This is the gate, and it is not optional. The recomputation is only as
    trustworthy as the annotations it reads, and a site can easily be paired
    with an annotation export from a different vintage than its clips — which
    produces plausible-looking numbers that are quietly wrong. Clips carrying a
    tag are ground truth for exactly this check: reproduce them, or do not
    trust the same computation on the clips that have none.

    -> (checked, ok, mismatches)
    """
    checked = ok = 0
    mism = []
    for r in rows[:limit] if limit else rows:
        if not int(r["tagged"]):
            continue
        stem = Path(r["video_path"]).stem
        got = source.fractions(r["case_id"], stem)
        if got is None:
            continue
        for a in spec.activities:
            want = float(r[f"frac_{a}"])
            checked += 1
            if abs(got[a] - want) <= FRAC_TOL:
                ok += 1
            elif len(mism) < 6:
                mism.append((stem, a, want, got[a]))
    return checked, ok, mism


def backfill_fractions(rows, site, ann_roots, spec: DataSpec, verify_rows,
                       min_agreement, allow_unverified, segment_ms):
    """Fill in `frac_*` for one site's untagged clips. Returns a report dict.

    Only TAG-BEARING buckets are touched: a bucket 0/4/5 clip has no activity
    overlap at all, so its all-zero fractions are already correct and rewriting
    them would be noise.

    A clip whose fractions cannot be recovered is LEFT untagged rather than
    guessed at, so DataSpec keeps labelling it from bucket + directory exactly
    as before. Partial success is therefore safe: the two mechanisms coexist
    per row, which is what `tagged` has always meant.
    """
    rep = {"site": site, "index": None, "checked": 0, "ok": 0, "mism": [],
           "verified": False, "filled": 0, "candidates": 0, "misses": Counter(),
           "skipped": None}
    index = AnnotationIndex.from_roots(ann_roots)
    rep["index"] = index
    if not len(index):
        rep["skipped"] = "no annotation file found under the given --annotations paths"
        return rep
    source = FractionSource(index, {a: spec.event_name(a) for a in spec.activities},
                            segment_ms=segment_ms)

    checked, ok, mism = verify_recompute(verify_rows, source, spec)
    rep.update(checked=checked, ok=ok, mism=mism)
    agreement = (ok / checked) if checked else None
    if checked == 0:
        rep["verified"] = False
        if not allow_unverified:
            rep["skipped"] = ("nothing to verify against — this site has no tagged clip "
                              "in the manifest. Point --verify-root at a tagged vintage "
                              "of this site, or pass --allow-unverified-backfill")
            return rep
    elif agreement < min_agreement:
        rep["skipped"] = (f"verification failed: {100*agreement:.1f}% of existing tags "
                          f"reproduce (need {100*min_agreement:.0f}%). The annotations "
                          f"and the clips are not from the same pipeline run")
        if not allow_unverified:
            return rep
    else:
        rep["verified"] = True

    for r in rows:
        if r["site"] != site or int(r["tagged"]):
            continue
        if int(r["bucket"]) not in TAG_BEARING_BUCKETS:
            continue
        rep["candidates"] += 1
        got = source.fractions(r["case_id"], Path(r["video_path"]).stem)
        if got is None:
            continue
        for a in spec.activities:
            r[f"frac_{a}"] = f"{got[a]:.2f}"
        r["tagged"] = 1
        rep["filled"] += 1
    rep["misses"] = source.misses
    return rep


def report_backfill(reports, spec: DataSpec):
    print("\n--- fraction backfill ----------------------------------------------")
    for rep in reports:
        site = rep["site"]
        print(f"\n  {site}")
        idx = rep["index"]
        if idx is not None and len(idx):
            print(f"    sources : {len(idx)} case key(s) over {len(idx.dirs)} "
                  f"director{'y' if len(idx.dirs) == 1 else 'ies'}")
            for d in idx.dirs[:4]:
                print(f"              {d}")
        if rep["checked"]:
            pct = 100 * rep["ok"] / rep["checked"]
            print(f"    verify  : {rep['ok']:,}/{rep['checked']:,} existing tags "
                  f"reproduced ({pct:.1f}%)")
            for stem, a, want, got in rep["mism"]:
                print(f"              MISMATCH {a:<12} tag={want:.2f} "
                      f"recomputed={got:.2f}  {stem[:52]}")
        elif rep["skipped"] is None:
            print("    verify  : no tagged clip available — UNVERIFIED (forced)")
        if rep["skipped"]:
            print(f"    SKIPPED : {rep['skipped']}")
            print("              clips stay untagged and keep their bucket+directory "
                  "label")
            continue
        print(f"    filled  : {rep['filled']:,} / {rep['candidates']:,} untagged "
              f"tag-bearing clips"
              + ("" if rep["verified"] else "   [UNVERIFIED]"))
        for reason, n in sorted(rep["misses"].items(), key=lambda kv: -kv[1]):
            print(f"              {n:,} not recovered: {reason}")
    if any(r["filled"] for r in reports):
        print("\n  A backfilled clip is indistinguishable from one the processor tagged:"
              "\n  its fractions are the same function of the same annotations, rounded"
              "\n  the same way. `thresholds` in the data config now apply to it.")
    else:
        print("\n  Nothing was backfilled — every clip keeps the label data_process.py"
              "\n  gave it, exactly as before this flag existed.")


def parse_root(spec_str: str):
    if "=" not in spec_str:
        raise argparse.ArgumentTypeError(f"--root must be SITE=PATH, got: {spec_str}")
    site, path = spec_str.split("=", 1)
    return site.strip(), Path(path).expanduser()


def report_tag_coverage(rows, sites, spec: DataSpec):
    """How much of each site carries fraction tags — and what that costs.

    A site cut before data_process.py wrote `_stim0.67`-style tags has no
    fractions at all. Its clips are still fully labellable (bucket + directory
    give the activity identity), but `thresholds` and `weak_threshold` are inert
    for them: the cut was made when the clips were produced and cannot be
    re-tuned. That is a real limitation and it is printed, not hidden.
    """
    print("\n--- fraction tags --------------------------------------------------")
    print(f"{'site':<14}{'tagged':>12}{'untagged':>12}{'':>4}note")
    any_untagged = False
    for s in sites:
        rs = [r for r in rows if r["site"] == s]
        tagged = sum(int(r["tagged"]) for r in rs)
        untagged = len(rs) - tagged
        any_untagged = any_untagged or untagged > 0
        note = ("labelled from bucket + directory; thresholds inert"
                if untagged else "fractions available; thresholds apply")
        print(f"{s:<14}{tagged:>12,}{untagged:>12,}{'':>4}{note}")
    if any_untagged:
        print("\n  Clips WITHOUT tags are labelled from their bucket and directory, which\n"
              "  carry the activity identity (see DataSpec.activities_from_path). What is\n"
              "  lost is only the ability to MOVE a threshold for those clips — the label\n"
              "  itself is exactly the one data_process.py assigned when it cut them.")


def report_directory_recovery(dir_census, spec: DataSpec):
    """directory -> bucket -> recovered activities, with counts.

    This is the audit trail for the path-based recovery: every row states what
    the code concluded from a directory, so a wrong conclusion is visible rather
    than buried in the labels.
    """
    print("\n--- directory -> activities (how untagged clips get their label) ----")
    print(f"{'directory':<44}{'bucket':>7}{'clips':>10}  recovered activities")
    for (rel_dir, bucket, acts), n in sorted(dir_census.items(),
                                             key=lambda kv: (-kv[1], kv[0][0])):
        shown = "+".join(acts) if acts else "-"
        print(f"{(rel_dir or '<root>'):<44}{bucket:>7}{n:>10,}  {shown}")


def report(rows, per_site_buckets, spec: DataSpec):
    """Print the bucket census and what the CURRENT data config would produce."""
    sites = list(per_site_buckets)
    total = len(rows)
    print("\n--- bucket census (as found on disk) -------------------------------")
    header = f"{'bucket':<26}" + "".join(f"{s:>12}" for s in sites) + f"{'TOTAL':>12}  policy"
    print(header)
    for b in sorted(BUCKET_NAMES):
        counts = [per_site_buckets[s].get(b, 0) for s in sites]
        policy = "keep" if spec.keeps_bucket(b) else "DROP"
        print(f"{b} {BUCKET_NAMES[b]:<24}" + "".join(f"{c:>12,}" for c in counts)
              + f"{sum(counts):>12,}  {policy}")

    print(f"\n--- resolved by {spec.source} ({spec.task}) ------------------")
    kept, dropped = Counter(), 0
    per_site_kept = {s: Counter() for s in sites}
    per_site_dropped = Counter()
    for r in rows:
        fracs = {a: float(r[f"frac_{a}"]) for a in spec.activities}
        label = spec.resolve(int(r["bucket"]), fracs, tagged=bool(int(r["tagged"])),
                             dir_activities=spec.activities_from_path(r["clip_dir"]))
        if label is None:
            dropped += 1
            per_site_dropped[r["site"]] += 1
        elif spec.is_multilabel:
            active = tuple(a for a, t, m in zip(spec.activities, label.targets, label.mask)
                           if t == 1.0 and m == 1.0)
            name = "+".join(active) or f"<{spec.negative_class}>"
            kept[name] += 1
            per_site_kept[r["site"]][name] += 1
        else:
            name = spec.class_names[label.class_index]
            kept[name] += 1
            per_site_kept[r["site"]][name] += 1
    print(f"{'':<34}" + "".join(f"{s:>12}" for s in sites) + f"{'TOTAL':>12}")
    for name, n in sorted(kept.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32}" + "".join(f"{per_site_kept[s][name]:>12,}" for s in sites)
              + f"{n:>12,}")
    n_kept = sum(kept.values())
    print(f"  {'(dropped by this config)':<32}"
          + "".join(f"{per_site_dropped[s]:>12,}" for s in sites) + f"{dropped:>12,}")
    print(f"\n  usable clips: {n_kept:,} / {total:,} ({100 * n_kept / max(total, 1):.1f}%)")
    print("  Buckets/thresholds are applied at TRAINING time — change "
          "configs/data.yaml and re-run\n  training without rebuilding this manifest.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", required=True, type=parse_root,
                   metavar="SITE=PATH", help="Repeatable. e.g. --root Haydom=/.../videos")
    p.add_argument("--out", required=True, type=Path, help="Output combined manifest CSV.")
    p.add_argument("--data-config", default=None,
                   help="Data/label config YAML (default: configs/data.yaml).")
    p.add_argument("--annotations", action="append", default=None, type=parse_root,
                   metavar="SITE=DIR", help="Repeatable. Recompute the per-activity "
                        "window fractions for SITE's UNTAGGED clips from the annotation "
                        "files under DIR (searched recursively). Use this to give a site "
                        "cut before fraction tags existed the same threshold-tuning "
                        "freedom as a tagged one. Refuses to run unless the same "
                        "computation reproduces tags that already exist — see "
                        "--verify-root.")
    p.add_argument("--verify-root", action="append", default=None, type=parse_root,
                   metavar="SITE=PATH", help="A TAGGED clip tree of SITE, used only to "
                        "verify the backfill and never indexed into the manifest. "
                        "Needed when the site's own clips carry no tags at all "
                        "(otherwise the manifest's own tagged rows are the reference).")
    p.add_argument("--segment-ms", type=int, default=SEGMENT_MS,
                   help=f"Clip length the fractions are relative to (default {SEGMENT_MS}).")
    p.add_argument("--min-agreement", type=float, default=0.99,
                   help="Fraction of existing tags a backfill must reproduce before it "
                        "is applied (default 0.99).")
    p.add_argument("--allow-unverified-backfill", action="store_true",
                   help="Apply the backfill even when verification fails or is "
                        "impossible. Every affected clip is marked tagged, so this "
                        "cannot be undone by re-reading the manifest — say why in your "
                        "notes if you use it.")
    args = p.parse_args()

    spec = DataSpec.load(args.data_config)
    print(spec.describe())

    all_rows, per_site_buckets, per_site_n = [], {}, {}
    dir_census, structural = Counter(), []
    for site, root in args.root:
        if not root.exists():
            print(f"[WARN] root does not exist: {root} (site={site}) — skipping")
            continue
        rows, buckets, unparsed, dirs, bad = scan_root(site, root, spec)
        dir_census.update(dirs)
        structural.extend(bad[:5 - len(structural)])
        per_site_buckets[site] = buckets
        per_site_n[site] = len(rows)
        all_rows.extend(rows)
        print(f"[INFO] {site}: {len(rows)} clips from {root}")
        for u in unparsed:
            print(f"       [WARN] unparseable filename, skipped: {u}")

    if not all_rows:
        raise SystemExit("No clips indexed — check the --root paths.")

    # ---- optional: recompute fractions for the sites that never got tags ----
    ann_by_site = {}
    for site, d in (args.annotations or []):
        ann_by_site.setdefault(site, []).append(d)
    backfill_reports = []
    for site, dirs in ann_by_site.items():
        if site not in per_site_buckets:
            print(f"[WARN] --annotations names site {site!r}, which has no --root — ignored")
            continue
        # Verify against this site's own tagged rows when it has any; otherwise
        # against a tagged vintage named by --verify-root. A site cut entirely
        # without tags has nothing internal to check itself against.
        verify_rows = [r for r in all_rows if r["site"] == site and int(r["tagged"])
                       and int(r["bucket"]) in TAG_BEARING_BUCKETS]
        for vsite, vpath in (args.verify_root or []):
            if vsite != site:
                continue
            if not vpath.exists():
                print(f"[WARN] --verify-root {vsite}={vpath} does not exist — ignored")
                continue
            vrows, *_ = scan_root(site, vpath, spec)
            verify_rows += [r for r in vrows if int(r["tagged"])]
            print(f"[INFO] {site}: verifying against {len(vrows):,} clips from {vpath}")
        backfill_reports.append(backfill_fractions(
            all_rows, site, dirs, spec, verify_rows,
            args.min_agreement, args.allow_unverified_backfill, args.segment_ms))

    fieldnames = (["video_path", "case_id", "site", "bucket", "clip_dir", "tagged"]
                  + spec.frac_columns())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n[DONE] {len(all_rows)} clips -> {args.out}")
    print("Per site:", per_site_n)
    if backfill_reports:
        report_backfill(backfill_reports, spec)
    report_tag_coverage(all_rows, list(per_site_buckets), spec)
    report_directory_recovery(dir_census, spec)
    if structural:
        print("\n[WARN] clips whose directory disagrees with their bucket — the "
              "path-based\n       recovery cannot be trusted for these:")
        for line in structural:
            print(f"       {line}")
    report(all_rows, per_site_buckets, spec)


if __name__ == "__main__":
    main()
