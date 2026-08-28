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

Each --root is SITE=PATH; PATH is a `videos/` directory containing the per-class
subfolders (or any tree of *.mp4 clips following the naming convention).
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from .spec import BUCKET_NAMES, DataSpec


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

    fieldnames = (["video_path", "case_id", "site", "bucket", "clip_dir", "tagged"]
                  + spec.frac_columns())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n[DONE] {len(all_rows)} clips -> {args.out}")
    print("Per site:", per_site_n)
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
