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
Every clip filename encodes both its original label bucket and the per-activity
share of the 3 s window that activity covered:

    {case}_interval_{n}_start_{ms}_end_{ms}[_stim0.67][_vent0.55]_{bucket}.mp4

Both are copied verbatim into the manifest:

    video_path, case_id, site, bucket, frac_<activity>...

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
    """Yield one row per clip under `root`, plus counters for the report."""
    rows, buckets, unparsed = [], Counter(), []
    for mp4 in sorted(root.rglob("*.mp4")):
        bucket, fracs = spec.parse_stem(mp4.stem)
        if bucket is None:
            if len(unparsed) < 5:
                unparsed.append(str(mp4))
            continue
        buckets[bucket] += 1
        row = {
            "video_path": str(mp4.resolve()),
            "case_id": spec.case_id_from_stem(mp4.stem),
            "site": site,
            "bucket": bucket,
        }
        for a in spec.activities:
            row[f"frac_{a}"] = f"{fracs.get(a, 0.0):.2f}"
        rows.append(row)
    return rows, buckets, unparsed


def parse_root(spec_str: str):
    if "=" not in spec_str:
        raise argparse.ArgumentTypeError(f"--root must be SITE=PATH, got: {spec_str}")
    site, path = spec_str.split("=", 1)
    return site.strip(), Path(path).expanduser()


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
    for r in rows:
        fracs = {a: float(r[f"frac_{a}"]) for a in spec.activities}
        label = spec.resolve(int(r["bucket"]), fracs)
        if label is None:
            dropped += 1
        elif spec.is_multilabel:
            active = tuple(a for a, t, m in zip(spec.activities, label.targets, label.mask)
                           if t == 1.0 and m == 1.0)
            kept["+".join(active) or f"<{spec.negative_class}>"] += 1
        else:
            kept[spec.class_names[label.class_index]] += 1
    for name, n in sorted(kept.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<34} {n:>10,}")
    n_kept = sum(kept.values())
    print(f"  {'(dropped by this config)':<34} {dropped:>10,}")
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
    for site, root in args.root:
        if not root.exists():
            print(f"[WARN] root does not exist: {root} (site={site}) — skipping")
            continue
        rows, buckets, unparsed = scan_root(site, root, spec)
        per_site_buckets[site] = buckets
        per_site_n[site] = len(rows)
        all_rows.extend(rows)
        print(f"[INFO] {site}: {len(rows)} clips from {root}")
        for u in unparsed:
            print(f"       [WARN] unparseable filename, skipped: {u}")

    if not all_rows:
        raise SystemExit("No clips indexed — check the --root paths.")

    fieldnames = ["video_path", "case_id", "site", "bucket"] + spec.frac_columns()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n[DONE] {len(all_rows)} clips -> {args.out}")
    print("Per site:", per_site_n)
    report(all_rows, per_site_buckets, spec)


if __name__ == "__main__":
    main()
