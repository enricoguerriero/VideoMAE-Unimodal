#!/usr/bin/env python3
"""
compare_ronald_data.py — put Ronald Paleczny's Haydom dataset and ours side by side.

Ronald's thesis trains the SAME backbones (videomae-base-finetuned-ssv2,
VideoMAEv2-giant) on the SAME hospital, so any gap in the reported numbers is a
DATA difference. This script quantifies the three that the code says should
matter most, and that only the CSVs can settle:

  1. how many clips / videos each split actually holds,
  2. what the LABEL DISTRIBUTION of each split is — in particular whether the
     TEST split carries the natural activity prevalence or a rebalanced one
     (Ronald's build_downsample_manifest.py caps largest_pool <= 20 x smallest
     and takes its drops from whichever split is over-represented, so his
     test.csv is thinned too; test_full.csv is the un-thinned reference),
  3. how much of the natural stream that thinning removed (test.csv vs
     test_full.csv), which is the factor his headline metrics are taken over.

READ-ONLY. Standard library only, so it runs in a bare environment; if the repo
is importable it additionally resolves OUR labels through the real DataSpec, so
the two sides are compared under the labelling rule actually used in training.

Usage:
    python scripts/compare_ronald_data.py \
        --ronald-dir /spo/LS-Haydom/ProcessedData/Ronald/code/Master-project/data \
        --ours-dir   data \
        --data-config configs/data_multilabel_thesis.yaml \
        --out        results/ronald_vs_ours.txt

Every argument has a default; with no arguments it uses the paths above.
Missing files are reported, never fatal — run it with whatever is mounted.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

ACTIVITIES = ("ventilation", "stimulation", "suction")
# {video_number}_Video_clip_{clip_number}_{VSS}.mp4  (write_csv.py's convention)
CLIP_RE = re.compile(r"^(\d+)_Video_clip_(\d+)_([01]{3})")


def read_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _int(row, key):
    v = (row.get(key) or "").strip()
    try:
        return int(float(v))
    except ValueError:
        return 0


def ronald_stats(rows):
    """Clips, videos, per-activity positives and the full 3-bit combo histogram."""
    combos, videos = Counter(), set()
    pos = Counter()
    for r in rows:
        bits = tuple(_int(r, a) for a in ACTIVITIES)
        combos["".join(str(b) for b in bits)] += 1
        for a, b in zip(ACTIVITIES, bits):
            if b:
                pos[a] += 1
        stem = os.path.basename(r.get("video_path", ""))
        m = CLIP_RE.match(stem)
        videos.add(m.group(1) if m else stem.split("_")[0])
    return {"n": len(rows), "videos": len(videos), "pos": pos, "combos": combos}


def fmt_block(name, st, out):
    n = max(st["n"], 1)
    out(f"  {name:<14} {st['n']:>8,} clips   {st['videos']:>4} videos")
    for a in ACTIVITIES:
        p = st["pos"][a]
        out(f"      {a:<13} {p:>8,} positives  ({100 * p / n:6.2f} %)")
    nz = st["combos"]
    multi = sum(v for k, v in nz.items() if k.count("1") >= 2)
    out(f"      {'all-zero':<13} {nz.get('000', 0):>8,}          "
        f"({100 * nz.get('000', 0) / n:6.2f} %)")
    out(f"      {'co-occurring':<13} {multi:>8,}          ({100 * multi / n:6.2f} %)")
    out(f"      combos: " + "  ".join(f"{k}={v:,}" for k, v in sorted(nz.items())))


def ours_stats(rows, spec):
    """Our manifests store EVIDENCE, not labels — resolve them the way training does."""
    pos, videos, combos = Counter(), set(), Counter()
    masked = Counter()
    dropped = 0
    for r in rows:
        bucket = _int(r, "bucket")
        fracs = {a: float(r.get(f"frac_{a}") or 0.0) for a in spec.activities}
        label = spec.resolve(
            bucket, fracs,
            tagged=bool(_int(r, "tagged")) if r.get("tagged") not in (None, "") else True,
            dir_activities=spec.activities_from_path(r.get("clip_dir", "")),
        )
        videos.add(r.get("case_id", ""))
        if label is None:
            dropped += 1
            continue
        bits = []
        for i, a in enumerate(spec.activities):
            if label.mask[i] == 0.0:
                masked[a] += 1
                bits.append("-")
            else:
                if label.targets[i] == 1.0:
                    pos[a] += 1
                bits.append(str(int(label.targets[i])))
        combos["".join(bits)] += 1
    kept = sum(combos.values())
    return {"n": kept, "dropped": dropped, "videos": len(videos),
            "pos": pos, "combos": combos, "masked": masked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ronald-dir",
                   default="/spo/LS-Haydom/ProcessedData/Ronald/code/Master-project/data")
    p.add_argument("--ours-dir", default="data")
    p.add_argument("--data-config", default="configs/data_multilabel_thesis.yaml")
    p.add_argument("--out", default="results/ronald_vs_ours.txt")
    a = p.parse_args()

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 78)
    out("RONALD'S DATA vs OURS — same hospital, same backbone, different clips")
    out("=" * 78)

    # ---------------------------------------------------------------- Ronald
    out("\n[1] RONALD  " + a.ronald_dir)
    found = {}
    for name in ("train", "validation", "test", "test_full"):
        rows = read_csv(os.path.join(a.ronald_dir, f"{name}.csv"))
        if rows is None:
            out(f"  {name + '.csv':<18} MISSING")
            continue
        found[name] = ronald_stats(rows)
        fmt_block(name + ".csv", found[name], out)

    if "test" in found and "test_full" in found:
        t, tf = found["test"], found["test_full"]
        out("\n  >>> DOWNSAMPLING APPLIED TO THE TEST SPLIT <<<")
        out(f"      test_full.csv {tf['n']:>8,} clips  (natural stream)")
        out(f"      test.csv      {t['n']:>8,} clips  (what the thesis scores)")
        if t["n"]:
            out(f"      kept {100 * t['n'] / max(tf['n'], 1):.1f} % of the stream "
                f"({tf['n'] / max(t['n'], 1):.2f}x thinned)")
        for act in ACTIVITIES:
            rt = 100 * t["pos"][act] / max(t["n"], 1)
            rf = 100 * tf["pos"][act] / max(tf["n"], 1)
            out(f"      {act:<13} prevalence {rf:6.2f} % -> {rt:6.2f} %  "
                f"({'inflated' if rt > rf else 'reduced'} {rt / max(rf, 1e-9):.2f}x)")

    # ---------------------------------------------------------------- ours
    out("\n[2] OURS  " + a.ours_dir)
    spec = None
    try:
        sys.path.insert(0, os.getcwd())
        from src.data import DataSpec
        spec = DataSpec.load(a.data_config)
        out(f"  labels resolved through {a.data_config}  (task={spec.task}, "
            f"ambiguous={spec.ambiguous}, kept buckets={spec.kept_buckets()})")
    except Exception as e:  # noqa: BLE001 — a bare env still gets the counts below
        out(f"  [WARN] could not load DataSpec ({e}); reporting buckets only")

    for name in ("train", "validation", "test_haydom", "test_drc", "test"):
        rows = read_csv(os.path.join(a.ours_dir, f"{name}.csv"))
        if rows is None:
            out(f"  {name + '.csv':<18} MISSING")
            continue
        if spec is None:
            b = Counter(_int(r, "bucket") for r in rows)
            out(f"  {name + '.csv':<18} {len(rows):>8,} clips   buckets " +
                " ".join(f"{k}:{v:,}" for k, v in sorted(b.items())))
            continue
        st = ours_stats(rows, spec)
        out(f"  {name + '.csv':<14} {st['n']:>8,} clips kept "
            f"({st['dropped']:,} dropped by the spec)   {st['videos']:>4} cases")
        for act in spec.activities:
            pos = st["pos"][act]
            out(f"      {act:<13} {pos:>8,} positives  "
                f"({100 * pos / max(st['n'], 1):6.2f} %)"
                + (f"   {st['masked'][act]:,} masked" if st["masked"][act] else ""))
        multi = sum(v for k, v in st["combos"].items() if k.count("1") >= 2)
        out(f"      {'co-occurring':<13} {multi:>8,}          "
            f"({100 * multi / max(st['n'], 1):6.2f} %)")

    out("\n" + "=" * 78)
    out("Read the TEST rows against each other: a prevalence that differs between")
    out("his test.csv and ours is a different QUESTION being scored, not a")
    out("different model. See build_downsample_manifest.py (--multiplier 20).")
    out("=" * 78)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
