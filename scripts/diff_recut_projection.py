#!/usr/bin/env python3
"""
diff_recut_projection.py — what the re-cut changes, case by case, BEFORE cutting.

recut_site.py's dry run prints one corpus-wide bucket census. That cannot
separate the two things a re-cut does at once:

    NEW      episodes the current tree does not have at all
    CHANGED  episodes it DOES have, whose labels the new annotation vintage
             moves between buckets

Only the second confounds a before/after comparison of model results, so it is
the one worth seeing first. Feed it the per-case projection:

    VIDEOS=/spo/LS-Haydom/ProcessedData/Ronald/data/Tanzania/videos_corrected \
        bash scripts/recut_haydom.sh --dump-projection results/projection.csv

    python scripts/diff_recut_projection.py \
        --projection results/projection.csv \
        --manifest   data/clips_all.csv \
        --site       Haydom \
        --out        results/recut_diff.txt

READ-ONLY, stdlib only. --top N sets how many changed cases are listed (default 25).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

BUCKETS = list(range(9))
NAMES = {0: "non_target", 1: "stimulation", 2: "ventilation", 3: "suction",
         4: "no_overlap", 5: "no_label", 6: "partial", 7: "target_overlap",
         8: "partial_overlap"}
# what a suction-positive label can come out of
SUCTION_BUCKETS = (3, 7)


def load_projection(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[str(r["case_id"]).strip()] = Counter(
                {b: int(r.get(f"bucket_{b}", 0) or 0) for b in BUCKETS})
    return out


def load_manifest(path, site):
    out = defaultdict(Counter)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if site and (r.get("site") or "").strip().lower() != site.lower():
                continue
            out[str(r.get("case_id", "")).strip()][int(r["bucket"])] += 1
    return dict(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projection", default="results/projection.csv")
    p.add_argument("--manifest", default="data/clips_all.csv")
    p.add_argument("--site", default="Haydom")
    p.add_argument("--out", default="results/recut_diff.txt")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    for f in (a.projection, a.manifest):
        if not os.path.exists(f):
            raise SystemExit(f"missing: {f}")

    new = load_projection(a.projection)
    old = load_manifest(a.manifest, a.site)
    nk, ok_ = set(new), set(old)
    shared, added, lost = nk & ok_, nk - ok_, ok_ - nk

    out("=" * 76)
    out(f"RE-CUT DIFF ({a.site}) — projection vs the tree you have today")
    out("=" * 76)
    out(f"  cases now            {len(ok_):>6}")
    out(f"  cases after          {len(nk):>6}")
    out(f"  NEW (added)          {len(added):>6}")
    out(f"  SHARED (may change)  {len(shared):>6}")
    out(f"  LOST (disappear)     {len(lost):>6}"
        + ("   <- check these" if lost else ""))
    if lost:
        out(f"    {', '.join(sorted(lost)[:20])}{' ...' if len(lost) > 20 else ''}")

    def tot(d, ks, bs=BUCKETS):
        return sum(d[k].get(b, 0) for k in ks for b in bs)

    out("\n  clips, split by where they come from")
    out(f"    {'':<22}{'now':>10}{'after':>10}{'change':>10}")
    out(f"    {'shared cases':<22}{tot(old, shared):>10,}{tot(new, shared):>10,}"
        f"{tot(new, shared) - tot(old, shared):>+10,}")
    out(f"    {'new cases':<22}{0:>10,}{tot(new, added):>10,}{tot(new, added):>+10,}")
    if lost:
        out(f"    {'lost cases':<22}{tot(old, lost):>10,}{0:>10,}{-tot(old, lost):>+10,}")

    out("\n  SUCTION-bearing clips (buckets 3 + 7)")
    out(f"    {'shared cases':<22}{tot(old, shared, SUCTION_BUCKETS):>10,}"
        f"{tot(new, shared, SUCTION_BUCKETS):>10,}"
        f"{tot(new, shared, SUCTION_BUCKETS) - tot(old, shared, SUCTION_BUCKETS):>+10,}")
    out(f"    {'new cases':<22}{0:>10,}{tot(new, added, SUCTION_BUCKETS):>10,}"
        f"{tot(new, added, SUCTION_BUCKETS):>+10,}")
    out("    ^ if most of the gain is on NEW cases, your old results stay")
    out("      broadly interpretable; if SHARED cases move a lot, the labels")
    out("      themselves changed and before/after is a two-variable comparison.")

    out("\n  per-bucket, SHARED cases only (isolates relabelling from new data)")
    out(f"    {'bucket':<20}{'now':>10}{'after':>10}{'change':>10}")
    for b in BUCKETS:
        o = sum(old[k].get(b, 0) for k in shared)
        n = sum(new[k].get(b, 0) for k in shared)
        out(f"    {str(b) + ' ' + NAMES[b]:<20}{o:>10,}{n:>10,}{n - o:>+10,}")

    out(f"\n  most-changed shared cases (by |total clip delta|), top {a.top}")
    deltas = sorted(((sum(new[k].values()) - sum(old[k].values()), k) for k in shared),
                    key=lambda t: -abs(t[0]))
    out(f"    {'case':<12}{'now':>8}{'after':>8}{'delta':>8}   suction now/after")
    for d, k in deltas[:a.top]:
        so = sum(old[k].get(b, 0) for b in SUCTION_BUCKETS)
        sn = sum(new[k].get(b, 0) for b in SUCTION_BUCKETS)
        out(f"    {k:<12}{sum(old[k].values()):>8,}{sum(new[k].values()):>8,}"
            f"{d:>+8,}   {so:>5,} / {sn:<5,}")

    gained = [k for k in shared
              if sum(new[k].get(b, 0) for b in SUCTION_BUCKETS) >
              sum(old[k].get(b, 0) for b in SUCTION_BUCKETS)]
    zeroed = [k for k in shared
              if sum(old[k].get(b, 0) for b in SUCTION_BUCKETS) > 0
              and sum(new[k].get(b, 0) for b in SUCTION_BUCKETS) == 0]
    out(f"\n  shared cases that GAIN suction : {len(gained):>4}")
    out(f"  shared cases that LOSE all suction: {len(zeroed):>4}"
        + (f"   {', '.join(sorted(zeroed)[:12])}" if zeroed else ""))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
