#!/usr/bin/env python3
"""
check_device_fractions.py — did the per-DEVICE suction backfill actually work?

READ-ONLY, stdlib only. Reads data/clips_all.csv (or a split CSV) and nothing
else, so it runs on any machine that has the manifest.

WHY THIS EXISTS
---------------
configs/data_suction3.yaml splits suction into penguin / bulb / tube. No filename
ever carried a per-device tag, so every one of those fractions comes from
src/data/annotations.py recomputing it from the annotation files
(`build_manifest.py --rebackfill-all`).

That recomputation can only see the device when the annotation row still NAMES
it. Two file vintages cannot:

  * a CLEANED file with only 4 columns — column 1 already holds the processor's
    category ("Suction") and there is no original-event column to read the
    device from;
  * a CLEANED 5-column file whose device string is not one of the exact
    spellings in annotations._EVENT_VARIANTS.

In both cases `intervals_by_category` returns "Suction" and NO "Suction
penguin", so `FractionSource.fractions` reports 0.00 for all three devices — a
plain dict, not None. `backfill_fractions` therefore writes those zeros AND sets
`tagged=1`, and DataSpec reads 0.00 <= weak_threshold as a CONFIDENT NEGATIVE. A
real penguin window becomes a supervised "no suction of any kind" example.

build_manifest's verification gate cannot catch this: it deliberately checks
only the processor's own `_stim/_vent/_suct` tags (LEGACY_TAGS), and the
aggregate `_suct` value reproduces perfectly on exactly the rows whose device
was lost.

WHAT IT PRINTS
--------------
The invariant that must hold, per site: legacy `frac_suction` (= penguin + bulb,
the value the filename tags WERE verified against) is non-zero only where
penguin or bulb is non-zero. Every row that breaks it is a suction window
relabelled as a confident negative for every device.

Usage:
    python scripts/check_device_fractions.py                       # data/clips_all.csv
    python scripts/check_device_fractions.py --manifest data/train.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

DEVICES = ["frac_suction_penguin", "frac_suction_bulb", "frac_suction_tube"]
LEGACY = "frac_suction"
BUCKET_NAMES = {0: "non_target", 1: "stimulation", 2: "ventilation", 3: "suction",
                4: "no_overlap", 5: "no_label", 6: "partial", 7: "target_overlap",
                8: "partial_overlap"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/clips_all.csv"))
    p.add_argument("--threshold", type=float, default=0.25,
                   help="thresholds.suction_* from the data config (default 0.25).")
    p.add_argument("--weak", type=float, default=0.20,
                   help="weak_threshold: at or below this a fraction is read as a "
                        "confident NEGATIVE (default 0.20).")
    p.add_argument("--cases", type=int, default=12, help="case ids to list per site")
    a = p.parse_args()

    with a.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in DEVICES + [LEGACY] if c not in cols]
        if missing:
            raise SystemExit(
                f"{a.manifest} has no {missing}. Rebuild it with the per-device config "
                f"and the legacy aggregate alongside it:\n"
                f"    bash scripts/build_data.sh configs/data_suction3.yaml\n"
                f"(build_data.sh passes --rebackfill-all --extra-frac suction, which is "
                f"what puts both in one manifest.)")
        rows = list(reader)

    sites = sorted({r["site"] for r in rows})
    n_clips = Counter(r["site"] for r in rows)
    n_suct = Counter()
    orphan_clips, orphan_harmful, orphan_strong = Counter(), Counter(), Counter()
    orphan_cases = defaultdict(set)
    orphan_buckets = Counter()
    pos_clips, pos_cases = Counter(), defaultdict(set)

    for r in rows:
        site, case = r["site"], r["case_id"]
        peng, bulb, tube = (float(r[c]) for c in DEVICES)
        legacy = float(r[LEGACY])
        # penguin + bulb ARE legacy "Suction"; tube was "Ignored label" to the
        # processor, so it is deliberately not part of this invariant.
        named = peng > 0 or bulb > 0
        if legacy > 0:
            n_suct[site] += 1
            if not named:
                orphan_clips[site] += 1
                orphan_cases[site].add(case)
                orphan_buckets[int(r["bucket"])] += 1
                # The rounding caveat: fractions are stored to 2 decimals, so a
                # sub-10 ms sliver can round the union to 0.01 and each device to
                # 0.00. Those rows are noise. `strong` counts the ones that WOULD
                # have been a positive clip, which is the number that matters.
                if legacy >= a.threshold:
                    orphan_strong[site] += 1
                # Only HARMFUL where the row is still marked tagged: an untagged
                # row falls back to bucket+directory, which drops it instead of
                # asserting a negative.
                if int(r.get("tagged", 1)):
                    orphan_harmful[site] += 1
        for c, v in zip(DEVICES, (peng, bulb, tube)):
            if v >= a.threshold:
                pos_clips[(c, site)] += 1
                pos_cases[(c, site)].add(case)

    print(f"manifest: {a.manifest}  ({len(rows):,} clips)\n")
    print("A. SUCTION WINDOWS WHOSE DEVICE WAS LOST")
    print("   frac_suction > 0 but neither penguin nor bulb claims it, i.e. the")
    print("   backfill wrote 0.00 for every device on a real suction window.\n")
    print(f"   {'site':<10}{'suction clips':>15}{'device lost':>13}{'share':>8}"
          f"{'cases hit':>11}{'lost positives':>16}{'-> negative':>13}")
    for s in sites:
        share = f"{100 * orphan_clips[s] / n_suct[s]:.1f}%" if n_suct[s] else "-"
        print(f"   {s:<10}{n_suct[s]:>15,}{orphan_clips[s]:>13,}{share:>8}"
              f"{len(orphan_cases[s]):>11}{orphan_strong[s]:>16,}{orphan_harmful[s]:>13,}")
    print(f"\n   'lost positives' = frac_suction >= {a.threshold:.2f}, i.e. clips that WOULD "
          f"have been a\n   positive for some device. The rest can be 2-decimal rounding "
          f"noise.")

    if orphan_buckets:
        print("\n   by bucket (where the lost windows sit):")
        for b in sorted(orphan_buckets):
            print(f"     {b} {BUCKET_NAMES.get(b, '?'):<18}{orphan_buckets[b]:>10,}")

    for s in sites:
        cases = sorted(orphan_cases[s])
        if not cases:
            continue
        print(f"\n   {s} cases affected ({len(cases)}): {cases[:a.cases]}"
              f"{' ...' if len(cases) > a.cases else ''}")
        print(f"   Which file vintage did one of them resolve to?")
        print(f"     python -c \"from src.data.annotations import *; "
              f"i=AnnotationIndex.from_roots(['<{s} annotation dir>',...]); "
              f"f=i.lookup({cases[0]!r}); print(f); "
              f"r,_=read_annotation(f); print(annotation_kind(r[0]), r[0][:3])\"")

    print("\nB. WHAT EACH SITE ACTUALLY SUPERVISES PER DEVICE")
    print(f"   positive = frac >= {a.threshold:.2f}; a frac of 0.00 is a confident "
          f"negative (<= {a.weak:.2f})\n")
    print(f"   {'device':<22}{'site':<10}{'positive':>11}{'clips':>11}{'pos rate':>10}"
          f"{'cases':>8}{'pos_weight':>12}{'bal. cut':>10}")
    for c in DEVICES:
        for s in sites:
            pos, tot = pos_clips[(c, s)], n_clips[s]
            rate = f"{100 * pos / tot:.2f}%" if tot else "-"
            if pos:
                pw = ((tot - pos) / pos) ** 0.5
                pw_s, cut = f"{pw:.1f}", f"{pw / (1 + pw):.2f}"
            else:
                pw_s, cut = "-", "-"
            print(f"   {c.replace('frac_', ''):<22}{s:<10}{pos:>11,}{tot:>11,}"
                  f"{rate:>10}{len(pos_cases[(c, s)]):>8}{pw_s:>12}{cut:>10}")

    print("\n   pos_weight is the sqrt_inv_freq value training derives from this, and")
    print("   `bal. cut` = pos_weight/(1+pos_weight) is where the balanced sigmoid")
    print("   threshold sits — NOT the 0.5 in data_suction3.yaml. A site with a lower")
    print("   positive rate is punished harder by an untuned cut, so compare `ap`, not")
    print("   `f1`, until src/tune_thresholds.py has been run on validation.")

    total_orphan = sum(orphan_clips.values())
    if total_orphan == 0:
        print("\nVERDICT: the device split is clean — every suction window names a device.")
        print("         A site gap is then a data-geometry / threshold question, not this.")
    else:
        print(f"\nVERDICT: {total_orphan:,} suction windows carry no device "
              f"({sum(orphan_strong.values()):,} of them above")
        print(f"         threshold), and {sum(orphan_harmful.values()):,} are supervised as "
              f"'no suction of any kind'.")
        print("         Fix the read (or drop those rows) before reading any per-device score.")


if __name__ == "__main__":
    main()
