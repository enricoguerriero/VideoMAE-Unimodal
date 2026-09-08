#!/usr/bin/env python3
"""
suction_devices.py — per-DEVICE breakdown of the suction annotations.

Answers one question: would `suction_penguin` and `suction_bulb` be two viable
classes, or is one of them a handful of intervals from a single case?

Clip counts and interval counts do not decide that — CASE counts do. A class
carried by two cases cannot be evaluated cross-site no matter how many clips it
has, because every clip in it shares one baby, one camera position and one
operator (this is `explore_data`'s [NARROW] warning, one level earlier).

READ-ONLY, stdlib only. Usage:

    python scripts/suction_devices.py \
        --site Haydom=/spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations \
        --site DRC=/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing/Unprocessed_data/anot_files
"""
import argparse
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "src" / "data" / "annotations.py"
_s = importlib.util.spec_from_file_location("_ann", _p)
ann = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ann)

#: the suction strings, grouped by the DEVICE they name rather than by category.
#: Everything here maps to "Suction" today; the point is to see it split.
DEVICES = {
    "penguin": ["Suction using penguin device", "Suction using penguine device",
                "Suction using Penguine Device", "Suction using Penguine device",
                "Suction using penguine devece", "Sunction using penguine device"],
    "bulb": ["Suction using bulb device"],
    "tube (currently IGNORED)": ["Suction using tube"],
}
BY_NORM = {ann.normalize_event(v): d for d, vs in DEVICES.items() for v in vs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", action="append", required=True, metavar="NAME=DIR")
    a = p.parse_args()

    print(f"{'site':<9}{'device':<26}{'cases':>7}{'intervals':>11}"
          f"{'total_s':>10}{'median_ms':>11}{'>=0.75s':>9}")
    for spec in a.site:
        name, d = spec.split("=", 1)
        idx = ann.AnnotationIndex.from_roots([Path(d)])
        per = defaultdict(list)
        cases = defaultdict(set)
        for entry in idx.entries:
            for f in sorted(set(entry["index"].values())):
                got, err = ann.read_annotation(f)
                if err:
                    continue
                rows = got[0]
                kind = ann.annotation_kind(rows)
                for event, s, e, original in rows:
                    # a cleaned file keeps the device name in column 5
                    text = original if (kind == "cleaned" and original) else event
                    dev = BY_NORM.get(ann.normalize_event(text))
                    if dev and e > s:
                        per[dev].append(e - s)
                        cases[dev].add(f.stem)
        for dev in ("penguin", "bulb", "tube (currently IGNORED)"):
            v = sorted(per.get(dev, []))
            if not v:
                print(f"{name:<9}{dev:<26}{0:>7}{0:>11}{0:>10}{'-':>11}{'-':>9}")
                continue
            # >= 750 ms is the only span that can clear suction_threshold 0.25
            # of a 3 s window, i.e. produce a positive clip at all.
            usable = sum(1 for x in v if x >= 750)
            print(f"{name:<9}{dev:<26}{len(cases[dev]):>7}{len(v):>11,}"
                  f"{sum(v)/1000:>10,.0f}{v[len(v)//2]:>11,}{usable:>9,}")
    print("\n  cases is the number that matters: a device carried by <10 cases cannot")
    print("  support a cross-site claim, however many clips it yields.")


if __name__ == "__main__":
    main()
