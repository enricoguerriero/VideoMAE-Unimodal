#!/usr/bin/env python3
"""
compare_ronald_episodes.py — do Ronald's episodes and ours overlap, and do the
annotations on the shared ones agree?

compare_ronald_data.py answered "how much data". This answers the two questions
that decide whether his numbers and ours are about the same recordings at all:

  1. EPISODE OVERLAP. His data_preprocessing.py renames each video by the FIRST
     continuous digit run in its stem; our build_manifest.py keys on the exact
     stem or any run of >= 5 digits. Those rules agree only when the case number
     is the first number in the filename, so the two corpora can disagree both
     on WHICH episodes exist and on what a given episode is CALLED.

  2. ANNOTATION AGREEMENT on the shared episodes. Clip-level comparison is
     impossible — he cuts at a 1500 ms stride with baby-visible gating, we
     reuse a 1000 ms-stride tree with a bucket policy — so this compares
     per-episode PREVALENCE (positive clips / clips in that episode), which is
     roughly stride-invariant. Exact agreement is not expected; what matters is
     GROSS disagreement, above all an episode where one side has an activity
     and the other has none of it.

Only Haydom rows of ours are considered (he never had DRC).

READ-ONLY. Usage:

    python scripts/compare_ronald_episodes.py \
        --ronald-dir /spo/LS-Haydom/ProcessedData/Ronald/code/Master-project/data \
        --ours-dir   data \
        --data-config configs/data_multilabel_thesis.yaml \
        --out        results/ronald_vs_ours_episodes.txt

Add --max-rows N to lengthen the per-episode table (default 40, 0 = all).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

ACTS = ("ventilation", "stimulation", "suction")
CLIP_RE = re.compile(r"^(\d+)_Video_clip_(\d+)_([01]{3})")


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key(raw):
    """Normalise an episode id so '0042' and '42' are the same episode."""
    s = str(raw).strip()
    return str(int(s)) if s.isdigit() else s.lower()


def _int(row, k):
    try:
        return int(float((row.get(k) or "").strip()))
    except ValueError:
        return 0


def load_ronald(d):
    """{episode: {split, n, vent, stim, suct}} — test taken from test_full.csv
    (the un-thinned stream) so prevalence is the natural one."""
    eps = defaultdict(lambda: {"split": set(), "n": 0, **{a: 0 for a in ACTS}})
    files = [("train", "train.csv"), ("val", "validation.csv"),
             ("test", "test_full.csv" if os.path.exists(os.path.join(d, "test_full.csv"))
              else "test.csv")]
    missing = []
    for split, fn in files:
        rows = read_csv(os.path.join(d, fn))
        if rows is None:
            missing.append(fn)
            continue
        for r in rows:
            stem = os.path.basename(r.get("video_path", ""))
            m = CLIP_RE.match(stem)
            e = key(m.group(1) if m else stem.split("_")[0])
            rec = eps[e]
            rec["split"].add(split)
            rec["n"] += 1
            for a in ACTS:
                rec[a] += _int(r, a)
    return eps, missing


def load_ours(d, spec):
    eps = defaultdict(lambda: {"split": set(), "n": 0, **{a: 0 for a in ACTS}})
    missing = []
    for split, fn in [("train", "train.csv"), ("val", "validation.csv"),
                      ("test", "test_haydom.csv")]:
        rows = read_csv(os.path.join(d, fn))
        if rows is None:
            missing.append(fn)
            continue
        for r in rows:
            if (r.get("site") or "").strip().lower() != "haydom":
                continue
            fracs = {a: float(r.get(f"frac_{a}") or 0.0) for a in spec.activities}
            tag = r.get("tagged")
            label = spec.resolve(
                _int(r, "bucket"), fracs,
                tagged=bool(_int(r, "tagged")) if tag not in (None, "") else True,
                dir_activities=spec.activities_from_path(r.get("clip_dir", "")))
            e = key(r.get("case_id", ""))
            rec = eps[e]
            rec["split"].add(split)
            if label is None:
                continue
            rec["n"] += 1
            for i, a in enumerate(spec.activities):
                if a in ACTS and label.mask[i] == 1.0 and label.targets[i] == 1.0:
                    rec[a] += 1
    return eps, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ronald-dir",
                   default="/spo/LS-Haydom/ProcessedData/Ronald/code/Master-project/data")
    p.add_argument("--ours-dir", default="data")
    p.add_argument("--data-config", default="configs/data_multilabel_thesis.yaml")
    p.add_argument("--out", default="results/ronald_vs_ours_episodes.txt")
    p.add_argument("--max-rows", type=int, default=40)
    a = p.parse_args()

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    sys.path.insert(0, os.getcwd())
    from src.data import DataSpec
    spec = DataSpec.load(a.data_config)

    R, rmiss = load_ronald(a.ronald_dir)
    O, omiss = load_ours(a.ours_dir, spec)
    for m in rmiss:
        out(f"[WARN] missing {a.ronald_dir}/{m}")
    for m in omiss:
        out(f"[WARN] missing {a.ours_dir}/{m}")

    rk, ok = set(R), set(O)
    both, only_r, only_o = rk & ok, rk - ok, ok - rk

    out("=" * 78)
    out("EPISODE OVERLAP — Ronald's videos vs our Haydom cases")
    out("=" * 78)
    out(f"  Ronald episodes      {len(rk):>5}")
    out(f"  Our Haydom cases     {len(ok):>5}")
    out(f"  SHARED               {len(both):>5}"
        f"   ({100 * len(both) / max(len(ok), 1):.1f} % of ours,"
        f" {100 * len(both) / max(len(rk), 1):.1f} % of his)")
    out(f"  only Ronald          {len(only_r):>5}")
    out(f"  only ours            {len(only_o):>5}")
    if only_r:
        out(f"    his-only ids  : {', '.join(sorted(only_r)[:15])}"
            f"{' ...' if len(only_r) > 15 else ''}")
    if only_o:
        out(f"    our-only ids  : {', '.join(sorted(only_o)[:15])}"
            f"{' ...' if len(only_o) > 15 else ''}")

    # id SHAPE — a systematic difference shows up as differing digit lengths
    out("\n  id length histogram (digits)")
    for nm, ks in (("Ronald", rk), ("ours", ok)):
        h = Counter(len(k) for k in ks if k.isdigit())
        out(f"    {nm:<8} " + "  ".join(f"{d}:{c}" for d, c in sorted(h.items())))

    if not both:
        out("\n  NO SHARED IDS — the two naming rules disagree (his = first digit")
        out("  run, ours = any run of >= 5 digits). Compare the id histograms")
        out("  above before concluding the episodes themselves differ.")
    else:
        out("\n" + "=" * 78)
        out("SPLIT AGREEMENT on shared episodes")
        out("=" * 78)
        x = Counter()
        for e in both:
            x[(",".join(sorted(R[e]["split"])), ",".join(sorted(O[e]["split"])))] += 1
        out(f"  {"his split":<18}{"our split":<18}{"episodes":>9}")
        for (hs, os_), c in sorted(x.items(), key=lambda kv: -kv[1]):
            flag = "" if hs == os_ else "   <- differs"
            out(f"  {hs:<18}{os_:<18}{c:>9}{flag}")

        out("\n" + "=" * 78)
        out("ANNOTATION AGREEMENT on shared episodes (prevalence = pos/clips)")
        out("=" * 78)
        dis = {a: {"his_only": [], "our_only": []} for a in ACTS}
        rows = []
        for e in sorted(both, key=lambda k: int(k) if k.isdigit() else 0):
            r, o = R[e], O[e]
            rows.append((e, r, o))
            for act in ACTS:
                hp, op = r[act], o[act]
                if hp > 0 and op == 0:
                    dis[act]["his_only"].append(e)
                elif op > 0 and hp == 0:
                    dis[act]["our_only"].append(e)

        out(f"  {'episode':<11}{'his n':>7}{'our n':>7}   " +
            "".join(f"{a[:4]+' his/our':>18}" for a in ACTS))
        shown = rows if a.max_rows == 0 else rows[:a.max_rows]
        for e, r, o in shown:
            cells = ""
            for act in ACTS:
                hp = 100 * r[act] / max(r["n"], 1)
                op = 100 * o[act] / max(o["n"], 1)
                cells += f"{hp:7.1f}%/{op:6.1f}%"
            out(f"  {e:<11}{r['n']:>7,}{o['n']:>7,}   {cells}")
        if a.max_rows and len(rows) > a.max_rows:
            out(f"  ... {len(rows) - a.max_rows} more (use --max-rows 0)")

        out("\n  ONE-SIDED EPISODES (the annotation-mismatch signal)")
        for act in ACTS:
            h, o_ = dis[act]["his_only"], dis[act]["our_only"]
            out(f"    {act:<13} his>0 & ours=0: {len(h):>4}   "
                f"ours>0 & his=0: {len(o_):>4}")
            if h:
                out(f"      his only: {', '.join(h[:12])}{' ...' if len(h) > 12 else ''}")
            if o_:
                out(f"      our only: {', '.join(o_[:12])}{' ...' if len(o_) > 12 else ''}")

        tot_h = {a: sum(R[e][a] for e in both) for a in ACTS}
        tot_o = {a: sum(O[e][a] for e in both) for a in ACTS}
        nh = sum(R[e]["n"] for e in both)
        no = sum(O[e]["n"] for e in both)
        out(f"\n  TOTALS over shared episodes   his {nh:,} clips | ours {no:,} clips"
            f"  (ratio {no / max(nh, 1):.2f}x)")
        for act in ACTS:
            out(f"    {act:<13} his {tot_h[act]:>7,} ({100 * tot_h[act] / max(nh, 1):5.2f} %)"
                f"   ours {tot_o[act]:>7,} ({100 * tot_o[act] / max(no, 1):5.2f} %)")

    out("\n" + "=" * 78)
    out("His stride is 1500 ms and gated on baby-visible; ours is 1000 ms and")
    out("gated on the bucket policy, so clip COUNTS differ by construction.")
    out("Prevalence should still broadly agree. A one-sided episode means the")
    out("two annotation vintages disagree about that recording.")
    out("=" * 78)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
