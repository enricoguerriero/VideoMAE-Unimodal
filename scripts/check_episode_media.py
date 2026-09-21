#!/usr/bin/env python3
"""
check_episode_media.py — which episodes can inference actually run on, and
where does each one's ground truth come from?

`src/infer_video.py` only lists an episode it can both DECODE and draw ground
truth for. Whether that works is a pure path question, and it is answered
differently per site:

    DRC     `Unprocessed_data/anot_files/<case_id>.txt` sits in the clip's own
            ancestry. Walking up from a clip finds it.
    Haydom  there is no such directory, and the raw exports are NOT named after
            the case. Resolution goes through `AnnotationIndex`, matching the
            case id against the roots in `src/data/sites.py` by KEY (exact stem,
            or any run of >= 5 digits).

Get that wrong and every Haydom episode reports "no ground truth" and silently
drops out of the menu — which looks exactly like "Haydom has no annotations"
when in fact the files are there and the lookup simply never reached them.

This prints, per site and per case, whether the video resolved, whether the
annotation resolved, and WHICH ROOT it came from — so the vintage is visible
rather than assumed. It calls the same `resolve_media` / `list_test_cases`
inference calls, so it cannot drift from what inference will do.

READ-ONLY. Usage:

    python scripts/check_episode_media.py
    python scripts/check_episode_media.py --test-csv data/test_haydom.csv
    python scripts/check_episode_media.py --ronald
    python scripts/check_episode_media.py --list-missing 20

Exit status is non-zero when a whole site resolves nothing, which is the
failure this exists to catch.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.infer_video import list_test_cases, resolve_media
except ModuleNotFoundError as exc:                           # pragma: no cover
    raise SystemExit(
        f"{exc.name} is required — src/infer_video.py pulls in cv2/torch. Run "
        f"this in the environment inference uses.") from exc
from src.data.annotations import AnnotationIndex
from src.data.ronald import DEFAULT_STRIP_PREFIX, split_path
from src.data.sites import annotation_dirs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", default=None,
                   help="Manifest to check (default: data/test.csv, or his under --ronald)")
    p.add_argument("--ronald", action="store_true",
                   help="Check Ronald Paleczny's test manifest and his tree instead")
    p.add_argument("--ronald-dir", default=None)
    p.add_argument("--annotation-dir", action="append", default=None, metavar="DIR",
                   help="Override the annotation roots (repeatable)")
    p.add_argument("--list-missing", type=int, default=10, metavar="N",
                   help="How many unresolved cases to name per site (0 = none)")
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    test_csv = Path(a.test_csv or (split_path("test", a.ronald_dir) if a.ronald
                                   else "data/test.csv"))
    if not test_csv.exists():
        raise SystemExit(f"{test_csv} not found — run scripts/build_data.sh, or pass "
                         f"--test-csv.")

    roots = ([Path(d).expanduser() for d in a.annotation_dir] if a.annotation_dir
             else annotation_dirs())
    print("=" * 78)
    print(f"episode media check — {test_csv}")
    print("=" * 78)
    print("\nannotation roots, in lookup order (first match wins):")
    if not roots:
        print("  (none exist on this machine — every Haydom episode will report no GT)")
    for d in roots:
        n = len(list(d.glob("*.txt"))) if d.is_dir() else 0
        print(f"  {n:>6} .txt  {d}")
    index = AnnotationIndex.from_roots(roots, prefer_order=True) if roots else None
    if index is not None:
        print(f"\n  indexed {len(index)} case key(s) across {len(index.dirs)} "
              f"directory(ies) actually holding files")

    cases = list_test_cases(
        test_csv, DEFAULT_STRIP_PREFIX if a.ronald else None, None,
        default_site="Haydom" if a.ronald else None)
    for c in cases:
        c["video"], c["annotation"] = resolve_media(c["anchor"], c["case_id"], index)

    by_site = defaultdict(list)
    for c in cases:
        by_site[c.get("site") or "?"].append(c)

    print(f"\n{'site':<10}{'cases':>7}{'video':>8}{'GT':>8}{'runnable':>10}   "
          f"annotation source(s)")
    bad = 0
    for site in sorted(by_site):
        cs = by_site[site]
        n_v = sum(1 for c in cs if c["video"])
        n_a = sum(1 for c in cs if c["annotation"])
        n_ok = sum(1 for c in cs if c["video"] and c["annotation"])
        srcs = Counter(Path(c["annotation"]).parent.name
                       for c in cs if c["annotation"])
        src = ", ".join(f"{k} ({v})" for k, v in srcs.most_common(3)) or "-"
        flag = "  <- NONE RUNNABLE" if n_ok == 0 else ""
        print(f"{site:<10}{len(cs):>7}{n_v:>8}{n_a:>8}{n_ok:>10}   {src}{flag}")
        if n_ok == 0:
            bad += 1

    # Full provenance: the point is to see the VINTAGE, not just a count.
    print("\nannotation provenance (full path -> cases resolved from it)")
    prov = Counter(str(Path(c["annotation"]).parent) for c in cases if c["annotation"])
    for d, n in prov.most_common():
        print(f"  {n:>5}  {d}")
    if not prov:
        print("  (nothing resolved)")

    if a.list_missing:
        for site in sorted(by_site):
            miss = [c for c in by_site[site] if not (c["video"] and c["annotation"])]
            if not miss:
                continue
            print(f"\n{site}: {len(miss)} case(s) not runnable — first "
                  f"{min(a.list_missing, len(miss))}:")
            for c in miss[:a.list_missing]:
                why = []
                if not c["video"]:
                    why.append("no video")
                if not c["annotation"]:
                    why.append("ambiguous annotation (two files disagree)"
                               if index is not None and index.is_ambiguous(c["case_id"])
                               else "no annotation")
                print(f"    {c['case_id']:<14} {', '.join(why)}")
                print(f"      clip: {c['anchor']}")

    print("\n" + "=" * 78)
    if bad:
        print("FAILED — a site resolved nothing. Its annotations are not on a path "
              "this can reach;\n         check src/data/sites.py, or pass "
              "--annotation-dir.")
    else:
        print("OK — every site has runnable episodes.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
