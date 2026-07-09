#!/usr/bin/env python3
"""
build_manifest.py

Scan the processed 3-second video clips produced by the multimodal thesis'
DataProcessor (for BOTH the Haydom and DRC sites) and emit ONE combined
clip-level manifest CSV for single-label 4-class training.

This is the PRIMARY data path for this repo: the processed clips already exist
on the VM (they are data, not code), so we simply index them — no re-cutting,
which guarantees the VideoMAE model trains on exactly the same clips as the
multimodal MoViNet video base model.

Label decoding (matches the thesis exactly):
    * The class integer is the trailing `_N` in the clip filename
      ({case}_interval_{n}_start_{ms}_end_{ms}[_tag]_{N}.mp4).
    * Included: 0 non_target, 1 stimulation, 2 ventilation, 3 suction.
    * no_overlap (4) is MERGED into non_target (0)  — as the thesis did.
    * Buckets 5 (no_label), 6 (partial), 7 (target_overlap), 8 (partial-combo)
      are DROPPED entirely.
    * case_id is recovered via filename.split('_interval_')[0].

Output CSV columns: video_path, label, case_id, site
    (`label` = the single class index; `site` tags the source for case-level
     splitting so identical case-id strings across sites never collide.)

Usage:
    python -m src.data.build_manifest \
        --root Haydom=/path/to/Haydom/Processed_.../videos \
        --root DRC=/path/to/DRC/Processed_.../videos \
        --out data/clips_all.csv

Each --root is SITE=PATH; PATH is a `videos/` directory containing the per-class
subfolders (or any tree of *.mp4 clips following the naming convention).
"""

import argparse
import csv
from pathlib import Path

# Trailing filename label_num -> training class index (None = drop the clip).
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}  # 4 (no_overlap) -> 0 (non_target)
DROP = {5, 6, 7, 8}


def decode_label(stem: str):
    """Return the training class index from a clip filename stem, or None to drop."""
    try:
        raw = int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return None
    if raw in DROP:
        return None
    return LABEL_REMAP.get(raw, None)


def case_id_from_stem(stem: str) -> str:
    """Recover the case id: everything before '_interval_'."""
    return stem.split("_interval_")[0]


def scan_root(site: str, root: Path):
    """Yield (video_path, label, case_id, site) for every valid clip under root."""
    rows = []
    for mp4 in sorted(root.rglob("*.mp4")):
        label = decode_label(mp4.stem)
        if label is None:
            continue
        rows.append({
            "video_path": str(mp4.resolve()),
            "label": label,
            "case_id": case_id_from_stem(mp4.stem),
            "site": site,
        })
    return rows


def parse_root(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--root must be SITE=PATH, got: {spec}")
    site, path = spec.split("=", 1)
    return site.strip(), Path(path).expanduser()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", required=True, type=parse_root,
                   metavar="SITE=PATH", help="Repeatable. e.g. --root Haydom=/.../videos")
    p.add_argument("--out", required=True, type=Path, help="Output combined manifest CSV.")
    args = p.parse_args()

    all_rows = []
    per_site = {}
    per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    for site, root in args.root:
        if not root.exists():
            print(f"[WARN] root does not exist: {root} (site={site}) — skipping")
            continue
        rows = scan_root(site, root)
        per_site[site] = len(rows)
        for r in rows:
            per_class[r["label"]] += 1
        all_rows.extend(rows)
        print(f"[INFO] {site}: {len(rows)} clips from {root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "label", "case_id", "site"])
        w.writeheader()
        w.writerows(all_rows)

    names = ["non_target", "stimulation", "ventilation", "suction"]
    print(f"\n[DONE] {len(all_rows)} clips -> {args.out}")
    print("Per site:", per_site)
    print("Per class:", {names[k]: v for k, v in per_class.items()})


if __name__ == "__main__":
    main()
