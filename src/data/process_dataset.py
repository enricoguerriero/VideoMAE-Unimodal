#!/usr/bin/env python3
"""
process_dataset.py

Driver for the OPTIONAL "regenerate clips from raw" path. For one site, it
discovers every case that has BOTH a video and a cleaned annotation file, then
cuts labeled 3-second video clips with VideoDataProcessor (segment_size=3,
shift=1) — reproducing the thesis' video output for that site.

Because the video path is offset-independent and identical whether a case has
accelerometer data or not, this single loop covers what the thesis split across
its `run_all` (3-modality) and `run_video_only` loops.

Input layout (per site, --base-dir):
    <base>/Unprocessed_data/videos/<case_id>.mp4
    <base>/Unprocessed_data/anot_files/<case_id>.txt   (cleaned 5-column)

Output:
    <folder-name>/videos/<class>/<case>_interval_..._<labelnum>.mp4

NOTE: this driver assumes annotation files are ALREADY cleaned (spelling
corrections, DRC breathing-label removal, event->category remap producing the
5-column format). That cleanup is site/data-specific in the original notebooks;
see the README. If you already have processed clips, prefer build_manifest.py.

Usage:
    python -m src.data.process_dataset \
        --base-dir /spo/LS-Haydom/ProcessedData/.../Data_processing \
        --folder-name /spo/LS-Haydom/ProcessedData/.../Processed_video_clips
"""

import argparse
from pathlib import Path

from .data_process import VideoDataProcessor

SEGMENT_SIZE = 3
SHIFT = 1


def discover_cases(base_dir: Path):
    videos = {p.stem for p in (base_dir / "Unprocessed_data" / "videos").glob("*.mp4")}
    annots = {p.stem for p in (base_dir / "Unprocessed_data" / "anot_files").glob("*.txt")}
    return sorted(videos & annots)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True, type=Path,
                   help="Site staging dir containing Unprocessed_data/{videos,anot_files}.")
    p.add_argument("--folder-name", required=True, type=Path,
                   help="Output clip dataset root (a 'videos/<class>/' tree is created under it).")
    args = p.parse_args()

    cases = discover_cases(args.base_dir)
    print(f"[INFO] {len(cases)} cases with video + annotation under {args.base_dir}")

    for i, case_id in enumerate(cases, 1):
        try:
            proc = VideoDataProcessor(
                video_file=f"{case_id}.mp4",
                annotation_file=f"{case_id}.txt",
                segment_size=SEGMENT_SIZE,
                shift=SHIFT,
                date_of_recording=case_id,
                folder_name=str(args.folder_name),
                for_predict=False,
            )
            proc.run_video_only()
            print(f"[{i}/{len(cases)}] {case_id} done")
        except Exception as exc:  # keep going on a bad case
            print(f"[{i}/{len(cases)}] {case_id} FAILED: {exc}")

    print(f"[DONE] clips written under {args.folder_name}/videos/")


if __name__ == "__main__":
    main()
