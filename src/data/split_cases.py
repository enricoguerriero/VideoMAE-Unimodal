#!/usr/bin/env python3
"""
split_cases.py

Turn the combined clip manifest (build_manifest.py) into train.csv / validation.csv
/ test.csv at the WHOLE-CASE level, so no case's clips leak across splits — the
same policy as the multimodal thesis.

Test set: the 14 fixed held-out cases used by the thesis (verbatim from its
`full_case_predictions_thesis/` outputs). These are frozen so the VideoMAE
results are evaluated on exactly the same cases as the multimodal system.
Override with --test-cases-file (one case_id per line) if needed.

Train/validation: the remaining cases are split by a stratified, deterministic
(seed=2025) whole-case assignment targeting TRAIN_RATIO (default 0.7, as in the
thesis's merged train/val split). For an EXACT reproduction you can instead pass
--train-cases-file and --val-cases-file with explicit case_id lists.

Output CSVs have columns: video_path, label   (consumed by VideoMAEDataset).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

# The 14 held-out test cases (DRC: 4, Haydom: 10) — verbatim from the thesis.
DEFAULT_TEST_CASES = [
    # DRC
    "2-33998-1", "2-34325-1", "2-37178-1", "2-37453-1",
    # Haydom
    "11848523", "15233524", "28631424", "37572224", "38037024",
    "38042423", "38714124", "40094725", "40386725", "40402325",
]


def read_case_list(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def majority_label(df_case: pd.DataFrame) -> int:
    return int(df_case["label"].value_counts().idxmax())


def stratified_case_split(case_to_major, train_ratio, seed):
    """Assign whole cases to train/val, stratified by each case's majority class."""
    by_class = defaultdict(list)
    for case, major in case_to_major.items():
        by_class[major].append(case)
    rng = random.Random(seed)
    train_cases, val_cases = set(), set()
    for cls in sorted(by_class):
        cases = sorted(by_class[cls])
        rng.shuffle(cases)
        n_train = round(len(cases) * train_ratio)
        train_cases.update(cases[:n_train])
        val_cases.update(cases[n_train:])
    return train_cases, val_cases


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, help="Combined manifest from build_manifest.py")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory to write train/validation/test CSVs")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--test-cases-file", type=Path, default=None,
                   help="Override the 14 default test case_ids (one per line).")
    p.add_argument("--train-cases-file", type=Path, default=None,
                   help="Explicit train case_ids (exact reproduction); overrides stratified split.")
    p.add_argument("--val-cases-file", type=Path, default=None,
                   help="Explicit val case_ids (exact reproduction); overrides stratified split.")
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    df["case_id"] = df["case_id"].astype(str)

    test_cases = set(read_case_list(args.test_cases_file) if args.test_cases_file else DEFAULT_TEST_CASES)
    present = set(df["case_id"].unique())
    missing = test_cases - present
    if missing:
        print(f"[WARN] {len(missing)} test case_ids not found in manifest: {sorted(missing)}")

    test_df = df[df["case_id"].isin(test_cases)].copy()
    rest_df = df[~df["case_id"].isin(test_cases)].copy()

    if args.train_cases_file and args.val_cases_file:
        train_cases = set(read_case_list(args.train_cases_file))
        val_cases = set(read_case_list(args.val_cases_file))
    else:
        case_to_major = {c: majority_label(g) for c, g in rest_df.groupby("case_id")}
        train_cases, val_cases = stratified_case_split(case_to_major, args.train_ratio, args.seed)

    train_df = rest_df[rest_df["case_id"].isin(train_cases)].copy()
    val_df = rest_df[rest_df["case_id"].isin(val_cases)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["video_path", "label"]
    for name, d in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        out = args.out_dir / f"{name}.csv"
        d[cols].to_csv(out, index=False)
        names = ["non_target", "stimulation", "ventilation", "suction"]
        dist = {names[k]: int((d["label"] == k).sum()) for k in range(4)}
        print(f"[{name}] {len(d)} clips | {d['case_id'].nunique()} cases | {dist} -> {out}")


if __name__ == "__main__":
    main()
