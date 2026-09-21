#!/usr/bin/env python3
"""
check_ronald_data.py — preflight for `--ronald`, before a GPU is touched.

`bash scripts/train.sh VideoMAE 0 --ronald` spends hours. Everything that can
make that time worthless is checkable in seconds, and all of it is checked here:

  1. his three manifests exist, parse, and have the columns his format promises;
  2. train.csv IS the thesis split — his README documents the pos_weight and
     prior bias it must produce, and a manifest that misses them is not the one
     Table 3.5 was measured on, so the comparison would be against the wrong
     baseline;
  3. the clip paths RESOLVE after the account prefix is stripped (the failure
     mode otherwise is a PermissionError per clip, hours in);
  4. configs/data_ronald.yaml loads, and — if torch is installed — the real
     `VideoMAEDataset._resolve_label_columns` turns his columns into targets,
     with the loss weights and head bias our training loop would actually use.

READ-ONLY. Usage:

    python scripts/check_ronald_data.py
    python scripts/check_ronald_data.py --ronald-dir /path/to/his/data
    python scripts/check_ronald_data.py --check-files 2000   # stat more clips

Exit status is 0 only if every check passes, so it can gate a job script.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.data.ronald import (DEFAULT_RONALD_DIR, DEFAULT_STRIP_PREFIX,
                                 LABEL_COLUMNS, SPLIT_FILES, load_split,
                                 split_stats, verify_split_stats)
except ModuleNotFoundError as exc:                           # pragma: no cover
    raise SystemExit(
        f"{exc.name} is required to read his manifests. This check has to run "
        f"where the data is — on the VM, in the environment training uses:\n"
        f"    python scripts/check_ronald_data.py") from exc
from src.data.spec import DataSpec

RONALD_DATA_CONFIG = "configs/data_ronald.yaml"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ronald-dir", default=None)
    p.add_argument("--strip-prefix", default=DEFAULT_STRIP_PREFIX)
    p.add_argument("--data-config", default=RONALD_DATA_CONFIG)
    p.add_argument("--check-files", type=int, default=200, metavar="N",
                   help="stat the first N clip paths per split (0 = skip)")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("check")
    ok = True

    print("=" * 72)
    print(f"Ronald data preflight — {a.ronald_dir or DEFAULT_RONALD_DIR}")
    print("=" * 72)

    # ---------------------------------------------------------------- 1. load
    splits = {}
    for name in SPLIT_FILES:
        try:
            splits[name] = load_split(name, a.ronald_dir, a.strip_prefix)
        except SystemExit as e:
            print(f"\n[FAIL] {name}: {e}")
            return 1

    print("\n--- split sizes " + "-" * 56)
    print(f"{'split':<12}{'clips':>10}   " + "".join(f"{c:>14}" for c in LABEL_COLUMNS))
    # His thesis Table 2.4. A size mismatch alone does not prove the wrong data,
    # but it is the first thing to look at when the stats check below fails.
    THESIS_SIZES = {"train": 34623, "validation": 8287, "test": 7938}
    for name, df in splits.items():
        st = split_stats(df)
        cells = "".join(f"{st['pos'][c]:>8,} {100*st['pos'][c]/max(len(df),1):>4.1f}%"
                        for c in LABEL_COLUMNS)
        want = THESIS_SIZES[name]
        flag = "" if len(df) == want else f"   <- thesis says {want:,}"
        print(f"{name:<12}{len(df):>10,}   {cells}{flag}")
        if len(df) != want:
            ok = False

    # ------------------------------------------------- 2. is it THE thesis split
    print("\n--- thesis-split verification " + "-" * 42)
    if not verify_split_stats(splits["train"], log):
        ok = False

    # ---------------------------------------------------------- 3. clips exist
    if a.check_files:
        print(f"\n--- clip paths (first {a.check_files} per split) " + "-" * 30)
        for name, df in splits.items():
            paths = df["video_path"].head(a.check_files).tolist()
            missing = [q for q in paths if not Path(q).exists()]
            if missing:
                ok = False
                print(f"  [FAIL] {name}: {len(missing)}/{len(paths)} missing, "
                      f"e.g. {missing[0]}")
                print(f"         His clip tree is not mounted here, or "
                      f"--strip-prefix is wrong for this account.")
            else:
                print(f"  [ok]   {name}: {len(paths)}/{len(paths)} resolve "
                      f"(e.g. {paths[0]})")

    # ------------------------------------------------- 4. the real label path
    print("\n--- data config + label resolution " + "-" * 37)
    try:
        spec = DataSpec.load(a.data_config)
    except Exception as e:                                   # noqa: BLE001
        print(f"  [FAIL] {a.data_config}: {e}")
        return 1
    print(f"  [ok]   {a.data_config}: label_source={spec.label_source}, "
          f"activities={list(spec.activities)}")
    if not spec.labels_from_columns:
        print(f"  [FAIL] expected label_source: columns — {a.data_config} will try to "
              f"read `bucket`/`frac_*`, which his manifests do not have.")
        return 1

    missing_cols = [c for c in spec.activities if c not in splits["train"].columns]
    if missing_cols:
        print(f"  [FAIL] the data config names activities his manifests lack: "
              f"{missing_cols}")
        return 1
    print(f"  [ok]   every activity in the config has a column in his manifests")

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("  [skip] torch not installed here — cannot exercise the real "
              "Dataset label path or print the loss weights. Re-run on the VM.")
        print("\n" + ("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED"))
        return 0 if ok else 1

    from src.data.videomae_dataset import VideoMAEDataset
    for name, df in splits.items():
        kept, targets, masks, dropped, gated = \
            VideoMAEDataset._resolve_label_columns(df, spec)
        if dropped or gated or len(kept) != len(df):
            ok = False
            print(f"  [FAIL] {name}: {len(df)} rows in, {len(kept)} out — a foreign "
                  f"manifest must never lose rows.")
        else:
            print(f"  [ok]   {name}: {len(kept):,} clips -> targets "
                  f"{tuple(targets.shape)}, all supervised "
                  f"({int(masks.sum())} == {masks.numel()})")

    # The numbers our training loop will actually use, next to his.
    print("\n--- what OUR loop derives from his train split " + "-" * 25)
    ds = object.__new__(VideoMAEDataset)
    ds.spec = spec
    ds.data, ds.labels, ds.masks, _, _ = \
        VideoMAEDataset._resolve_label_columns(splits["train"], spec)
    bias = ds.compute_bias()
    print(f"  prior bias         : {[round(v, 4) for v in bias.tolist()]}")
    print(f"     (his documented : [-0.0694, -2.2965, -3.6654])")
    for w in ("inv_freq", "sqrt_inv_freq"):
        pw = ds.compute_pos_weight(w)
        note = "  <- HIS formula" if w == "inv_freq" else "  <- our default"
        print(f"  pos_weight {w:<14}: {[round(v, 4) for v in pw.tolist()]}{note}")
    print(f"     (his documented : [1.0719, 9.9393, 39.0729])")
    print("\n  To reproduce his loss exactly, train with a config that sets "
          "`class_weighting: inv_freq`.")

    print("\n" + "=" * 72)
    print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED — see the lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
