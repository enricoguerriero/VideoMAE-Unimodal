"""
ronald.py

Load Ronald Paleczny's Haydom manifests so this repo's model trains on EXACTLY
his data.

--------------------------------------------------------------------------
What this is for
--------------------------------------------------------------------------
Our pipeline and his disagree in several places at once — the baby-visible gate,
clip resolution, stride, pool downsampling, the split itself — and every fix
lands on top of the others, so a run that is still worse does not say which
difference is responsible. This module removes the data from the comparison
entirely: same clips, same labels, same splits as his thesis, driven through our
model, our training loop and our metrics.

That makes the result interpretable in one step:

    our model on HIS data  ~=  his numbers   ->  the gap was DATA. Keep working
                                                 on the pipeline.
    our model on HIS data  <<  his numbers   ->  the gap is MODEL or TRAINING.
                                                 The pipeline was never the issue.

It is a diagnostic, not a destination — his manifests point at his clip tree, so
nothing trained this way can be deployed on our own data without a retrain.

--------------------------------------------------------------------------
His manifest format
--------------------------------------------------------------------------
    video_path, ventilation, stimulation, suction

`video_path` is absolute under the account that generated the dataset
(`/home/u269483/spo/...`), which no other account can read — it fails with
`PermissionError: [Errno 13]`. The prefix is stripped at load time, leaving the
readable `/spo/...` mount path. This is his own `strip_path_prefix` behaviour,
reproduced (Ronald_code/src/dataset.py).

The three columns are FINAL binary labels, decided by his
`clips_and_video_stats.py` at his thresholds, under his visible gate. Nothing
here re-derives them: `configs/data_ronald.yaml` sets `label_source: columns`,
and DataSpec then takes the columns verbatim. Column ORDER does not matter —
they are matched by name — which is what keeps his `[ventilation, stimulation,
suction]` from being silently transposed onto our usual stimulation-first order.

Deliberately stdlib + pandas only: no torch, no av. `verify_split_stats` is the
gate worth running before a GPU is touched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Where his three manifests live on the VM (Ronald_code/configs/config.yaml).
DEFAULT_RONALD_DIR = "/spo/LS-Haydom/ProcessedData/Ronald/code/Master-project/data"

#: Stripped from every `video_path`. His config's `strip_path_prefix`.
DEFAULT_STRIP_PREFIX = "/home/u269483"

#: His activity columns, in HIS order. Used only to validate the file; the
#: targets are selected by name against `spec.activities`.
LABEL_COLUMNS = ("ventilation", "stimulation", "suction")

SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}

#: Ronald_code/README.md, "Check the data first". These are his THESIS SPLITS'
#: statistics (Table 2.4: train 34,623 / val 8,287 / test 7,938), and his thesis
#: independently reports the stimulation bias as b = -2.30 (3.2.1), matching
#: -2.2965. If a loaded train.csv does not reproduce them, it is not the split
#: the published numbers came from and nothing downstream will match.
#:
#: Keyed by his column order [ventilation, stimulation, suction].
THESIS_TRAIN_ROWS = 34623
THESIS_POS_WEIGHT = {"ventilation": 1.0719, "stimulation": 9.9393, "suction": 39.0729}
THESIS_PRIOR_BIAS = {"ventilation": -0.0694, "stimulation": -2.2965, "suction": -3.6654}
#: His pos_weight is a raw neg/pos ratio spanning 1 -> 39, so a relative
#: tolerance is the only one that means the same thing at both ends. 1 % absorbs
#: his 4-decimal rounding with room to spare while still catching a split that
#: differs by even a few hundred clips.
STATS_RTOL = 0.01


def split_path(split: str, ronald_dir=None) -> Path:
    """Absolute path to one of his three manifests."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    return Path(ronald_dir or DEFAULT_RONALD_DIR) / SPLIT_FILES[split]


def load_split(split: str, ronald_dir=None, strip_prefix=DEFAULT_STRIP_PREFIX,
               check_files=False) -> pd.DataFrame:
    """Read one of his manifests into a DataFrame this repo's Dataset accepts.

    Returns the frame with `video_path` rewritten and the three label columns
    coerced to int. Raises with an actionable message rather than letting a
    missing mount surface later as a PyAV traceback.
    """
    path = split_path(split, ronald_dir)
    if not path.exists():
        raise SystemExit(
            f"Ronald's {split} manifest not found: {path}\n"
            f"  Pass --ronald-dir if his data lives elsewhere, and check the /spo "
            f"mount is present on this machine. These manifests are his, not ours "
            f"— they are not produced by scripts/build_data.sh.")

    df = pd.read_csv(path)
    missing = [c for c in ("video_path",) + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing column(s) {missing}. Expected his format: "
            f"video_path,{','.join(LABEL_COLUMNS)}")

    if strip_prefix:
        before = df["video_path"].astype(str)
        after = before.str.replace(f"^{strip_prefix}", "", regex=True)
        n = int((before != after).sum())
        if n:
            logger.info(f"[ronald:{split}] stripped {strip_prefix!r} from {n:,}/{len(df):,} "
                        f"clip paths (e.g. {after.iloc[0]})")
        df["video_path"] = after

    for c in LABEL_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if df[list(LABEL_COLUMNS)].isna().any().any():
        raise SystemExit(f"{path} has non-numeric values in its label columns")
    for c in LABEL_COLUMNS:
        df[c] = df[c].astype(int)

    if check_files:
        miss = [p for p in df["video_path"].head(200) if not Path(p).exists()]
        if miss:
            raise SystemExit(
                f"[ronald:{split}] {len(miss)} of the first 200 clips do not exist, "
                f"e.g. {miss[0]}\n  His clip tree is not mounted here, or "
                f"--ronald-strip-prefix is wrong for this account.")

    logger.info(f"[ronald:{split}] {len(df):,} clips from {path}")
    return df


def split_stats(df: pd.DataFrame) -> dict:
    """His pos_weight and prior bias for one manifest, by HIS formulas.

    pos_weight = neg/pos and bias = log(p/(1-p)), which is `inv_freq` rather
    than this repo's default `sqrt_inv_freq`. Computed here in his form ONLY so
    the numbers can be compared against the ones his README documents; it is not
    what trains the model. See `verify_split_stats`.
    """
    import math
    n = len(df)
    out = {"n": n, "pos": {}, "pos_weight": {}, "prior_bias": {}}
    for c in LABEL_COLUMNS:
        pos = int(df[c].sum())
        neg = n - pos
        p = min(max(pos / n if n else 0.0, 1e-6), 1 - 1e-6)
        out["pos"][c] = pos
        out["pos_weight"][c] = neg / (pos + 1e-6)
        out["prior_bias"][c] = math.log(p / (1 - p))
    return out


def verify_split_stats(df: pd.DataFrame, logger=logger, strict=False) -> bool:
    """Is this train.csv the split his published numbers came from?

    His README makes this checkable: the thesis splits produce exactly

        pos_weight per label: [1.0719, 9.9393, 39.0729]
        prior bias per label: [-0.0694, -2.2965, -3.6654]

    Two manifests can have the same row count and different content, and a
    mismatch here means every later comparison is against the wrong baseline —
    so it is worth the two seconds it costs, before the backbone downloads.

    Returns True on a match. Warns by default rather than raising: a deliberate
    re-derivation of his splits is a legitimate thing to be doing, and it should
    not be blocked, only made impossible to do by accident. `strict=True` turns
    it into an error.
    """
    st = split_stats(df)
    rows = []
    ok = st["n"] == THESIS_TRAIN_ROWS
    rows.append(("clips", float(st["n"]), float(THESIS_TRAIN_ROWS), ok))
    for c in LABEL_COLUMNS:
        for name, got, want in (("pos_weight", st["pos_weight"][c], THESIS_POS_WEIGHT[c]),
                                ("prior_bias", st["prior_bias"][c], THESIS_PRIOR_BIAS[c])):
            hit = abs(got - want) <= STATS_RTOL * max(abs(want), 1e-9)
            ok &= hit
            rows.append((f"{name}/{c}", got, want, hit))

    logger.info("[ronald] thesis-split check (Ronald_code/README.md 'Check the data first')")
    for name, got, want, hit in rows:
        logger.info(f"    {'OK ' if hit else 'XX '} {name:<24} got {got:>12.4f}   "
                    f"thesis {want:>10.4f}")
    if ok:
        logger.info("[ronald] this IS the thesis split — his Table 3.5 is the right target.")
        return True

    msg = ("[ronald] these manifests are NOT the thesis splits: the statistics above "
           "disagree. His published numbers were measured on a different set of "
           "clips, so comparing against Table 3.5 would be comparing two different "
           "experiments. Check --ronald-dir points at the manifests his config.yaml "
           "names.")
    if strict:
        raise SystemExit(msg)
    logger.warning(msg)
    return False
