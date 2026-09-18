"""
manifest.py

DataFrame-level helpers shared by the manifest's two consumers, the splitter
(split_cases.py) and the auditor (explore_data.py). Kept here so neither has to
import the other.

Both deal with the same wrinkle: the two sites were cut by different generations
of data_process.py. DRC's filenames carry per-activity fraction tags; Haydom's do
not. The manifest records that per clip (`tagged`), along with the clip's
directory (`clip_dir`), which names the activities involved regardless — so an
untagged clip is still fully labellable, just not re-thresholdable.
"""

from __future__ import annotations

import pandas as pd

from .spec import DataSpec, parse_visible


def read_manifest(path) -> pd.DataFrame:
    """Read a manifest/split CSV with the dtypes it actually needs.

    `case_id` mixes bare digits (Haydom, "11848523") with hyphenated ids (DRC,
    "2-33998-1"), so pandas infers a mixed-type column and warns. Forcing str
    also stops "11848523" becoming an int and losing any leading zero.

    `clip_dir` is read with keep_default_na off: an empty directory string is a
    clip at the site root, not a missing value.
    """
    return pd.read_csv(path, dtype={"case_id": str, "clip_dir": str},
                       keep_default_na=False)


def label_keys(df: pd.DataFrame, spec: DataSpec) -> list:
    """Memo keys for `spec.resolve`, one per row.

        (bucket, tagged, clip_dir, frac_visible, *fracs)

    The manifest rounds every fraction to 2 decimals, so 10^5 clips collapse to
    a few thousand distinct keys and `resolve` runs once per key instead of once
    per clip. Four reporting paths built this tuple by hand and each had to be
    edited in step whenever a column joined the evidence — `frac_visible` is the
    second time that happened, so the shape lives here now and `resolve_key`
    below is the only place that knows which slot is which.

    Columns absent from an older manifest fall back to the values that reproduce
    the pre-column behaviour: tagged=1, clip_dir="", frac_visible=None (unknown).
    """
    n = len(df)
    tagged = (df["tagged"].astype(int) if "tagged" in df.columns
              else pd.Series(1, index=df.index))
    clip_dir = df["clip_dir"] if "clip_dir" in df.columns else pd.Series("", index=df.index)
    visible = ([parse_visible(v) for v in df["frac_visible"]]
               if "frac_visible" in df.columns else [None] * n)
    return list(zip(df["bucket"].astype(int), tagged, clip_dir, visible,
                    *(df[c].astype(float) for c in spec.frac_columns())))


def resolve_key(spec: DataSpec, key):
    """`spec.resolve` applied to one `label_keys` tuple -> ClipLabel | None."""
    return spec.resolve(int(key[0]),
                        dict(zip(spec.activities, key[4:])),
                        tagged=bool(key[1]),
                        dir_activities=spec.activities_from_path(key[2]),
                        frac_visible=key[3])


def evidence_masses(df: pd.DataFrame, spec: DataSpec) -> pd.DataFrame:
    """Per-clip, per-activity mass — real fractions where the filename carried
    tags, a nominal value from bucket + directory where it did not.

    Without this an untagged site reports zero mass for every activity, and the
    selector below would balance it on clip counts alone while believing its
    activity mix was empty. See DataSpec.evidence_mass.
    """
    has_tag_cols = "tagged" in df.columns and "clip_dir" in df.columns
    if not has_tag_cols:
        return df[spec.frac_columns()].astype(float).set_axis(list(spec.activities), axis=1)
    rows = []
    for r in df.itertuples(index=False):
        fracs = {a: float(getattr(r, f"frac_{a}")) for a in spec.activities}
        m = spec.evidence_mass(int(r.bucket), fracs, tagged=bool(int(r.tagged)),
                               dir_activities=spec.activities_from_path(r.clip_dir))
        rows.append([m[a] for a in spec.activities])
    return pd.DataFrame(rows, columns=list(spec.activities), index=df.index)



def explain_bad_manifest(path, df, missing) -> str:
    """Tell the user which failure this is, and the command that fixes it.

    A LEGACY manifest (the pre-DataSpec `video_path,label,case_id,site` format)
    cannot be upgraded in place: its scanner resolved the label at scan time AND
    dropped buckets 5-8 outright, so the rows simply are not in the file. Only a
    rescan of the clip roots recovers them.
    """
    lines = [f"{path} is missing columns {missing}."]
    if "label" in df.columns and "bucket" not in df.columns:
        lines += [
            "",
            "This is a LEGACY manifest: it stores a resolved 4-class `label` instead of",
            "the evidence (`bucket` + `frac_*`) the DataSpec needs, and its scanner threw",
            "buckets 5-8 away, so those clips are not in the file at all. It cannot be",
            "converted — the clip roots have to be rescanned.",
        ]
    else:
        lines += ["",
                  "It was built by an older build_manifest.py, or with a different",
                  "`activities` list than the current data config."]
    lines += [
        "",
        "Fix (rescans the clips on disk; expect MORE rows than before, because",
        "buckets 5-8 are now indexed and filtered later at load time):",
        "",
        "    bash scripts/build_data.sh          # manifest + split + this audit",
        "",
        "or just the manifest step:",
        "",
        "    python -m src.data.build_manifest \\",
        "        --root Haydom=/.../videos --root DRC=/.../videos \\",
        "        --out data/clips_all.csv",
    ]
    return "\n".join(lines)
