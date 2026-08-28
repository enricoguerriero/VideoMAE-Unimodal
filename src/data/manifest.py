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

from .spec import DataSpec


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
