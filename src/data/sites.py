"""
sites.py

Where each hospital's ANNOTATIONS live, for Python callers.

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
The two sites are not laid out the same way, and the difference is invisible
until something silently finds nothing:

    DRC     <DRC_BASE>/Unprocessed_data/anot_files/<case_id>.txt
            — a sibling of the clips, named after the case. Anything walking UP
              from a clip path finds it on its own.

    Haydom  there is NO `Unprocessed_data/anot_files/` at all, and the raw
            annotation files are NOT named after the case. They live in
            separate export directories, and the case id has to be matched by
            KEY (exact stem, or any run of >= 5 digits) — which is what
            `annotations.AnnotationIndex` does.

            The re-cut also STAGES a per-case copy at
            <HAYDOM_BASE>/Data_processing_recut/Unprocessed_data/anot_files/
            <case_id>.txt. That one IS named after the case, but it is a
            SIBLING of the clip tree (Processed_data_recut), not an ancestor,
            so walking up from a clip still never reaches it.

So a tool that resolves annotations by walking up from a clip finds every DRC
episode and no Haydom one. That is not a missing-data problem — the
annotations are there — it is a layout problem, and the fix is to search these
roots by case key rather than by path adjacency.

`scripts/site_paths.sh` is the bash twin of this file, used by the data-prep
scripts. The two lists are kept in step by hand; that duplication is
deliberate and already documented there (scripts/build_data.sh carries a third
copy for the same reason). If you change a path, change it in both.

Deliberately stdlib only, so `src/data/__init__.py`'s no-torch guarantee holds.
"""

from __future__ import annotations

from pathlib import Path

HAYDOM_BASE = Path("/spo/LS-Haydom/ProcessedData/Athavan_Frida/Data_processing")
DRC_BASE = Path("/spo/LS-DRC/ProcessedData/Athavan_Frida/Data_processing")

#: Haydom annotation roots, most-preferred first. Element 0 is the re-cut's
#: STAGED copy: one cleaned file per case, keyed by case id, and — the reason it
#: comes first — the exact annotations the current clip tree was cut from, so an
#: overlay drawn from it cannot disagree with the labels the model trained on.
#: The rest are the raw exports, ordered by the case coverage
#: scripts/audit_source_data.py measured (240-242 of 246).
HAYDOM_ANNOTATION_DIRS = [
    HAYDOM_BASE / "Data_processing_recut" / "Unprocessed_data" / "anot_files",
    Path("/spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations"),
    Path("/spo/LS-Haydom/ProcessedData/Ronald/data/Tanzania/annotations_corrected"),
    Path("/spo/LS-Haydom/Data/FullDataset/2025-2026/March2026Sync/annotations"),
    Path("/spo/LS-Haydom/ProcessedData/Athavan_Frida/FullDataset_Combined/Annotations"),
]

#: DRC needs no list — this is found by walking up from a clip — but naming it
#: keeps a caller that wants every root from special-casing one site.
DRC_ANNOTATION_DIRS = [DRC_BASE / "Unprocessed_data" / "anot_files"]

#: Ronald Paleczny's own tree, for a checkpoint trained on his manifests.
RONALD_ANNOTATION_DIRS = [
    Path("/spo/LS-Haydom/ProcessedData/Ronald/data/Tanzania/annotations_corrected"),
]


def annotation_dirs(existing_only: bool = True) -> list[Path]:
    """Every known annotation root, best-first, de-duplicated.

    A LOOKUP-only caller (an inference overlay) should take all of them:
    matching is by case key, coverage is what matters, and a qualitative
    reference is not a training label. A caller that produces LABELS should
    pick ONE — see the note in scripts/recut_haydom.sh about mixing export
    vintages.
    """
    out, seen = [], set()
    for d in HAYDOM_ANNOTATION_DIRS + DRC_ANNOTATION_DIRS + RONALD_ANNOTATION_DIRS:
        if d in seen:
            continue
        seen.add(d)
        if not existing_only or d.is_dir():
            out.append(d)
    return out
