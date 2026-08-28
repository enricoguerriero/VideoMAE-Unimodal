# Lazy exports: importing the package must NOT pull heavy deps (av/torch), so the
# pure-stdlib+PyYAML data-prep scripts (spec, build_manifest, split_cases) run on
# a machine without the ML stack installed. VideoMAEDataset is imported on first
# access.

from .spec import DataSpec, ClipLabel, load_spec, spec_from_checkpoint, BUCKET_NAMES

__all__ = ["VideoMAEDataset", "DataSpec", "ClipLabel", "load_spec",
           "spec_from_checkpoint", "BUCKET_NAMES"]


def __getattr__(name):
    if name == "VideoMAEDataset":
        from .videomae_dataset import VideoMAEDataset
        return VideoMAEDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
