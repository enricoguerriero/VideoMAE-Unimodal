# Lazy exports: importing the package must NOT pull heavy deps (av/torch), so the
# pure-stdlib data-prep scripts (build_manifest, split_cases) run on a machine
# without the ML stack installed. VideoMAEDataset is imported on first access.

__all__ = ["VideoMAEDataset"]


def __getattr__(name):
    if name == "VideoMAEDataset":
        from .videomae_dataset import VideoMAEDataset
        return VideoMAEDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
