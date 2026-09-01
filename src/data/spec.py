"""
spec.py

DataSpec — the single source of truth for WHAT the labels are.

Everything task-dependent in this repo derives from one `configs/data.yaml`:
the number of logits, the head's output activation (softmax vs sigmoid), the
loss (CrossEntropy vs masked BCE), the metric set, the class names, and which
clips are eligible at all. Swapping `task: multiclass` -> `task: multilabel`,
moving a threshold, or admitting a dropped bucket needs no code change.

Deliberately stdlib + PyYAML only (no torch / numpy / av), so the data-prep
scripts (build_manifest.py, split_cases.py) run on a machine without the ML
stack — see src/data/__init__.py.

--------------------------------------------------------------------------
The label pipeline, in two stages
--------------------------------------------------------------------------
Stage 1 — TASK-AGNOSTIC (build_manifest.py, cached in the manifest CSV):
    clip filename  ->  (bucket, {activity: fraction of the 3 s window})
    e.g. "…_start_6000_end_9000_stim0.67_vent0.55_7.mp4"
         -> bucket 7, {stimulation: 0.67, ventilation: 0.55, suction: 0.0}

Stage 2 — TASK-SPECIFIC (`DataSpec.resolve`, applied at Dataset load time):
    (bucket, fractions)  ->  ClipLabel | None      (None = drop this clip)

Stage 2 runs at load time, not at manifest time, so `task`, `thresholds`,
`ambiguous` and `decision_thresholds` can all be changed WITHOUT rebuilding the
manifest. Only `buckets` and `tag_keys` affect Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

MULTICLASS = "multiclass"
MULTILABEL = "multilabel"
TASKS = (MULTICLASS, MULTILABEL)

AMBIGUOUS_POLICIES = ("drop", "negative", "mask")
OVERLAP_POLICIES = ("dominant", "drop")
BUCKET_POLICIES = ("keep", "drop")

#: The nine buckets data_process.py writes as the trailing `_N` of every clip.
BUCKET_NAMES = {
    0: "non_target",
    1: "activity_strong_1",
    2: "activity_strong_2",
    3: "activity_strong_3",
    4: "no_overlap",
    5: "no_label",
    6: "partial",
    7: "target_overlap",
    8: "partial_overlap",
}

DEFAULT_DATA_CONFIG = "configs/data.yaml"

#: Buckets whose clips have, by construction, a non-zero overlap with at least
#: one activity — so data_process.py's `_overlap_suffix` MUST have written a
#: `_stim0.67`-style tag for them. A clip in one of these buckets with no tag
#: therefore comes from an older processing run that did not write tags at all
#: (this is the case for the Haydom tree), and its fractions are UNKNOWN rather
#: than zero. See `resolve` for how those clips are labelled instead.
TAG_BEARING_BUCKETS = frozenset({1, 2, 3, 6, 7, 8})

#: Stand-in fractions for an untagged clip, used only where a NUMBER is needed
#: (the split's balancing masses). Buckets 1/2/3/7 assert "at or above
#: threshold", buckets 6/8 assert "present but below it" — the midpoint of the
#: ambiguous band is the least-wrong single value for the latter.
NOMINAL_STRONG_FRAC = 1.0
NOMINAL_PARTIAL_FRAC = 0.35

# Filename fractions are written with `f"{frac:.2f}"`, so a value that should be
# exactly at a threshold can land 0.005 below it. Absorb the rounding.
_FRAC_EPS = 1e-6


@dataclass(frozen=True)
class ClipLabel:
    """Resolved target for one clip.

    multiclass: `class_index` is set, `targets`/`mask` are None.
    multilabel: `targets` (one float per activity, 0.0/1.0) and `mask`
                (1.0 = supervised, 0.0 = excluded from the loss) are set,
                `class_index` is None.
    """

    class_index: int | None = None
    targets: tuple[float, ...] | None = None
    mask: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DataSpec:
    task: str
    activities: tuple[str, ...]
    negative_class: str
    tag_keys: dict[str, str]
    thresholds: dict[str, float]
    weak_threshold: float
    ambiguous: str
    overlap_resolution: str
    buckets: dict[int, str]
    decision_thresholds: dict[str, float]
    annotation_events: dict[str, str] = field(default_factory=dict)
    source: str | None = field(default=None, compare=False)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None) -> "DataSpec":
        """Read a data config YAML (default: configs/data.yaml)."""
        path = Path(path or DEFAULT_DATA_CONFIG)
        if not path.exists():
            raise FileNotFoundError(
                f"data config not found: {path}. Copy configs/data.yaml or pass "
                f"--data-config / set `data_config:` in configs/config.yaml.")
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: dict, source: str | None = None) -> "DataSpec":
        task = str(raw.get("task", MULTICLASS)).strip().lower()
        activities = tuple(raw.get("activities") or ())
        negative_class = str(raw.get("negative_class", "non_target"))
        tag_keys = {str(k): str(v) for k, v in (raw.get("tag_keys") or {}).items()}
        thresholds = {str(k): float(v) for k, v in (raw.get("thresholds") or {}).items()}
        weak = float(raw.get("weak_threshold", 0.20))
        ambiguous = str(raw.get("ambiguous", "drop")).strip().lower()
        overlap = str(raw.get("overlap_resolution", "drop")).strip().lower()
        buckets = {int(k): str(v).strip().lower() for k, v in (raw.get("buckets") or {}).items()}
        dec = {str(k): float(v) for k, v in (raw.get("decision_thresholds") or {}).items()}
        events = {str(k): str(v) for k, v in (raw.get("annotation_events") or {}).items()}

        spec = cls(task=task, activities=activities, negative_class=negative_class,
                   tag_keys=tag_keys, thresholds=thresholds, weak_threshold=weak,
                   ambiguous=ambiguous, overlap_resolution=overlap, buckets=buckets,
                   decision_thresholds=dec, annotation_events=events, source=source)
        spec.validate()
        return spec

    def to_dict(self) -> dict:
        """Round-trippable plain dict — stored in every checkpoint."""
        return {
            "task": self.task,
            "activities": list(self.activities),
            "negative_class": self.negative_class,
            "tag_keys": dict(self.tag_keys),
            "thresholds": dict(self.thresholds),
            "weak_threshold": self.weak_threshold,
            "ambiguous": self.ambiguous,
            "overlap_resolution": self.overlap_resolution,
            "buckets": dict(self.buckets),
            "decision_thresholds": dict(self.decision_thresholds),
            "annotation_events": dict(self.annotation_events),
        }

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if not self.activities:
            raise ValueError("`activities` must list at least one activity")
        if len(set(self.activities)) != len(self.activities):
            raise ValueError(f"duplicate entries in activities: {self.activities}")
        if self.negative_class in self.activities:
            raise ValueError(
                f"negative_class {self.negative_class!r} must not also be an activity")
        if self.ambiguous not in AMBIGUOUS_POLICIES:
            raise ValueError(f"ambiguous must be one of {AMBIGUOUS_POLICIES}")
        if self.overlap_resolution not in OVERLAP_POLICIES:
            raise ValueError(f"overlap_resolution must be one of {OVERLAP_POLICIES}")
        for a in self.activities:
            if a not in self.tag_keys:
                raise ValueError(f"tag_keys is missing activity {a!r}")
            if a not in self.thresholds:
                raise ValueError(f"thresholds is missing activity {a!r}")
            if not 0.0 < self.thresholds[a] <= 1.0:
                raise ValueError(f"thresholds[{a}] must be in (0, 1], got {self.thresholds[a]}")
            if self.thresholds[a] <= self.weak_threshold:
                raise ValueError(
                    f"thresholds[{a}]={self.thresholds[a]} must exceed "
                    f"weak_threshold={self.weak_threshold}; otherwise the "
                    f"positive and negative bands overlap")
        if len(set(self.tag_keys[a] for a in self.activities)) != len(self.activities):
            raise ValueError(f"tag_keys must be unique per activity: {self.tag_keys}")
        if not 0.0 <= self.weak_threshold < 1.0:
            raise ValueError(f"weak_threshold must be in [0, 1), got {self.weak_threshold}")
        for b, policy in self.buckets.items():
            if policy not in BUCKET_POLICIES:
                raise ValueError(f"buckets[{b}] must be one of {BUCKET_POLICIES}, got {policy!r}")
            if b not in BUCKET_NAMES:
                raise ValueError(f"unknown bucket {b}; valid buckets are {sorted(BUCKET_NAMES)}")
        missing = sorted(set(BUCKET_NAMES) - set(self.buckets))
        if missing:
            raise ValueError(
                f"buckets must cover every bucket 0-8; missing {missing}. "
                f"Be explicit — a silently dropped bucket is a silently smaller dataset.")
        for a in self.activities:
            t = self.decision_thresholds.get(a, 0.5)
            if not 0.0 < t < 1.0:
                raise ValueError(f"decision_thresholds[{a}] must be in (0, 1), got {t}")
        unknown = sorted(set(self.annotation_events) - set(self.activities))
        if unknown:
            raise ValueError(f"annotation_events names non-activities: {unknown}")

    # ------------------------------------------------------------------ shape
    @property
    def is_multilabel(self) -> bool:
        return self.task == MULTILABEL

    @property
    def class_names(self) -> list[str]:
        """The model's output units, in logit order."""
        if self.is_multilabel:
            return list(self.activities)
        return [self.negative_class] + list(self.activities)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def projected_class_names(self) -> list[str]:
        """Single-label class list used for the thesis-comparable confusion
        matrix. Identical to `class_names` in multiclass."""
        return [self.negative_class] + list(self.activities)

    @property
    def activation(self) -> str:
        return "sigmoid" if self.is_multilabel else "softmax"

    def sigmoid_thresholds(self) -> list[float]:
        """Per-activity decision thresholds in logit order (multilabel)."""
        return [float(self.decision_thresholds.get(a, 0.5)) for a in self.activities]

    def event_name(self, activity: str) -> str:
        """The activity's `Event` string in the 5-column annotation TSV.

        Defaults to the title-cased activity name ("stimulation" ->
        "Stimulation"), which matches the Haydom/DRC annotation files. Override
        per activity via `annotation_events` in the data config when a site
        spells an event differently.
        """
        if activity in self.annotation_events:
            return self.annotation_events[activity]
        return activity.replace("_", " ").capitalize()

    def event_to_activity_index(self) -> dict[str, int]:
        """{annotation Event string -> activity index} for GT parsing."""
        return {self.event_name(a): i for i, a in enumerate(self.activities)}

    def frac_columns(self) -> list[str]:
        """Manifest column names holding the per-activity window fractions."""
        return [f"frac_{a}" for a in self.activities]

    def keeps_bucket(self, bucket: int) -> bool:
        return self.buckets.get(int(bucket), "drop") == "keep"

    def kept_buckets(self) -> list[int]:
        return sorted(b for b, p in self.buckets.items() if p == "keep")

    # ------------------------------------------------------------------ stage 1
    def parse_stem(self, stem: str) -> tuple[int | None, dict[str, float], bool]:
        """Filename stem -> (bucket, {activity: fraction}, tagged).

        Clip stems look like
            {case}_interval_{n}_start_{ms}_end_{ms}[_stim0.67][_vent0.55]_{bucket}

        `tagged` is False when the stem carries NO fraction tag but its bucket
        requires one (see TAG_BEARING_BUCKETS) — i.e. the clip predates the
        tag-writing processor and its fractions are unknown, not zero. For every
        other clip `tagged` is True and the (possibly all-zero) fractions are
        real. Returns (None, {}, False) when the stem breaks the convention.
        """
        try:
            bucket = int(stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None, {}, False
        if bucket not in BUCKET_NAMES:
            return None, {}, False
        fracs = {a: 0.0 for a in self.activities}
        by_tag = {self.tag_keys[a]: a for a in self.activities}
        n_tags = 0
        for tag, value in self._tag_re().findall(stem):
            fracs[by_tag[tag]] = float(value)
            n_tags += 1
        tagged = n_tags > 0 or bucket not in TAG_BEARING_BUCKETS
        return bucket, fracs, tagged

    def activities_from_path(self, rel_dir) -> tuple[str, ...]:
        """Activities named by the clip's DIRECTORY, in `activities` order.

        data_process.py files every clip under a directory that names the
        activities involved — `ventilation/`, `partial/stimulation/`,
        `target_overlap/stimulation+ventilation/` — independently of the
        filename tag. That makes the activity IDENTITY recoverable even for an
        untagged corpus; only the exact fractions are lost.

        Matching is case-insensitive and tolerant of separators, so a tree that
        spells the combo differently still resolves. An activity name is matched
        as a CONTIGUOUS RUN OF TOKENS, not as a single token: the directory is
        split on every non-alphanumeric character, so a multi-word activity like
        `chest_compression` would never equal one token and was previously
        unmatchable — which silently dropped every untagged clip of that activity.
        """
        if not isinstance(rel_dir, str) or not rel_dir:
            return ()  # missing, NaN, or a clip sitting at the site root
        parts = [p for p in re.split(r"[^a-z0-9]+", rel_dir.lower()) if p]
        found = set()
        for a in self.activities:
            want = [t for t in re.split(r"[^a-z0-9]+", a.lower()) if t]
            n = len(want)
            if n and any(parts[i:i + n] == want for i in range(len(parts) - n + 1)):
                found.add(a)
        return tuple(a for a in self.activities if a in found)

    @staticmethod
    def expects_tag(bucket: int) -> bool:
        """Would data_process.py have written a fraction tag for this bucket?"""
        return int(bucket) in TAG_BEARING_BUCKETS

    def _tag_re(self):
        cached = getattr(self, "_tag_re_cache", None)
        if cached is None:
            alt = "|".join(re.escape(self.tag_keys[a]) for a in self.activities)
            cached = re.compile(rf"_({alt})(\d+\.\d+)")
            object.__setattr__(self, "_tag_re_cache", cached)
        return cached

    @staticmethod
    def case_id_from_stem(stem: str) -> str:
        """Case id = everything before '_interval_'."""
        return stem.split("_interval_")[0]

    # ------------------------------------------------------------------ stage 2
    def _activity_state(self, activity: str, frac: float) -> str:
        """'positive' | 'negative' | 'ambiguous' for one activity."""
        if frac >= self.thresholds[activity] - _FRAC_EPS:
            return "positive"
        if frac <= self.weak_threshold + _FRAC_EPS:
            return "negative"
        return "ambiguous"

    def _states_from_fracs(self, fracs: dict[str, float]) -> dict[str, str]:
        return {a: self._activity_state(a, float(fracs.get(a, 0.0)))
                for a in self.activities}

    def _states_from_bucket(self, bucket: int, dir_activities) -> dict[str, str] | None:
        """Per-activity state for an UNTAGGED clip, from bucket + directory.

        The bucket IS the processor's labelling decision, so it carries the same
        information the fractions would have been thresholded into — just
        already thresholded, at the cut the processor used:

            0 / 4 / 5   nothing annotated            -> every activity negative
            1 / 2 / 3   this activity at/above its threshold, the others below
                        `weak_threshold` (the processor's purity guard)
                                                     -> positive / negative
            7           every activity in the combo is above threshold; a further
                        activity may be weakly present with no evidence either way
                                                     -> positive / AMBIGUOUS
            6 / 8       every activity named is present but BELOW threshold
                                                     -> ambiguous / negative

        Returns None when the directory names no activity for a bucket that
        needs one — the identity is then genuinely unrecoverable and the clip
        must be dropped rather than guessed at.
        """
        present = set(dir_activities)
        if bucket in (0, 4, 5):
            return {a: "negative" for a in self.activities}
        if not present:
            return None
        if bucket in (1, 2, 3):
            return {a: ("positive" if a in present else "negative") for a in self.activities}
        if bucket == 7:
            return {a: ("positive" if a in present else "ambiguous") for a in self.activities}
        if bucket in (6, 8):
            return {a: ("ambiguous" if a in present else "negative") for a in self.activities}
        return None

    def resolve(self, bucket: int, fracs: dict[str, float], tagged: bool = True,
                dir_activities=()) -> ClipLabel | None:
        """(bucket, fractions) -> ClipLabel, or None if the clip is dropped.

        `tagged=False` means the clip's filename carries no fraction tags although
        its bucket requires them, so the fractions are UNKNOWN (not zero) and the
        label comes from the bucket and the clip's directory instead — see
        `_states_from_bucket`. Two consequences worth knowing:

          * `thresholds` and `weak_threshold` are inert for such clips. They were
            effectively applied when the clips were cut and cannot be re-tuned.
          * `overlap_resolution: dominant` has nothing to rank by, so a multiclass
            clip with two positives is dropped rather than assigned arbitrarily.
        """
        if not self.keeps_bucket(bucket):
            return None

        states = (self._states_from_fracs(fracs) if tagged
                  else self._states_from_bucket(int(bucket), dir_activities))
        if states is None:
            return None

        targets, mask = [], []
        for a in self.activities:
            state = states[a]
            if state == "positive":
                targets.append(1.0)
                mask.append(1.0)
            elif state == "negative":
                targets.append(0.0)
                mask.append(1.0)
            elif self.ambiguous == "drop":
                return None
            elif self.ambiguous == "negative":
                targets.append(0.0)
                mask.append(1.0)
            else:  # mask
                targets.append(0.0)
                mask.append(0.0)

        if self.is_multilabel:
            if not any(mask):
                return None  # nothing left to supervise
            return ClipLabel(targets=tuple(targets), mask=tuple(mask))

        # ---- multiclass: collapse to exactly one class index --------------
        positives = [i for i, (t, m) in enumerate(zip(targets, mask)) if t == 1.0 and m == 1.0]
        if len(positives) == 1:
            return ClipLabel(class_index=1 + positives[0])
        if not positives:
            # "no activity" cannot be asserted while some activity is unknown.
            if any(m == 0.0 for m in mask):
                return None
            return ClipLabel(class_index=0)
        if self.overlap_resolution == "dominant" and tagged:
            best = max(positives, key=lambda i: float(fracs.get(self.activities[i], 0.0)))
            return ClipLabel(class_index=1 + best)
        return None

    def evidence_mass(self, bucket: int, fracs: dict[str, float], tagged: bool = True,
                      dir_activities=()) -> dict[str, float]:
        """Per-activity "how much of this 3 s window was activity a", for the SPLIT.

        Tagged clips report their real fractions. Untagged ones report a nominal
        value from the bucket — the split needs a comparable NUMBER per activity
        to balance sites against each other, and a site whose fractions are all
        zero would otherwise be balanced on clip counts alone, ignoring its
        activity mix entirely.

        Config-independent by construction (no threshold is consulted), which is
        what keeps the split reproducible across data-config edits.
        """
        if tagged:
            return {a: float(fracs.get(a, 0.0)) for a in self.activities}
        present = set(dir_activities)
        if not present:
            return {a: 0.0 for a in self.activities}
        nominal = NOMINAL_PARTIAL_FRAC if int(bucket) in (6, 8) else NOMINAL_STRONG_FRAC
        return {a: (nominal if a in present else 0.0) for a in self.activities}

    def resolve_stem(self, stem: str) -> ClipLabel | None:
        """Convenience: filename stem -> ClipLabel (both stages at once)."""
        bucket, fracs, tagged = self.parse_stem(stem)
        if bucket is None:
            return None
        return self.resolve(bucket, fracs, tagged=tagged)

    # ------------------------------------------------------------------ report
    def describe(self) -> str:
        kept = self.kept_buckets()
        lines = [
            f"DataSpec ({self.source or 'inline'})",
            f"  task              : {self.task}  ({self.activation} head, "
            f"{'masked BCE' if self.is_multilabel else 'weighted CE'})",
            f"  outputs           : {self.num_classes} — {', '.join(self.class_names)}",
            "  LABEL cut (frac of the 3 s window that must be the activity)",
            "    positive if     : " + ", ".join(
                f"{a}>={self.thresholds[a]:.2f}" for a in self.activities),
            f"    negative if     : frac <= {self.weak_threshold:.2f}",
            f"  ambiguous band    : {self.ambiguous}",
            f"  buckets kept      : {kept}  (dropped: "
            f"{[b for b in sorted(BUCKET_NAMES) if b not in kept]})",
        ]
        if not self.is_multilabel:
            lines.append(f"  >=2 activities    : {self.overlap_resolution}")
        else:
            lines.append("  PREDICT cut (sigmoid probability, unrelated to the label cut)")
            lines.append("    predict 1 if    : " + ", ".join(
                f"p({a})>={self.decision_thresholds.get(a, 0.5):.2f}"
                for a in self.activities))
        return "\n".join(lines)


def load_spec(path: str | Path | None = None) -> DataSpec:
    """Module-level convenience wrapper around DataSpec.load."""
    return DataSpec.load(path)


def spec_from_checkpoint(saved: dict, fallback: str | Path | None = None) -> DataSpec:
    """Recover the DataSpec a checkpoint was trained with.

    Checkpoints written by training.py carry it under "data_spec". Older
    checkpoints do not; those fall back to the given YAML (or configs/data.yaml)
    and MUST be verified by hand — a mismatched spec silently changes the head
    width and the meaning of every logit.
    """
    raw = saved.get("data_spec")
    if raw:
        return DataSpec.from_dict(raw, source="checkpoint")
    return DataSpec.load(fallback)
