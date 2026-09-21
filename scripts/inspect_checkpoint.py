#!/usr/bin/env python3
"""
inspect_checkpoint.py — what is inside a .pt, without loading a backbone.

Every checkpoint records the DataSpec it was trained with, the training config
and its best metrics. When a run fails with "checkpoint head emits 4 logits but
this run's data config asks for 3", this is the two-second way to see which
checkpoint you actually have and which data config it belongs to.

READ-ONLY. Usage:

    python scripts/inspect_checkpoint.py checkpoints/*.pt
    python scripts/inspect_checkpoint.py models/VideoMAE_final_*.pt --verbose

Prints one block per checkpoint: logit count, task, class names, label source,
the data config and training data it came from, whether it is a `--ronald` run,
and the validation metrics it was saved at.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ModuleNotFoundError as exc:                           # pragma: no cover
    raise SystemExit(
        f"{exc.name} is required. Run this in the environment training uses.") from exc


def head_width(saved: dict):
    """Logits the stored head actually emits, read from its output layer.

    The ground truth for "what does this checkpoint predict" — the stored spec
    says what it was MEANT to be, this says what the weights are. They agree
    unless a checkpoint was hand-edited, and when they disagree that is the
    finding.
    """
    head = saved.get("classifier") or {}
    idxs = [int(k.split(".")[1]) for k in head
            if len(k.split(".")) > 2 and k.split(".")[1].isdigit()]
    if not idxs:
        return None
    w = head.get(f"seq.{max(idxs)}.weight")
    return None if w is None else int(w.shape[0])


def describe(path: Path, verbose: bool) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved.get("config", {}) or {}
    spec = saved.get("data_spec") or {}
    n_head = head_width(saved)

    print(f"\n{'=' * 74}\n{path}\n{'=' * 74}")
    if not spec:
        print("  data_spec        : ABSENT — trained before it was recorded.")
        print("                     Its task/classes are a guess from the data")
        print("                     config on disk; pass --data-config explicitly.")
    else:
        acts = spec.get("activities", [])
        task = spec.get("task")
        expected = len(acts) if task == "multilabel" else len(acts) + 1
        print(f"  task             : {task}")
        print(f"  activities       : {acts}")
        print(f"  logits (spec)    : {expected}"
              + ("" if task == "multilabel"
                 else f"   ({spec.get('negative_class')} + {len(acts)} activities)"))
        print(f"  label_source     : {spec.get('label_source', 'evidence')}")
        if spec.get("min_visible_fraction"):
            print(f"  visible gate     : >= {spec['min_visible_fraction']} "
                  f"(unknown: {spec.get('unknown_visibility')})")
        if spec.get("decision_thresholds"):
            print(f"  decision thresh. : {spec['decision_thresholds']}")

    print(f"  logits (weights) : {n_head}")
    if spec and n_head is not None:
        acts = spec.get("activities", [])
        expected = len(acts) if spec.get("task") == "multilabel" else len(acts) + 1
        if expected != n_head:
            print("  !! the stored spec and the stored weights DISAGREE — this "
                  "checkpoint is inconsistent")

    print(f"  data config      : {cfg.get('data_config')}")
    print(f"  trained on       : {cfg.get('train_data')}")
    if cfg.get("ronald"):
        print("  --ronald         : YES — Ronald Paleczny's manifests")
        print("                     score with: scripts/test.sh <model> <ckpt>")
        print("                     episode  : RONALD=1 scripts/infer_video.sh ...")
    if cfg.get("sites"):
        print(f"  --sites          : {cfg['sites']}")
    print(f"  class_weighting  : {cfg.get('class_weighting', 'sqrt_inv_freq')}")
    if saved.get("epoch") is not None:
        print(f"  saved at epoch   : {saved['epoch']}   val_loss="
              f"{saved.get('val_loss')}")

    m = saved.get("metrics") or {}
    if m:
        keys = [k for k in ("macro/f1", "macro/ap", "macro/accuracy") if k in m]
        keys += sorted(k for k in m if k.endswith("/f1") and not k.startswith("macro"))
        shown = keys if verbose else keys[:8]
        print("  metrics          : " + ", ".join(
            f"{k}={m[k]:.4f}" for k in shown if isinstance(m[k], (int, float))))
    if verbose:
        print(f"  config keys      : {sorted(cfg)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args()

    bad = 0
    for path in a.checkpoints:
        if not path.exists():
            print(f"\n[missing] {path}")
            bad += 1
            continue
        try:
            describe(path, a.verbose)
        except Exception as exc:                             # noqa: BLE001
            print(f"\n[unreadable] {path}: {type(exc).__name__}: {exc}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
