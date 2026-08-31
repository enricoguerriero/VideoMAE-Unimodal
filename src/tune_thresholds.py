#!/usr/bin/env python3
"""
tune_thresholds.py

Pick the per-activity sigmoid cuts for a multilabel checkpoint, from a
`scores_*.npz` written by src/test.py.

    # 1. score VALIDATION as if it were a test set
    bash scripts/test.sh VideoMAE checkpoints/<ckpt>.pt 0 "" \
        --test_data val=data/validation.csv

    # 2. tune on that, and paste the result into the data config
    python -m src.tune_thresholds results/scores_<ckpt>_val_<ts>.npz --write configs/data_multilabel.yaml

--------------------------------------------------------------------------
Why 0.5 is the wrong default here
--------------------------------------------------------------------------
Training uses `pos_weight` (class_weighting: sqrt_inv_freq), which deliberately
inflates the odds of the positive class so rare activities are not ignored. A
model fitted that way does not output calibrated probabilities: with weight w the
balanced operating point sits near w/(1+w), not 0.5. Left at 0.5 a rare class
collapses to "predict positive everywhere" — recall 1, precision = prevalence,
F1 = 2p/(1+p) — which looks like the model learned nothing even when its ranking
is good. Average precision, printed here alongside, is what tells the two apart:
it is threshold-free, so a high AP with a terrible F1 IS this problem.

--------------------------------------------------------------------------
Tune on validation, report on test
--------------------------------------------------------------------------
Thresholds fitted on the same clips you report are threshold-shopped, and the
number stops meaning anything. This script warns when the file it is given does
not look like validation. Report the thresholded F1 AND the AP, and say which
split the cuts came from.

Each activity is optimised independently — in multilabel the per-activity
decisions do not interact — by sweeping every distinct score as a candidate cut
(O(n log n) via cumulative counts, so it is exact, not a grid approximation).
Masked entries are excluded: a clip that does not supervise an activity must not
vote on its threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


def fbeta_curve(y: np.ndarray, p: np.ndarray, beta: float):
    """Exact best F-beta over all cuts. Returns (threshold, fbeta, prec, rec).

    Sorting by score descending and walking the cumulative true/false positive
    counts evaluates every distinct cut in one pass, so this finds the true
    optimum rather than the best point on some arbitrary grid.
    """
    order = np.argsort(-p, kind="stable")
    y_s, p_s = y[order], p[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1.0 - y_s)
    n_pos = float(y.sum())
    fn = n_pos - tp
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    f = np.divide((1 + b2) * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    k = int(np.argmax(f))

    # Put the cut between the last kept score and the first rejected one, so it
    # does not sit exactly on a sample and flip with floating-point noise.
    hi = p_s[k]
    lo = p_s[k + 1] if k + 1 < len(p_s) else 0.0
    thr = float((hi + lo) / 2) if lo < hi else float(hi)
    prec = tp[k] / max(tp[k] + fp[k], 1e-12)
    rec = tp[k] / max(n_pos, 1e-12)
    return thr, float(f[k]), float(prec), float(rec)


def score_at(y: np.ndarray, p: np.ndarray, thr: float, beta: float):
    pred = (p >= thr).astype(float)
    tp = float((pred * y).sum())
    fp = float((pred * (1 - y)).sum())
    fn = float(((1 - pred) * y).sum())
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return ((1 + b2) * tp / denom) if denom > 0 else 0.0


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold-free ranking quality — the number that says whether the model
    learned the class at all, independent of where the cut is put."""
    order = np.argsort(-p, kind="stable")
    y_s = y[order]
    tp = np.cumsum(y_s)
    prec = tp / np.arange(1, len(y_s) + 1)
    n_pos = float(y.sum())
    return float((prec * y_s).sum() / n_pos) if n_pos else float("nan")


def write_thresholds(path: Path, thresholds: dict[str, float]) -> bool:
    """Rewrite the `decision_thresholds:` block in place, comments intact.

    Only the activity lines inside that block are touched, and only when every
    activity is found exactly once. Anything unexpected leaves the file alone —
    a config is hand-maintained and worth more than the convenience.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"^decision_thresholds:\s*$", l)), None)
    if start is None:
        return False
    seen, i = set(), start + 1
    while i < len(lines):
        m = re.match(r"^(\s+)([A-Za-z_][\w]*):\s*[\d.]+\s*$", lines[i])
        if not m:
            if lines[i].strip() and not lines[i].lstrip().startswith("#"):
                break          # end of the block
            i += 1
            continue
        indent, name = m.group(1), m.group(2)
        if name not in thresholds:
            return False
        lines[i] = f"{indent}{name}: {thresholds[name]:.3f}\n"
        seen.add(name)
        i += 1
    if seen != set(thresholds):
        return False
    path.write_text("".join(lines))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", type=Path, help="scores_*.npz from src/test.py")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="F-beta to maximise. 1 = F1; >1 favours recall (2.0 is a "
                         "reasonable choice when missing an event costs more than a "
                         "false alarm); <1 favours precision.")
    ap.add_argument("--write", type=Path, default=None,
                    help="Data config YAML whose `decision_thresholds:` block to update "
                         "in place. Without this the block is only printed.")
    ap.add_argument("--allow-test", action="store_true",
                    help="Silence the warning about tuning on a non-validation split.")
    args = ap.parse_args()

    d = np.load(args.scores, allow_pickle=False)
    task = str(d["task"])
    if task != "multilabel":
        raise SystemExit(f"{args.scores} is a {task} run. decision_thresholds only "
                         f"apply to multilabel — multiclass predicts by argmax and has "
                         f"no cut to tune.")

    classes = [str(c) for c in d["classes"]]
    logits, labels, masks = d["logits"], d["labels"], d["masks"]
    probs = 1.0 / (1.0 + np.exp(-logits))

    if "val" not in args.scores.stem and not args.allow_test:
        print(f"[WARN] {args.scores.name} does not look like a validation run. "
              f"Thresholds fitted on\n       the split you then report are "
              f"threshold-shopped. Score data/validation.csv\n       instead "
              f"(--test_data val=data/validation.csv), or pass --allow-test.\n")

    print(f"tuning on {len(labels):,} clips from {args.scores.name}  (F{args.beta:g})\n")
    header = (f"{'activity':<16}{'supervised':>11}{'pos':>9}{'prev':>8}"
              f"{'thr*':>8}{'F@thr*':>9}{'F@0.50':>9}{'prec':>8}{'rec':>8}{'AP':>8}")
    print(header)
    print("-" * len(header))

    chosen = {}
    for i, name in enumerate(classes):
        sup = masks[:, i] == 1.0
        y, p = labels[sup, i].astype(float), probs[sup, i].astype(float)
        if y.sum() == 0:
            print(f"{name:<16}{int(sup.sum()):>11,}{0:>9}{'-':>8}  no positives — "
                  f"threshold left unchanged")
            continue
        thr, f_best, prec, rec = fbeta_curve(y, p, args.beta)
        f_half = score_at(y, p, 0.5, args.beta)
        chosen[name] = thr
        print(f"{name:<16}{int(sup.sum()):>11,}{int(y.sum()):>9,}"
              f"{y.mean():>8.3f}{thr:>8.3f}{f_best:>9.4f}{f_half:>9.4f}"
              f"{prec:>8.3f}{rec:>8.3f}{average_precision(y, p):>8.4f}")

    if not chosen:
        raise SystemExit("\nno activity had a positive example — nothing to tune.")

    print("\nA large gap between F@thr* and F@0.50 with a healthy AP means the model "
          "\nranks well and only the cut was wrong. A low AP means no threshold saves "
          "\nit — that is a training problem, not a calibration one.\n")
    print("decision_thresholds:")
    for name in classes:
        print(f"  {name}: {chosen.get(name, 0.5):.3f}")

    if args.write:
        if write_thresholds(args.write, chosen):
            print(f"\n[written] {args.write} — decision_thresholds updated in place.")
            print("Re-score the TEST sets with this config to see the effect:\n"
                  f"  bash scripts/test.sh VideoMAE <ckpt>.pt 0 {args.write}")
        else:
            print(f"\n[SKIPPED] could not find a `decision_thresholds:` block in "
                  f"{args.write} listing exactly {sorted(chosen)}.\n"
                  f"          Paste the block above by hand.")


if __name__ == "__main__":
    main()
