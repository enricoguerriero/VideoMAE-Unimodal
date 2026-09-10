#!/usr/bin/env python3
"""
inference_demo.py

Side-by-side qualitative demo: ONE episode from each hospital, rendered as a
video with the model's per-second activity probabilities plotted underneath.

    python -m src.inference_demo --model VideoMAE --model_path checkpoints/<ckpt>.pt

With no video paths it picks a random case from each site's test set
(data/test_haydom.csv, data/test_drc.csv) and resolves the full-episode video
from the sibling `Unprocessed_data` tree, the same way src/infer_video.py does.
Give --haydom-video / --drc-video to pin specific episodes.

Everything lands in `inference_output/` (--out-dir):

    inference_output/
      haydom_<case>/
        source.mp4          copy of the original episode video
        annotated.mp4       the episode with the probability plot underneath
        probabilities.csv   one row per second, one column per activity
      drc_<case>/ ...

--------------------------------------------------------------------------
Why the plot looks the way it does
--------------------------------------------------------------------------
Three activities, each with its own decision threshold, over a long timeline.
Drawing them as three lines on one axis makes them cross and occlude, and a
single threshold line cannot serve three different cuts. So it is SMALL
MULTIPLES: one thin panel per activity, each with its own threshold drawn where
it actually sits, and the spans where the model says "performed" shaded in.
Each panel is titled with its activity name, so identity never depends on colour
alone.

The inference itself is src/infer_video.py's — a 3 s window at 1 s stride, each
window's probabilities assigned to the second nearest its centre — so this demo
and the quantitative evaluation cannot drift apart.
"""

from argparse import ArgumentParser
import csv
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from .infer_video import (build_model, gt_per_second, list_test_cases,
                          load_gt_intervals, resolve_media, run_inference,
                          windows_to_per_second)
from .data.annotations import AnnotationIndex

logger = logging.getLogger(__name__)

# Categorical slots of the validated reference palette (light surface), in
# fixed order. A colour belongs to an activity, not to a position in some list,
# so this is indexed directly and NOT cycled: with `%` the 5-activity per-device
# config drew suction_bulb in stimulation's blue and suction_tube in
# ventilation's orange. Beyond this many activities the panels fall back to a
# neutral ink rather than aliasing onto a colour that already means something.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8c5cd6", "#c2405a"]
GT_INK = "#0b0b0b"          # ground truth is drawn in ink, never in a series colour
GT_BAND = (-0.17, -0.07)    # y range of the truth ribbon, below the 0 gridline
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2df"

PLOT_H_PER_PANEL = 112      # px per activity panel (incl. its title row)
PLOT_PAD = 66               # px for the shared x axis + label
TARGET_W = 960              # output width; video and plot are both scaled to it


def build_plot_image(per_second, spec, width, height, title, gt_second=None):
    """Render the static probability panels once, as an RGB array.

    Drawn ONCE and reused for every frame — only the playhead moves, and that is
    a cheap line drawn per frame with cv2. Rendering matplotlib per frame would
    take longer than the inference.

    `gt_second` (optional, from infer_video.gt_per_second) adds the annotated
    truth as a solid ink ribbon under each panel. It is deliberately NOT a
    second translucent span like the prediction: where the two agree the ribbon
    sits inside the shaded span and you see one block, and where they disagree
    the ribbon sticks out past the shading or the shading floats with no ribbon
    beneath it. Overlaying two translucent bands would make agreement — the
    common case — the hardest thing to read. The ribbon is binary and derived
    with the SAME threshold rule as the training targets, so it is the label the
    model was asked to reproduce, not the raw annotation span.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acts = list(spec.activities)
    secs = np.array([e["t"] for e in per_second], dtype=float)
    probs = np.array([e["probs"] for e in per_second], dtype=float)

    # Truth as a boolean per (second, activity). The two tracks are generated
    # from the same duration so they are the same length, but they are aligned
    # by index and truncated rather than trusted to match.
    gt = None
    if gt_second:
        n = min(len(per_second), len(gt_second))
        gt = np.zeros((len(secs), len(acts)), dtype=bool)
        for row, e in enumerate(gt_second[:n]):
            if "active" in e:                      # multilabel
                for i in e["active"]:
                    gt[row, i] = True
            elif e.get("label"):                   # multiclass: 0 = negative
                gt[row, e["label"] - 1] = True
    thresholds = (spec.sigmoid_thresholds() if spec.is_multilabel
                  else [0.5] * len(acts))
    # multiclass logits include the negative class at index 0; the activities we
    # plot are the remaining columns, in `activities` order.
    offset = 0 if spec.is_multilabel else 1

    dpi = 100
    fig, axes = plt.subplots(len(acts), 1, sharex=True, dpi=dpi,
                             figsize=(width / dpi, height / dpi))
    if len(acts) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for i, (ax, act) in enumerate(zip(axes, acts)):
        color = SERIES_COLORS[i] if i < len(SERIES_COLORS) else TEXT_SECONDARY
        p = probs[:, i + offset]
        thr = thresholds[i]

        # Spans the model calls "performed" — the binary decision, shown as
        # context behind the continuous probability rather than as a second line.
        active = p >= thr
        if active.any():
            edges = np.diff(active.astype(int))
            starts = list(np.where(edges == 1)[0] + 1) + ([0] if active[0] else [])
            ends = list(np.where(edges == -1)[0] + 1) + ([len(active)] if active[-1] else [])
            for a, b in zip(sorted(starts), sorted(ends)):
                ax.axvspan(secs[a], secs[min(b, len(secs) - 1)],
                           color=color, alpha=0.16, linewidth=0)

        ax.axhline(thr, color=TEXT_SECONDARY, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
        ax.plot(secs, p, color=color, lw=2.0, solid_capstyle="round")

        if gt is not None:
            ax.fill_between(secs, GT_BAND[0], GT_BAND[1], where=gt[:, i],
                            step="post", color=GT_INK, alpha=0.82, linewidth=0)
            ax.axhline(GT_BAND[1] + 0.015, color=GRID, lw=0.8)

        ax.set_facecolor(SURFACE)
        ax.set_ylim(GT_BAND[0] - 0.03 if gt is not None else -0.04, 1.04)
        ax.set_xlim(secs[0], secs[-1] if len(secs) > 1 else secs[0] + 1)
        ax.set_yticks([0, thr, 1])
        ax.set_yticklabels(["0", f"{thr:g}", "1"], fontsize=8, color=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)

        # Direct label ABOVE the panel, never inside it — a label placed over the
        # plot area collides with whatever the curve happens to do there. The
        # name is in text ink and the colour is carried by a swatch beside it,
        # so identity never rests on colour alone. That swatch is also the
        # relief the palette validator requires for the aqua slot, whose
        # contrast against the surface is below 3:1.
        ax.set_title(act, loc="left", fontsize=10.5, color=TEXT_PRIMARY, pad=5, x=0.022)
        ax.text(0.0, 1.045, "■", transform=ax.transAxes, color=color,
                fontsize=10, va="bottom", ha="left", clip_on=False)

    axes[-1].set_xlabel("time in episode (s)", fontsize=9, color=TEXT_SECONDARY)
    axes[-1].tick_params(axis="x", labelsize=8, colors=TEXT_SECONDARY, length=0)
    if gt is not None:
        title = f"{title}   ·   ink bar under each panel = annotated ground truth"
    fig.suptitle(title, fontsize=10, color=TEXT_SECONDARY, x=0.006, ha="left", y=0.992)

    fig.tight_layout(pad=0.9, rect=(0, 0, 1, 0.965))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    img = cv2.cvtColor(buf.copy(), cv2.COLOR_RGB2BGR)
    plt.close(fig)

    # Where t maps to in pixels, so the playhead lands on the right column.
    x0 = axes[-1].get_position().x0 * img.shape[1]
    x1 = axes[-1].get_position().x1 * img.shape[1]
    return img, float(x0), float(x1), float(secs[0]), float(secs[-1] if len(secs) > 1 else secs[0] + 1)


def render(video_path, per_second, spec, out_path, fps, title, gt_second=None):
    """Write <video on top, probability panels underneath> with a moving playhead."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or TARGET_W
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    vid_h = max(1, int(round(src_h * TARGET_W / src_w)))
    plot_h = PLOT_H_PER_PANEL * len(spec.activities) + PLOT_PAD

    plot, px0, px1, t0, t1 = build_plot_image(per_second, spec, TARGET_W, plot_h,
                                             title, gt_second)
    plot_h = plot.shape[0]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (TARGET_W, vid_h + plot_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open the writer for {out_path}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        panel = plot.copy()
        frac = 0.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        x = int(round(px0 + frac * (px1 - px0)))
        cv2.line(panel, (x, 0), (x, plot_h - 1), (11, 11, 11), 1, cv2.LINE_AA)
        out = np.vstack([cv2.resize(frame, (TARGET_W, vid_h),
                                    interpolation=cv2.INTER_AREA), panel])
        writer.write(out)
        n += 1
        if n % 2000 == 0:
            logger.info(f"    {n} frames written")
    cap.release()
    writer.release()
    return n


def gt_row(entry, n_act):
    """One ground-truth second -> a 0/1 list per activity, either task."""
    row = [0] * n_act
    if entry is None:
        return row
    if "active" in entry:                       # multilabel
        for i in entry["active"]:
            row[i] = 1
    elif entry.get("label"):                    # multiclass: 0 = negative class
        row[entry["label"] - 1] = 1
    return row


def write_csv(per_second, spec, path, gt_second=None):
    offset = 0 if spec.is_multilabel else 1
    n_act = len(spec.activities)
    thresholds = (spec.sigmoid_thresholds() if spec.is_multilabel
                  else [0.5] * n_act)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["second"] + [f"p_{a}" for a in spec.activities]
                  + [f"active_{a}" for a in spec.activities])
        if gt_second:
            header += [f"gt_{a}" for a in spec.activities]
        w.writerow(header)
        for k, e in enumerate(per_second):
            p = [float(e["probs"][i + offset]) for i in range(n_act)]
            row = ([e["t"]] + [round(v, 4) for v in p]
                   + [int(v >= t) for v, t in zip(p, thresholds)])
            if gt_second:
                row += gt_row(gt_second[k] if k < len(gt_second) else None, n_act)
            w.writerow(row)


def pick_random_case(test_csv: Path, rng: random.Random):
    """A random case from a per-site test CSV, resolved to its episode video.

    -> (case_id, video, annotation). `annotation` is None when the sibling
    Unprocessed_data/anot_files/ has no file for the case, which is the normal
    state at Haydom — that directory does not exist there. Pass --gt-dir to
    supply one from anywhere.
    """
    if not test_csv.exists():
        logger.warning(f"{test_csv} not found — skipping this site.")
        return None, None, None
    cases = list_test_cases(test_csv)
    rng.shuffle(cases)
    for c in cases:
        video, annotation = resolve_media(c["anchor"], c["case_id"])
        if video:
            return c["case_id"], video, annotation
    logger.warning(f"no episode video resolved for any case in {test_csv} — "
                   f"is the sibling Unprocessed_data/videos tree present?")
    return None, None, None


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="VideoMAE", choices=["VideoMAE", "VideoMAEGiant"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--haydom-video", default=None, help="Episode video (default: random from data/test_haydom.csv)")
    ap.add_argument("--drc-video", default=None, help="Episode video (default: random from data/test_drc.csv)")
    ap.add_argument("--gt-dir", action="append", default=None, metavar="DIR",
                    help="Extra annotation directory to draw ground truth from, "
                         "searched recursively and matched by case KEY (exact "
                         "stem or any run of >= 5 digits). Repeatable. Use it for "
                         "Haydom, whose Unprocessed_data/anot_files/ does not "
                         "exist and whose files are not named after the case — "
                         "e.g. --gt-dir /spo/LS-Haydom/Data/FullDataset/2023-2025/Annotations")
    ap.add_argument("--splits-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("inference_output"))
    ap.add_argument("--data-config", default=None,
                    help="Override the checkpoint's DataSpec (e.g. tuned decision_thresholds).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None, help="Fix the random case choice.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, spec = build_model(args.model, args.model_path, device, args.data_config)
    logger.info(spec.describe())

    # One index over every --gt-dir, consulted when the sibling anot_files/ has
    # nothing. Case-KEY matching is what makes this work at Haydom, where the
    # annotation files are not named after the case.
    gt_index = AnnotationIndex.from_roots(args.gt_dir) if args.gt_dir else None
    if gt_index is not None:
        logger.info(f"ground-truth fallback: {len(gt_index):,} case key(s) over "
                    f"{len(gt_index.dirs)} director(y/ies)")

    targets = []
    for site, given, csv_name in [("haydom", args.haydom_video, "test_haydom.csv"),
                                  ("drc", args.drc_video, "test_drc.csv")]:
        if given:
            video = Path(given).expanduser()
            if not video.exists():
                raise SystemExit(f"{site}: video not found: {video}")
            _, annotation = resolve_media(str(video), video.stem)
            targets.append((site, video.stem, video, annotation))
        else:
            case_id, video, annotation = pick_random_case(
                args.splits_dir / csv_name, rng)
            if video:
                targets.append((site, case_id, video, annotation))
    if not targets:
        raise SystemExit("no episodes to run — pass --haydom-video/--drc-video, or "
                         "check that the test CSVs and Unprocessed_data tree exist.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for site, case_id, video, annotation in targets:
        dest = args.out_dir / f"{site}_{case_id}"
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== {site}: {case_id} ===\n  source: {video}")
        if annotation is None and gt_index is not None:
            annotation = gt_index.lookup(case_id)
        if annotation:
            logger.info(f"  ground truth: {annotation}")
        else:
            logger.warning("  no annotation found — the plot will show predictions "
                           "only. Pass --gt-dir to point at an annotation store.")

        source_copy = dest / f"source{video.suffix.lower() if video.suffix else '.mp4'}"
        if not source_copy.exists():
            logger.info(f"  copying source -> {source_copy}")
            shutil.copy2(video, source_copy)

        fps, duration_s, results = run_inference(model, model.processor, video,
                                                 device, args.batch_size)
        per_second = windows_to_per_second(results, duration_s, spec)
        if not per_second:
            logger.warning(f"  no predictions for {case_id} — skipped")
            continue

        gt_second = None
        if annotation:
            gt_second = gt_per_second(load_gt_intervals(annotation, spec),
                                      duration_s, spec)

        write_csv(per_second, spec, dest / "probabilities.csv", gt_second)
        out_mp4 = dest / "annotated.mp4"
        title = f"{site.upper()} · case {case_id} · per-second activity probability"
        n = render(source_copy, per_second, spec, out_mp4, fps, title, gt_second)
        logger.info(f"  {n} frames -> {out_mp4}")
        logger.info(f"  per-second probabilities -> {dest / 'probabilities.csv'}")

    print(f"\nDone. Everything is under {args.out_dir}/")
    print("Copy it off the VM and play the annotated.mp4 files in any player:")
    print(f"  scp -r <user>@<vm>:{args.out_dir.resolve()} .")


if __name__ == "__main__":
    main()
