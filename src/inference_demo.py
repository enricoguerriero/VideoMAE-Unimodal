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

from .infer_video import (build_model, list_test_cases, resolve_media,
                          run_inference, windows_to_per_second)

logger = logging.getLogger(__name__)

# Categorical slots 1-3 of the validated reference palette (light surface), in
# fixed order — stimulation, ventilation, suction. Never cycled, never reordered:
# a colour belongs to an activity, not to a position in some list.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2df"

PLOT_H_PER_PANEL = 112      # px per activity panel (incl. its title row)
PLOT_PAD = 66               # px for the shared x axis + label
TARGET_W = 960              # output width; video and plot are both scaled to it


def build_plot_image(per_second, spec, width, height, title):
    """Render the static probability panels once, as an RGB array.

    Drawn ONCE and reused for every frame — only the playhead moves, and that is
    a cheap line drawn per frame with cv2. Rendering matplotlib per frame would
    take longer than the inference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acts = list(spec.activities)
    secs = np.array([e["t"] for e in per_second], dtype=float)
    probs = np.array([e["probs"] for e in per_second], dtype=float)
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
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
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

        ax.set_facecolor(SURFACE)
        ax.set_ylim(-0.04, 1.04)
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


def render(video_path, per_second, spec, out_path, fps, title):
    """Write <video on top, probability panels underneath> with a moving playhead."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or TARGET_W
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    vid_h = max(1, int(round(src_h * TARGET_W / src_w)))
    plot_h = PLOT_H_PER_PANEL * len(spec.activities) + PLOT_PAD

    plot, px0, px1, t0, t1 = build_plot_image(per_second, spec, TARGET_W, plot_h, title)
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


def write_csv(per_second, spec, path):
    offset = 0 if spec.is_multilabel else 1
    thresholds = (spec.sigmoid_thresholds() if spec.is_multilabel
                  else [0.5] * len(spec.activities))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["second"] + [f"p_{a}" for a in spec.activities]
                   + [f"active_{a}" for a in spec.activities])
        for e in per_second:
            p = [float(e["probs"][i + offset]) for i in range(len(spec.activities))]
            w.writerow([e["t"]] + [round(v, 4) for v in p]
                       + [int(v >= t) for v, t in zip(p, thresholds)])


def pick_random_case(test_csv: Path, rng: random.Random):
    """A random case from a per-site test CSV, resolved to its episode video."""
    if not test_csv.exists():
        logger.warning(f"{test_csv} not found — skipping this site.")
        return None, None
    cases = list_test_cases(test_csv)
    rng.shuffle(cases)
    for c in cases:
        video, _ = resolve_media(c["anchor"], c["case_id"])
        if video:
            return c["case_id"], video
    logger.warning(f"no episode video resolved for any case in {test_csv} — "
                   f"is the sibling Unprocessed_data/videos tree present?")
    return None, None


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="VideoMAE", choices=["VideoMAE", "VideoMAEGiant"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--haydom-video", default=None, help="Episode video (default: random from data/test_haydom.csv)")
    ap.add_argument("--drc-video", default=None, help="Episode video (default: random from data/test_drc.csv)")
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

    targets = []
    for site, given, csv_name in [("haydom", args.haydom_video, "test_haydom.csv"),
                                  ("drc", args.drc_video, "test_drc.csv")]:
        if given:
            video = Path(given).expanduser()
            if not video.exists():
                raise SystemExit(f"{site}: video not found: {video}")
            targets.append((site, video.stem, video))
        else:
            case_id, video = pick_random_case(args.splits_dir / csv_name, rng)
            if video:
                targets.append((site, case_id, video))
    if not targets:
        raise SystemExit("no episodes to run — pass --haydom-video/--drc-video, or "
                         "check that the test CSVs and Unprocessed_data tree exist.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for site, case_id, video in targets:
        dest = args.out_dir / f"{site}_{case_id}"
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== {site}: {case_id} ===\n  source: {video}")

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

        write_csv(per_second, spec, dest / "probabilities.csv")
        out_mp4 = dest / "annotated.mp4"
        title = f"{site.upper()} · case {case_id} · per-second activity probability"
        n = render(source_copy, per_second, spec, out_mp4, fps, title)
        logger.info(f"  {n} frames -> {out_mp4}")
        logger.info(f"  per-second probabilities -> {dest / 'probabilities.csv'}")

    print(f"\nDone. Everything is under {args.out_dir}/")
    print("Copy it off the VM and play the annotated.mp4 files in any player:")
    print(f"  scp -r <user>@<vm>:{args.out_dir.resolve()} .")


if __name__ == "__main__":
    main()
