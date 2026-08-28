"""
infer_video.py

Run a trained VideoMAE / VideoMAEv2-giant checkpoint over an ENTIRE episode
video (not the pre-cut 3-second clips) and produce a synced viewer.

It reproduces the thesis' clip scheme at inference time — a 3-second window slid
with a 1-second stride over the whole video — and classifies each window. Each
window's prediction is mapped to a per-second timeline so you get one prediction
per second of the episode.

The task comes from the checkpoint's stored DataSpec (configs/data.yaml at
training time), so both modes are supported end to end:

    multiclass : one label per second (argmax over the softmax logits), drawn as
                 a single colour-coded timeline.
    multilabel : one INDEPENDENT on/off decision per activity per second (sigmoid
                 vs `decision_thresholds`), drawn as one timeline PER ACTIVITY so
                 co-occurring activities are visible as overlapping bars.

Outputs (into --out-dir, default: viewer_out/<case-or-video-stem>/):
    annotated.mp4      (with --render-video) STANDALONE video with the label +
                       timeline burned onto every frame — offline, no server
    predictions.json   per-second + per-window predictions + metadata
    predictions.csv    per-window (start,end,label,confidence,probs...)
    viewer.html        self-contained page: plays the video with the predicted
                       label following the playhead + a colour-coded timeline
    video.mp4          symlink (or copy with --copy-video) to the source video

Easiest use — pick a case from the test set (no paths to type):
    python -m src.infer_video --model VideoMAE --model_path <ckpt.pt> --render-video
This lists the cases in data/test.csv; once you pick one it auto-resolves the
full-episode video AND its annotation from the sibling `Unprocessed_data` tree
(see src/data/data_process.py). Add --case <case_id> to skip the menu.

Or point it at any video directly:
    python -m src.infer_video --model VideoMAE --model_path <ckpt.pt> \
        --video /path/to/<case_id>.mp4 [--annotation /path/to/<case_id>.txt] --render-video

OFFLINE / headless VM (no localhost): use --render-video. It writes a single
annotated.mp4 you copy off the VM (scp) and play in any media player (VLC) — no
browser or network needed.

ONLINE (a browser can reach the VM): use --serve instead for an interactive HTML
viewer at http://localhost:<port>/viewer.html (forward the port over SSH / VS Code
Remote). --serve uses a Range-capable server so scrubbing works.

Ground truth is overlaid automatically when an annotation file is found (the
5-column TSV from Unprocessed_data/anot_files), computed with the SAME rule the
training labels use — the per-activity share of each 3 s window, thresholded by
the data config. In multiclass that reduces to the thesis' `for_predict` rule
(dominant of the activities if >= its threshold, else the negative class).
--no-gt disables it.
"""

from argparse import ArgumentParser
import csv
import json
import logging
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

from src.utils import load_model
from src.data import DataSpec, spec_from_checkpoint

# Distinct, colour-blind-friendly-ish palette. The negative/"no activity" state
# is always grey; each activity takes the next palette colour in spec order, so
# the colours stay stable whether the run is multiclass or multilabel.
NEGATIVE_COLOR = "#6b7280"
ACTIVITY_PALETTE = ["#3b82f6", "#22c55e", "#f97316", "#a855f7", "#eab308", "#ec4899"]

WINDOW_S = 3        # clip length, seconds (thesis segment_size)
STRIDE_S = 1        # slide, seconds (thesis shift)
NUM_FRAMES = 16     # VideoMAE requirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task-driven presentation
# ---------------------------------------------------------------------------
def class_colors(spec):
    """{output index -> hex colour} in logit order."""
    if spec.is_multilabel:
        return {i: ACTIVITY_PALETTE[i % len(ACTIVITY_PALETTE)]
                for i in range(len(spec.activities))}
    return {0: NEGATIVE_COLOR, **{i + 1: ACTIVITY_PALETTE[i % len(ACTIVITY_PALETTE)]
                                  for i in range(len(spec.activities))}}


def track_specs(spec):
    """The timeline rows to draw.

    multiclass: a single 'argmax' row coloured by the predicted class.
    multilabel: one 'binary' row per activity, on when that activity fires —
                which is the only honest way to show two at once.
    """
    if spec.is_multilabel:
        return [{"name": a, "kind": "binary", "index": i}
                for i, a in enumerate(spec.activities)]
    return [{"name": "predicted", "kind": "argmax", "index": None}]


def entry_text(entry, spec):
    """The label chip's text for one per-second entry."""
    if spec.is_multilabel:
        names = [spec.activities[i] for i in entry.get("active", [])]
        return " + ".join(names) if names else f"no {spec.negative_class.replace('_', ' ')}"
    return spec.class_names[entry["label"]]


def entry_is_on(entry, track):
    """Is `track` lit for this per-second entry? (multilabel rows)"""
    return track["index"] in entry.get("active", [])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(model_name, model_path, device, data_config=None):
    """Load a trained checkpoint exactly like src/test.py.

    The DataSpec is read back from the checkpoint, so head width and output
    activation match how the model was trained. `data_config` overrides it — use
    that only deliberately, e.g. to re-score at different decision thresholds.
    """
    saved = torch.load(model_path, map_location=device, weights_only=False)
    config = saved.get("config", {})
    if data_config:
        spec = DataSpec.load(data_config)
        logger.warning(f"overriding the checkpoint's DataSpec with {data_config}")
    else:
        spec = spec_from_checkpoint(saved, config.get("data_config"))
        if "data_spec" not in saved:
            logger.warning("checkpoint has no stored DataSpec — falling back to the "
                           "data config on disk; verify it matches this checkpoint.")
    logger.info(spec.describe())
    model = load_model(model_name, spec=spec).to(device)
    model.load_classifier(saved, config)
    model.load_backbone(saved, config)
    if config.get("attention_pooling", False):
        model.load_attention_pooling(saved)
    model.eval()
    return model, config, spec


# ---------------------------------------------------------------------------
# Video windows + frame sampling
# ---------------------------------------------------------------------------
def build_windows(duration_s):
    """(start_s, end_s) at 1 s stride / 3 s window, covering the whole video."""
    windows = []
    start = 0.0
    while start + WINDOW_S <= duration_s + 1e-6:
        windows.append((start, start + WINDOW_S))
        start += STRIDE_S
    if not windows:
        windows.append((0.0, duration_s))            # video shorter than a window
    elif windows[-1][1] < duration_s - 1e-6:
        tail = max(0.0, duration_s - WINDOW_S)        # cover the trailing seconds
        windows.append((tail, duration_s))
    return windows


def read_window_frames(cap, start_s, end_s, fps):
    """Return NUM_FRAMES uniformly spaced RGB frames from [start_s, end_s)."""
    start_f = int(round(start_s * fps))
    end_f = int(round(end_s * fps))
    idxs = np.linspace(start_f, max(start_f, end_f - 1), NUM_FRAMES).astype(int)
    needed = sorted(set(int(i) for i in idxs))

    cap.set(cv2.CAP_PROP_POS_FRAMES, needed[0])
    grabbed, cur, ptr = {}, needed[0], 0
    while ptr < len(needed):
        ret, frame = cap.read()
        if not ret:
            break
        if cur == needed[ptr]:
            grabbed[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ptr += 1
        cur += 1

    frames, last = [], None
    for i in idxs:                     # keep temporal order, pad short reads
        f = grabbed.get(int(i), last)
        if f is not None:
            last = f
        frames.append(f)
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    while len(frames) < NUM_FRAMES:    # pad with the last decoded frame
        frames.append(frames[-1])
    return frames[:NUM_FRAMES]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, processor, video_path, device, batch_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps else 0.0
    logger.info(f"Video: {duration_s:.1f}s @ {fps:.2f} fps ({total_frames} frames)")

    windows = build_windows(duration_s)
    logger.info(f"{len(windows)} windows (win={WINDOW_S}s, stride={STRIDE_S}s)")

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    results = []          # (start, end, probs)
    buf_meta, buf_px = [], []

    def flush():
        if not buf_px:
            return
        pixel_values = torch.cat(buf_px, dim=0).to(device)
        if device.type == "cuda":
            with autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixel_values=pixel_values)
        else:
            logits = model(pixel_values=pixel_values)
        # model.probs applies the task's activation: softmax (multiclass) or
        # independent sigmoids (multilabel, so rows need not sum to 1).
        probs = model.probs(logits.float()).cpu().numpy()
        for (s, e), p in zip(buf_meta, probs):
            results.append((s, e, p))
        buf_meta.clear()
        buf_px.clear()

    for wi, (s, e) in enumerate(windows):
        frames = read_window_frames(cap, s, e, fps)
        if frames is None:
            logger.warning(f"window {wi} [{s:.1f},{e:.1f}] decoded 0 frames — skipped")
            continue
        inputs = processor(frames, return_tensors="pt")
        buf_meta.append((s, e))
        buf_px.append(inputs.pixel_values)   # (1, 16, 3, 224, 224)
        if len(buf_px) >= batch_size:
            flush()
        if (wi + 1) % 50 == 0:
            logger.info(f"  {wi + 1}/{len(windows)} windows")
    flush()
    cap.release()
    return fps, duration_s, results


# ---------------------------------------------------------------------------
# Per-second mapping
# ---------------------------------------------------------------------------
def probs_to_entry(sec, probs, spec, thresholds):
    """One per-second (or per-window) prediction record, shaped by the task.

    multiclass: {"t", "label", "conf", "probs"} — argmax and its softmax prob.
    multilabel: {"t", "active", "conf", "probs"} — every activity clearing its
                own sigmoid threshold. `active` may hold 0, 1 or several indices;
                an empty list is "no activity", which is a genuine prediction
                here rather than a class.
    """
    probs = [round(float(x), 4) for x in probs]
    if spec.is_multilabel:
        active = [i for i, p in enumerate(probs) if p >= thresholds[i]]
        conf = max((probs[i] for i in active), default=max(probs))
        return {"t": sec, "active": active, "conf": round(float(conf), 4), "probs": probs}
    label = int(np.argmax(probs))
    return {"t": sec, "label": label, "conf": probs[label], "probs": probs}


def windows_to_per_second(results, duration_s, spec):
    """Assign each second the prediction of the window whose centre is nearest."""
    if not results:
        return []
    thresholds = spec.sigmoid_thresholds() if spec.is_multilabel else None
    centers = np.array([(s + e) / 2.0 for s, e, _ in results])
    per_second = []
    for sec in range(int(math.ceil(duration_s))):
        target = sec + 0.5
        j = int(np.argmin(np.abs(centers - target)))
        per_second.append(probs_to_entry(sec, results[j][2], spec, thresholds))
    return per_second


# ---------------------------------------------------------------------------
# Optional ground truth (5-column annotation TSV)
# ---------------------------------------------------------------------------
def load_gt_intervals(annotation_path, spec):
    """Parse the 5-col TSV into {activity_index: [(start_ms, end_ms), ...]}.

    Which `Event` string maps to which activity comes from the data config
    (`annotation_events`, defaulting to the title-cased activity name), so a site
    that spells an event differently is a config change, not a code change.
    """
    event_to_idx = spec.event_to_activity_index()
    intervals = {i: [] for i in range(len(spec.activities))}
    with open(annotation_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            event, start, end = parts[0], parts[1], parts[2]
            if len(parts) >= 5 and parts[4] == "Newborn visible in video frame":
                continue
            idx = event_to_idx.get(event)
            if idx is None:
                continue
            try:
                intervals[idx].append((int(start), int(end)))
            except ValueError:
                continue
    return intervals


def _overlap_ms(a0, a1, ivs):
    return sum(min(a1, e) - max(a0, s) for s, e in ivs if a0 < e and a1 > s)


def gt_per_second(intervals, duration_s, spec):
    """Reference labels per second, using the SAME rule as the training targets.

    For each second's 3 s window, measure each activity's share of the window and
    threshold it with the data config's `thresholds` — the same numbers
    build_manifest/DataSpec use on the pre-cut clips.

    multilabel: every activity clearing its threshold is active (co-occurrence
                shows up here too).
    multiclass: the dominant activity if it clears its threshold, else the
                negative class — i.e. the thesis' `for_predict` rule, with the
                threshold read from config instead of hard-coded at 0.50.
    """
    out = []
    win_ms = WINDOW_S * 1000
    n_act = len(spec.activities)
    for sec in range(int(math.ceil(duration_s))):
        s_ms = sec * 1000
        e_ms = s_ms + win_ms
        fracs = [_overlap_ms(s_ms, e_ms, intervals[i]) / win_ms for i in range(n_act)]
        over = [i for i in range(n_act)
                if fracs[i] >= spec.thresholds[spec.activities[i]]]
        if spec.is_multilabel:
            out.append({"t": sec, "active": over,
                        "probs": [round(f, 4) for f in fracs]})
        else:
            best = max(over, key=lambda i: fracs[i]) if over else None
            out.append({"t": sec, "label": 0 if best is None else best + 1,
                        "probs": [round(f, 4) for f in fracs]})
    return out


# ---------------------------------------------------------------------------
# Serving (Range-capable, so video scrubbing works)
# ---------------------------------------------------------------------------
def serve(directory, port):
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class RangeHandler(SimpleHTTPRequestHandler):
        def send_head(self):
            rng = self.headers.get("Range")
            if rng is None:
                return super().send_head()
            path = self.translate_path(self.path)
            try:
                f = open(path, "rb")
            except OSError:
                self.send_error(404, "File not found")
                return None
            fs = os.fstat(f.fileno())
            size = fs[6]
            try:
                unit, rangespec = rng.split("=")
                first, last = rangespec.split("-")
                first = int(first)
                last = int(last) if last else size - 1
            except ValueError:
                self.send_error(400, "Invalid Range")
                f.close()
                return None
            last = min(last, size - 1)
            length = last - first + 1
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            f.seek(first)
            self._range_remaining = length
            return f

        def copyfile(self, source, outputfile):
            remaining = getattr(self, "_range_remaining", None)
            if remaining is None:
                return super().copyfile(source, outputfile)
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)

    handler = functools.partial(RangeHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    logger.info(f"Serving {directory} at http://localhost:{port}/viewer.html  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped")
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Offline output: burn the predictions into a standalone .mp4
# ---------------------------------------------------------------------------
def _hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _fit_text(text, font, scale, max_w):
    """Trim `text` until it fits `max_w` pixels (empty string if nothing fits)."""
    if cv2.getTextSize(text, font, scale, 1)[0][0] <= max_w:
        return text
    for n in range(len(text) - 1, 0, -1):
        if cv2.getTextSize(text[:n], font, scale, 1)[0][0] <= max_w:
            return text[:n]
    return ""


def render_annotated_video(src_video, dst_video, fps, per_second, gt_second,
                           spec, colors_hex):
    """Write a self-contained mp4 with the predictions + timelines drawn on every
    frame — no server/browser needed, plays in any local media player.

    Layout: original frame on top, then a footer with one timeline row per
    prediction track (see `track_specs`), the same rows again for ground truth
    when available, a moving playhead, and a colour legend. A label chip is
    overlaid on the top-left of each frame.

    In multilabel mode there is one row PER ACTIVITY, so a second where two
    activities fire shows two lit bars stacked — the thing a single-label
    timeline structurally cannot display.
    """
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for rendering: {src_video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps else 0.0
    bgr = {int(k): _hex_to_bgr(v) for k, v in colors_hex.items()}

    font = cv2.FONT_HERSHEY_SIMPLEX
    tracks = track_specs(spec)

    # (row label, per-second array, track) — predictions first, then ground truth.
    rows = [(f"P:{t['name']}" if spec.is_multilabel else "PRED", per_second, t)
            for t in tracks]
    if gt_second:
        rows += [(f"G:{t['name']}" if spec.is_multilabel else "GT", gt_second, t)
                 for t in tracks]

    # Scale the layout to the frame: full-episode videos are wide, but the
    # processed clips are only 256 px, and a fixed gutter would swallow them.
    # The gutter is sized to the row labels, capped at a quarter of the frame,
    # and labels are trimmed to whatever that leaves — never drawn over a bar.
    bar_h = 18 if len(rows) > 2 else 20
    gap, pad = 4, 8
    row_label_scale = 0.35
    widest = max(cv2.getTextSize(n, font, row_label_scale, 1)[0][0] for n, _, _ in rows)
    left = max(22, min(widest + 8, W // 4))
    rows = [(_fit_text(n, font, row_label_scale, left - 6), arr, t) for n, arr, t in rows]
    legend_h = 20 if W >= 200 else 0
    footer = pad + len(rows) * (bar_h + gap) + legend_h + pad
    track_w = max(W - left - 8, 1)
    seg = track_w / max(dur, 1.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst_video), fourcc, fps, (W, H + footer))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {dst_video}")

    def draw_row(canvas, arr, track, y):
        cv2.rectangle(canvas, (left, y), (left + track_w, y + bar_h), (44, 40, 38), -1)
        for d in arr:
            if track["kind"] == "argmax":
                color = bgr[d["label"]]
            elif entry_is_on(d, track):
                color = bgr[track["index"]]
            else:
                continue
            x0 = left + int(d["t"] * seg)
            x1 = left + int((d["t"] + 1) * seg)
            cv2.rectangle(canvas, (x0, y), (max(x1, x0 + 1), y + bar_h), color, -1)

    row_y = [H + pad + i * (bar_h + gap) for i in range(len(rows))]
    y_legend = (row_y[-1] + bar_h if row_y else H + pad) + gap

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps if fps else 0.0
        canvas = np.full((H + footer, W, 3), (24, 20, 18), np.uint8)
        canvas[:H, :, :] = frame

        # current-prediction chip (top-left of the frame)
        sec = min(len(per_second) - 1, int(t)) if per_second else 0
        d = per_second[max(0, sec)] if per_second else None
        if d is not None:
            conf = d.get("conf")
            text = entry_text(d, spec) + (f"  {conf:.2f}" if conf is not None else "")
            if spec.is_multilabel:
                active = d.get("active", [])
                chip_bgr = bgr[active[0]] if active else _hex_to_bgr(NEGATIVE_COLOR)
            else:
                chip_bgr = bgr[d["label"]]
            # Shrink the chip until it fits — a truncated label is worse than a
            # small one, especially when it is a multi-activity "a + b".
            scale, thick = 0.8, 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            while tw + 28 > W - 10 and scale > 0.3:
                scale -= 0.05
                thick = 2 if scale >= 0.55 else 1
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(canvas, (10, 10), (10 + tw + 18, 10 + th + 16), chip_bgr, -1)
            cv2.putText(canvas, text, (19, 10 + th + 9), font, scale, (0, 0, 0),
                        thick, cv2.LINE_AA)

        # timeline rows + labels
        for (name, arr, track), y in zip(rows, row_y):
            draw_row(canvas, arr, track, y)
            cv2.putText(canvas, name, (3, y + bar_h - 5), font, row_label_scale,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # playhead across every row
        if row_y:
            px = left + int(t / max(dur, 1.0) * track_w)
            cv2.line(canvas, (px, row_y[0]), (px, row_y[-1] + bar_h), (255, 255, 255), 1)

        # legend (dropped entirely on very narrow frames rather than clipped)
        if legend_h:
            lx = 8
            names = spec.activities if spec.is_multilabel else spec.class_names
            for i, name in enumerate(names):
                (nw, _), _ = cv2.getTextSize(name, font, 0.4, 1)
                if lx + 18 + nw > W - 4:
                    break
                cv2.rectangle(canvas, (lx, y_legend), (lx + 14, y_legend + 14), bgr[i], -1)
                cv2.putText(canvas, name, (lx + 18, y_legend + 12), font, 0.4,
                            (220, 220, 220), 1, cv2.LINE_AA)
                lx += 18 + nw + 22

        out.write(canvas)
        idx += 1
        if fps and idx % (int(fps) * 30 or 1) == 0:
            logger.info(f"  rendered {idx}/{total} frames ({idx / fps:.0f}s)")

    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# Viewer HTML
# ---------------------------------------------------------------------------
def write_viewer(out_dir, data):
    html = _VIEWER_TEMPLATE.replace("__DATA__", json.dumps(data))
    (out_dir / "viewer.html").write_text(html)


_VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VideoMAE prediction viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0b0f17; color: #e5e7eb;
         font: 15px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 16px; font-weight: 600; color: #9ca3af; margin: 0 0 12px; }
  video { width: 100%; background: #000; border-radius: 8px; display: block; }
  .now { display: flex; align-items: center; gap: 14px; margin: 14px 0 6px; }
  .chip { padding: 6px 14px; border-radius: 999px; font-weight: 700;
          font-size: 18px; color: #0b0f17; letter-spacing: .3px; }
  .time { font-variant-numeric: tabular-nums; color: #9ca3af; }
  .conf { margin-left: auto; color: #9ca3af; font-variant-numeric: tabular-nums; }
  .section { font-size: 12px; color: #9ca3af; margin: 16px 0 2px;
             text-transform: uppercase; letter-spacing: .6px; }
  .track-label { font-size: 12px; color: #6b7280; margin: 8px 0 3px; }
  canvas { width: 100%; height: 30px; display: block; border-radius: 4px;
           cursor: pointer; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 16px;
            font-size: 13px; color: #9ca3af; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .meta { margin-top: 14px; font-size: 12px; color: #6b7280; }
</style>
</head>
<body>
<div class="wrap">
  <h1 id="title">prediction viewer</h1>
  <video id="vid" controls preload="metadata"></video>

  <div class="now">
    <span class="time" id="time">0:00</span>
    <span class="chip" id="chip">—</span>
    <span class="conf" id="conf"></span>
  </div>

  <div class="section" id="predSection">predicted (per second)</div>
  <div id="predTracks"></div>

  <div class="section" id="gtSection" style="display:none">ground truth (per second)</div>
  <div id="gtTracks"></div>

  <div class="legend" id="legend"></div>
  <div class="meta" id="meta"></div>
</div>

<script>
const DATA = __DATA__;
const CLASSES = DATA.classes, ACTIVITIES = DATA.activities, COLORS = DATA.colors;
const TRACKS = DATA.tracks, DUR = DATA.duration, MULTILABEL = DATA.multilabel;
const NEG_COLOR = DATA.negative_color;
const perSec = DATA.per_second, gtSec = DATA.ground_truth_per_second;
const TRACK_H = 30;   // fixed: never read back off the canvas, which would
                      // compound the devicePixelRatio scale on every redraw

const vid = document.getElementById('vid');
vid.src = DATA.video;
document.getElementById('title').textContent =
  DATA.title + '  ·  ' + DATA.model + '  ·  ' + DATA.task;

// One row per track. multiclass -> a single colour-coded row; multilabel -> one
// on/off row per activity, so simultaneous activities are visible at once.
function buildRows(container, arr, showNames){
  container.innerHTML = '';
  return TRACKS.map(tr => {
    if (showNames){
      const lab = document.createElement('div');
      lab.className = 'track-label';
      lab.textContent = tr.name;
      container.appendChild(lab);
    }
    const cv = document.createElement('canvas');
    cv.height = TRACK_H;
    cv.addEventListener('click', e => seekFromCanvas(e.currentTarget, e));
    container.appendChild(cv);
    return {track: tr, canvas: cv, arr: arr};
  });
}

const predRows = buildRows(document.getElementById('predTracks'), perSec, MULTILABEL);
let gtRows = [];
if (gtSec){
  document.getElementById('gtSection').style.display = '';
  gtRows = buildRows(document.getElementById('gtTracks'), gtSec, MULTILABEL);
}

// legend
const legend = document.getElementById('legend');
(MULTILABEL ? ACTIVITIES : CLASSES).forEach((c, i) => {
  const s = document.createElement('span');
  s.innerHTML = '<span class="sw" style="background:' + COLORS[i] + '"></span>' + c;
  legend.appendChild(s);
});
document.getElementById('meta').textContent =
  DATA.n_windows + ' windows · ' + DATA.window.size + 's window / '
  + DATA.window.stride + 's stride'
  + (MULTILABEL ? ' · thresholds ' + ACTIVITIES.map(
        (a, i) => a + '@' + DATA.decision_thresholds[i]).join(', ') : '');

function fmt(t){ const m = Math.floor(t/60), s = Math.floor(t%60);
  return m + ':' + String(s).padStart(2,'0'); }

function entryColor(d, track){
  if (track.kind === 'argmax') return COLORS[d.label];
  return (d.active || []).indexOf(track.index) >= 0 ? COLORS[track.index] : null;
}

function chipText(d){
  if (!MULTILABEL) return CLASSES[d.label];
  const names = (d.active || []).map(i => ACTIVITIES[i]);
  return names.length ? names.join(' + ') : 'no activity';
}
function chipColor(d){
  if (!MULTILABEL) return COLORS[d.label];
  const a = d.active || [];
  return a.length ? COLORS[a[0]] : NEG_COLOR;
}

function drawRow(row){
  const canvas = row.canvas, dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = TRACK_H;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1f2b'; ctx.fillRect(0, 0, w, h);
  const px = w / Math.max(DUR, 1);
  row.arr.forEach(d => {
    const color = entryColor(d, row.track);
    if (!color) return;
    ctx.fillStyle = color;
    ctx.fillRect(d.t * px, 0, Math.max(1, px) + 0.5, h);
  });
  return {w, h, px, ctx};
}

function playhead(view, t){
  const x = t * view.px;
  view.ctx.fillStyle = '#ffffff';
  view.ctx.fillRect(x - 1, 0, 2, view.h);
}

function render(){
  const t = vid.currentTime;
  const sec = Math.min(perSec.length - 1, Math.floor(t));
  const d = perSec[Math.max(0, sec)];
  if (d){
    const chip = document.getElementById('chip');
    chip.textContent = chipText(d);
    chip.style.background = chipColor(d);
    document.getElementById('conf').textContent =
      'conf ' + (d.conf != null ? d.conf.toFixed(2) : '—');
  }
  document.getElementById('time').textContent = fmt(t);
  predRows.concat(gtRows).forEach(row => playhead(drawRow(row), t));
}

function seekFromCanvas(canvas, ev){
  const r = canvas.getBoundingClientRect();
  const frac = (ev.clientX - r.left) / r.width;
  vid.currentTime = Math.max(0, Math.min(DUR, frac * DUR));
}

vid.addEventListener('timeupdate', render);
vid.addEventListener('loadedmetadata', render);
window.addEventListener('resize', render);
render();
</script>
</body>
</html>

"""


# ---------------------------------------------------------------------------
# Test-set case selection (auto-resolve raw video + annotation from a clip path)
# ---------------------------------------------------------------------------
# The full-episode videos + annotations live in a sibling `Unprocessed_data`
# tree of the processed clips (see src/data/data_process.py):
#     <base>/Unprocessed_data/videos/<case_id>.mp4
#     <base>/Unprocessed_data/anot_files/<case_id>.txt
# while a clip path is  <base>/<Processed_...>/videos/<class>/<case>_interval_...
# so we recover the case id from the clip filename and walk up its ancestors
# to find the matching raw video/annotation.
VIDEO_EXTS = [".mp4", ".MP4", ".avi", ".mkv", ".mov", ".MOV"]


def recover_case_id(clip_path: str) -> str:
    """Case id = clip filename stem before '_interval_' (matches build_manifest)."""
    return Path(clip_path).stem.split("_interval_")[0]


def list_test_cases(test_csv: Path):
    """Read the test manifest → ordered unique cases with clip counts + an anchor."""
    cases = {}
    with open(test_csv, newline="") as f:
        for row in csv.DictReader(f):
            vp = row.get("video_path")
            if not vp:
                continue
            cid = recover_case_id(vp)
            c = cases.setdefault(cid, {"case_id": cid, "n_clips": 0, "anchor": vp})
            c["n_clips"] += 1
    return sorted(cases.values(), key=lambda c: c["case_id"])


def resolve_media(anchor_clip: str, case_id: str):
    """Walk up the clip path for a sibling Unprocessed_data/{videos,anot_files}."""
    p = Path(anchor_clip).expanduser().resolve()
    for anc in p.parents:
        base = anc / "Unprocessed_data"
        vids, anots = base / "videos", base / "anot_files"
        if not vids.is_dir():
            continue
        for ext in VIDEO_EXTS:
            cand = vids / f"{case_id}{ext}"
            if cand.exists():
                anot = anots / f"{case_id}.txt"
                return cand, (anot if anot.exists() else None)
    return None, None


def choose_case(cases):
    """Print a numbered menu and return the selected case dict (interactive)."""
    print("\nTest-set cases:")
    for i, c in enumerate(cases, 1):
        v = "video ✓" if c["video"] else "video ✗ (raw not found)"
        g = "GT ✓" if c["annotation"] else "GT ✗"
        print(f"  [{i:2d}] {c['case_id']:14s} {c['n_clips']:5d} clips   {v:26s} {g}")
    while True:
        sel = input(f"\nSelect a case [1-{len(cases)}] (q to quit): ").strip()
        if sel.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        if sel.isdigit() and 1 <= int(sel) <= len(cases):
            return cases[int(sel) - 1]
        print("  invalid selection")


def select_from_test_set(args):
    """Resolve (video_path, annotation) via the test manifest + user selection."""
    test_csv = Path(args.test_csv).expanduser()
    if not test_csv.exists():
        raise FileNotFoundError(
            f"{test_csv} not found — run scripts/build_data.sh first, or pass --video.")
    cases = list_test_cases(test_csv)
    if not cases:
        raise RuntimeError(f"No cases found in {test_csv}.")
    for c in cases:
        c["video"], c["annotation"] = resolve_media(c["anchor"], c["case_id"])

    if args.case:
        chosen = next((c for c in cases if c["case_id"] == args.case), None)
        if chosen is None:
            raise SystemExit(f"case '{args.case}' not in {test_csv}. "
                             f"Available: {[c['case_id'] for c in cases]}")
    else:
        chosen = choose_case(cases)

    if not chosen["video"]:
        raise SystemExit(
            f"No raw video found for case '{chosen['case_id']}'. Its clips are at\n"
            f"  {chosen['anchor']}\n"
            f"but no sibling Unprocessed_data/videos/{chosen['case_id']}.* exists. "
            f"Pass the full video explicitly with --video (and --annotation).")
    logger.info(f"Selected case {chosen['case_id']}: {chosen['video']}")
    return chosen["case_id"], chosen["video"], chosen["annotation"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["VideoMAE", "VideoMAEGiant"])
    ap.add_argument("--model_path", required=True, help="Trained checkpoint .pt")
    ap.add_argument("--video", default=None,
                    help="Full episode video. If omitted, pick a case from --test-csv.")
    ap.add_argument("--test-csv", default="data/test.csv",
                    help="Test manifest to pick a case from (default: data/test.csv).")
    ap.add_argument("--case", default=None,
                    help="Case id to run non-interactively (skips the menu).")
    ap.add_argument("--annotation", default=None,
                    help="5-col TSV to overlay ground truth (auto-resolved for test cases).")
    ap.add_argument("--no-gt", action="store_true",
                    help="Do not overlay ground truth even if an annotation is found.")
    ap.add_argument("--data-config", default=None,
                    help="Override the DataSpec stored in the checkpoint (e.g. to re-score "
                         "a multilabel model at different decision_thresholds).")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: viewer_out/<case-or-video-stem>/).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--copy-video", action="store_true",
                    help="Copy the video into out-dir instead of symlinking.")
    ap.add_argument("--render-video", action="store_true",
                    help="Burn predictions into a standalone annotated.mp4 (offline: "
                         "no server/browser needed — just copy the file off the VM).")
    ap.add_argument("--serve", action="store_true", help="Serve the viewer over HTTP.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")

    # Resolve the target video (+ optional annotation): explicit --video, or an
    # interactive pick from the test set.
    stem = None
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        annotation = args.annotation
    else:
        stem, video_path, auto_annotation = select_from_test_set(args)
        annotation = args.annotation or auto_annotation

    if args.no_gt:
        annotation = None
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    stem = stem or video_path.stem
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path("viewer_out") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model, config, spec = build_model(args.model, args.model_path, device, args.data_config)
    colors = class_colors(spec)
    tracks = track_specs(spec)

    fps, duration_s, results = run_inference(
        model, model.processor, video_path, device, args.batch_size)
    if not results:
        raise RuntimeError("No windows produced predictions — check the video/codec.")

    per_second = windows_to_per_second(results, duration_s, spec)
    gt_second = None
    if annotation:
        logger.info(f"Ground truth: {annotation}")
        gt_second = gt_per_second(load_gt_intervals(annotation, spec), duration_s, spec)

    # ---- make the video reachable by the browser ----
    local_video = out_dir / "video.mp4"
    if local_video.exists() or local_video.is_symlink():
        local_video.unlink()
    if args.copy_video:
        shutil.copy2(video_path, local_video)
    else:
        os.symlink(video_path, local_video)

    # ---- write predictions.json ----
    thresholds = spec.sigmoid_thresholds() if spec.is_multilabel else None
    data = {
        "title": stem,
        "model": args.model,
        "task": spec.task,
        "multilabel": spec.is_multilabel,
        "video": "video.mp4",
        "fps": round(float(fps), 3),
        "duration": round(float(duration_s), 3),
        "classes": spec.class_names,
        "activities": list(spec.activities),
        "negative_class": spec.negative_class,
        "negative_color": NEGATIVE_COLOR,
        "colors": colors,
        "tracks": tracks,
        "decision_thresholds": thresholds,
        "window": {"size": WINDOW_S, "stride": STRIDE_S},
        "n_windows": len(results),
        "per_second": per_second,
        "ground_truth_per_second": gt_second,
        "windows": [
            {"start": round(ws, 3), "end": round(we, 3),
             **{k: v for k, v in probs_to_entry(None, p, spec, thresholds).items()
                if k != "t"}}
            for ws, we, p in results
        ],
    }
    (out_dir / "predictions.json").write_text(json.dumps(data, indent=2))

    # ---- write predictions.csv (per window) ----
    # multiclass: one predicted class per window.
    # multilabel: `pred` is the '+'-joined set of activities over threshold (empty
    #             string = no activity), plus the raw per-activity probabilities.
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "pred", "confidence", *spec.class_names])
        for ws, we, p in results:
            entry = probs_to_entry(None, p, spec, thresholds)
            if spec.is_multilabel:
                pred = "+".join(spec.activities[i] for i in entry["active"])
            else:
                pred = spec.class_names[entry["label"]]
            w.writerow([round(ws, 3), round(we, 3), pred, entry["conf"],
                        *[round(float(x), 4) for x in p]])

    write_viewer(out_dir, data)
    logger.info(f"Wrote viewer -> {out_dir}/viewer.html")
    logger.info(f"Predictions  -> {out_dir}/predictions.json  (+ .csv)")

    # ---- offline: burn predictions into a standalone annotated.mp4 ----
    if args.render_video:
        annotated = out_dir / "annotated.mp4"
        logger.info("Rendering annotated video (this decodes every frame)…")
        render_annotated_video(video_path, annotated, fps, per_second, gt_second,
                               spec, colors)
        logger.info(f"Annotated MP4 -> {annotated}")
        logger.info("Copy it off the VM and play it in any media player (VLC), e.g.:")
        logger.info(f"  scp <user>@<vm>:{annotated.resolve()} .")

    if args.serve:
        serve(out_dir, args.port)
    elif not args.render_video:
        logger.info("No display? Re-run with --render-video for a standalone annotated.mp4,")
        logger.info("or copy the whole folder off the VM and open viewer.html locally:")
        logger.info(f"  python -m src.infer_video ... --render-video   (offline, single file)")
        logger.info(f"  (viewer.html needs --copy-video so video.mp4 is a real file, not a symlink)")


if __name__ == "__main__":
    main()
