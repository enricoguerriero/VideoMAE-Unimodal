"""
infer_video.py

Run a trained VideoMAE / VideoMAEv2-giant checkpoint over an ENTIRE episode
video (not the pre-cut 3-second clips) and produce a synced viewer.

It reproduces the thesis' clip scheme at inference time — a 3-second window slid
with a 1-second stride over the whole video — and classifies each window into one
of the 4 activities (non_target / stimulation / ventilation / suction). Each
window's argmax is mapped to a per-second timeline so you get one predicted label
per second of the episode.

Outputs (into --out-dir, default: viewer_out/<video-stem>/):
    predictions.json   per-second + per-window predictions + metadata
    predictions.csv    per-window (start,end,label,confidence,probs...)
    viewer.html        self-contained page: plays the video with the predicted
                       label following the playhead + a colour-coded timeline
    video.mp4          symlink (or copy with --copy-video) to the source video

View it with:
    python -m src.infer_video --model VideoMAE --model_path <ckpt.pt> \
        --video /path/to/<case_id>.mp4 --serve
then open the printed http://localhost:<port>/viewer.html (forward the port over
SSH / VS Code Remote). --serve uses a Range-capable server so scrubbing works.

Ground truth (optional): pass --annotation /path/to/<case_id>.txt (the 5-column
TSV from Unprocessed_data/anot_files) to overlay a second timeline of the
reference labels, computed with the thesis' `for_predict` rule (dominant of
stim/vent/suction over the 3 s window if >= 50%, else non_target).
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

from src.utils import load_model, CLASSES

# Distinct, colour-blind-friendly-ish palette, one per class index.
CLASS_COLORS = {
    0: "#6b7280",  # non_target  — grey
    1: "#3b82f6",  # stimulation — blue
    2: "#22c55e",  # ventilation — green
    3: "#f97316",  # suction     — orange
}

WINDOW_S = 3        # clip length, seconds (thesis segment_size)
STRIDE_S = 1        # slide, seconds (thesis shift)
NUM_FRAMES = 16     # VideoMAE requirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(model_name, model_path, device):
    """Load a trained checkpoint exactly like src/test.py."""
    saved = torch.load(model_path, map_location=device, weights_only=False)
    config = saved.get("config", {})
    model = load_model(model_name, num_classes=4).to(device)
    model.load_classifier(saved, config)
    model.load_backbone(saved, config)
    if config.get("attention_pooling", False):
        model.load_attention_pooling(saved)
    model.eval()
    return model, config


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
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
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
def windows_to_per_second(results, duration_s):
    """Assign each second the prediction of the window whose centre is nearest."""
    if not results:
        return []
    centers = np.array([(s + e) / 2.0 for s, e, _ in results])
    per_second = []
    for sec in range(int(math.ceil(duration_s))):
        target = sec + 0.5
        j = int(np.argmin(np.abs(centers - target)))
        probs = results[j][2]
        label = int(np.argmax(probs))
        per_second.append({"t": sec, "label": label, "conf": round(float(probs[label]), 4)})
    return per_second


# ---------------------------------------------------------------------------
# Optional ground truth (5-column annotation TSV)
# ---------------------------------------------------------------------------
EVENT_TO_CLASS = {"Non-target": 0, "Stimulation": 1, "Ventilation": 2, "Suction": 3}


def load_gt_intervals(annotation_path):
    """Parse the 5-col TSV into {class_idx: [(start_ms, end_ms), ...]}."""
    intervals = {0: [], 1: [], 2: [], 3: []}
    with open(annotation_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            event, start, end = parts[0], parts[1], parts[2]
            if len(parts) >= 5 and parts[4] == "Newborn visible in video frame":
                continue
            cls = EVENT_TO_CLASS.get(event)
            if cls is None:
                continue
            try:
                intervals[cls].append((int(start), int(end)))
            except ValueError:
                continue
    return intervals


def _overlap_ms(a0, a1, ivs):
    return sum(min(a1, e) - max(a0, s) for s, e in ivs if a0 < e and a1 > s)


def gt_per_second(intervals, duration_s):
    """Label each second via the thesis' `for_predict` rule over its 3 s window."""
    out = []
    win_ms = WINDOW_S * 1000
    for sec in range(int(math.ceil(duration_s))):
        s_ms = sec * 1000
        e_ms = s_ms + win_ms
        ov = {c: _overlap_ms(s_ms, e_ms, intervals[c]) for c in (1, 2, 3)}
        best = max(ov, key=ov.get)
        label = best if ov[best] >= win_ms * 0.5 else 0
        out.append({"t": sec, "label": label})
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
  .track-label { font-size: 12px; color: #6b7280; margin: 12px 0 4px; }
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

  <div class="track-label">predicted (per second)</div>
  <canvas id="pred" height="30"></canvas>

  <div class="track-label gt-label" id="gtLabel" style="display:none">ground truth (per second)</div>
  <canvas id="gt" height="30" style="display:none"></canvas>

  <div class="legend" id="legend"></div>
  <div class="meta" id="meta"></div>
</div>

<script>
const DATA = __DATA__;
const CLASSES = DATA.classes, COLORS = DATA.colors, DUR = DATA.duration;
const perSec = DATA.per_second, gtSec = DATA.ground_truth_per_second;

const vid = document.getElementById('vid');
vid.src = DATA.video;
document.getElementById('title').textContent =
  DATA.title + '  ·  ' + DATA.model;

// legend
const legend = document.getElementById('legend');
CLASSES.forEach((c, i) => {
  const s = document.createElement('span');
  s.innerHTML = '<span class="sw" style="background:' + COLORS[i] + '"></span>' + c;
  legend.appendChild(s);
});
document.getElementById('meta').textContent =
  DATA.n_windows + ' windows · ' + WINDOW_txt();
function WINDOW_txt(){ return DATA.window.size + 's window / ' + DATA.window.stride + 's stride'; }

function fmt(t){ const m = Math.floor(t/60), s = Math.floor(t%60);
  return m + ':' + String(s).padStart(2,'0'); }

// draw a per-second track onto a canvas
function drawTrack(canvas, arr){
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.height;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  ctx.clearRect(0,0,w,h);
  const px = w / Math.max(DUR, 1);
  arr.forEach(d => {
    ctx.fillStyle = COLORS[d.label];
    ctx.fillRect(d.t * px, 0, Math.max(1, px) + 0.5, h);
  });
  return {w, h, px, ctx};
}

if (gtSec){
  document.getElementById('gtLabel').style.display = '';
  document.getElementById('gt').style.display = '';
}

// redraw tracks + the moving playhead, and update the current-label chip
function render(){
  const t = vid.currentTime;
  const sec = Math.min(perSec.length - 1, Math.floor(t));
  const d = perSec[Math.max(0, sec)] || {label:0, conf:0};
  const chip = document.getElementById('chip');
  chip.textContent = CLASSES[d.label];
  chip.style.background = COLORS[d.label];
  document.getElementById('time').textContent = fmt(t);
  document.getElementById('conf').textContent =
    'conf ' + (d.conf!=null ? d.conf.toFixed(2) : '—');

  // redraw tracks + playhead
  const pv = drawTrack(document.getElementById('pred'), perSec);
  playhead(pv, t);
  if (gtSec){
    const gv = drawTrack(document.getElementById('gt'), gtSec);
    playhead(gv, t);
  }
}
function playhead(view, t){
  const x = t * view.px;
  view.ctx.fillStyle = '#ffffff';
  view.ctx.fillRect(x - 1, 0, 2, view.h);
}

function seekFromCanvas(canvas, ev){
  const r = canvas.getBoundingClientRect();
  const frac = (ev.clientX - r.left) / r.width;
  vid.currentTime = Math.max(0, Math.min(DUR, frac * DUR));
}
document.getElementById('pred').addEventListener('click', e =>
  seekFromCanvas(e.currentTarget, e));
document.getElementById('gt').addEventListener('click', e =>
  seekFromCanvas(e.currentTarget, e));

vid.addEventListener('timeupdate', render);
vid.addEventListener('loadedmetadata', render);
window.addEventListener('resize', render);
render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["VideoMAE", "VideoMAEGiant"])
    ap.add_argument("--model_path", required=True, help="Trained checkpoint .pt")
    ap.add_argument("--video", required=True, help="Path to the ENTIRE episode video.")
    ap.add_argument("--annotation", default=None,
                    help="Optional 5-col TSV to overlay ground-truth labels.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: viewer_out/<video-stem>/).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--copy-video", action="store_true",
                    help="Copy the video into out-dir instead of symlinking.")
    ap.add_argument("--serve", action="store_true", help="Serve the viewer over HTTP.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path("viewer_out") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model, config = build_model(args.model, args.model_path, device)

    fps, duration_s, results = run_inference(
        model, model.processor, video_path, device, args.batch_size)
    if not results:
        raise RuntimeError("No windows produced predictions — check the video/codec.")

    per_second = windows_to_per_second(results, duration_s)
    gt_second = None
    if args.annotation:
        gt_second = gt_per_second(load_gt_intervals(args.annotation), duration_s)

    # ---- make the video reachable by the browser ----
    local_video = out_dir / "video.mp4"
    if local_video.exists() or local_video.is_symlink():
        local_video.unlink()
    if args.copy_video:
        shutil.copy2(video_path, local_video)
    else:
        os.symlink(video_path, local_video)

    # ---- write predictions.json ----
    data = {
        "title": video_path.stem,
        "model": args.model,
        "video": "video.mp4",
        "fps": round(float(fps), 3),
        "duration": round(float(duration_s), 3),
        "classes": CLASSES,
        "colors": CLASS_COLORS,
        "window": {"size": WINDOW_S, "stride": STRIDE_S},
        "n_windows": len(results),
        "per_second": per_second,
        "ground_truth_per_second": gt_second,
        "windows": [
            {"start": round(s, 3), "end": round(e, 3),
             "label": int(np.argmax(p)), "probs": [round(float(x), 4) for x in p]}
            for s, e, p in results
        ],
    }
    (out_dir / "predictions.json").write_text(json.dumps(data, indent=2))

    # ---- write predictions.csv (per window) ----
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "pred_label", "pred_class", "confidence", *CLASSES])
        for s, e, p in results:
            lab = int(np.argmax(p))
            w.writerow([round(s, 3), round(e, 3), lab, CLASSES[lab],
                        round(float(p[lab]), 4), *[round(float(x), 4) for x in p]])

    write_viewer(out_dir, data)
    logger.info(f"Wrote viewer -> {out_dir}/viewer.html")
    logger.info(f"Predictions  -> {out_dir}/predictions.json  (+ .csv)")

    if args.serve:
        serve(out_dir, args.port)
    else:
        logger.info("Open the viewer with:")
        logger.info(f"  python -m src.infer_video ... --serve   (or)")
        logger.info(f"  python -m http.server -d {out_dir} {args.port}  # note: no video scrubbing")


if __name__ == "__main__":
    main()
