# VideoMAE Unimodal — Neonatal Resuscitation Activity Recognition (Haydom + DRC)

Single-label **4-class** video activity recognition with **VideoMAE** /
**VideoMAEv2-giant**, trained and evaluated on the **combined Haydom + DRC**
neonatal-resuscitation dataset used by the multimodal thesis of Tharmaratnam &
Wagner (UiS, in collaboration with Laerdal Medical).

Classes: `0 non_target · 1 stimulation · 2 ventilation · 3 suction`.

The goal is a clean **backbone-swap benchmark**: VideoMAE in place of the
thesis' MoViNet-A2 *video base model*, on **the same clips, same labels, same
whole-case split, same metrics** — so the numbers are directly comparable to the
multimodal work's video modality. This repo is **fully self-contained**: it has
no code dependency on the two original repositories. It only *reads* the
processed clip files that already exist on the VM.

---

## Why this is comparable (and where it deliberately differs)

Reproduced from the thesis (see `docs`-style notes inline in each module):

| Aspect | This repo | Thesis (MoViNet video model) |
|---|---|---|
| Clips | reuse the exact processed 3 s clips (1 s stride, 256×192) | same |
| Classes | 4-class single-label; `no_overlap`→`non_target`; buckets 5–8 dropped | same |
| Split | whole-case; **14 fixed held-out test cases**; 70/30 train/val | same |
| Loss | weighted CE, **sqrt** inverse-freq weights, `label_smoothing=0.1` | same |
| Model selection | best **macro-F1** + best **minority (suction) F1** checkpoints | same |
| Metrics | clip-level macro/per-class/minority F1, accuracy, confusion matrix (argmax) | same |

Intentional differences (the experimental variable + backbone constraints):

- **Backbone**: VideoMAE / VideoMAEv2-giant instead of MoViNet-A2 (the point).
- **Frame preprocessing**: VideoMAE's own processor — **16 frames**, 224², ImageNet
  normalisation — vs MoViNet's 50 frames, 280→224 crop, [0,1] scaling. This is
  fixed by the VideoMAE architecture and cannot be matched.
- **No MoViNet-style train augmentation** (flip/brightness/greyscale/speed) and
  **no progressive block-unfreezing**; this repo does full-backbone fine-tuning
  (configurable). Document these when reporting.

---

## Repository structure

```
videomae-unimodal/
├── configs/config.yaml          # single-label 4-class config (edit paths/LR)
├── requirements.txt
├── scripts/                     # build_data.sh, train.sh, test.sh, infer_video.sh, *.slurm
├── data/                        # generated manifests land here (git-ignored)
└── src/
    ├── training.py              # weighted-CE training; dual best-checkpoints
    ├── test.py                  # argmax eval on the 14 held-out cases
    ├── infer_video.py           # whole-episode inference + synced HTML viewer
    ├── data/
    │   ├── build_manifest.py    # [PRIMARY] index existing clips -> combined CSV
    │   ├── split_cases.py       # [PRIMARY] whole-case train/val/test split
    │   ├── videomae_dataset.py  # single-label Dataset (labels = class index)
    │   ├── data_process.py      # [OPTIONAL] video-only clip cutter (from raw)
    │   └── process_dataset.py   # [OPTIONAL] driver for data_process.py
    ├── models/
    │   ├── base.py              # trimmed base (no LoRA/VLM)
    │   ├── videomae.py          # MCG-NJU/videomae-base-finetuned-ssv2
    │   ├── videomae_giant.py    # OpenGVLab/VideoMAEv2-giant
    │   ├── classifier.py        # 4-logit MLP head (log-prior bias init)
    │   └── attentionpooling.py
    └── utils/
        ├── metrics.py           # 4-class argmax metrics
        ├── collate.py
        └── model_loading.py
```

---

## Setup

```bash
conda create -n videomae-unimodal python=3.11 -y
conda activate videomae-unimodal
pip install -r requirements.txt      # transformers is pinned to a GitHub commit
```

`wandb` is optional — set `wandb_mode: disabled` in `configs/config.yaml` to skip it.

---

## Pipeline

### Step 1 — Build the dataset manifests (recommended path)

The thesis' processed clips already exist on the VM; we just index them. Point
the two roots at each site's `.../videos` directory (the one with the per-class
subfolders), then run:

```bash
bash scripts/build_data.sh
```

This runs:

```bash
python -m src.data.build_manifest \
    --root Haydom=/…/Processed_…_chestmov/videos \
    --root DRC=/…/Processed_…_chestmov/videos \
    --out data/clips_all.csv

python -m src.data.split_cases \
    --manifest data/clips_all.csv --out-dir data \
    --train-ratio 0.7 --seed 2025
```

Producing `data/train.csv`, `data/validation.csv`, `data/test.csv`
(columns: `video_path,label`).

**Test cases are frozen** to the thesis' 14 (DRC `2-33998-1, 2-34325-1,
2-37178-1, 2-37453-1`; Haydom `11848523, 15233524, 28631424, 37572224,
38037024, 38042423, 38714124, 40094725, 40386725, 40402325`). For an exact
train/val reproduction, pass `--train-cases-file`/`--val-cases-file` instead of
the stratified split.

### Step 1 (alternative) — Regenerate clips from raw

Only if you must re-cut clips (needs raw videos + **cleaned** 5-column
annotation files under `<base>/Unprocessed_data/{videos,anot_files}`):

```bash
python -m src.data.process_dataset \
    --base-dir /…/Data_processing \
    --folder-name /…/Processed_video_clips
```

`data_process.py` reproduces the thesis' video labeling exactly (thresholds
strong=0.50, suction=0.25, non_target=0.20; 3 s / 1 s / 256×192). The
annotation-cleaning chain (spelling fixes, DRC breathing-label removal,
event→category remap) is **site/data-specific and not ported** — supply
already-cleaned annotations, or reuse the existing processed clips (Step 1).

### Step 2 — Train

```bash
# edit configs/config.yaml (paths, batch_size, LR, epochs) first
bash scripts/train.sh VideoMAE 0          # or: VideoMAEGiant
```

Saves best-macro-F1 and best-suction-F1 checkpoints to `checkpoints/`, a final
model to `models/`, and per-epoch metrics to `results/metrics_*.csv`.

### Step 3 — Test

```bash
bash scripts/test.sh VideoMAE checkpoints/VideoMAE_best_macro_<ts>.pt 0
```

Writes `results/results_*.csv` (per-class + macro + minority F1 + confusion
matrix) and `results/scores_*.npz` (raw logits + ground-truth class indices).

### Step 4 (optional) — Inference over a full video + synced viewer

Runs a trained checkpoint over an **entire episode** (not the pre-cut clips) and
produces a self-contained web viewer that plays the video with the predicted
label following the playhead. It re-creates the thesis clip scheme at inference
time — a **3 s window slid with a 1 s stride** over the whole video — classifies
each window (16-frame / 224² VideoMAE processor), and maps each window's argmax
to a **per-second** label.

```bash
bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro_<ts>.pt \
    /…/Unprocessed_data/videos/<case_id>.mp4 0 8000 \
    /…/Unprocessed_data/anot_files/<case_id>.txt      # last arg (GT) is optional
```

Args: `MODEL CKPT VIDEO [GPU] [PORT] [ANNOTATION]`. This calls:

```bash
python -m src.infer_video \
    --model VideoMAE --model_path <ckpt.pt> \
    --video /…/<case_id>.mp4 \
    --annotation /…/<case_id>.txt \   # optional: overlay ground-truth timeline
    --serve --port 8000
```

Outputs land in `viewer_out/<case_id>/`:

- `viewer.html` — plays the video with a live class chip + confidence, a
  colour-coded **per-second timeline** (click to seek), and — when
  `--annotation` is given — a **ground-truth** strip below it for comparison.
  Ground truth uses the thesis' `for_predict` rule (dominant of
  stim/vent/suction over the 3 s window if ≥ 0.50, else non_target).
- `predictions.json` — per-second + per-window predictions + metadata.
- `predictions.csv` — per-window `start,end,label,confidence,` + the 4 class probs.
- `video.mp4` — symlink to the source (use `--copy-video` to copy instead).

`--serve` starts a **Range-capable** HTTP server on `0.0.0.0`, so video scrubbing
works and the port forwards cleanly over SSH / VS Code Remote — open
`http://localhost:8000/viewer.html`. Without `--serve`, the files are written but
not served (plain `python -m http.server` won't support video seeking).

> Predictions use VideoMAE's own 16-frame / 224² preprocessing (not MoViNet's
> 50-frame pipeline), the same backbone-driven difference documented above.

---

## Weights & Biases logging

Logging is optional and controlled by `wandb_project` / `wandb_mode` in
`configs/config.yaml` (`online` | `offline` | `disabled`). `wandb login` once
before online runs. With wandb absent or `disabled`, everything runs unchanged
(all logging is a no-op via `src/utils/wandb_utils.py`).

**Training** (`job_type: train`) logs, with `epoch` as the chart x-axis:
- `train/loss` (per 50 steps, x = `train/global_step`), `train/loss_epoch`, `lr`
- `val/loss` and every metric under `val/*` (per-class + `macro/*` + `minority/f1`)
- `val/confusion_matrix` plot each epoch (and `val_step/*` if `validation_step` is set)
- run **summary**: `best/macro_f1`, `best/<minority>_f1` and the epochs they occurred.

**Testing** (`job_type: eval`) logs `test/*` metrics, a `test/confusion_matrix`
plot, writes them to the run summary, and **stores `results_*.csv` + `scores_*.npz`
as a wandb Artifact** (`type=test-results`) so each evaluation is preserved with
its run.

## Notes

- `num_classes` is fixed at 4 everywhere; the head bias is initialised to the
  per-class log-priors and the loss uses sqrt inverse-frequency class weights,
  both derived from the train split at runtime.
- VideoMAEv2-giant needs `timm` + `easydict` and ~24 GB VRAM (use bf16/fp16, the
  default AMP path).
- Cross-site experiment (train Haydom → test DRC): build separate manifests per
  site and pass the DRC-only CSV as `test_data`, mirroring the thesis' Haydom-only
  generalisation study.
