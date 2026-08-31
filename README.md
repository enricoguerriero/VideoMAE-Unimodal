# VideoMAE Unimodal — Neonatal Resuscitation Activity Recognition (Haydom + DRC)

Video activity recognition with **VideoMAE** / **VideoMAEv2-giant**, trained and
evaluated on the **combined Haydom + DRC** neonatal-resuscitation dataset used by
the multimodal thesis of Tharmaratnam & Wagner (UiS, in collaboration with
Laerdal Medical).

The **label regime is configuration, not code** (`configs/data.yaml`):

| | `task: multiclass` (default) | `task: multilabel` |
|---|---|---|
| outputs | 4: `non_target · stimulation · ventilation · suction` | 3: `stimulation · ventilation · suction` |
| "no activity" | its own class (index 0) | the **all-zero vector** |
| head | softmax | independent sigmoids |
| loss | weighted CrossEntropy | masked BCEWithLogits |
| co-occurring activities | not representable | **representable and trainable** |
| thesis-comparable | yes, exactly | via a projected 4-class table |

The default reproduces the thesis exactly. `configs/data_multilabel.yaml` is a
ready-to-run alternative: three independent activities, keeping the
`target_overlap` clips where two activities genuinely co-occur.

The goal is a clean **backbone-swap benchmark**: VideoMAE in place of the
thesis' MoViNet-A2 *video base model*, on **the same clips, same labels, same
whole-case split, same metrics** — so the numbers are directly comparable to the
multimodal work's video modality. This repo is **fully self-contained**: it has
no code dependency on the two original repositories. It only *reads* the
processed clip files that already exist on the VM.

---

## Why this is comparable (and where it deliberately differs)

Reproduced from the thesis (see `docs`-style notes inline in each module):

| Aspect | This repo (with the default `configs/data.yaml`) | Thesis (MoViNet video model) |
|---|---|---|
| Clips | reuse the exact processed 3 s clips (1 s stride, 256×192) | same |
| Classes | 4-class single-label; `no_overlap`→`non_target`; buckets 5–8 dropped | same |
| Split | whole-case; **per-hospital test sets seeded with the thesis' 14 cases**; 80/20 train/val | whole-case; the 14 cases; 70/30 |
| Loss | weighted CE, **sqrt** inverse-freq weights, `label_smoothing=0.1` | same |
| Model selection | best **macro-F1** + best **minority (suction) F1** checkpoints | same |
| Metrics | clip-level macro/per-class/minority F1, accuracy, confusion matrix (argmax) | same |

The default data config was written to be a **no-op** relative to the original
hard-coded behaviour: same thresholds (0.50 / 0.50 / 0.25), same `weak_threshold`
purity guard, same kept buckets. Verify with the census
`python -m src.data.build_manifest` prints.

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
├── configs/
│   ├── config.yaml              # HOW to train (paths, LR, epochs, batch size)
│   ├── data.yaml                # WHAT the labels are — multiclass, thesis-exact
│   └── data_multilabel.yaml     # WHAT the labels are — 3 independent activities
├── requirements.txt
├── scripts/                     # build_data.sh, train.sh, test.sh, infer_video.sh, *.slurm
├── data/                        # generated manifests land here (git-ignored)
└── src/
    ├── training.py              # task-blind training loop; dual best-checkpoints
    ├── test.py                  # eval on every per-site test set (task from the ckpt)
    ├── tune_thresholds.py       # pick multilabel decision thresholds on validation
    ├── inference_demo.py        # one episode per hospital + probability plot under it
    ├── infer_video.py           # whole-episode inference + annotated mp4 / HTML viewer
    ├── data/
    │   ├── spec.py              # [CORE] DataSpec: reads data.yaml, resolves targets
    │   ├── build_manifest.py    # [PRIMARY] index existing clips -> task-agnostic CSV
    │   ├── split_cases.py       # [PRIMARY] whole-case split; one test set per site
    │   ├── explore_data.py      # audit the manifest + the splits (read-only)
    │   ├── videomae_dataset.py  # Dataset; applies the DataSpec at load time
    │   ├── data_process.py      # [OPTIONAL] video-only clip cutter (from raw)
    │   └── process_dataset.py   # [OPTIONAL] driver for data_process.py
    ├── models/
    │   ├── base.py              # trimmed base (no LoRA/VLM); owns probs() per task
    │   ├── videomae.py          # MCG-NJU/videomae-base-finetuned-ssv2
    │   ├── videomae_giant.py    # OpenGVLab/VideoMAEv2-giant
    │   ├── classifier.py        # MLP head, width + bias init from the DataSpec
    │   └── attentionpooling.py
    └── utils/
        ├── metrics.py           # argmax OR per-activity-threshold metrics
        ├── losses.py            # weighted CE / masked BCE factory
        ├── collate.py
        └── model_loading.py
```

### Where each decision lives

`configs/data.yaml` is the only place the label regime is defined. Everything
downstream reads it through `DataSpec` (`src/data/spec.py`):

```
configs/data.yaml
      │
      ├─ task ─────────────► head width, output activation, loss, metric set
      ├─ activities ───────► class names, logit order, every metric key
      ├─ thresholds ───────► which activities count as present in a clip
      ├─ weak_threshold ───► which count as absent  (the band between is ambiguous)
      ├─ ambiguous ────────► drop the clip / call it negative / mask that activity
      ├─ buckets ──────────► which of the nine label buckets are eligible at all
      └─ decision_thresholds ► the sigmoid cut at inference (multilabel)
```

Every checkpoint stores the DataSpec it was trained with, so `test.py` and
`infer_video.py` reconstruct the right head and the right logit meaning without
being told.

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
    --test-ratio 0.20 --train-ratio 0.80 --seed 2025

python -m src.data.explore_data \
    --manifest data/clips_all.csv --splits-dir data --out-dir results/data_report
```

Producing

```
data/train.csv           data/test_haydom.csv    ~20 % of Haydom's clips
data/validation.csv      data/test_drc.csv       ~20 % of DRC's clips
                         data/test.csv           their union
data/split_assignment.csv   case -> split, for auditing
data/split_params.json      exactly how this split was made
```

with columns

```
video_path, case_id, site, bucket, clip_dir, tagged,
frac_stimulation, frac_ventilation, frac_suction
```

(the three test CSVs carry one more, `thesis_test`, see below).

These manifests are **task-agnostic**: they record the *evidence* — each clip's
original label bucket and the share of its 3 s window each activity covered, both
recovered from the filename — not a resolved label. Labels are derived from them
at Dataset load time by the DataSpec.

> **So switching task, moving a threshold, or admitting a dropped bucket requires
> no rescan and no re-split.** Edit the data config (or pass a different
> `--data-config`) and re-run training. Only a change to the clip roots, the
> activity list, or the split policy needs `build_data.sh` again.

The split assignment is deliberately independent of the data config too — cases
are stratified on raw fraction-mass, so nudging a threshold cannot silently
reshuffle cases between train and val.

`build_manifest` also prints a **bucket census** and what the current config
would resolve it into — the fastest way to see how much data each regime keeps:

```
--- bucket census (as found on disk) ---
bucket                     Haydom     DRC    TOTAL  policy
0 non_target                  ...     ...      ...  keep
...
7 target_overlap              ...     ...      ...  DROP      <- co-occurrence
...
--- resolved by configs/data.yaml (multiclass) ---
  non_target                  ...
  (dropped by this config)    ...
  usable clips: ... / ... (..%)
```

### Two generations of clips (read this before trusting any label)

The Haydom and DRC trees were cut by **different versions of `data_process.py`**,
and it matters:

| | DRC | Haydom |
|---|---|---|
| filename fraction tags (`_vent0.55`) | yes | **no** |
| activity directory (`partial/stimulation/`) | yes | yes |
| `thresholds` in the data config | apply | **inert** — frozen at the cut |

`DataSpec.resolve` originally labelled purely from the filename fractions, so
every untagged Haydom clip read as "all fractions zero" → `non_target`. That
silently mislabelled **~35,000 Haydom activity clips**, and it is why an early
audit showed Haydom contributing 0 cases to all three activities.

The identity was never actually lost: `data_process.py` files every clip under a
directory that names the activities involved (`ventilation/`,
`partial/stimulation/`, `target_overlap/stimulation+ventilation/`). So the
manifest now records `clip_dir` and `tagged` per clip, and `DataSpec.resolve`
falls back to bucket + directory when the fractions are absent:

| bucket | untagged clip resolves to |
|---|---|
| 0, 4, 5 | every activity negative |
| 1, 2, 3 | that activity positive, the others negative (the processor's purity guard) |
| 7 | every activity in the combo positive; any *other* activity ambiguous |
| 6, 8 | every activity named ambiguous; the rest negative |

What an untagged site loses is only re-thresholding — its labels are exactly the
ones `data_process.py` assigned when it cut the clips. `build_manifest` prints a
`directory → bucket → recovered activities` table so the recovery is auditable,
and warns on any clip whose directory disagrees with its bucket.

**For multilabel this is the difference between a usable Haydom and no Haydom at
all:** bucket 6 (`partial`) is 25,354 clips, 20,854 of them Haydom, and with
`ambiguous: mask` each one still supervises the activities it *can* speak to.

### One test set per hospital

Haydom and DRC differ in camera, lighting, staff and protocol, so a single pooled
test score hides the only number that matters for deployment: **does the model
work at *this* hospital.** Each site therefore gets its own test set, sized
independently as `--test-ratio` of *that site's* clips — they are deliberately
**not** the same size, because the sites hold very different amounts of data.

Cases are picked per site by a greedy + hill-climbing selector (with randomized
restarts, so it does not get stuck on a whale case) that matches the site's own
profile on five dimensions at once:

| dimension | why it is in the objective |
|---|---|
| number of cases | without it the selector hits the clip target with many *short* episodes, and the test set ends up systematically shorter than train |
| number of clips | the size target itself |
| clips with no annotated activity | keeps the negative/positive balance |
| fraction-mass of each activity | what puts the rare classes in test in usable numbers. For an untagged site this is a *nominal* mass from bucket + directory (strong 1.0, partial 0.35) — without it Haydom reports zero activity mass and gets balanced on clip count alone |

All five are **raw evidence, never resolved labels** — so nudging a threshold in
`configs/data.yaml` cannot reshuffle cases between splits. Validation is chosen
from the remaining cases the same way, per site, so it is size-balanced and
always covers both hospitals; train is everything left.

**The thesis' 14 cases are seeds, not the whole test set** (DRC `2-33998-1,
2-34325-1, 2-37178-1, 2-37453-1`; Haydom `11848523, 15233524, 28631424,
37572224, 38037024, 38042423, 38714124, 40094725, 40386725, 40402325`). They are
always placed in their own site's test set and never moved, and every test row is
flagged `thesis_test` (1/0), so the exact thesis-comparable evaluation survives
as a one-line filter:

```bash
bash scripts/test.sh VideoMAE <ckpt>.pt 0 "" --thesis-only   # scores only those 14
```

To go back to the old behaviour entirely, pass `--freeze-test` (test = the 14
seed cases, still written per site). For an exact train/val reproduction, pass
`--train-cases-file`/`--val-cases-file`.

### Auditing the data and the split

```bash
python -m src.data.explore_data --manifest data/clips_all.csv --splits-dir data
```

Read-only; it changes nothing. Six sections: the corpus and its bucket census
per site; what the current data config resolves that into; the **case geometry**
(clips-per-case distribution, Gini, top-5 concentration — the lumpiness that
makes a whole-case split hard to balance); **where each class actually lives**
(a class in 500 clips but 3 cases has an effective sample size of 3, not 500);
an audit of the existing splits (sizes, per-site composition, label mix vs. the
corpus in percentage points, leakage, and per-class clip *and case* counts); and
what a re-split at a given `--target-test-ratio` would have to move.

Its `findings` block is the part to read first:

```
findings
  [LEAK]   case 40402325 appears in ['train', 'test_haydom'] — clips from one
           episode on both sides of the split
  [THIN]   'suction' has 27 clips in test_drc (<100): one clip moves its F1 by ~3.7 points
  [NARROW] 'suction' in test_drc comes from only 2 case(s) — that score measures
           those episodes, not the class
```

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
bash scripts/train.sh VideoMAE 0                                # multiclass (default)
bash scripts/train.sh VideoMAE 0 configs/data_multilabel.yaml    # 3 independent activities
```

The third argument (or `--data-config`) picks the label regime; nothing in
`configs/config.yaml` has to change. The run logs the resolved spec, the class
distribution, the loss weights and the head bias before the first epoch — read
those three lines to confirm you got the regime you meant.

Saves best-macro-F1 and best-minority-F1 checkpoints to `checkpoints/`, a final
model to `models/`, and per-epoch metrics to `results/metrics_*.csv`. Every
checkpoint embeds its DataSpec.

### Qualitative demo — one episode per hospital

```bash
bash scripts/inference_demo.sh VideoMAE checkpoints/<ckpt>.pt
```

Picks a random case from `data/test_haydom.csv` and one from `data/test_drc.csv`,
resolves each full episode from the sibling `Unprocessed_data` tree, and writes
into `inference_output/`:

```
inference_output/haydom_<case>/
  source.mp4          copy of the original episode
  annotated.mp4       episode on top, per-second probabilities underneath
  probabilities.csv   one row per second: p_<activity> and active_<activity>
inference_output/drc_<case>/ ...
```

Pin specific episodes with `--haydom-video` / `--drc-video`, or make the random
choice reproducible with `--seed`.

The plot is **small multiples** — one thin panel per activity rather than three
lines on shared axes. Three activities each carry their own decision threshold,
so a single threshold line cannot serve all of them, and overlaid lines occlude
each other exactly where two activities co-occur (the case worth looking at).
Each panel shows its probability curve, its own threshold, and the spans the
model calls "performed"; a playhead tracks the video. Panels are titled with the
activity name so identity never depends on colour alone.

Inference is `src/infer_video.py`'s — 3 s window, 1 s stride, each window
assigned to the second nearest its centre — so the demo and the quantitative
evaluation cannot disagree. Pass `--data-config` to use tuned
`decision_thresholds`; the shaded "performed" spans follow them.

### Step 3 — Test

```bash
bash scripts/test.sh VideoMAE checkpoints/VideoMAE_best_macro_<ts>.pt 0
```

This evaluates **every test set listed under `test_data:` in the checkpoint's
config** — by default `data/test_haydom.csv` and `data/test_drc.csv` — one after
the other, then prints a side-by-side table:

```
PER-TEST-SET COMPARISON
test set              macro/f1   macro/accuracy      suction/f1
haydom                  0.7xxx           0.8xxx          0.4xxx
drc                     0.6xxx           0.7xxx          0.2xxx
  spread in macro/f1: 0.0xxx
```

**That spread is a result, not noise.** It is the cross-site generalisation gap;
read it next to each set's clip count, because the smaller site's F1 moves in
much coarser steps (`explore_data`'s `[THIN]` warnings tell you how coarse).

Variants:

```bash
bash scripts/test.sh VideoMAE <ckpt>.pt 0 "" --thesis-only          # just the 14 frozen cases
bash scripts/test.sh VideoMAE <ckpt>.pt 0 "" --test_data data/test.csv   # one pooled score
```

#### Which clips get scored: `--full-coverage`

`ambiguous: mask` is right for training — an activity whose window coverage falls
between `weak_threshold` and its `thresholds` value has no defensible binary
label, and inventing one puts noise in the loss. For **reporting** it makes the
score optimistic: deployment is a continuous stream of 3 s windows, many of them
transitional, and those are exactly the ones being set aside.

It is also not applied evenly. Haydom's clips carry no fraction tags, so every
bucket-6 clip means "present but sub-threshold" and gets masked; DRC's fractions
resolve many of the same clips outright. Grading the hospitals on subsets of
different difficulty contaminates the very comparison the per-site test sets
exist to make.

```bash
bash scripts/test.sh VideoMAE <ckpt>.pt --full-coverage
```

One inference pass, both conventions, printed side by side:

```
[haydom] AMBIGUITY CONVENTIONS
convention                  clips  supervised          macro/f1        suction/f1
full coverage (all)        13,224      39,672            0.xxxx            0.xxxx
confident subset           13,180      35,380            0.xxxx            0.xxxx

  clips dropped entirely : 44
  activity decisions set aside : 4,292 (10.8% of all)
  clips with >=1 masked activity : 4,292 (32.5% of this site)
```

A **per-class table** follows it on every run (with or without the flag) — macro
averages hide a collapsed rare class, since ventilation is ~43 % of clips and
suction ~5 %:

```
[haydom] PER-CLASS
                    full coverage (all clips)              confident subset
class          pos     sup   prec    rec     F1     AP     pos     sup   prec    rec     F1     AP
stimulation  1,102  13,224 0.4120 0.8910 0.5632 0.6431   1,102   9,800 0.5030 0.8890 0.6425 0.6702
ventilation  5,680  13,224 0.7740 0.9020 0.8331 0.9012   5,680  10,108 0.8610 0.9050 0.8824 0.9105
suction        171  13,224 0.0930 0.7600 0.1657 0.4128     171  12,990 0.1180 0.7600 0.2043 0.4210
MACRO                      0.4260 0.8510 0.5207 0.6524                   0.4940 0.8510 0.5764 0.6672
```

`sup` is the number of clips whose label for that activity is supervised, so the
gap between the two `sup` columns is exactly what the confident convention set
aside — per class, which is where the site asymmetry shows up.

Full coverage reads a sub-threshold activity as **not performed** and scores every
clip. Both go to wandb (`test/<site>/…` and `test/<site>/confident/…`) and into
the results CSV, and the file gets a `_fullcov` suffix so the two never overwrite
each other. Compare hospitals on the same convention, and say which one you used.

Note this removes the **ambiguity** exclusion, not the **bucket** one: bucket 5
(`no_label`) is still dropped, because those clips overlap spans the annotators
marked untrustworthy — admitting them as all-zero negatives asserts "nothing is
happening" where the annotation declines to say. The run prints how many are
still excluded that way. To include them too, copy the data config with
`buckets: {5: keep}` and pass it as `DATA_CONFIG`.

#### Multilabel: tune the decision thresholds, do not leave them at 0.5

Training uses `pos_weight` (`class_weighting: sqrt_inv_freq`), which inflates the
odds of the positive class on purpose so rare activities are not ignored. The
model therefore does **not** output calibrated probabilities: with weight `w` the
balanced operating point sits near `w/(1+w)`, not 0.5. Left at 0.5 a rare class
degenerates into "positive everywhere" — recall 1, precision = prevalence, so
`F1 = 2p/(1+p)`. A suction F1 of 0.24 next to a suction **AP of 0.86** is exactly
this: good ranking, wrong cut.

```bash
# 1. score VALIDATION as if it were a test set
bash scripts/test.sh VideoMAE checkpoints/<ckpt>.pt 0 "" \
    --test_data val=data/validation.csv

# 2. tune on it, writing the block back into the data config
python -m src.tune_thresholds results/scores_<ckpt>_val_<ts>.npz \
    --write configs/data_multilabel.yaml

# 3. report on TEST with the tuned cuts
bash scripts/test.sh VideoMAE checkpoints/<ckpt>.pt 0 configs/data_multilabel.yaml
```

Each activity is optimised independently over every distinct score (exact, not a
grid), masked entries excluded. `--beta 2` favours recall if a missed event costs
more than a false alarm. The script warns if you hand it anything that does not
look like validation — thresholds fitted on the split you then report are
threshold-shopped, and `test.py` prints AP alongside F1 so that stays visible.

The task is read back from the checkpoint — a multilabel model needs no extra
flags. Writes `results/results_<ckpt>_<site>_<ts>.csv` and the matching
`results/scores_*.npz` (raw logits, ground truth, and the supervision mask) per
test set, and logs each under its own `test/<site>/` wandb prefix.

**multiclass** reports per-class + macro + minority F1, accuracy, and the N×N
confusion matrix — the thesis' table.

**multilabel** reports, per activity, precision/recall/F1 **plus average
precision** (threshold-free, so results are not threshold-shopped), the 2×2
counts, macro averages, exact-match and Hamming accuracy, how many clips truly
have ≥ 2 activities versus how many were predicted so — and a **projected 4-class
confusion matrix** so one table stays comparable with the single-label results.
Clips whose ground truth has ≥ 2 activities are *excluded* from that projection
rather than tie-broken; `proj/excluded` says how many.

Re-scoring an existing multilabel checkpoint at different sigmoid cuts needs no
retraining — edit `decision_thresholds` and pass the config explicitly:

```bash
bash scripts/test.sh VideoMAE <ckpt>.pt 0 configs/data_multilabel.yaml
```

### Step 4 (optional) — Inference over a full video + viewer

Runs a trained checkpoint over an **entire episode** (not the pre-cut clips) and
shows the video together with the prediction per second. A multiclass checkpoint
draws one colour-coded timeline; a **multilabel checkpoint draws one timeline per
activity**, so seconds where two activities fire show two lit bars at once — with
the ground-truth rows stacked underneath in the same layout. It re-creates the
thesis clip scheme at inference time — a **3 s window slid with a 1 s stride**
over the whole video — classifies each window (16-frame / 224² VideoMAE
processor), and maps each window's argmax to a **per-second** label.

Just pick a case from the test set — **no paths to type**:

```bash
bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro_<ts>.pt
```

It lists the cases in `data/test.csv` (the held-out episodes of both sites) with their clip
counts and whether the raw video / annotation were found, then waits for you to
pick one:

```
Test-set cases:
  [ 1] 11848523         412 clips   video ✓                    GT ✓
  [ 2] 15233524         388 clips   video ✓                    GT ✓
  ...
Select a case [1-N] (q to quit):
```

Once picked, it **auto-resolves the full-episode video and its annotation** by
recovering the case id from the clip filename (`<case>_interval_…`) and walking
up to the sibling `Unprocessed_data/{videos,anot_files}/<case_id>.*` tree
(the layout produced by `data_process.py`). No need to know where the raw files
live. Pass a `CASE` id as the 5th arg to skip the menu
(`… VideoMAE <ckpt> 0 8000 11848523`).

#### Two output modes

**Offline / headless VM (default) — a standalone annotated `.mp4`.** The script
burns the predictions onto every frame (current label + confidence chip, a
colour-coded per-second timeline with a moving playhead, and — when an annotation
exists — a ground-truth strip below it) and writes
`viewer_out/<case_id>/annotated.mp4`. **No server, no browser, no network** — just
copy the one file off the VM and play it in any media player (VLC):

```bash
bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro_<ts>.pt
# then, from your laptop:
scp <user>@<vm>:/…/videomae-unimodal/viewer_out/11848523/annotated.mp4 .
```

**Interactive HTML viewer (needs a reachable browser).** If you *can* forward a
port (SSH / VS Code Remote), set `SERVE=1` to serve a page that plays the video
with the label following the playhead and a clickable timeline:

```bash
SERVE=1 bash scripts/infer_video.sh VideoMAE checkpoints/VideoMAE_best_macro_<ts>.pt 0 8000
# open http://localhost:8000/viewer.html  (Range-capable server, so scrubbing works)
```

Under the hood both modes are `python -m src.infer_video`:

```bash
python -m src.infer_video --model VideoMAE --model_path <ckpt.pt> --render-video  # offline mp4
python -m src.infer_video --model VideoMAE --model_path <ckpt.pt> --serve          # html viewer
# run any video directly, bypassing the test set:
python -m src.infer_video --model VideoMAE --model_path <ckpt.pt> \
    --video /…/<case_id>.mp4 [--annotation /…/<case_id>.txt] --render-video
```

Every run also writes, into `viewer_out/<case_id>/`:

- `predictions.json` — per-second + per-window predictions + metadata.
- `predictions.csv` — per-window `start,end,pred,confidence` + one column per
  output. In multilabel, `pred` is the `+`-joined set of activities over threshold
  (empty = no activity).
- `viewer.html` + `video.mp4` — the HTML player and the source video (a symlink;
  use `--copy-video` so the folder is portable and `viewer.html` can be opened
  locally via `file://` after copying it off the VM).

Ground truth is overlaid automatically when an annotation is found (disable with
`--no-gt`). It uses **the same rule as the training targets** — each activity's
share of the 3 s window, thresholded by the data config — which in multiclass
reduces to the thesis' `for_predict` rule (dominant activity if it clears its
threshold, else the negative class) and in multilabel records co-occurrence in
the ground truth too.

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

The metric *keys* follow the activity names, and `macro/f1`, `macro/accuracy` and
`minority/f1` exist in both tasks (so the charts and the checkpoint-selection
logic are unchanged). Two readings differ in multilabel: `macro/accuracy` is
**exact-match** (subset) accuracy rather than plain accuracy, and the confusion
matrix is the projected 4-class one.

## Choosing / editing a label regime

Everything below is `configs/data.yaml`. The two shipped configs are just two
points in this space.

**Duration threshold per activity.** How much of a 3 s window an activity must
cover to count as present:

```yaml
thresholds:
  stimulation: 0.50    # >= 1.5 s of the window
  ventilation: 0.50
  suction:     0.25    # >= 0.75 s — suction is brief; the thesis used a lower bar
weak_threshold: 0.20   # <= 0.6 s counts as ABSENT
```

Anything strictly between the two is **ambiguous**, and `ambiguous:` decides
what happens to it:

| value | effect |
|---|---|
| `drop` | discard the clip (the thesis' purity guard) |
| `negative` | call it a 0 — more data, more label noise |
| `mask` | *multilabel only:* supervise the clip's other activities and exclude this one from the loss |

`mask` is what makes the previously-discarded clips usable. A window that is 40 %
stimulation and 60 % ventilation is a confident ventilation **positive** and a
genuinely unknown stimulation label; masking the stimulation term keeps the
clip's real supervision without inventing a negative.

**Which buckets to keep.** `data_process.py` sorted every window into one of nine
buckets and wrote all of them to disk. `buckets:` decides which are eligible:

| bucket | what it holds | default |
|---|---|---|
| 0 `non_target` | no activity, cleanly annotated | keep |
| 1 / 2 / 3 | one activity above threshold, pure | keep |
| 4 `no_overlap` | nothing annotated at all → negative | keep |
| 5 `no_label` | no activity, but sub-threshold `nt` or an *Ignored label* span | drop |
| 6 `partial` | one activity present, sub-threshold | drop |
| 7 `target_overlap` | **≥ 2 activities all above threshold** | drop (multiclass) / keep (multilabel) |
| 8 `partial_overlap` | ≥ 2 present, none above threshold | drop |

Two things worth knowing before you change these:

- **Bucket 7 is the co-occurrence.** It is only representable under
  `task: multilabel`; a single label cannot express it, which is why the
  multiclass config drops it (or, with `overlap_resolution: dominant`, collapses
  it to the largest-fraction activity).
- **Bucket 6 is not only sub-threshold clips.** `data_process.py`'s ventilation
  branch also requires `other == 0`, so a window with `vent >= 0.50` contaminated
  by an *Ignored label* interval was demoted to `partial/ventilation/`. Keeping
  bucket 6 readmits those at full strength — along with the annotation
  contamination they were excluded for. That trade-off is invisible from the
  filenames (the `nt` and `other` overlaps are not encoded there); only a
  re-derivation from the annotation TSVs would expose it.
- **Bucket 5 stays dropped by default** even in multilabel. Those clips have zero
  activity overlap but were rejected for containing *Ignored label* spans —
  annotator-flagged unreliable regions. Admitting them as all-zero negatives
  would teach the model that unannotated activity is "nothing".

**Renaming or adding an activity** is a config change: extend `activities`, add
its `tag_keys` abbreviation (the token `data_process.py` writes in the filename),
its `thresholds` entry, and — if inference should read it out of the annotation
TSVs — an `annotation_events` override when the event string is not just the
title-cased name.

**Precision caveat.** Filename fractions are rounded to 2 decimals
(`f"{frac:.2f}"`), i.e. ±0.005 ≈ ±15 ms of a 3 s window. Irrelevant at the
default thresholds; worth remembering if you set something like `0.335`.

---

## Notes

- `num_classes`, the output activation, the loss and the metric set are all
  derived from `configs/data.yaml` — 4 logits + softmax + weighted CE in
  multiclass, 3 logits + sigmoid + masked BCE in multilabel. The head bias is
  initialised to the per-class log-priors (softmax) or logit-priors (sigmoid), and
  the loss weights to sqrt inverse-frequency (`class_weighting` in
  `configs/config.yaml`), both derived from the train split at runtime.
- A **legacy** `video_path,label` manifest still loads in multiclass mode, with a
  warning that the bucket/threshold settings are being ignored. Multilabel needs
  the fraction columns, so rebuild the manifest for it.
- VideoMAEv2-giant needs `timm` + `easydict` and ~24 GB VRAM (use bf16/fp16, the
  default AMP path).
- Cross-site experiment (train Haydom → test DRC): the per-site test sets already
  give you the *evaluation* half of this for free (`test/haydom/*` vs `test/drc/*`
  in one run). For the training half, filter `data/train.csv` and
  `data/validation.csv` on `site == "Haydom"`, mirroring the thesis' Haydom-only
  generalisation study.
