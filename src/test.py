"""
test.py

Evaluate a trained VideoMAE / VideoMAEv2-giant checkpoint on the held-out test
splits and write clip-level metrics + raw scores for each.

There is ONE TEST SET PER HOSPITAL (data/test_haydom.csv, data/test_drc.csv —
see src/data/split_cases.py), because the sites differ in camera, lighting and
protocol and a pooled score hides the number that actually matters: does this
model work at THIS hospital. Every set listed under `test_data:` in the config
is evaluated in turn, each writing its own results_*.csv / scores_*.npz and
logging under its own `test/<name>/` wandb prefix, followed by a comparison
table. Pass `--test_data` to evaluate something else (repeatable, `NAME=PATH`
allowed); pass `--thesis-only` to score just the thesis' 14 frozen cases, which
are still flagged inside the test CSVs by the `thesis_test` column.

The task comes from the CHECKPOINT, not from the current configs/data.yaml:
training.py stores the DataSpec it trained with under "data_spec", and that is
what fixes the head width, the output activation and the meaning of every logit.
Pass --data-config only to override it deliberately — e.g. to re-score an
existing multilabel checkpoint at different `decision_thresholds` (that changes
predictions but not the model).

    multiclass : predictions = argmax over the softmax logits.
    multilabel : predictions = independent per-activity sigmoid cuts, plus
                 threshold-free average precision and a projected single-label
                 confusion matrix for comparability with the 4-class results.

Outputs: results_*.csv (all metrics incl. confusion matrix) and scores_*.npz
(raw logits + ground truth + supervision mask) for later analysis.
"""

from argparse import ArgumentParser
import csv
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except Exception:  # pragma: no cover
    _HAS_WANDB = False

import pandas as pd

from src.utils import (load_model, collate_fn, compute_metrics,
                       DEFAULT_MINORITY_CLASS, wandb_utils as wu)
from src.data import VideoMAEDataset, DataSpec, spec_from_checkpoint

VIT_MODELS = ["VideoMAE", "VideoMAEGiant"]

DEFAULT_TEST_SETS = {"haydom": "data/test_haydom.csv", "drc": "data/test_drc.csv"}
THESIS_COLUMN = "thesis_test"


def resolve_test_sets(cli, config) -> list[tuple[str, str]]:
    """[(name, csv_path)] from --test_data, else the checkpoint config, else the
    per-site defaults.

    Accepts `PATH` or `NAME=PATH` on the CLI, and either a plain string (legacy,
    one pooled test set) or a {name: path} mapping in the config. Names become
    the wandb metric prefix and part of every output filename, so a run over two
    hospitals never overwrites its own results.
    """
    if cli:
        entries = [e.split("=", 1) if "=" in e else [Path(e).stem, e] for e in cli]
        return [(n.replace("test_", "") or "test", p) for n, p in entries]
    cfg = config.get("test_data")
    if isinstance(cfg, dict):
        return [(str(k).lower(), str(v)) for k, v in cfg.items()]
    if isinstance(cfg, str):
        return [(Path(cfg).stem.replace("test_", "") or "test", cfg)]
    return list(DEFAULT_TEST_SETS.items())


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VIT_MODELS)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, nargs="*", default=None,
                        help="Override the test CSVs from the checkpoint config. "
                             "Repeatable; each entry is PATH or NAME=PATH.")
    parser.add_argument("--thesis-only", action="store_true",
                        help=f"Score only the thesis' frozen cases (rows with "
                             f"{THESIS_COLUMN} == 1), for a like-for-like comparison "
                             f"with the multimodal results.")
    parser.add_argument("--data-config", type=str, default=None,
                        help="Override the DataSpec stored in the checkpoint. Only do this "
                             "on purpose — a mismatched spec changes what every logit means.")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--minority_class", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    saved = torch.load(args.model_path, map_location=device, weights_only=False)
    config = saved.get("config", {})
    minority_class = args.minority_class or config.get("minority_class", DEFAULT_MINORITY_CLASS)

    if args.data_config:
        spec = DataSpec.load(args.data_config)
        logger.warning(f"overriding the checkpoint's DataSpec with {args.data_config}")
    else:
        spec = spec_from_checkpoint(saved, config.get("data_config"))
        if "data_spec" not in saved:
            logger.warning("checkpoint has no stored DataSpec (trained before this was "
                           "recorded) — falling back to the data config on disk. Verify "
                           "it matches how the checkpoint was trained.")
    logger.info(spec.describe())

    model = load_model(args.model, spec=spec)
    model = model.to(device)

    test_sets = resolve_test_sets(args.test_data, config)
    logger.info("Test sets: " + ", ".join(f"{n} -> {c}" for n, c in test_sets))

    model.load_classifier(saved, config)
    model.load_backbone(saved, config)
    if config.get("attention_pooling", False):
        model.load_attention_pooling(saved)
    model.eval()

    if _HAS_WANDB:
        wandb.init(project=config.get("wandb_project", "videomae-unimodal"),
                   name=f"test_{model.model_name}_{spec.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   config={**config, "test_data": dict(test_sets),
                           "thesis_only": args.thesis_only,
                           "data_spec": spec.to_dict()},
                   mode=config.get("wandb_mode", "online"), job_type="eval")

    os.makedirs(args.results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.model_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_metrics, all_files = {}, []
    for name, test_csv in test_sets:
        metrics, files = run_test_set(name, test_csv, model=model, spec=spec, args=args,
                                      config=config, device=device, amp_dtype=amp_dtype,
                                      minority_class=minority_class, base=base, ts=ts,
                                      logger=logger)
        if metrics is None:
            continue
        all_metrics[name] = metrics
        all_files += files

    if not all_metrics:
        raise SystemExit("no test set produced any metrics — check the paths above.")

    report_comparison(all_metrics, spec, minority_class, logger)

    # Store every test set's outputs in wandb so they persist with the run.
    wu.log_artifact(
        name=f"test-results-{base}",
        artifact_type="test-results",
        files=all_files,
        metadata={"task": spec.task, "test_sets": dict(test_sets),
                  "thesis_only": args.thesis_only,
                  **{f"{n}/macro_f1": float(m["macro/f1"]) for n, m in all_metrics.items()},
                  **{f"{n}/{minority_class}_f1": m.get("minority/f1")
                     for n, m in all_metrics.items()}},
    )
    wu.finish()


def run_test_set(name, test_csv, *, model, spec, args, config, device, amp_dtype,
                 minority_class, base, ts, logger):
    """Evaluate one test CSV. Returns (metrics, written files), or (None, []) if
    the CSV is missing — a missing site should not abort the other site's score."""
    if not os.path.exists(test_csv):
        logger.warning(f"[{name}] {test_csv} not found — skipped. Run "
                       f"scripts/build_data.sh to regenerate the splits.")
        return None, []

    logger.info(f"\n{'=' * 70}\n[{name}] {test_csv}\n{'=' * 70}")
    rows = pd.read_csv(test_csv)
    if args.thesis_only:
        if THESIS_COLUMN not in rows.columns:
            logger.warning(f"[{name}] no `{THESIS_COLUMN}` column in {test_csv} — it "
                           f"predates the per-site split; scoring every row instead.")
        else:
            rows = rows[rows[THESIS_COLUMN] == 1]
            logger.info(f"[{name}] --thesis-only: {len(rows)} clips from "
                        f"{rows['case_id'].nunique()} frozen cases")
            if rows.empty:
                logger.warning(f"[{name}] no thesis cases in this set — skipped.")
                return None, []

    test_dataset = VideoMAEDataset(rows, processor=model.processor, spec=spec, num_frames=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=config.get("num_workers", 4),
                             collate_fn=collate_fn)
    logger.info(f"[{name}] {len(test_dataset)} clips")
    logger.info(f"[{name}] label distribution:\n" + test_dataset.describe_labels())

    n, c = len(test_dataset), spec.num_classes
    logits_t = torch.empty((n, c), dtype=torch.float32)
    if spec.is_multilabel:
        labels_t = torch.empty((n, c), dtype=torch.float32)
        masks_t = torch.empty((n, c), dtype=torch.float32)
    else:
        labels_t = torch.empty((n,), dtype=torch.long)
        masks_t = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Testing[{name}]")):
            labels = batch.pop("labels").to(device)
            mask = batch.pop("label_mask", None)
            with autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch)
            logits_t[i] = logits.detach().float().cpu().squeeze(0)
            labels_t[i] = (labels.detach().float() if spec.is_multilabel
                           else labels.detach()).cpu().squeeze(0)
            if masks_t is not None:
                masks_t[i] = (torch.ones(c) if mask is None
                              else mask.detach().float().cpu().squeeze(0))

    metrics = compute_metrics(logits_t, labels_t, spec, masks=masks_t,
                              minority_class=minority_class)
    logger.info(f"[{name}] macro/f1={metrics['macro/f1']:.4f}  "
                f"macro/accuracy={metrics['macro/accuracy']:.4f}  "
                f"{minority_class}/f1={metrics.get('minority/f1', float('nan')):.4f}")
    if spec.is_multilabel:
        logger.info(f"[{name}] macro/ap={metrics['macro/ap']:.4f}  "
                    f"hamming/accuracy={metrics['hamming/accuracy']:.4f}  "
                    f"projected macro/f1={metrics.get('proj/macro_f1', float('nan')):.4f} "
                    f"(excluded {metrics.get('proj/excluded', 0)} ambiguous clips)")
        logger.info(f"[{name}] co-occurrence: {metrics['true/multi_active']} clips truly "
                    f"have >=2 activities, {metrics['pred/multi_active']} were predicted so")

    wu.log_metrics(metrics, prefix=f"test/{name}/")
    wu.log_confusion_matrix(logits_t, labels_t, spec,
                            key=f"test/{name}/confusion_matrix", masks=masks_t)
    wu.update_summary({f"test/{name}/{k}": float(v) for k, v in metrics.items()
                       if not k.startswith("cm/")})

    suffix = f"{base}_{name}{'_thesis' if args.thesis_only else ''}_{ts}"
    csv_path = os.path.join(args.results_dir, f"results_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["model", base])
        w.writerow(["test_set", name])
        w.writerow(["test_data", test_csv])
        w.writerow(["n_clips", n])
        w.writerow(["task", spec.task])
        w.writerow(["classes", "|".join(spec.class_names)])
        if spec.is_multilabel:
            w.writerow(["decision_thresholds",
                        "|".join(f"{a}={t}" for a, t in
                                 zip(spec.activities, spec.sigmoid_thresholds()))])
        for k, v in metrics.items():
            w.writerow([k, round(float(v), 6) if not k.startswith("cm/") else int(v)])
    logger.info(f"[{name}] results -> {csv_path}")

    scores_path = os.path.join(args.results_dir, f"scores_{suffix}.npz")
    np.savez(scores_path,
             logits=logits_t.numpy(),                 # (N, C) raw logits
             labels=labels_t.numpy(),                 # (N,) indices | (N, C) 0/1
             masks=(np.ones((n, c), dtype=np.float32) if masks_t is None
                    else masks_t.numpy()),            # (N, C) 1 = supervised
             classes=np.array(spec.class_names),
             task=np.array(spec.task))
    logger.info(f"[{name}] scores -> {scores_path}")
    return metrics, [csv_path, scores_path]


def report_comparison(all_metrics, spec, minority_class, logger):
    """Side-by-side table across test sets — the point of splitting them by site.

    A gap between the hospitals here is a generalisation result, not noise to be
    averaged away: read it together with each set's clip count, since the smaller
    site's F1 moves in much coarser steps.
    """
    keys = ["macro/f1", "macro/accuracy", "minority/f1"]
    if spec.is_multilabel:
        keys += ["macro/ap", "hamming/accuracy"]
    width = max(len(n) for n in all_metrics) + 2
    header = f"{'test set':<{width}}" + "".join(
        f"{(minority_class + '/f1' if k == 'minority/f1' else k):>18}" for k in keys)
    lines = ["", "=" * len(header), "PER-TEST-SET COMPARISON", "=" * len(header), header]
    for name, m in all_metrics.items():
        lines.append(f"{name:<{width}}" + "".join(
            f"{m.get(k, float('nan')):>18.4f}" for k in keys))
    if len(all_metrics) > 1:
        for k in keys:
            vals = [m.get(k) for m in all_metrics.values() if m.get(k) is not None]
            if len(vals) > 1:
                lines.append(f"  spread in {k}: {max(vals) - min(vals):.4f}")
    logger.info("\n".join(lines))
    wu.update_summary({f"test/{n}/{k}": float(m[k]) for n, m in all_metrics.items()
                       for k in keys if k in m})


if __name__ == "__main__":
    main()
