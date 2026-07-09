"""
test.py

Evaluate a trained VideoMAE / VideoMAEv2-giant checkpoint on the held-out test
split (the 14 fixed thesis test cases) and write clip-level metrics + raw scores.

Single-label adaptation of the multimodal repo's test.py:
    * Predictions = argmax over 4 softmax logits (no per-class thresholds).
    * Saves per-class + macro + minority F1 + confusion matrix, plus a .npz of
      raw logits (N,4) and ground-truth class indices (N,) for later analysis.
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

from src.utils import load_model, collate_fn, compute_metrics, CLASSES, DEFAULT_MINORITY_CLASS, wandb_utils as wu
from src.data import VideoMAEDataset

VIT_MODELS = ["VideoMAE", "VideoMAEGiant"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VIT_MODELS)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None,
                        help="Override the test CSV path stored in the checkpoint config.")
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

    model = load_model(args.model, num_classes=4)
    model = model.to(device)

    test_csv = args.test_data or config.get("test_data", "data/test.csv")
    logger.info(f"Test CSV: {test_csv}")
    test_dataset = VideoMAEDataset(test_csv, processor=model.processor, num_frames=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=config.get("num_workers", 4),
                             collate_fn=collate_fn)
    logger.info(f"Test size: {len(test_dataset)}")

    model.load_classifier(saved, config)
    model.load_backbone(saved, config)
    if config.get("attention_pooling", False):
        model.load_attention_pooling(saved)
    model.eval()

    if _HAS_WANDB:
        wandb.init(project=config.get("wandb_project", "videomae-unimodal"),
                   name=f"test_{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   config={**config, "test_data": test_csv}, mode=config.get("wandb_mode", "online"),
                   job_type="eval")

    n, c = len(test_dataset), model.num_classes
    logits_t = torch.empty((n, c), dtype=torch.float32)
    labels_t = torch.empty((n,), dtype=torch.long)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            labels = batch.pop("labels").to(device)
            with autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch)
            logits_t[i] = logits.detach().float().cpu().squeeze(0)
            labels_t[i] = labels.detach().cpu().squeeze(0)

    metrics = compute_metrics(logits_t, labels_t, minority_class=minority_class)
    logger.info(f"macro/f1={metrics['macro/f1']:.4f}  macro/accuracy={metrics['macro/accuracy']:.4f}  "
                f"{minority_class}/f1={metrics.get('minority/f1', float('nan')):.4f}")
    wu.log_metrics(metrics, prefix="test/")
    wu.log_confusion_matrix(logits_t, labels_t, key="test/confusion_matrix")
    wu.update_summary({f"test/{k}": float(v) for k, v in metrics.items() if not k.startswith("cm/")})

    os.makedirs(args.results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.model_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(args.results_dir, f"results_{base}_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["model", base])
        w.writerow(["classes", "|".join(CLASSES)])
        for k, v in metrics.items():
            w.writerow([k, round(float(v), 6) if not k.startswith("cm/") else int(v)])
    logger.info(f"Results -> {csv_path}")

    scores_path = os.path.join(args.results_dir, f"scores_{base}_{ts}.npz")
    np.savez(scores_path,
             logits=logits_t.numpy(),           # (N, 4) raw logits
             labels=labels_t.numpy(),           # (N,) class indices
             classes=np.array(CLASSES))
    logger.info(f"Scores -> {scores_path}")

    # Store the test outputs in wandb so they persist with the run.
    wu.log_artifact(
        name=f"test-results-{base}",
        artifact_type="test-results",
        files=[csv_path, scores_path],
        metadata={"macro_f1": metrics["macro/f1"], "accuracy": metrics["macro/accuracy"],
                  f"{minority_class}_f1": metrics.get("minority/f1"), "test_data": test_csv},
    )
    wu.finish()


if __name__ == "__main__":
    main()
