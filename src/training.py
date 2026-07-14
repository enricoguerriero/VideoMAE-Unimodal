"""
training.py

Train VideoMAE / VideoMAEv2-giant for SINGLE-LABEL 4-class neonatal
resuscitation activity recognition (non_target / stimulation / ventilation /
suction) on 3-second clips.

This is the single-label adaptation of the multimodal repo's training.py, made
comparable to that thesis's MoViNet video base model:
    * Loss: weighted CrossEntropyLoss with sqrt inverse-frequency class weights
      and label_smoothing=0.1 (matches the thesis) — NOT BCEWithLogitsLoss.
    * Metrics: argmax over 4 softmax logits (no per-class thresholds).
    * Model selection: keeps the best macro-F1 checkpoint AND the best
      minority-class (suction) F1 checkpoint, as the thesis did.
    * LR schedule: ReduceLROnPlateau on the minority-class val F1, plain cosine
      decay, or linear-warmup -> cosine decay (warmup_cosine).

Config is read from configs/config.yaml. CLI flags: --model (required),
--debug, --only_train, --attention_pooling.
"""

from argparse import ArgumentParser
import csv
import logging
import os
from datetime import datetime

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except Exception:  # pragma: no cover
    _HAS_WANDB = False

from src.utils import load_model, collate_fn, compute_metrics, DEFAULT_MINORITY_CLASS, wandb_utils as wu
from src.data import VideoMAEDataset

VIT_MODELS = ["VideoMAE", "VideoMAEGiant"]


def save_metrics_to_csv(csv_path, metrics, val_loss, epoch, split, batch=None):
    """Append one row of validation metrics to a persistent CSV (W&B-independent)."""
    scalar = {k: v for k, v in metrics.items() if not k.startswith("cm/")}
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "split", "epoch", "batch", "val_loss"] + sorted(scalar.keys()))
        writer.writerow(
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), split, epoch,
             batch if batch is not None else "epoch_end", round(float(val_loss), 6)]
            + [round(float(scalar[k]), 6) for k in sorted(scalar.keys())]
        )


def run_validation(model, val_loader, criterion, device, amp_dtype, n_val, n_classes, minority_class):
    """Full pass over the validation loader; returns (metrics, mean_val_loss)."""
    model.eval()
    logits_t = torch.empty((n_val, n_classes), dtype=torch.float32)
    labels_t = torch.empty((n_val,), dtype=torch.long)
    val_loss, seen = 0.0, 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            labels = batch.pop("labels").to(device)
            logits = model(**batch)
            loss = criterion(logits, labels)
            bs = labels.size(0)
            logits_t[seen:seen + bs] = logits.detach().float().cpu()
            labels_t[seen:seen + bs] = labels.detach().cpu()
            val_loss += loss.item() * bs
            seen += bs
    val_loss /= max(seen, 1)
    metrics = compute_metrics(logits_t, labels_t, minority_class=minority_class)
    return metrics, val_loss, logits_t, labels_t


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VIT_MODELS)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--only_train", action="store_true", default=False)
    parser.add_argument("--attention_pooling", action="store_true", default=False)
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if args.attention_pooling:
        config["attention_pooling"] = True

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Training {args.model} (single-label 4-class)")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = config.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    metrics_csv_path = os.path.join(results_dir, f"metrics_{args.model}_{run_ts}.csv")

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.model, num_classes=4, **config.get("model_params", {}))
    model = model.to(device)

    if _HAS_WANDB:
        wandb.init(project=config.get("wandb_project", "videomae-unimodal"),
                   name=f"train_{model.model_name}_{run_ts}", config=config,
                   mode=config.get("wandb_mode", "online"),
                   job_type="train")
        wu.define_epoch_metrics()

    minority_class = config.get("minority_class", DEFAULT_MINORITY_CLASS)

    # ------------------------------------------------------------------ datasets
    train_dataset = VideoMAEDataset(config["train_data"], processor=model.processor, num_frames=16)
    if not args.only_train:
        val_dataset = VideoMAEDataset(config["validation_data"], processor=model.processor, num_frames=16)

    logger.info(f"Train size: {len(train_dataset)}"
                + ("" if args.only_train else f" | Val size: {len(val_dataset)}"))

    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 8), shuffle=True,
                              num_workers=config.get("num_workers", 4),
                              pin_memory=config.get("num_workers", 4) > 0,
                              collate_fn=collate_fn, drop_last=True)
    if not args.only_train:
        val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 8), shuffle=False,
                                num_workers=config.get("num_workers", 4),
                                pin_memory=config.get("num_workers", 4) > 0,
                                collate_fn=collate_fn)

    # ------------------------------------------------- class weights & head bias
    class_weights = train_dataset.compute_class_weights()
    bias = train_dataset.compute_bias()
    logger.info(f"Class weights (sqrt inv-freq): {class_weights.tolist()}")
    logger.info(f"Head bias (log-priors): {bias.tolist()}")

    model.build_classifier(classifier_config=config.get("classifier_config", {}), bias=bias)
    if config.get("attention_pooling", False):
        model.build_attention_pooling()

    # -------------------------------------------------------- freeze / unfreeze
    for p in model.parameters():
        p.requires_grad = False
    if config.get("train_backbone", True):
        for p in model.backbone.parameters():
            p.requires_grad = True
    if config.get("attention_pooling", False):
        for p in model.attn_pool.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    model.to(device)
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------- optimizer
    if config.get("learning_rate", None) is not None:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=config["learning_rate"], weight_decay=config.get("weight_decay", 1e-3))
    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = list(model.classifier.parameters())
        if config.get("attention_pooling", False):
            head_params += list(model.attn_pool.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": config.get("backbone_lr", 1e-5)},
             {"params": head_params, "lr": config.get("classifier_lr", 5e-5)}],
            weight_decay=config.get("weight_decay", 1e-3))

    num_epochs = config.get("num_epochs", 80)
    scheduler_type = config.get("scheduler", "cosine")
    if scheduler_type == "plateau" and not args.only_train:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=config.get("plateau_patience", 8), min_lr=1e-8)
    elif scheduler_type == "warmup_cosine":
        # Linear warmup for `warmup_epochs`, then cosine decay over the rest.
        # Deterministic (not tied to noisy val F1): LR reliably winds down so the
        # model settles into the minimum instead of oscillating at a high LR.
        warmup_epochs = max(1, min(int(config.get("warmup_epochs", 5)), num_epochs - 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.get("warmup_start_factor", 0.01),
            end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        logger.info(f"Scheduler: warmup_cosine (warmup_epochs={warmup_epochs}, "
                    f"start_factor={config.get('warmup_start_factor', 0.01)})")
    else:
        scheduler_type = "cosine"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=config.get("label_smoothing", 0.1))

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    N, C = len(train_dataset), model.num_classes
    N_val = len(val_dataset) if not args.only_train else 0
    val_step = config.get("validation_step", None) if not args.only_train else None

    best_macro_f1, best_minority_f1 = -1.0, -1.0
    best_epoch_macro, best_epoch_minority = -1, -1
    epochs_no_improve = 0
    global_step = 0
    ckpt_dir = config.get("checkpoint_path", "checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_ckpt(tag, epoch, metrics, val_loss):
        path = os.path.join(ckpt_dir, f"{model.model_name}_{tag}_{run_ts}.pt")
        torch.save({
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "attention_pooling": model.attn_pool.state_dict() if model.attn_pool is not None else None,
            "processor": model.processor,
            "epoch": epoch, "val_loss": val_loss,
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("cm/")},
            "classifier_config": config.get("classifier_config", {}),
            "config": config,
        }, path)
        logger.info(f"Saved {tag} checkpoint -> {path}")

    # ------------------------------------------------------------------- loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss, seen = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} train", total=N // config.get("batch_size", 8)):
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            global_step += 1
            if global_step % 50 == 0:
                wu.log({"train/loss": train_loss / seen, "train/global_step": global_step,
                        "epoch": epoch + 1})

            if val_step is not None and seen % val_step == 0:
                metrics, val_loss, vl, vt = run_validation(model, val_loader, criterion, device,
                                                           amp_dtype, N_val, C, minority_class)
                model.train()
                save_metrics_to_csv(metrics_csv_path, metrics, val_loss, epoch + 1, "val_step", seen)
                wu.log_metrics(metrics, prefix="val_step/",
                               extra={"val_step/loss": val_loss, "train/global_step": global_step})

        train_loss /= max(seen, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        wu.log({"train/loss_epoch": train_loss, "lr": current_lr, "epoch": epoch + 1})

        if args.only_train:
            if scheduler_type != "plateau":
                scheduler.step()
            continue

        # ------------------------------------------------- end-of-epoch validation
        metrics, val_loss, val_logits, val_labels = run_validation(
            model, val_loader, criterion, device, amp_dtype, N_val, C, minority_class)
        save_metrics_to_csv(metrics_csv_path, metrics, val_loss, epoch + 1, "val_epoch")
        macro_f1 = metrics["macro/f1"]
        minority_f1 = metrics.get("minority/f1", 0.0)
        logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"macro_f1={macro_f1:.4f} {minority_class}_f1={minority_f1:.4f}")
        wu.log_metrics(metrics, prefix="val/", extra={"val/loss": val_loss, "epoch": epoch + 1})
        wu.log_confusion_matrix(val_logits, val_labels, key="val/confusion_matrix",
                                extra={"epoch": epoch + 1})

        improved = False
        if macro_f1 > best_macro_f1:
            best_macro_f1, best_epoch_macro = macro_f1, epoch + 1
            save_ckpt("best_macro", epoch + 1, metrics, val_loss)
            improved = True
        if minority_f1 > best_minority_f1:
            best_minority_f1, best_epoch_minority = minority_f1, epoch + 1
            save_ckpt(f"best_{minority_class}", epoch + 1, metrics, val_loss)
            improved = True
        epochs_no_improve = 0 if improved else epochs_no_improve + 1
        wu.update_summary({
            "best/macro_f1": best_macro_f1, "best/macro_f1_epoch": best_epoch_macro,
            f"best/{minority_class}_f1": best_minority_f1,
            f"best/{minority_class}_f1_epoch": best_epoch_minority,
        })

        if scheduler_type == "plateau":
            scheduler.step(minority_f1)
        else:
            scheduler.step()

        patience = config.get("early_stopping_patience", 20)
        if patience and epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (no {minority_class}/macro F1 gain in {patience} epochs)")
            break

    # -------------------------------------------------------------- final model
    final_path = os.path.join(config.get("save_path", "models/"), f"{model.model_name}_final_{run_ts}.pt")
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    torch.save({
        "backbone": model.backbone.state_dict(),
        "classifier": model.classifier.state_dict(),
        "attention_pooling": model.attn_pool.state_dict() if model.attn_pool is not None else None,
        "processor": model.processor,
        "classifier_config": config.get("classifier_config", {}),
        "config": config,
    }, final_path)
    logger.info(f"Final model saved -> {final_path}")
    wu.finish()


if __name__ == "__main__":
    main()
