"""
training.py

Train VideoMAE / VideoMAEv2-giant for neonatal resuscitation activity
recognition on 3-second clips, in whichever mode configs/data.yaml selects:

    task: multiclass   softmax head over [non_target] + activities,
                       weighted CrossEntropyLoss with sqrt inverse-frequency
                       class weights and label smoothing.  <- the thesis' setup
    task: multilabel   sigmoid head, one independent logit per activity,
                       masked BCEWithLogitsLoss with sqrt(neg/pos) pos_weight.
                       "No activity" is the all-zero vector, not a class.

The training loop itself is task-blind. Everything that differs is resolved
before the loop starts:
    * DataSpec           -> head width, output activation, targets, masks
    * build_criterion    -> CE vs masked BCE, weights from the TRAIN split
    * compute_metrics    -> argmax vs per-activity thresholds
and both losses are called through the same (logits, labels, mask) signature.

Model selection is unchanged: it keeps the best macro-F1 checkpoint AND the best
minority-class F1 checkpoint, as the thesis did. In multilabel runs macro-F1 is
the mean over activities and `macro/accuracy` is exact-match accuracy.

Config: configs/config.yaml (+ configs/data.yaml, or --data-config).
CLI flags: --model (required), --data-config, --debug, --only_train,
--attention_pooling.
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

from src.utils import (load_model, collate_fn, compute_metrics, build_criterion,
                       DEFAULT_MINORITY_CLASS, wandb_utils as wu)
from src.data import VideoMAEDataset, DataSpec
from src.data.manifest import read_manifest

VIT_MODELS = ["VideoMAE", "VideoMAEGiant"]


FIXED_CSV_COLUMNS = ["timestamp", "split", "epoch", "batch", "val_loss"]


def save_metrics_to_csv(csv_path, metrics, val_loss, epoch, split, batch=None):
    """Append one row of validation metrics to a persistent CSV (W&B-independent).

    Written through a DictWriter keyed on the file's own header, so a column
    always means the same metric. The previous version wrote the header once and
    then emitted every later row in that row's own `sorted(keys())` order, which
    silently shifted every column after any key that came or went. compute_metrics
    does not guarantee a fixed key set — `proj/macro_f1`, `proj/accuracy` and the
    projected `cm/...` counts only exist when the projected single-label view has
    a clip to score (see metrics._multilabel_metrics). In practice that depends on
    the ground truth alone, so it holds still within one run; it is not something
    the writer should be relying on.

    A genuinely new key rewrites the file with a widened header rather than
    dropping the value or misaligning the row.
    """
    scalar = {k: round(float(v), 6) for k, v in metrics.items() if not k.startswith("cm/")}
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
        "epoch": epoch,
        "batch": batch if batch is not None else "epoch_end",
        "val_loss": round(float(val_loss), 6),
        **scalar,
    }

    header, old_rows, widen = [], [], True
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            widen = bool(set(scalar) - set(header)) or not header
            if widen and header:
                old_rows = list(reader)   # only pay the re-read when the header grows

    if not widen:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header, restval="",
                           extrasaction="ignore").writerow(row)
        return

    header = FIXED_CSV_COLUMNS + sorted((set(header) - set(FIXED_CSV_COLUMNS)) | set(scalar))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def select_sites(csv_path, sites, split_name, logger):
    """Restrict a split CSV to `sites`, or hand the path straight through.

    The hospitals differ in camera, lighting, staff and protocol, so training a
    single-site model is a question worth asking on its own. The split CSVs are
    built at WHOLE-CASE level (src/data/split_cases.py), so filtering by site
    keeps the no-leakage guarantee intact — a case belongs to one site and one
    split.

    Returns the path itself when no filter is asked for, so an unfiltered run
    reads the file exactly as it did before. With a filter it returns the
    DataFrame, read through `read_manifest` so `case_id` keeps its leading zeros
    and an empty `clip_dir` stays "" rather than becoming NaN.

    Matching is case-insensitive ("haydom" finds the manifest's "Haydom"), but an
    unrecognised site is an ERROR: a typo that silently produced an empty split
    would look exactly like a site with no data.
    """
    if not sites:
        return csv_path
    df = read_manifest(csv_path)
    if "site" not in df.columns:
        raise SystemExit(
            f"--sites was given but {csv_path} has no `site` column — it predates "
            f"the per-site manifest. Rebuild it with `bash scripts/build_data.sh`.")
    available = {str(v).lower(): str(v) for v in df["site"].unique()}
    unknown = [s for s in sites if s.lower() not in available]
    if unknown:
        raise SystemExit(
            f"--sites {unknown} not present in {csv_path}. "
            f"Available: {sorted(available.values())}")
    keep = sorted({available[s.lower()] for s in sites})
    out = df[df["site"].isin(keep)].reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"no {split_name} clips left after --sites {keep}")
    cases = f", {out['case_id'].nunique()} cases" if "case_id" in out.columns else ""
    logger.info(f"{split_name}: --sites {keep} keeps {len(out):,}/{len(df):,} clips{cases}")
    return out


def alloc_targets(n, spec):
    """Pre-allocate the ground-truth buffers for one evaluation pass.

    multiclass -> (labels (N,) int64, masks None)
    multilabel -> (labels (N,C) float32, masks (N,C) float32)
    """
    if spec.is_multilabel:
        return (torch.empty((n, spec.num_classes), dtype=torch.float32),
                torch.empty((n, spec.num_classes), dtype=torch.float32))
    return torch.empty((n,), dtype=torch.long), None


def run_validation(model, val_loader, criterion, device, amp_dtype, n_val, spec,
                   minority_class):
    """Full pass over the validation loader.

    Returns (metrics, mean_val_loss, logits, labels, masks) — masks is None in
    multiclass.
    """
    model.eval()
    logits_t = torch.empty((n_val, spec.num_classes), dtype=torch.float32)
    labels_t, masks_t = alloc_targets(n_val, spec)
    val_loss, seen = 0.0, 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            labels = batch.pop("labels").to(device)
            mask = batch.pop("label_mask", None)
            mask = None if mask is None else mask.to(device)
            logits = model(**batch)
            loss = criterion(logits, labels, mask)
            bs = labels.size(0)
            logits_t[seen:seen + bs] = logits.detach().float().cpu()
            labels_t[seen:seen + bs] = labels.detach().float().cpu() if spec.is_multilabel \
                else labels.detach().cpu()
            if masks_t is not None:
                masks_t[seen:seen + bs] = (torch.ones(bs, spec.num_classes) if mask is None
                                           else mask.detach().float().cpu())
            val_loss += loss.item() * bs
            seen += bs
    val_loss /= max(seen, 1)
    metrics = compute_metrics(logits_t, labels_t, spec, masks=masks_t,
                              minority_class=minority_class)
    return metrics, val_loss, logits_t, labels_t, masks_t


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VIT_MODELS)
    parser.add_argument("--data-config", type=str, default=None,
                        help="Data/label config YAML. Defaults to `data_config:` in "
                             "configs/config.yaml, else configs/data.yaml.")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--only_train", action="store_true", default=False)
    parser.add_argument("--attention_pooling", action="store_true", default=False)
    parser.add_argument("--sites", nargs="+", default=None, metavar="SITE",
                        help="Train and validate on these hospitals only, e.g. "
                             "`--sites Haydom`. Case-insensitive; repeatable "
                             "(`--sites Haydom DRC`). Filters `train_data` and "
                             "`validation_data` on their `site` column. Omit to use "
                             "every site in the split CSVs. Test sets are already "
                             "per-hospital files — see `test_data:` in the config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override `num_epochs` from configs/config.yaml.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override `early_stopping_patience`. 0 disables early "
                             "stopping.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train the classifier head ONLY: the backbone keeps its "
                             "pretrained weights and receives no gradient. Equivalent "
                             "to `train_backbone: false` in the config.")
    parser.add_argument("--run-name", default=None,
                        help="Name this run. Used for the checkpoint / metrics "
                             "filenames and the W&B run name in place of the model "
                             "name; a timestamp is still appended, so runs never "
                             "overwrite each other. Useful when a sweep writes many "
                             "checkpoints into one directory.")
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if args.attention_pooling:
        config["attention_pooling"] = True
    # Recorded in the run config so the checkpoint, and W&B, say which hospitals
    # the model actually saw — the class weights and the head's prior are derived
    # from this split, so it is part of what the model IS.
    config["sites"] = list(args.sites) if args.sites else None
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.patience is not None:          # `is not None`, so --patience 0 disables
        config["early_stopping_patience"] = args.patience
    if args.freeze_backbone:
        config["train_backbone"] = False

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    spec = DataSpec.load(args.data_config or config.get("data_config"))
    logger.info(f"Training {args.model}\n{spec.describe()}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    site_tag = "_" + "-".join(sorted(s.lower() for s in args.sites)) if args.sites else ""
    regime = "full" if config.get("train_backbone", True) else "head"
    # Defaults to the model name, so naming is unchanged unless --run-name is given.
    run_label = args.run_name or args.model
    results_dir = config.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    metrics_csv_path = os.path.join(results_dir, f"metrics_{run_label}_{run_ts}.csv")

    # Resolve (and validate) the split sources BEFORE the backbone is built: a
    # mistyped --sites should fail in a second, not after a 1B-parameter download.
    sites_note = f" [sites={'+'.join(sorted(args.sites))}]" if args.sites else ""
    train_source = select_sites(config["train_data"], args.sites, "train", logger)
    val_source = (None if args.only_train else
                  select_sites(config["validation_data"], args.sites, "validation", logger))

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.model, spec=spec, **config.get("model_params", {}))
    model = model.to(device)

    if _HAS_WANDB:
        wandb.init(project=config.get("wandb_project", "videomae-unimodal"),
                   name=(f"{args.run_name}_{run_ts}" if args.run_name else
                         f"train_{model.model_name}_{spec.task}{site_tag}_{regime}_{run_ts}"),
                   config={**config, "data_spec": spec.to_dict()},
                   mode=config.get("wandb_mode", "online"),
                   job_type="train")
        wu.define_epoch_metrics()

    minority_class = config.get("minority_class", DEFAULT_MINORITY_CLASS)
    if minority_class not in spec.class_names:
        logger.warning(f"minority_class={minority_class!r} is not one of "
                       f"{spec.class_names} — 'minority/f1' will read 0.0 and the "
                       f"best-minority checkpoint will be meaningless.")

    # ------------------------------------------------------------------ datasets
    train_dataset = VideoMAEDataset(train_source, processor=model.processor,
                                    spec=spec, num_frames=16,
                                    source=f"{config['train_data']}{sites_note}")
    if not args.only_train:
        val_dataset = VideoMAEDataset(val_source, processor=model.processor,
                                      spec=spec, num_frames=16,
                                      source=f"{config['validation_data']}{sites_note}")

    logger.info(f"Train size: {len(train_dataset)}"
                + ("" if args.only_train else f" | Val size: {len(val_dataset)}"))
    logger.info("Train label distribution:\n" + train_dataset.describe_labels())

    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 8), shuffle=True,
                              num_workers=config.get("num_workers", 4),
                              pin_memory=config.get("num_workers", 4) > 0,
                              collate_fn=collate_fn, drop_last=True)
    if not args.only_train:
        val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 8), shuffle=False,
                                num_workers=config.get("num_workers", 4),
                                pin_memory=config.get("num_workers", 4) > 0,
                                collate_fn=collate_fn)

    # ------------------------------------------- loss weights & head bias init
    bias = train_dataset.compute_bias()
    criterion, loss_weights = build_criterion(spec, train_dataset, config, device)
    weight_name = "pos_weight (sqrt neg/pos)" if spec.is_multilabel else "class weights (sqrt inv-freq)"
    logger.info(f"Loss: {type(criterion).__name__} | {weight_name}: "
                f"{[round(w, 4) for w in loss_weights.tolist()]}")
    logger.info(f"Head bias ({'logit' if spec.is_multilabel else 'log'}-priors): "
                f"{[round(b, 4) for b in bias.tolist()]}")

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

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    N = len(train_dataset)
    N_val = len(val_dataset) if not args.only_train else 0
    val_step = config.get("validation_step", None) if not args.only_train else None

    best_macro_f1, best_minority_f1 = -1.0, -1.0
    best_epoch_macro, best_epoch_minority = -1, -1
    epochs_no_improve = 0
    global_step = 0
    ckpt_dir = config.get("checkpoint_path", "checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_ckpt(tag, epoch, metrics, val_loss):
        path = os.path.join(ckpt_dir, f"{run_label}_{tag}_{run_ts}.pt")
        torch.save({
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "attention_pooling": model.attn_pool.state_dict() if model.attn_pool is not None else None,
            "processor": model.processor,
            "epoch": epoch, "val_loss": val_loss,
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("cm/")},
            "classifier_config": config.get("classifier_config", {}),
            "config": config,
            # The data config is part of the model: it fixes the head width, the
            # output activation and the meaning of every logit. test.py and
            # infer_video.py read it back instead of guessing.
            "data_spec": spec.to_dict(),
        }, path)
        logger.info(f"Saved {tag} checkpoint -> {path}")

    # ------------------------------------------------------------------- loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss, seen = 0.0, 0
        # `seen` counts SAMPLES and advances in whole batches, so `seen % val_step`
        # only ever hit multiples of lcm(batch_size, val_step) — with batch_size 6
        # and validation_step 1000 that is every 3000 samples, and for batch_size 3
        # / val_step 5000 it never fires inside a normal epoch at all. Track the
        # next threshold instead, so validation runs as soon as the step is passed.
        next_val_at = val_step if val_step is not None else None

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} train", total=N // config.get("batch_size", 8)):
            labels = batch.pop("labels").to(device)
            mask = batch.pop("label_mask", None)
            mask = None if mask is None else mask.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch)
                loss = criterion(logits, labels, mask)
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

            if next_val_at is not None and seen >= next_val_at:
                next_val_at += val_step
                metrics, val_loss, _, _, _ = run_validation(
                    model, val_loader, criterion, device, amp_dtype, N_val, spec, minority_class)
                model.train()
                save_metrics_to_csv(metrics_csv_path, metrics, val_loss, epoch + 1, "val_step", seen)
                wu.log_metrics(metrics, prefix="val_step/",
                               extra={"val_step/loss": val_loss, "train/global_step": global_step})

        train_loss /= max(seen, 1)
        # First group that actually holds parameters: with --freeze-backbone the
        # backbone group is empty, and charting its LR would plot a number that is
        # training nothing.
        current_lr = next((g["lr"] for g in optimizer.param_groups if g["params"]),
                          optimizer.param_groups[0]["lr"])
        wu.log({"train/loss_epoch": train_loss, "lr": current_lr, "epoch": epoch + 1})

        if args.only_train:
            if scheduler_type != "plateau":
                scheduler.step()
            continue

        # ------------------------------------------------- end-of-epoch validation
        metrics, val_loss, val_logits, val_labels, val_masks = run_validation(
            model, val_loader, criterion, device, amp_dtype, N_val, spec, minority_class)
        save_metrics_to_csv(metrics_csv_path, metrics, val_loss, epoch + 1, "val_epoch")
        macro_f1 = metrics["macro/f1"]
        minority_f1 = metrics.get("minority/f1", 0.0)
        logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"macro_f1={macro_f1:.4f} {minority_class}_f1={minority_f1:.4f}")
        wu.log_metrics(metrics, prefix="val/", extra={"val/loss": val_loss, "epoch": epoch + 1})
        wu.log_confusion_matrix(val_logits, val_labels, spec, key="val/confusion_matrix",
                                masks=val_masks, extra={"epoch": epoch + 1})

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
    final_path = os.path.join(config.get("save_path", "models/"), f"{run_label}_final_{run_ts}.pt")
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    torch.save({
        "backbone": model.backbone.state_dict(),
        "classifier": model.classifier.state_dict(),
        "attention_pooling": model.attn_pool.state_dict() if model.attn_pool is not None else None,
        "processor": model.processor,
        "classifier_config": config.get("classifier_config", {}),
        "config": config,
        "data_spec": spec.to_dict(),
    }, final_path)
    logger.info(f"Final model saved -> {final_path}")
    wu.finish()


if __name__ == "__main__":
    main()
