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
import sys
from datetime import datetime
from pathlib import Path

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
from src.data import (VideoMAEDataset, DataSpec, spec_from_checkpoint,
                      parse_visible)
from src.data.ronald import (DEFAULT_RONALD_DIR, DEFAULT_STRIP_PREFIX, load_split,
                             split_path)

VIT_MODELS = ["VideoMAE", "VideoMAEGiant"]

DEFAULT_TEST_SETS = {"haydom": "data/test_haydom.csv", "drc": "data/test_drc.csv"}
THESIS_COLUMN = "thesis_test"


def full_coverage_spec(spec: DataSpec) -> DataSpec:
    """The same spec, but with the ambiguous band read as "not performed".

    `ambiguous: mask` is right for TRAINING — an activity whose window coverage
    lands between `weak_threshold` and `thresholds` has no defensible binary
    label, and inventing one puts noise in the loss. For REPORTING it makes the
    score optimistic: deployment is a continuous stream of 3 s windows and many
    of them are transitional, so measuring only the clean ones answers an easier
    question than the one you care about.

    Worse, the masking rate is not equal across sites. Haydom's clips carry no
    fraction tags, so every bucket-6 clip means "present but sub-threshold" and
    is masked; DRC's fractions resolve many of the same clips outright. Grading
    the two hospitals on subsets of different difficulty contaminates exactly the
    comparison the per-site test sets exist to make. This spec applies one rule
    to both.
    """
    d = spec.to_dict()
    d["ambiguous"] = "negative"
    return DataSpec.from_dict(d, source=f"{spec.source} [full-coverage]")


def confident_subset(rows, spec: DataSpec):
    """Rows the ORIGINAL spec would keep, with its labels/masks.

    Returns (indices into `rows`, labels, masks). Used to recompute the
    confident-subset metrics from the same logits, so the two conventions are
    compared on one inference pass and cannot drift apart.
    """
    idx, targets, masks = [], [], []
    for i, row in enumerate(rows.itertuples(index=False)):
        fracs = {a: float(getattr(row, f"frac_{a}")) for a in spec.activities}
        label = spec.resolve(int(row.bucket), fracs,
                             tagged=bool(int(getattr(row, "tagged", 1))),
                             dir_activities=spec.activities_from_path(
                                 getattr(row, "clip_dir", "")),
                             frac_visible=parse_visible(
                                 getattr(row, "frac_visible", None)))
        if label is None:
            continue
        idx.append(i)
        if spec.is_multilabel:
            targets.append(label.targets)
            masks.append(label.mask)
        else:
            targets.append(label.class_index)
    if not idx:
        return [], None, None
    if spec.is_multilabel:
        return (idx, torch.tensor(targets, dtype=torch.float32),
                torch.tensor(masks, dtype=torch.float32))
    return idx, torch.tensor(targets, dtype=torch.long), None


def per_case_errors(rows, logits, labels, masks, spec):
    """Per EPISODE: how many decisions the model got wrong, and where.

    A "decision" is one clip in multiclass, one clip x activity in multilabel —
    the same unit the metrics are computed over, so the counts here add up to
    the confusion matrix rather than telling a second story.

    Returns a list of dicts sorted worst-first by raw error count, each with
    per-activity false positives and false negatives. FP and FN are kept apart
    on purpose: an episode with 200 false positives on suction and one with 200
    false negatives look identical under an error count and need completely
    different explanations.
    """
    acts = list(spec.activities)
    if spec.is_multilabel:
        pred = (torch.sigmoid(logits) >= torch.tensor(spec.sigmoid_thresholds())).float()
        sup = masks if masks is not None else torch.ones_like(pred)
        wrong = ((pred != labels) & (sup > 0))
        fp = ((pred == 1) & (labels == 0) & (sup > 0))
        fn = ((pred == 0) & (labels == 1) & (sup > 0))
    else:
        # Project onto the activity axis so the report reads the same in both
        # tasks: class 0 is "no activity", classes 1..N are the activities.
        top = logits.argmax(dim=1)
        pred = torch.zeros((len(top), len(acts)))
        tgt = torch.zeros_like(pred)
        for i, a in enumerate(acts):
            pred[:, i] = (top == i + 1).float()
            tgt[:, i] = (labels == i + 1).float()
        labels, sup = tgt, torch.ones_like(pred)
        wrong = (pred != labels)
        fp = ((pred == 1) & (labels == 0))
        fn = ((pred == 0) & (labels == 1))

    case_ids = (rows["case_id"].astype(str).tolist() if "case_id" in rows.columns
                else [DataSpec.case_id_from_stem(Path(p).stem) for p in rows["video_path"]])
    sites = (rows["site"].astype(str).tolist() if "site" in rows.columns
             else [""] * len(case_ids))

    agg = {}
    for i, cid in enumerate(case_ids):
        c = agg.setdefault(cid, {"case_id": cid, "site": sites[i], "clips": 0,
                                 "decisions": 0, "errors": 0,
                                 "fp": dict.fromkeys(acts, 0),
                                 "fn": dict.fromkeys(acts, 0),
                                 "pos": dict.fromkeys(acts, 0),
                                 "sup": dict.fromkeys(acts, 0)})
        c["clips"] += 1
        c["decisions"] += int(sup[i].sum().item())
        c["errors"] += int(wrong[i].sum().item())
        for j, a in enumerate(acts):
            c["fp"][a] += int(fp[i, j].item())
            c["fn"][a] += int(fn[i, j].item())
            c["pos"][a] += int(labels[i, j].item())
            c["sup"][a] += int(sup[i, j].item())
    for c in agg.values():
        c["rate"] = c["errors"] / max(c["decisions"], 1)
        # Per activity: errors on it over the decisions actually SUPERVISED for
        # it. Not over its positives — an episode with no suction at all would
        # divide by zero, and those are exactly the episodes a false-positive
        # problem shows up in, so that denominator is undefined where it matters
        # most.
        c["act_rate"] = {a: (c["fp"][a] + c["fn"][a]) / max(c["sup"][a], 1)
                         for a in acts}
    return sorted(agg.values(), key=lambda c: -c["errors"])


def rank_worst(cases, by, spec):
    """Re-sort `per_case_errors` output by the requested criterion.

    `errors` is the default because it finds where the bulk of the damage is.
    `rate` finds the episodes the model handles worst, which are often short and
    contribute little overall — useful, but a different question. An ACTIVITY
    name ranks by that activity's errors alone, which is the one to use when a
    single class is the problem (suction, here), and `<activity>:rate` does the
    same normalised by that activity's supervised decisions — so a long
    ventilation-heavy recording stops crowding out a short one that gets suction
    badly wrong.
    """
    if by == "rate":
        return sorted(cases, key=lambda c: (-c["rate"], -c["errors"]))
    act, _, mode = by.partition(":")
    if act in spec.activities:
        if mode == "rate":
            # Tie-break on the raw count so two episodes at the same rate are
            # ordered by how much they actually contribute.
            return sorted(cases, key=lambda c: (-c["act_rate"][act],
                                                -(c["fp"][act] + c["fn"][act])))
        return sorted(cases, key=lambda c: -(c["fp"][act] + c["fn"][act]))
    return sorted(cases, key=lambda c: -c["errors"])


def report_worst(cases, spec, name, by, top, logger):
    """Print the worst episodes, with FP/FN split per activity."""
    acts = list(spec.activities)
    ranked_act, _, mode = by.partition(":")
    show_act = ranked_act if (ranked_act in acts and mode == "rate") else None

    head = "".join(f"{a[:7]+' FP/FN':>16}" for a in acts)
    extra = f"{ranked_act[:9]+' err%':>15}" if show_act else ""
    logger.info(f"\n[{name}] WORST EPISODES by {by}")
    logger.info(f"  {'case':<14}{'site':<8}{'clips':>7}{'errors':>8}{'rate':>7}"
                f"{head}{extra}")
    for c in cases[:top]:
        cells = "".join(f"{c['fp'][a]:>8,}/{c['fn'][a]:<7,}" for a in acts)
        tail = (f"{100*c['act_rate'][show_act]:>14.1f}%" if show_act else "")
        logger.info(f"  {c['case_id']:<14}{c['site']:<8}{c['clips']:>7,}"
                    f"{c['errors']:>8,}{100*c['rate']:>6.1f}%{cells}{tail}")
    logger.info("  FP = predicted the activity where the ground truth says no; "
                "FN = missed it.")
    if show_act:
        logger.info(f"  {ranked_act} err% = ({ranked_act} FP + FN) / decisions "
                    f"supervised for {ranked_act} in that episode.")


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


def resolve_sites(cli_sites, config, logger):
    """Which hospitals to score, as a lowercase set — or None for "all of them".

    A model trained with `--sites Haydom` has never seen DRC, so scoring it there
    by default answers a question nobody asked and buries the number that matters
    in a two-row comparison table. training.py records the flag in the
    checkpoint's config, so the default here is simply "the sites this model was
    trained on".

    That default is a convenience, never a restriction: `--sites all` scores
    every test set regardless, which IS the cross-site generalisation experiment
    and is worth running deliberately. An explicit `--sites` wins over both.
    """
    if cli_sites:
        if len(cli_sites) == 1 and cli_sites[0].lower() == "all":
            logger.info("--sites all: scoring every test set, including hospitals "
                        "this model was not trained on")
            return None
        return {s.lower() for s in cli_sites}
    trained = config.get("sites")
    if trained:
        logger.info(f"checkpoint was trained with --sites {sorted(trained)}, so only "
                    f"those are scored. Pass `--sites all` for the cross-site number.")
        return {str(s).lower() for s in trained}
    return None


def filter_rows_to_sites(rows, sites, name, logger):
    """Restrict one test CSV's rows to `sites`. Returns None if nothing is left.

    Filtering on the `site` COLUMN rather than on the test set's NAME, so this is
    correct for a pooled file (`data/test.csv`, `data/validation.csv`) as well as
    for the per-site ones — and a per-site file filtered to its own site is a
    no-op rather than a special case.
    """
    if not sites:
        return rows
    if "site" not in rows.columns:
        logger.warning(f"[{name}] --sites given but this CSV has no `site` column — "
                       f"it predates the per-site manifest; scoring every row.")
        return rows
    out = rows[rows["site"].str.lower().isin(sites)]
    if out.empty:
        logger.info(f"[{name}] no clips from {sorted(sites)} — skipped.")
        return None
    if len(out) < len(rows):
        logger.info(f"[{name}] --sites {sorted(sites)} keeps {len(out):,}/{len(rows):,} clips")
    return out.reset_index(drop=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VIT_MODELS)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, nargs="*", default=None,
                        help="Override the test CSVs from the checkpoint config. "
                             "Repeatable; each entry is PATH or NAME=PATH.")
    parser.add_argument("--full-coverage", action="store_true",
                        help="Score EVERY clip: an activity below its threshold counts "
                             "as not performed instead of being masked out, and no clip "
                             "is dropped for ambiguity. Reports this alongside the "
                             "confident-subset numbers so the two are comparable.")
    parser.add_argument("--render-worst", type=int, default=0, metavar="N",
                        help="After scoring, render a full-episode inference video "
                             "for the N episodes the model got most wrong, via "
                             "src.infer_video. 0 (default) renders nothing; the "
                             "per-episode error table is printed either way.")
    parser.add_argument("--worst-by", default="errors", metavar="CRITERION",
                        help="How to rank episodes: `errors` (total wrong decisions, "
                             "the default), `rate` (errors per decision — finds the "
                             "episodes handled worst, often short ones), an ACTIVITY "
                             "name for that class's error COUNT (e.g. `suction`), or "
                             "`<activity>:rate` for that class's error RATE (e.g. "
                             "`suction:rate`), which stops long recordings crowding "
                             "out short ones that get the class badly wrong.")
    parser.add_argument("--worst-dir", default="results/worst_episodes", metavar="DIR",
                        help="Where the rendered episodes are written.")
    parser.add_argument("--ronald", action="store_true",
                        help="Score on Ronald Paleczny's test manifest. Implied "
                             "for a checkpoint trained with --ronald (the path is "
                             "recorded in it), so you rarely need this; pass it to "
                             "score OUR model on HIS test set deliberately.")
    parser.add_argument("--ronald-dir", default=None, metavar="DIR",
                        help=f"Where his manifests live (default {DEFAULT_RONALD_DIR}).")
    parser.add_argument("--ronald-strip-prefix", default=DEFAULT_STRIP_PREFIX,
                        metavar="PREFIX",
                        help="Leading path prefix stripped from his clip paths.")
    parser.add_argument("--sites", nargs="+", default=None, metavar="SITE",
                        help="Score only these hospitals, e.g. `--sites Haydom`. "
                             "Case-insensitive, repeatable, and applied to the "
                             "`site` column so it works on pooled CSVs too. "
                             "DEFAULT: the sites the checkpoint was TRAINED on "
                             "(training.py records --sites), so a Haydom-only "
                             "model is scored on Haydom only. Pass `--sites all` "
                             "to score every test set, which is the cross-site "
                             "generalisation experiment.")
    parser.add_argument("--thesis-only", action="store_true",
                        help=f"Score only the thesis' frozen cases (rows with "
                             f"{THESIS_COLUMN} == 1), for a like-for-like comparison "
                             f"with the multimodal results.")
    parser.add_argument("--data-config", type=str, default=None,
                        help="Override the DataSpec stored in the checkpoint. Only do this "
                             "on purpose — a mismatched spec changes what every logit means.")
    parser.add_argument("--legacy-pooling", choices=["auto", "on", "off"], default="auto",
                        help="How to pool patch tokens for VideoMAE checkpoints trained "
                             "before the 2026-08-31 pooling fix, whose head was fitted on "
                             "last_hidden_state[:, 0] instead of the mean over all tokens. "
                             "auto (default): detect them by the missing `fc_norm` and use "
                             "the old path, so they stay evaluable. on: force it. off: "
                             "always mean-pool — the pre-fix head then sees a feature space "
                             "it was never trained on and its predictions are meaningless.")
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

    # --full-coverage widens the AMBIGUOUS band to "not performed". A
    # label_source: columns manifest has no ambiguous band — its labels are
    # already final — so the flag has nothing to widen, and full_coverage_spec()
    # would be rejected by DataSpec.validate for setting an inert key. Say that
    # here rather than surfacing it as a validation error three frames down.
    if args.full_coverage and spec.labels_from_columns:
        raise SystemExit(
            f"--full-coverage does not apply to {spec.source}: it sets "
            f"`label_source: columns`, so every clip already carries a final "
            f"0/1 for every activity and nothing is masked or dropped for "
            f"ambiguity. The score you would get is the one you already get.")

    # Only VideoMAE has the two pooling paths; the giant pools inside its own
    # trunk, so passing the kwarg to it would be a TypeError.
    pooling_kwargs = {}
    if args.model == "VideoMAE":
        pooling_kwargs["pooling"] = {"auto": "auto", "on": "patch0_legacy",
                                     "off": "mean"}[args.legacy_pooling]
    model = load_model(args.model, spec=spec, **pooling_kwargs)
    model = model.to(device)

    _act, _, _mode = args.worst_by.partition(":")
    if not (args.worst_by in ("errors", "rate")
            or (_act in spec.activities and _mode in ("", "rate"))):
        raise SystemExit(
            f"--worst-by {args.worst_by!r} is not `errors`, `rate`, one of this "
            f"checkpoint's activities {list(spec.activities)}, or "
            f"`<activity>:rate`.")
    args._ckpt_ronald = bool(config.get("ronald"))

    ronald = args.ronald or bool(config.get("ronald"))
    if args.ronald and not args.test_data:
        config = {**config, "test_data":
                  {"ronald_test": str(split_path("test", args.ronald_dir))}}
    test_sets = resolve_test_sets(args.test_data, config)
    sites = None if ronald else resolve_sites(args.sites, config, logger)
    if ronald:
        logger.warning(
            "--ronald: scoring on Ronald Paleczny's test manifest. Compare against "
            "his thesis Table 3.5 (macro P/R/F1 0.878/0.846/0.861 at thresholds "
            "0.5/0.5/0.5) — and note AP is ours, he did not report it.")
        if args.sites:
            raise SystemExit("--ronald and --sites are incompatible: his manifests "
                             "are Haydom-only and have no `site` column.")
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
                           # `sites` in the checkpoint config is what the model was
                           # TRAINED on; this is what it is being SCORED on. They
                           # differ exactly when --sites all asks the cross-site
                           # question, which is the run you most need to tell apart.
                           "eval_sites": sorted(sites) if sites else "all",
                           "thesis_only": args.thesis_only,
                           "pooling": getattr(model, "pooling", "n/a"),
                           "data_spec": spec.to_dict()},
                   mode=config.get("wandb_mode", "online"), job_type="eval")

    os.makedirs(args.results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.model_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_metrics, all_files, all_worst = {}, [], {}
    for name, test_csv in test_sets:
        metrics, files, worst = run_test_set(name, test_csv, model=model, spec=spec, args=args,
                                      config=config, device=device, amp_dtype=amp_dtype,
                                      minority_class=minority_class, base=base, ts=ts,
                                      logger=logger, sites=sites, ronald=ronald)
        if metrics is None:
            continue
        all_metrics[name] = metrics
        all_files += files
        all_worst[name] = worst

    if not all_metrics:
        raise SystemExit(
            "no test set produced any metrics — check the paths above."
            + (f" Every set was empty after --sites {sorted(sites)}; pass "
               f"`--sites all` to score the hospitals this model was not trained on."
               if sites else ""))

    report_comparison(all_metrics, spec, minority_class, logger)
    # The same numbers as one sortable grid instead of ~40 scalar panels. This is
    # what scripts/wandb_report.py points its table panel at.
    wu.log_results_table(all_metrics, spec)

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

    # LAST, after every metric is printed, logged and stored: rendering is slow
    # and optional, and nothing above it should be waiting on a video encoder.
    if args.render_worst:
        render_worst_episodes(all_worst, args, spec, logger)


def render_worst_episodes(worst_by_set, args, spec, logger):
    """Render a full-episode video for the worst episodes, via src.infer_video.

    Run as a SUBPROCESS rather than by importing infer_video's pipeline. The
    request is "the same video infer_video produces", and shelling out to it is
    the only way to guarantee that stays true as either side changes — there is
    no second copy of the window/overlay/render logic to drift. The cost is
    reloading the backbone per episode (tens of seconds), which is small next to
    decoding a 20-minute recording.

    A failure to render one episode is reported and skipped, never fatal: the
    metrics are the point of the run and they are already computed by here.
    """
    import subprocess
    out_root = Path(args.worst_dir)
    rendered, failed = [], []
    for set_name, cases in worst_by_set.items():
        for c in cases[:args.render_worst]:
            out_dir = out_root / f"{set_name}_{c['case_id']}"
            cmd = [sys.executable, "-m", "src.infer_video",
                   "--model", args.model, "--model_path", args.model_path,
                   "--case", c["case_id"], "--out-dir", str(out_dir),
                   "--render-video", "--all-cases"]
            if args.ronald or bool(getattr(args, "_ckpt_ronald", False)):
                cmd += ["--ronald"]
            if args.data_config:
                cmd += ["--data-config", args.data_config]
            if args.legacy_pooling != "auto":
                cmd += ["--legacy-pooling", args.legacy_pooling]
            logger.info(f"[worst] rendering {set_name}/{c['case_id']} "
                        f"({c['errors']:,} errors over {c['clips']:,} clips) -> {out_dir}")
            try:
                subprocess.run(cmd, check=True)
                rendered.append(out_dir / "annotated.mp4")
            except subprocess.CalledProcessError as exc:
                failed.append((set_name, c["case_id"], exc.returncode))
                logger.warning(f"[worst] {c['case_id']} did not render "
                               f"(exit {exc.returncode}) — skipped. Its raw video or "
                               f"annotation may not resolve on this machine; "
                               f"`python scripts/check_episode_media.py` says which.")
    if rendered:
        logger.info(f"\n[worst] {len(rendered)} episode(s) rendered under {out_root}:")
        for f in rendered:
            logger.info(f"    {f}")
        logger.info("  Copy them off the VM and play in VLC — the prediction track "
                    "sits above the ground-truth track, so a systematic error reads "
                    "as one band being consistently wider or shifted.")
    if failed:
        logger.warning(f"[worst] {len(failed)} episode(s) failed to render: "
                       f"{[f'{a}/{b}' for a, b, _ in failed]}")


def run_test_set(name, test_csv, *, model, spec, args, config, device, amp_dtype,
                 minority_class, base, ts, logger, sites=None, ronald=False):
    """Evaluate one test CSV. Returns (metrics, written files), or (None, []) if
    the CSV is missing — a missing site should not abort the other site's score."""
    if not os.path.exists(test_csv):
        logger.warning(f"[{name}] {test_csv} not found — skipped. Run "
                       f"scripts/build_data.sh to regenerate the splits.")
        return None, [], []

    logger.info(f"\n{'=' * 70}\n[{name}] {test_csv}\n{'=' * 70}")
    if ronald:
        # NOT pd.read_csv: his manifests store clip paths under the account that
        # generated them, and the prefix has to come off or every decode is a
        # PermissionError.
        rows = load_split("test", os.path.dirname(test_csv) or None,
                          args.ronald_strip_prefix)
    else:
        rows = pd.read_csv(test_csv)
    rows = filter_rows_to_sites(rows, sites, name, logger)
    if rows is None:
        return None, [], []
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
                return None, [], []

    # With --full-coverage the DATASET is built from the permissive spec, so no
    # clip is dropped for ambiguity; the strict spec is then re-applied to the
    # same logits below. One inference pass, two conventions.
    eval_spec = full_coverage_spec(spec) if args.full_coverage else spec
    test_dataset = VideoMAEDataset(rows, processor=model.processor, spec=eval_spec,
                                   num_frames=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=config.get("num_workers", 4),
                             collate_fn=collate_fn)
    logger.info(f"[{name}] {len(test_dataset)} clips")
    if args.full_coverage and test_dataset.n_dropped:
        logger.info(f"[{name}] {test_dataset.n_dropped} clips still excluded by BUCKET "
                    f"policy (buckets {sorted(set(range(9)) - set(spec.kept_buckets()))} "
                    f"are dropped in {spec.source}) — --full-coverage removes the "
                    f"ambiguity exclusion, not the bucket one.")
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

    # Per-episode errors: printed on every run, because "which recordings is
    # this failing on" is a different and often more actionable question than
    # the aggregate, and it costs nothing once the logits exist.
    worst = rank_worst(
        per_case_errors(test_dataset.data, logits_t, labels_t, masks_t, spec),
        args.worst_by, spec)

    sub_metrics, sub_idx = None, None
    if args.full_coverage and spec.ambiguous != "negative":
        sub_idx, sub_labels, sub_masks_t = confident_subset(test_dataset.data, spec)
        if sub_idx:
            sub_metrics = compute_metrics(logits_t[sub_idx], sub_labels, spec,
                                          masks=sub_masks_t, minority_class=minority_class)
        else:
            logger.warning(f"[{name}] the strict spec keeps no clips — "
                           f"confident-subset metrics skipped.")
    elif args.full_coverage:
        logger.info(f"[{name}] {spec.source} already uses `ambiguous: negative`, so "
                    f"full coverage and the confident subset are the same set.")

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

    report_worst(worst, spec, name, args.worst_by,
                 max(args.render_worst, 10), logger)

    if sub_metrics is not None:
        # Under `ambiguous: mask` a clip is DROPPED only when every activity is
        # ambiguous; far more often it is kept with individual activities masked
        # out. Counting dropped clips alone would badly understate what the
        # confident convention sets aside, so account for both: whole clips, and
        # individual activity decisions.
        n_all, n_sub = len(test_dataset), len(sub_idx)
        ent_all = n_all * spec.num_classes if spec.is_multilabel else n_all
        if spec.is_multilabel:
            ent_sub = int(sub_masks_t.sum().item())
            clips_touched = int((sub_masks_t.min(dim=1).values == 0).sum().item())
        else:
            ent_sub, clips_touched = n_sub, 0

        keys = ["macro/f1", "minority/f1", "macro/ap", "hamming/accuracy"]
        label = {"minority/f1": f"{minority_class}/f1"}
        head = (f"{'convention':<24}{'clips':>9}{'supervised':>12}"
                + "".join(f"{label.get(k, k):>18}" for k in keys))
        lines = ["", f"[{name}] AMBIGUITY CONVENTIONS", "-" * len(head), head]
        lines.append(f"{'full coverage (all)':<24}{n_all:>9,}{ent_all:>12,}" + "".join(
            f"{metrics.get(k, float('nan')):>18.4f}" for k in keys))
        lines.append(f"{'confident subset':<24}{n_sub:>9,}{ent_sub:>12,}" + "".join(
            f"{sub_metrics.get(k, float('nan')):>18.4f}" for k in keys))
        lines.append("")
        lines.append(f"  clips dropped entirely : {n_all - n_sub:,}")
        if spec.is_multilabel:
            lines.append(f"  activity decisions set aside : {ent_all - ent_sub:,} "
                         f"({100 * (ent_all - ent_sub) / max(ent_all, 1):.1f}% of all)")
            lines.append(f"  clips with >=1 masked activity : {clips_touched:,} "
                         f"({100 * clips_touched / max(n_all, 1):.1f}% of this site)")
        lines.append("")
        lines.append("Full coverage reads a sub-threshold activity as NOT PERFORMED and "
                     "scores every")
        lines.append("clip — the deployment question. The confident subset scores only "
                     "decisions whose")
        lines.append("label is unambiguous — the cleaner question, and an easier one. The "
                     "set-aside")
        lines.append("rate differs per site, so compare hospitals on the SAME convention.")
        logger.info("\n".join(lines))

        wu.log_metrics(sub_metrics, prefix=f"test/{name}/confident/")
        wu.update_summary({f"test/{name}/confident/{k}": float(v)
                           for k, v in sub_metrics.items() if not k.startswith("cm/")})
        wu.update_summary({
            f"test/{name}/confident/n_clips": n_sub,
            f"test/{name}/confident/n_supervised": ent_sub,
            f"test/{name}/set_aside_frac": (ent_all - ent_sub) / max(ent_all, 1),
            f"test/{name}/clips_with_masked_frac": clips_touched / max(n_all, 1)})

    # Always shown: the macro numbers average three very differently sized
    # classes, so the per-class rows are where a collapsed rare activity is
    # actually visible.
    report_per_class(name, metrics, sub_metrics, spec, logger)

    wu.log_metrics(metrics, prefix=f"test/{name}/")
    wu.log_confusion_matrix(logits_t, labels_t, spec,
                            key=f"test/{name}/confusion_matrix", masks=masks_t)
    wu.update_summary({f"test/{name}/{k}": float(v) for k, v in metrics.items()
                       if not k.startswith("cm/")})

    suffix = (f"{base}_{name}{'_thesis' if args.thesis_only else ''}"
              f"{'_fullcov' if args.full_coverage else ''}_{ts}")
    csv_path = os.path.join(args.results_dir, f"results_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["model", base])
        w.writerow(["test_set", name])
        w.writerow(["test_data", test_csv])
        w.writerow(["n_clips", n])
        w.writerow(["ambiguity", "negative (full coverage)" if args.full_coverage
                    else spec.ambiguous])
        w.writerow(["task", spec.task])
        w.writerow(["classes", "|".join(spec.class_names)])
        if spec.is_multilabel:
            w.writerow(["decision_thresholds",
                        "|".join(f"{a}={t}" for a, t in
                                 zip(spec.activities, spec.sigmoid_thresholds()))])
        for k, v in metrics.items():
            w.writerow([k, round(float(v), 6) if not k.startswith("cm/") else int(v)])
        if sub_metrics is not None:
            w.writerow(["confident/n_clips", len(sub_idx)])
            for k, v in sub_metrics.items():
                w.writerow([f"confident/{k}",
                            round(float(v), 6) if not k.startswith("cm/") else int(v)])
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
    return metrics, [csv_path, scores_path], worst


def report_per_class(name, metrics, sub_metrics, spec, logger):
    """Per-class table — what the macro averages are hiding.

    A macro F1 is three very different numbers averaged: with prevalences of
    ~43 %, ~7 % and ~5 % here, one collapsed rare class barely moves it. The
    per-class row is where you see which activity actually failed, and `sup`
    (supervised decisions) is where you see how much of it was set aside.

    With --full-coverage both conventions are shown side by side, computed from
    the same logits, so the columns are directly comparable.
    """
    classes = spec.class_names
    if spec.is_multilabel:
        cols = [("support", "pos"), ("n_supervised", "sup"), ("precision", "prec"),
                ("recall", "rec"), ("f1", "F1"), ("ap", "AP")]
    else:
        cols = [("precision", "prec"), ("recall", "rec"), ("f1", "F1")]

    def cells(m, cls):
        out = ""
        for key, _ in cols:
            v = m.get(f"{cls}/{key}")
            if v is None:
                out += f"{'-':>8}"
            elif key in ("support", "n_supervised"):
                out += f"{int(v):>8,}"
            else:
                out += f"{v:>8.4f}"
        return out

    width = 15 + 8 * len(cols)
    both = sub_metrics is not None
    lines = ["", f"[{name}] PER-CLASS"]
    if both:
        lines.append(f"{'':<15}{'full coverage (all clips)':^{8 * len(cols)}}   "
                     f"{'confident subset':^{8 * len(cols)}}")
    header = f"{'class':<15}" + "".join(f"{h:>8}" for _, h in cols)
    if both:
        header += "   " + "".join(f"{h:>8}" for _, h in cols)
    lines.append(header)
    lines.append("-" * (width + (3 + 8 * len(cols) if both else 0)))
    for cls in classes:
        row = f"{cls:<15}" + cells(metrics, cls)
        if both:
            row += "   " + cells(sub_metrics, cls)
        lines.append(row)

    macro = f"{'MACRO':<15}" + "".join(
        f"{metrics.get('macro/' + k, float('nan')):>8.4f}"
        if k in ("precision", "recall", "f1", "ap") else f"{'':>8}" for k, _ in cols)
    if both:
        macro += "   " + "".join(
            f"{sub_metrics.get('macro/' + k, float('nan')):>8.4f}"
            if k in ("precision", "recall", "f1", "ap") else f"{'':>8}" for k, _ in cols)
    lines.append(macro)
    if spec.is_multilabel:
        lines.append("pos = positive clips | sup = clips whose label for this activity is "
                     "supervised")
        if both:
            lines.append("AP is threshold-free: if it barely moves between the two "
                         "conventions while F1 does,\nthe change is about which clips "
                         "were scored, not about the model.")
    logger.info("\n".join(lines))


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
