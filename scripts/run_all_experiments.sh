#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — the full VideoMAE experiment matrix
# =============================================================================
# 12 trainings + their evaluations, run back to back:
#
#     site  x  task        x  regime
#     ----     ----           ------
#     Haydom   multiclass     full   (backbone + head, 20 epochs, patience 5)
#     DRC      multilabel     head   (head only,       50 epochs, patience 10)
#     both
#
# "both" trains on the two hospitals pooled and is then scored on EACH test set
# separately — src/test.py writes one results_*.csv / scores_*.npz per set and
# prints a side-by-side comparison, so you get two independent statistics rather
# than one pooled number. The single-site runs are scored on their own site.
#
# Every run gets its own directory under $RUNS_DIR with the training log, the
# evaluation log, and a manifest naming the artefacts it produced.
#
# -----------------------------------------------------------------------------
# RUN IT DETACHED — this is a long job
# -----------------------------------------------------------------------------
#     tmux new -s exp
#     bash scripts/run_all_experiments.sh 2>&1 | tee runs/all.log
#     # detach with Ctrl-b d
#
# or:  nohup bash scripts/run_all_experiments.sh > runs/all.log 2>&1 &
#
# It is RESUMABLE: a finished experiment drops a DONE marker and is skipped on
# the next invocation, so if the machine dies at experiment 7 you just re-run
# this script. Delete $RUNS_DIR/<name>/DONE to force one to repeat.
#
# A failing experiment does NOT abort the rest; failures are collected and
# reported in the summary at the end.
#
# -----------------------------------------------------------------------------
# Knobs (environment variables)
# -----------------------------------------------------------------------------
#   GPU=0                which CUDA device
#   MODEL=VideoMAE       VideoMAE (ViT-B base, the default) | VideoMAEGiant (ViT-g)
#   CKPT_TAGS="best_macro"
#                        which saved checkpoints to evaluate. Training always
#                        keeps best_macro AND best_<minority_class>; add
#                        "best_suction" to score both.
#   CROSS_SITE=0         1 = also score each single-site model on the OTHER
#                        hospital (the transfer number). Cheap — inference only.
#   ONLY=<regex>         run just the experiments whose name matches, e.g.
#                        ONLY='haydom.*full' . Handy for retrying one.
#   DRY_RUN=1            print the commands and exit without running anything.
#   RUNS_DIR=runs        where the per-experiment logs go
#
# -----------------------------------------------------------------------------
# Before you start
# -----------------------------------------------------------------------------
# * DISK: each run writes three state_dicts (best_macro, best_<minority>, final).
#   VideoMAE base is ~0.35 GB each (~2 GB per run, ~24 GB for the matrix);
#   VideoMAEGiant is ~4 GB each (~12 GB per run, ~146 GB). The preflight sizes
#   this from $MODEL and checks you have room.
# * W&B: if `wandb_mode: online` and you are not logged in, wandb.init() blocks
#   on an interactive prompt and the whole unattended job hangs. The preflight
#   refuses to start in that state — run `wandb login`, or set
#   `wandb_mode: offline` (or `disabled`) in configs/config.yaml.
# * HEAD-ONLY RUNS use `classifier_lr` from configs/config.yaml (5e-5). That is
#   tuned for fine-tuning alongside a moving backbone; a frozen-backbone probe
#   usually wants it 10-100x higher. Consider raising it before the head-only
#   runs if their loss barely moves — it is a real knob, not a formality.
# =============================================================================
set -uo pipefail          # deliberately NOT -e: one bad run must not kill the rest

GPU="${GPU:-0}"
MODEL="${MODEL:-VideoMAE}"
CKPT_TAGS="${CKPT_TAGS:-best_macro}"
CROSS_SITE="${CROSS_SITE:-0}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
RUNS_DIR="${RUNS_DIR:-runs}"

EPOCHS_FULL=20;  PATIENCE_FULL=5
EPOCHS_HEAD=50;  PATIENCE_HEAD=10

MULTICLASS_CONFIG="configs/data.yaml"
MULTILABEL_CONFIG="configs/data_multilabel.yaml"
HAYDOM_TEST="data/test_haydom.csv"
DRC_TEST="data/test_drc.csv"

# Manifest site names (capitalised, as build_data.sh writes them) per site key.
site_filter() { case "$1" in haydom) echo "Haydom";; drc) echo "DRC";; both) echo "";; esac; }

# Which test sets a given site key is scored on. NAME=PATH so each writes its
# own results/scores files and its own test/<name>/ W&B prefix.
test_sets() {
    case "$1" in
        haydom) [[ "$CROSS_SITE" == "1" ]] \
                    && echo "haydom=$HAYDOM_TEST drc=$DRC_TEST" \
                    || echo "haydom=$HAYDOM_TEST" ;;
        drc)    [[ "$CROSS_SITE" == "1" ]] \
                    && echo "drc=$DRC_TEST haydom=$HAYDOM_TEST" \
                    || echo "drc=$DRC_TEST" ;;
        both)   echo "haydom=$HAYDOM_TEST drc=$DRC_TEST" ;;
    esac
}

# ---------------------------------------------------------------- preflight
preflight() {
    local fail=0
    echo "── preflight ────────────────────────────────────────────────────────"
    echo "  python  $(python -c 'import sys; print(sys.executable)' 2>/dev/null \
                      || echo 'NOT FOUND on PATH')"

    for f in configs/config.yaml "$MULTICLASS_CONFIG" "$MULTILABEL_CONFIG" \
             data/train.csv data/validation.csv "$HAYDOM_TEST" "$DRC_TEST"; do
        if [[ -f "$f" ]]; then
            echo "  ok      $f"
        else
            echo "  MISSING $f"; fail=1
        fi
    done
    [[ $fail -eq 1 ]] && {
        echo
        echo "  Split CSVs are built by:  bash scripts/build_data.sh"
        return 1
    }

    # Both hospitals must actually be in the train/val splits, or --sites fails
    # partway through the matrix instead of now.
    python - "$MULTICLASS_CONFIG" <<'PY' || fail=1
import sys
sys.path.insert(0, ".")

# Inventory the WHOLE import stack in one pass. training.py pulls src.utils and
# src.data at module load, so torch / transformers / sklearn / pandas / av / yaml
# / tqdm all have to work before epoch 1 — reporting them one failure per run is
# a slow way to find out. cv2 is only needed by the clip re-cutting and the
# viewers, so it is reported but not fatal.
REQUIRED = ["torch", "transformers", "huggingface_hub", "safetensors",
            "numpy", "pandas", "sklearn", "yaml", "tqdm"]
OPTIONAL = ["cv2", "av", "wandb", "matplotlib"]

def probe(mod):
    try:
        m = __import__(mod)
        return None, getattr(m, "__version__", "?"), getattr(m, "__file__", "") or ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", None, None

broken = {}
versions = {}
for mod in REQUIRED + OPTIONAL:
    err, ver, path = probe(mod)
    if err is None:
        versions[mod] = ver
    elif mod in REQUIRED:
        broken[mod] = err
    else:
        versions[mod] = f"absent ({mod} is optional)"

if broken:
    print(f"  FAIL    {len(broken)} required package(s) unusable in this interpreter.")
    print(f"          interpreter: {sys.executable}")
    print()
    for mod in REQUIRED + OPTIONAL:
        err, ver, path = probe(mod)
        tag = "FAIL" if err else " ok "
        if err:
            print(f"          [{tag}] {mod:<16} {err[:74]}")
        else:
            print(f"          [{tag}] {mod:<16} {ver:<12} {path}")
    print()
    print("          Training imports this same stack — it would fail identically.")
    print("          Compare the paths above: packages split across ~/.local and a")
    print("          conda prefix, with different numpy major versions, is the usual")
    print("          cause (conda envs inherit ~/.local; venvs do not).")
    print("          Durable fix — a venv, which ignores ~/.local:")
    print("            python -m venv .venv && source .venv/bin/activate")
    print("            pip install -r requirements.txt")
    raise SystemExit(1)

print(f"  deps    numpy {versions['numpy']}, pandas {versions['pandas']}, "
      f"torch {versions['torch']}, transformers {versions['transformers']}")

try:
    from src.data.manifest import read_manifest
except Exception as exc:
    print(f"  FAIL    cannot import src.data.manifest ({type(exc).__name__}: {exc})")
    print(f"          Run from the repo root; this needs ./src on the path.")
    raise SystemExit(1)

need = {"Haydom", "DRC"}
for split in ("train", "validation"):
    df = read_manifest(f"data/{split}.csv")
    if "site" not in df.columns:
        print(f"  FAIL    data/{split}.csv has no `site` column — rebuild the splits")
        raise SystemExit(1)
    have = set(map(str, df["site"].unique()))
    missing = need - have
    n = {s: int((df["site"] == s).sum()) for s in sorted(have)}
    print(f"  ok      data/{split}.csv sites={n}")
    if missing:
        print(f"  FAIL    data/{split}.csv is missing {sorted(missing)} — the "
              f"per-site runs cannot work")
        raise SystemExit(1)
PY

    # W&B must not be able to block on an interactive login prompt.
    python - <<'PY' || fail=1
import os, sys, yaml
mode = (yaml.safe_load(open("configs/config.yaml")) or {}).get("wandb_mode", "online")
if mode != "online":
    print(f"  ok      wandb_mode={mode} (no login needed)"); raise SystemExit(0)
try:
    import wandb
except Exception:
    print("  ok      wandb_mode=online but wandb is not installed — logging is skipped")
    raise SystemExit(0)
key = os.environ.get("WANDB_API_KEY") or wandb.api.api_key
if key:
    print("  ok      wandb_mode=online and an API key was found"); raise SystemExit(0)
print("  FAIL    wandb_mode=online but no API key. wandb.init() would block on an")
print("          interactive prompt and hang this unattended job. Fix with either:")
print("            wandb login")
print("            # or set `wandb_mode: offline` in configs/config.yaml")
raise SystemExit(1)
PY

    # Disk: three fp32 state_dicts per run, sized from the backbone.
    local runs_planned free_gb need_gb gb_per_run
    case "$MODEL" in
        VideoMAEGiant) gb_per_run=12 ;;   # 3 x ~4.1 GB  (ViT-g, 1.01B params)
        VideoMAE)      gb_per_run=2  ;;   # 3 x ~0.35 GB (ViT-B, 87M params)
        *)             gb_per_run=12 ;;   # unknown: assume the worst
    esac
    runs_planned=$(( ${#EXPERIMENTS[@]} ))
    need_gb=$(( runs_planned * gb_per_run ))
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    echo "  disk    ${free_gb} GB free, ~${need_gb} GB needed (${runs_planned} runs x ~${gb_per_run} GB of checkpoints)"
    if [[ "$free_gb" -lt "$need_gb" ]]; then
        echo "  FAIL    not enough space. Point checkpoint_path:/save_path: in"
        echo "          configs/config.yaml at a bigger disk, or trim CKPT_TAGS."
        fail=1
    fi

    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
        | sed 's/^/  gpu     /' || echo "  gpu     nvidia-smi unavailable"

    # VRAM sanity. Full fine-tuning holds fp32 weights + grads + two AdamW moments
    # (16 bytes/param) before a single activation is stored; a frozen backbone holds
    # only the weights and stores no backbone activations at all. The two regimes
    # have wildly different footprints, and it is worth knowing which of them fits
    # BEFORE a 12-run job spends its first hours OOM-ing.
    MODEL="$MODEL" GPU="$GPU" python - <<'VRAM' || true
import os, subprocess, yaml

# Coarse but honest peak-VRAM model, per regime:
#   optimiser state = 16 B/param full fine-tune (fp32 weights + grads + 2 AdamW
#                     moments), 4 B/param frozen (weights only).
#   activations     = what autograd must keep for the backward pass, which only
#                     exists for layers that get a gradient. A frozen backbone
#                     builds no graph at all, which is why head-only runs are
#                     cheap however big the backbone is.
MODELS = {   # params, GB of stored activations per clip when the backbone trains
    "VideoMAEGiant": (1.013e9, 3.6),   # ViT-g: 2048 tokens x 40 layers x 1408 dim
    "VideoMAE":      (0.087e9, 0.5),   # ViT-B: 1568 tokens x 12 layers x 768 dim
}
OVERHEAD = 2.0   # CUDA context, workspace, fragmentation

model = os.environ["MODEL"]
if model not in MODELS:
    print(f"  vram    unknown model {model}; skipping the estimate"); raise SystemExit(0)
params, act_per_clip = MODELS[model]
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    total = int(q.stdout.split()[int(os.environ.get("GPU", "0"))]) / 1024
except Exception:
    print("  vram    could not read GPU memory; skipping the estimate"); raise SystemExit(0)

bs = (yaml.safe_load(open("configs/config.yaml")) or {}).get("batch_size", 8)
full_state, head_state = params * 16 / 1e9, params * 4 / 1e9
full_peak = full_state + act_per_clip * bs + OVERHEAD
head_peak = head_state + 0.15 * bs + OVERHEAD

print(f"  vram    {total:.0f} GB total | at batch_size={bs}: "
      f"full ~{full_peak:.0f} GB, head-only ~{head_peak:.0f} GB")
if head_peak > total:
    print(f"  WARN    even the HEAD-ONLY runs may not fit. Lower batch_size.")
if full_peak > total:
    fits = int((total - full_state - OVERHEAD) // act_per_clip)
    print(f"  WARN    the 6 FULL fine-tuning runs will very likely OOM.")
    print(f"          {full_state:.1f} GB of optimiser state + "
          f"{act_per_clip * bs:.1f} GB of activations at batch_size={bs} "
          f"exceeds {total:.0f} GB.")
    if fits >= 1:
        print(f"          Largest batch_size that should fit: ~{fits}. Set it in "
              f"configs/config.yaml,")
        print(f"          and consider raising the LRs to match the smaller batch.")
    else:
        print(f"          Not even batch_size=1 fits ({full_state:.1f} GB of state on a "
              f"{total:.0f} GB card).")
        print(f"          Use MODEL=VideoMAE (ViT-B, {0.087e9*16/1e9:.1f} GB of state) "
              f"or a larger GPU.")
    print(f"          The 6 HEAD-ONLY runs are unaffected: a frozen backbone stores no")
    print(f"          activations for backward, so they fit in ~{head_peak:.0f} GB.")
VRAM

    echo "─────────────────────────────────────────────────────────────────────"
    return $fail
}

# ---------------------------------------------------------------- one experiment
run_experiment() {
    local site="$1" task="$2" regime="$3"
    local name="${MODEL}_${task}_${site}_${regime}"
    local dir="$RUNS_DIR/$name"

    if [[ -n "$ONLY" && ! "$name" =~ $ONLY ]]; then return 0; fi
    if [[ -f "$dir/DONE" ]]; then
        echo "[skip]  $name (already done — rm $dir/DONE to redo)"
        SKIPPED+=("$name"); return 0
    fi
    mkdir -p "$dir"

    local data_config epochs patience
    [[ "$task" == "multiclass" ]] && data_config="$MULTICLASS_CONFIG" || data_config="$MULTILABEL_CONFIG"
    if [[ "$regime" == "full" ]]; then
        epochs=$EPOCHS_FULL;  patience=$PATIENCE_FULL
    else
        epochs=$EPOCHS_HEAD;  patience=$PATIENCE_HEAD
    fi

    local -a cmd=(python -m src.training
                  --model "$MODEL" --data-config "$data_config"
                  --run-name "$name" --epochs "$epochs" --patience "$patience")
    local filter; filter="$(site_filter "$site")"
    [[ -n "$filter" ]] && cmd+=(--sites "$filter")
    [[ "$regime" == "head" ]] && cmd+=(--freeze-backbone)

    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "[$((++COUNT))/${#EXPERIMENTS[@]}] $name"
    echo "  task=$task  site=${filter:-Haydom+DRC}  regime=$regime  epochs=$epochs  patience=$patience"
    echo "  train: ${cmd[*]}"
    echo "════════════════════════════════════════════════════════════════════"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  test : python -m src.test --model $MODEL --model_path <ckpt> \\"
        echo "             --test_data $(test_sets "$site")"
        return 0
    fi

    local started; started=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "$dir/train.log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "[FAIL]  $name — training exited $rc (see $dir/train.log)"
        FAILED+=("$name: training rc=$rc"); return 1
    fi

    : > "$dir/manifest.txt"
    {   echo "experiment   : $name"
        echo "task         : $task ($data_config)"
        echo "sites        : ${filter:-Haydom+DRC}"
        echo "regime       : $regime (epochs=$epochs patience=$patience)"
        echo "train minutes: $(( ($(date +%s) - started) / 60 ))"
    } >> "$dir/manifest.txt"

    # Recover the checkpoints from the training log — exact, and immune to any
    # stale file that happens to be newer in checkpoints/.
    local scored=0 tag ckpt
    for tag in $CKPT_TAGS; do
        ckpt=$(grep -F "Saved $tag checkpoint -> " "$dir/train.log" | tail -1 | sed "s/.*-> //")
        if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
            echo "[warn]  $name — no '$tag' checkpoint in the log; skipping its evaluation"
            echo "checkpoint($tag): NOT PRODUCED" >> "$dir/manifest.txt"
            continue
        fi
        echo "checkpoint($tag): $ckpt" >> "$dir/manifest.txt"
        echo
        echo "  ── evaluating $tag: $ckpt"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$GPU" python -m src.test \
            --model "$MODEL" --model_path "$ckpt" --results_dir results/ \
            --test_data $(test_sets "$site") 2>&1 | tee "$dir/test_${tag}.log"
        local trc=${PIPESTATUS[0]}
        if [[ $trc -ne 0 ]]; then
            echo "[FAIL]  $name — evaluation of $tag exited $trc"
            FAILED+=("$name: test($tag) rc=$trc")
        else
            scored=$((scored + 1))
            grep -F "results -> " "$dir/test_${tag}.log" | sed "s/.*results -> /results($tag): /" \
                >> "$dir/manifest.txt"
            grep -F "scores -> "  "$dir/test_${tag}.log" | sed "s/.*scores -> /scores($tag) : /" \
                >> "$dir/manifest.txt"
        fi
    done

    if [[ $scored -eq 0 ]]; then
        echo "[FAIL]  $name — trained but nothing was scored"
        FAILED+=("$name: nothing scored"); return 1
    fi
    date -Iseconds > "$dir/DONE"
    OK+=("$name")
    echo "[done]  $name -> $dir/"
}

# ---------------------------------------------------------------- the matrix
EXPERIMENTS=()
for site in haydom drc both; do
    for task in multiclass multilabel; do
        for regime in full head; do
            EXPERIMENTS+=("$site $task $regime")
        done
    done
done

mkdir -p "$RUNS_DIR" results checkpoints models
OK=(); FAILED=(); SKIPPED=(); COUNT=0

echo "VideoMAE experiment matrix — ${#EXPERIMENTS[@]} experiments"
echo "  model=$MODEL gpu=$GPU ckpt_tags='$CKPT_TAGS' cross_site=$CROSS_SITE runs_dir=$RUNS_DIR"
[[ -n "$ONLY" ]] && echo "  ONLY='$ONLY' (filtering)"
echo

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    if ! preflight; then
        echo
        echo "preflight failed — nothing was run. Fix the above, or set"
        echo "SKIP_PREFLIGHT=1 to start anyway (not recommended for an unattended job)."
        exit 1
    fi
fi

STARTED=$(date +%s)
for e in "${EXPERIMENTS[@]}"; do
    # shellcheck disable=SC2086
    run_experiment $e
done

# ---------------------------------------------------------------- summary
echo
echo "═════════════════════════════════════════════════════════════════════"
echo "SUMMARY   ($(( ($(date +%s) - STARTED) / 60 )) minutes)"
echo "═════════════════════════════════════════════════════════════════════"
printf '  completed : %d\n' "${#OK[@]}"
printf '  skipped   : %d\n' "${#SKIPPED[@]}"
printf '  failed    : %d\n' "${#FAILED[@]}"
[[ ${#FAILED[@]} -gt 0 ]] && { echo; printf '    FAILED %s\n' "${FAILED[@]}"; }
echo
echo "  per-run logs and artefact manifests : $RUNS_DIR/<experiment>/"
echo "  metrics / scores                    : results/"
echo
echo "  Collect every headline number with:"
echo "    grep -H -e '^macro/f1' -e '^minority/f1' -e '^macro/ap' results/results_${MODEL}_*.csv"
echo
[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
