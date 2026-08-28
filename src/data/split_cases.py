#!/usr/bin/env python3
"""
split_cases.py

Turn the combined clip manifest (build_manifest.py) into train / validation /
per-site test CSVs at the WHOLE-CASE level, so no case's clips leak across
splits — the same policy as the multimodal thesis.

--------------------------------------------------------------------------
Two test sets, one per hospital
--------------------------------------------------------------------------
Haydom and DRC differ in camera, lighting, staff and protocol, so a single
pooled test score hides the only number that matters for deployment: does the
model work at THIS hospital. Each site therefore gets its own test set, sized
independently as `--test-ratio` of that site's clips (they are NOT forced to the
same size — the sites have very different amounts of data).

    data/test_haydom.csv    Haydom cases only
    data/test_drc.csv       DRC cases only
    data/test.csv           their union, for anything that wants one test set

The thesis' 14 frozen cases are used as SEEDS: they are always placed in their
own site's test set, and the extra cases are chosen around them. Every test row
carries `thesis_test` (1/0), so the exact thesis-comparable evaluation is still
one filter away:

    df[df.thesis_test == 1]

--------------------------------------------------------------------------
How cases are chosen
--------------------------------------------------------------------------
Greedy fill, then hill-climbing with randomized restarts, on a
size-and-composition objective, per site. For each site, the target is
`--test-ratio` of that site's:

    * case count      — without it the selector hits the clip target with many
                        SHORT episodes, making test systematically shorter than train,
    * clip count      — the size target itself,
    * count of clips with no annotated activity at all,
    * fraction-mass of every activity (sum of frac_a over its clips).

Each dimension's error is measured RELATIVE to its target, so being 30 % short on
suction mass costs exactly as much as being 30 % short on clips — that is what
puts the rare classes into the test sets in usable numbers, instead of letting
them fall wherever the clip count happens to land.

Case sizes are heavy-tailed, so a single greedy pass reliably ends up stuck on an
overshoot. Hence: greedy fill, then a best-improvement hill-climb over
add/drop/swap moves, then `--restarts` more hill-climbs from random starting
subsets, keeping the best. In practice every dimension lands within a few percent
of its target.

Validation is chosen from the remaining cases the same way (targeting
`1 - --train-ratio` of the pool, per site), so val is size-balanced and always
covers both hospitals. Train is everything left.

--------------------------------------------------------------------------
The split is deliberately INDEPENDENT of configs/data.yaml
--------------------------------------------------------------------------
The objective uses raw fraction-mass, NOT resolved labels. If the split depended
on the thresholds, nudging one threshold would reshuffle cases between splits
and quietly destroy run-to-run comparability.

The split CSVs stay TASK-AGNOSTIC too: every clip of every case is written with
its `bucket` and `frac_*` columns intact, exactly as in the manifest. Bucket
filtering and thresholding are applied later by DataSpec at Dataset load time.
So a change to configs/data.yaml requires NO rebuild and NO re-split — this
script only ever needs re-running when the manifest or the split policy changes.

The data config is read only to REPORT the resulting label distribution.

Audit the result with:
    python -m src.data.explore_data --manifest data/clips_all.csv --splits-dir data
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .manifest import evidence_masses, explain_bad_manifest, read_manifest
from .spec import DataSpec

# The thesis' 14 held-out cases (DRC: 4, Haydom: 10). They are SEEDS for the
# per-site test sets, never moved elsewhere, and are flagged `thesis_test=1`.
DEFAULT_TEST_CASES = [
    # DRC
    "2-33998-1", "2-34325-1", "2-37178-1", "2-37453-1",
    # Haydom
    "11848523", "15233524", "28631424", "37572224", "38037024",
    "38042423", "38714124", "40094725", "40386725", "40402325",
]

NEGATIVE_STRATUM = "__none__"
MAX_MOVES = 500      # hill-climb iterations; it converges long before this
DEFAULT_RESTARTS = 24  # randomized restarts per site (see greedy_select)


def read_case_list(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def slug(site: str) -> str:
    """'Haydom' -> 'haydom'; safe for a filename."""
    return re.sub(r"[^a-z0-9]+", "_", site.lower()).strip("_") or "site"


# --------------------------------------------------------------------------
# case features — config-independent by construction
# --------------------------------------------------------------------------
def case_features(df: pd.DataFrame, spec: DataSpec) -> dict[str, dict]:
    """{case_id: {site, clips, mass_<a>..., negatives}}.

    `mass_a` = sum(frac_a) over the case's clips, i.e. how many 3 s windows'
    worth of activity `a` the case holds. `negatives` = clips with no annotated
    activity. Neither depends on thresholds or bucket policy — see the module
    docstring on why that matters.
    """
    feats = {}
    mass = evidence_masses(df, spec)
    for case, idx in df.groupby("case_id", sort=True).groups.items():
        g = mass.loc[idx]
        f = {"site": df.loc[idx, "site"].iloc[0], "cases": 1.0, "clips": float(len(idx))}
        for a in spec.activities:
            f[f"mass_{a}"] = float(g[a].sum())
        f["negatives"] = float((g.sum(axis=1) == 0).sum())
        feats[case] = f
    return feats


def dims(spec: DataSpec) -> list[str]:
    """The dimensions the selection is balanced on.

    `cases` is in here on purpose. Without it the selector hits the clip and mass
    targets by hoovering up many SHORT episodes — arithmetically perfect, but it
    would make the test set systematically shorter than the training set. Asking
    for the expected NUMBER of cases too keeps the episode-length mix honest.
    """
    return ["cases", "clips", "negatives"] + [f"mass_{a}" for a in spec.activities]


def cost_from_sums(sums: list[float], targets: list[float]) -> float:
    """Relative squared distance from the per-dimension targets.

    Normalising by the target makes a rare activity count as much as the clip
    count: being 30 % short on suction mass is as expensive as being 30 % short
    on clips, which is exactly the trade the split should be making. Dimensions
    the site has none of (target 0) are skipped rather than dividing by zero.
    """
    return sum(((g - t) / t) ** 2 for g, t in zip(sums, targets) if t > 0)


def cost(selected, feats: dict, targets: dict, keys: list[str]) -> float:
    """`cost_from_sums` for an arbitrary case set — used by the reports."""
    sums = [sum(feats[c][k] for c in selected) for k in keys]
    return cost_from_sums(sums, [targets[k] for k in keys])


def _hill_climb(selected: set[str], pool: list[str], vec: dict, tgt: list[float],
                seeds: set[str], min_cases: int):
    """Best-improvement local search over add / drop / swap moves.

    Case sizes are heavy-tailed, so a plain greedy fill regularly stops on an
    overshoot (one whale case past the target) that only a drop can undo — hence
    all three move types. `seeds` are never dropped. Returns (selected, sums, cost).
    """
    selected, pool = set(selected), sorted(pool)
    sums = [0.0] * len(tgt)
    for c in selected:
        sums = [s + v for s, v in zip(sums, vec[c])]
    best = cost_from_sums(sums, tgt)

    for _ in range(MAX_MOVES):
        moves = [(cost_from_sums([s + v for s, v in zip(sums, vec[i])], tgt), "", i)
                 for i in pool]
        droppable = sorted(selected - seeds)
        if len(selected) > max(min_cases, 1):
            moves += [(cost_from_sums([s - v for s, v in zip(sums, vec[o])], tgt), o, "")
                      for o in droppable]
        moves += [(cost_from_sums([s - o + i for s, o, i in zip(sums, vec[out], vec[inn])], tgt),
                   out, inn)
                  for out in droppable for inn in pool]
        if not moves:
            break
        new_cost, out, inn = min(moves)
        if new_cost >= best - 1e-12:
            break
        if out:
            selected.discard(out)
            sums = [s - v for s, v in zip(sums, vec[out])]
            pool = sorted(pool + [out])
        if inn:
            selected.add(inn)
            sums = [s + v for s, v in zip(sums, vec[inn])]
            pool.remove(inn)
        best = new_cost
    return selected, sums, best


def greedy_select(candidates: list[str], feats: dict, targets: dict, keys: list[str],
                  seeds: set[str] = frozenset(), min_cases: int = 0,
                  restarts: int = DEFAULT_RESTARTS, seed: int = 2025) -> set[str]:
    """The set of cases whose summed features sits closest to `targets`.

    This is a multi-dimensional subset-sum, and a single greedy pass lands in
    poor local optima when a few cases hold most of the clips. So: one
    deterministic greedy fill, then `restarts` hill-climbs from random starting
    subsets, keeping the best result. Cheap — every move is scored from running
    per-dimension sums in O(dimensions).

    Deterministic given `seed`: the restarts draw from a local RNG and ties break
    on the case id, so the same manifest and flags always give the same split.
    """
    candidates = sorted(candidates)
    seeds = {c for c in candidates if c in seeds}
    vec = {c: [feats[c][k] for k in keys] for c in candidates}
    tgt = [targets[k] for k in keys]

    # deterministic greedy fill: add the best case while that improves
    selected, pool = set(seeds), [c for c in candidates if c not in seeds]
    sums = [0.0] * len(keys)
    for c in selected:
        sums = [s + v for s, v in zip(sums, vec[c])]
    best = cost_from_sums(sums, tgt)
    while pool:
        cand_cost, cand = min(
            (cost_from_sums([s + v for s, v in zip(sums, vec[c])], tgt), c) for c in pool)
        if cand_cost >= best and len(selected) >= min_cases:
            break
        selected.add(cand)
        pool.remove(cand)
        sums = [s + v for s, v in zip(sums, vec[cand])]
        best = cand_cost

    best_set, _, best_cost = _hill_climb(selected, pool, vec, tgt, seeds, min_cases)

    rng = random.Random(seed)
    free = [c for c in candidates if c not in seeds]
    for _ in range(max(0, restarts)):
        k = rng.randint(min(min_cases, len(free)), len(free))
        start = seeds | set(rng.sample(free, k))
        cand_set, _, cand_cost = _hill_climb(
            start, [c for c in candidates if c not in start], vec, tgt, seeds, min_cases)
        if cand_cost < best_cost - 1e-12:
            best_set, best_cost = cand_set, cand_cost
    return best_set


def balance_report(selected: set[str], feats: dict, targets: dict,
                   keys: list[str], indent: str = "    ") -> None:
    print(f"{indent}{'dimension':<20}{'selected':>12}{'target':>12}{'error':>10}")
    for k in keys:
        got = sum(feats[c][k] for c in selected)
        t = targets[k]
        err = f"{100 * (got - t) / t:>+9.1f}%" if t > 0 else "        -"
        print(f"{indent}{k:<20}{got:>12,.1f}{t:>12,.1f}{err}")


# --------------------------------------------------------------------------
# train/val fallback stratification (used only with --no-balanced-val)
# --------------------------------------------------------------------------
def case_stratum(feats_case: dict, spec: DataSpec) -> str:
    mass = {a: feats_case[f"mass_{a}"] for a in spec.activities}
    best = max(mass, key=lambda a: (mass[a], a))
    dominant = best if mass[best] > 0 else NEGATIVE_STRATUM
    return f"{feats_case['site']}|{dominant}"


def stratified_case_split(case_to_stratum, train_ratio, seed):
    """Whole-case train/val assignment, stratified by site + dominant activity.

    Counts CASES, not clips, so a whale case can still skew the realised ratio —
    which is why the balanced selector is the default.
    """
    by_stratum = defaultdict(list)
    for case, stratum in case_to_stratum.items():
        by_stratum[stratum].append(case)
    rng = random.Random(seed)
    train_cases, val_cases = set(), set()
    for stratum in sorted(by_stratum):
        cases = sorted(by_stratum[stratum])
        rng.shuffle(cases)
        n_train = round(len(cases) * train_ratio)
        train_cases.update(cases[:n_train])
        val_cases.update(cases[n_train:])
    return train_cases, val_cases


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def label_distribution(df: pd.DataFrame, spec: DataSpec):
    """What the CURRENT data config resolves this split into. Report only."""
    kept, dropped = Counter(), 0
    memo: dict[tuple, object] = {}
    frac_cols = spec.frac_columns()
    tagged_col = (df["tagged"].astype(int) if "tagged" in df.columns
                  else pd.Series(1, index=df.index))
    dir_col = df["clip_dir"] if "clip_dir" in df.columns else pd.Series("", index=df.index)
    for key in zip(df["bucket"].astype(int), tagged_col, dir_col,
                   *(df[c].astype(float) for c in frac_cols)):
        if key not in memo:
            fracs = dict(zip(spec.activities, key[3:]))
            label = spec.resolve(int(key[0]), fracs, tagged=bool(key[1]),
                                 dir_activities=spec.activities_from_path(key[2]))
            if label is None:
                memo[key] = None
            elif spec.is_multilabel:
                active = [a for a, t, m in zip(spec.activities, label.targets, label.mask)
                          if t == 1.0 and m == 1.0]
                memo[key] = "+".join(active) or spec.negative_class
            else:
                memo[key] = spec.class_names[label.class_index]
        name = memo[key]
        if name is None:
            dropped += 1
        else:
            kept[name] += 1
    return kept, dropped


def describe_split(name: str, d: pd.DataFrame, spec: DataSpec, out: Path, total_clips: int):
    kept, dropped = label_distribution(d, spec)
    usable = sum(kept.values())
    sites = ", ".join(f"{s}: {int(d['site'].eq(s).sum()):,} clips / "
                      f"{d[d['site'].eq(s)]['case_id'].nunique()} cases"
                      for s in sorted(d["site"].unique()))
    print(f"\n[{name}] {len(d):,} clips ({100 * len(d) / max(total_clips, 1):.1f}% of corpus) "
          f"| {d['case_id'].nunique()} cases -> {out}")
    print(f"         {sites}")
    print(f"         usable under the current config: {usable:,} (dropped {dropped:,})")
    for k, v in sorted(kept.items(), key=lambda kv: -kv[1]):
        print(f"           {k:<32} {v:>8,}  {100 * v / max(usable, 1):>5.1f}%")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, help="Combined manifest from build_manifest.py")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory to write the split CSVs")
    p.add_argument("--test-ratio", type=float, default=0.20,
                   help="Share of EACH SITE's clips held out as that site's test set. "
                        "0 = use only the seed cases (the old 14-case behaviour).")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Share of the REMAINING (non-test) clips used for training.")
    p.add_argument("--min-test-cases", type=int, default=4,
                   help="Floor on cases per site test set, even if the ratio is met sooner.")
    p.add_argument("--seed", type=int, default=2025,
                   help="Seeds the selector's randomized restarts. The split is fully "
                        "determined by (manifest, flags, seed).")
    p.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS,
                   help="Randomized restarts per site in the case selector. More = "
                        "better-balanced splits, linearly slower.")
    p.add_argument("--data-config", default=None,
                   help="Data/label config YAML (default: configs/data.yaml). Used for "
                        "reporting and for the activity list only — never for the "
                        "assignment itself.")
    p.add_argument("--test-cases-file", type=Path, default=None,
                   help="Override the 14 default SEED cases (one case_id per line).")
    p.add_argument("--freeze-test", action="store_true",
                   help="Use the seed cases as the complete test set (equivalent to "
                        "--test-ratio 0), still written per site.")
    p.add_argument("--no-balanced-val", action="store_true",
                   help="Pick validation by stratified case sampling (counts cases, "
                        "ignores case size) instead of the balanced selector.")
    p.add_argument("--train-cases-file", type=Path, default=None,
                   help="Explicit train case_ids (exact reproduction); overrides the val selector.")
    p.add_argument("--val-cases-file", type=Path, default=None,
                   help="Explicit val case_ids (exact reproduction); overrides the val selector.")
    args = p.parse_args()

    spec = DataSpec.load(args.data_config)
    df = read_manifest(args.manifest)
    df["case_id"] = df["case_id"].astype(str)

    required = ["video_path", "case_id", "site", "bucket"] + spec.frac_columns()
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(explain_bad_manifest(args.manifest, df, missing_cols))

    feats = case_features(df, spec)
    keys = dims(spec)
    sites = sorted(df["site"].unique())
    test_ratio = 0.0 if args.freeze_test else args.test_ratio

    seed_cases = set(read_case_list(args.test_cases_file) if args.test_cases_file
                     else DEFAULT_TEST_CASES)
    missing = seed_cases - set(feats)
    if missing:
        print(f"[WARN] {len(missing)} seed test case_ids not found in manifest: {sorted(missing)}")
    seed_cases &= set(feats)

    # ---------------------------------------------------------------- test
    print(f"\n{'=' * 78}\nPER-SITE TEST SELECTION  (target: {test_ratio:.0%} of each site's clips)\n{'=' * 78}")
    test_by_site: dict[str, set[str]] = {}
    for site in sites:
        site_cases = [c for c, f in feats.items() if f["site"] == site]
        site_seeds = {c for c in seed_cases if feats[c]["site"] == site}
        totals = {k: sum(feats[c][k] for c in site_cases) for k in keys}
        targets = {k: test_ratio * v for k, v in totals.items()}
        if test_ratio <= 0:
            chosen = set(site_seeds)
        else:
            chosen = greedy_select(site_cases, feats, targets, keys,
                                   seeds=site_seeds, min_cases=args.min_test_cases,
                                   restarts=args.restarts, seed=args.seed)
        test_by_site[site] = chosen
        n_clips = int(sum(feats[c]["clips"] for c in chosen))
        print(f"\n{site}: {len(chosen)} cases ({len(site_seeds)} thesis seeds + "
              f"{len(chosen) - len(site_seeds)} added), {n_clips:,} clips "
              f"({100 * n_clips / max(totals['clips'], 1):.1f}% of the site)")
        if test_ratio > 0:
            balance_report(chosen, feats, targets, keys)
        print(f"    cases: {', '.join(sorted(chosen))}")

    test_cases = set().union(*test_by_site.values()) if test_by_site else set()

    # ---------------------------------------------------------------- train/val
    rest = [c for c in feats if c not in test_cases]
    if args.train_cases_file and args.val_cases_file:
        train_cases = set(read_case_list(args.train_cases_file))
        val_cases = set(read_case_list(args.val_cases_file))
    elif args.no_balanced_val:
        strata = {c: case_stratum(feats[c], spec) for c in rest}
        train_cases, val_cases = stratified_case_split(strata, args.train_ratio, args.seed)
    else:
        val_cases = set()
        print(f"\n{'=' * 78}\nVALIDATION SELECTION  (target: {1 - args.train_ratio:.0%} "
              f"of each site's remaining clips)\n{'=' * 78}")
        for site in sites:
            site_rest = [c for c in rest if feats[c]["site"] == site]
            totals = {k: sum(feats[c][k] for c in site_rest) for k in keys}
            targets = {k: (1 - args.train_ratio) * v for k, v in totals.items()}
            chosen = greedy_select(site_rest, feats, targets, keys, min_cases=1,
                                   restarts=args.restarts, seed=args.seed)
            val_cases |= chosen
            n_clips = int(sum(feats[c]["clips"] for c in chosen))
            print(f"\n{site}: {len(chosen)} cases, {n_clips:,} clips "
                  f"({100 * n_clips / max(totals['clips'], 1):.1f}% of the site's remainder)")
            balance_report(chosen, feats, targets, keys)
        train_cases = set(rest) - val_cases

    # ---------------------------------------------------------------- write
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    outputs: list[tuple[str, pd.DataFrame, Path]] = []

    train_df = df[df["case_id"].isin(train_cases)][cols].copy()
    val_df = df[df["case_id"].isin(val_cases)][cols].copy()
    outputs.append(("train", train_df, args.out_dir / "train.csv"))
    outputs.append(("validation", val_df, args.out_dir / "validation.csv"))

    test_all = df[df["case_id"].isin(test_cases)][cols].copy()
    test_all["thesis_test"] = test_all["case_id"].isin(seed_cases).astype(int)
    for site in sites:
        d = test_all[test_all["site"] == site].copy()
        outputs.append((f"test_{slug(site)}", d, args.out_dir / f"test_{slug(site)}.csv"))
    outputs.append(("test", test_all, args.out_dir / "test.csv"))

    print(f"\n{'=' * 78}\nSPLITS  (task-agnostic; labels resolved at load time by "
          f"{spec.source}, {spec.task})\n{'=' * 78}")
    for name, d, out in outputs:
        d.to_csv(out, index=False)
        describe_split(name, d, spec, out, len(df))

    # a machine-readable record of who went where, for the audit and for reruns
    assignment = []
    for case, f in sorted(feats.items()):
        where = ("test" if case in test_cases else
                 "validation" if case in val_cases else
                 "train" if case in train_cases else "unassigned")
        assignment.append({"case_id": case, "site": f["site"], "split": where,
                           "clips": int(f["clips"]),
                           "thesis_test": int(case in seed_cases)})
    pd.DataFrame(assignment).to_csv(args.out_dir / "split_assignment.csv", index=False)
    (args.out_dir / "split_params.json").write_text(json.dumps({
        "manifest": str(args.manifest), "test_ratio": test_ratio,
        "train_ratio": args.train_ratio, "min_test_cases": args.min_test_cases,
        "seed": args.seed, "restarts": args.restarts, "data_config": spec.source,
        "seed_test_cases": sorted(seed_cases),
        "test_cases_by_site": {s: sorted(c) for s, c in test_by_site.items()},
    }, indent=2))
    print(f"\n[written] {args.out_dir / 'split_assignment.csv'}  (case -> split)")
    print(f"[written] {args.out_dir / 'split_params.json'}  (how this split was made)")
    print(f"\nThe thesis-comparable subset is still in there: "
          f"data/test.csv rows with thesis_test == 1 "
          f"({int(test_all['thesis_test'].sum()):,} clips, {len(seed_cases)} cases).")
    print("Audit it with:\n  python -m src.data.explore_data "
          f"--manifest {args.manifest} --splits-dir {args.out_dir}")


if __name__ == "__main__":
    main()
