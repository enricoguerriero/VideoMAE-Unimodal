#!/usr/bin/env python3
"""
explore_data.py

READ-ONLY audit of the clip manifest and of whatever split CSVs exist next to
it. Writes nothing to the dataset; it only reports. Use it before and after
re-splitting (src/data/split_cases.py) to see what actually changed.

    python -m src.data.explore_data --manifest data/clips_all.csv --splits-dir data

What it answers, in order:

  1. How much data is there, per site and per label bucket?
  2. What does the CURRENT data config resolve that into?
  3. How is it distributed over CASES — the unit the split works on? A handful
     of "whale" cases holding most clips is what makes a case-level split
     lumpy and hard to balance.
  4. Where does each class actually live? A class present in 500 clips but
     only 3 cases has an effective sample size of 3, not 500.
  5. What do the existing splits look like — sizes, per-site composition, label
     mix vs. the corpus, leakage, and whether each split can support the
     metrics computed on it.
  6. Given a target test share, how far off is the current test set and what
     would a re-split have to move?

Everything class-related is reported through the DataSpec in configs/data.yaml
(override with --data-config), because "how many suction clips" only has an
answer once thresholds and bucket policy are fixed. The CASE-level geometry
(sections 3-4's fraction masses) is config-independent, which is why the
splitter balances on it.

Pure pandas + PyYAML — no torch, no av. Runs anywhere the manifest does.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .spec import BUCKET_NAMES, DataSpec

# A class with fewer than this many clips in an eval split gives an F1 whose
# smallest possible change is coarse enough to be noise. Warned about, not enforced.
MIN_EVAL_CLIPS = 100
# Same idea one level up: clips inside a case are correlated, so the number of
# CASES carrying a class is closer to its true sample size.
MIN_EVAL_CASES = 3

DROPPED = "(dropped by config)"


# --------------------------------------------------------------------------
# resolution helpers
# --------------------------------------------------------------------------
def resolve_labels(df: pd.DataFrame, spec: DataSpec) -> pd.Series:
    """One label name per row under the current config; DROPPED where the
    config discards the clip.

    Memoised on (bucket, fractions): the manifest stores fractions rounded to
    2 decimals, so a corpus of 10^5 clips has only a few thousand distinct
    (bucket, fracs) keys and `spec.resolve` runs once per key.
    """
    frac_cols = spec.frac_columns()
    memo: dict[tuple, str] = {}

    def name_for(key):
        bucket, fracs = int(key[0]), dict(zip(spec.activities, key[1:]))
        label = spec.resolve(bucket, fracs)
        if label is None:
            return DROPPED
        if spec.is_multilabel:
            active = [a for a, t, m in zip(spec.activities, label.targets, label.mask)
                      if t == 1.0 and m == 1.0]
            return "+".join(active) or spec.negative_class
        return spec.class_names[label.class_index]

    keys = list(zip(df["bucket"].astype(int), *(df[c].astype(float) for c in frac_cols)))
    out = []
    for k in keys:
        if k not in memo:
            memo[k] = name_for(k)
        out.append(memo[k])
    return pd.Series(out, index=df.index, name="label")


def case_table(df: pd.DataFrame, spec: DataSpec) -> pd.DataFrame:
    """One row per case: size, config-independent activity mass, resolved labels.

    `mass_<a>` is sum(frac_a) over the case's clips — the number of 3 s windows'
    worth of activity `a` the case contains. It does not depend on thresholds or
    on bucket policy, which is exactly why the splitter balances on it: moving a
    threshold must not reshuffle cases between splits.
    """
    rows = []
    for (case, site), g in df.groupby(["case_id", "site"], sort=True):
        row = {"case_id": case, "site": site, "clips": len(g)}
        for a in spec.activities:
            row[f"mass_{a}"] = float(g[f"frac_{a}"].sum())
        row["usable"] = int((g["label"] != DROPPED).sum())
        for name in [spec.negative_class, *spec.activities]:
            row[f"n_{name}"] = int((g["label"] == name).sum())
        mass = {a: row[f"mass_{a}"] for a in spec.activities}
        best = max(mass, key=lambda a: (mass[a], a))
        row["dominant"] = best if mass[best] > 0 else "none"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("clips", ascending=False).reset_index(drop=True)


def gini(values) -> float:
    """0 = every case the same size, 1 = one case holds everything."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def pct(part, whole) -> str:
    return f"{100 * part / whole:>5.1f}%" if whole else "    -"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# 1-2. corpus
# --------------------------------------------------------------------------
def report_corpus(df: pd.DataFrame, spec: DataSpec, sites: list[str]) -> None:
    rule("1. CORPUS")
    print(f"{'':<14}{'clips':>12}{'cases':>10}")
    for s in sites:
        d = df[df["site"] == s]
        print(f"{s:<14}{len(d):>12,}{d['case_id'].nunique():>10}")
    print(f"{'TOTAL':<14}{len(df):>12,}{df['case_id'].nunique():>10}")

    print("\nbucket census (as found on disk)")
    print(f"{'bucket':<26}" + "".join(f"{s:>12}" for s in sites) + f"{'TOTAL':>12}  policy")
    for b in sorted(BUCKET_NAMES):
        counts = [int((df["site"].eq(s) & df["bucket"].eq(b)).sum()) for s in sites]
        policy = "keep" if spec.keeps_bucket(b) else "DROP"
        print(f"{b} {BUCKET_NAMES[b]:<24}" + "".join(f"{c:>12,}" for c in counts)
              + f"{sum(counts):>12,}  {policy}")

    rule(f"2. LABELS under {spec.source} ({spec.task})")
    order = [spec.negative_class, *spec.activities]
    seen = list(dict.fromkeys(order + sorted(set(df["label"]) - set(order) - {DROPPED})))
    print(f"{'label':<26}" + "".join(f"{s:>12}" for s in sites) + f"{'TOTAL':>12}{'share':>9}")
    usable = int((df["label"] != DROPPED).sum())
    for name in seen:
        counts = [int((df["site"].eq(s) & df["label"].eq(name)).sum()) for s in sites]
        if sum(counts) == 0:
            continue
        print(f"{name:<26}" + "".join(f"{c:>12,}" for c in counts)
              + f"{sum(counts):>12,}{pct(sum(counts), usable):>9}")
    dropped = [int((df["site"].eq(s) & df["label"].eq(DROPPED)).sum()) for s in sites]
    print(f"{DROPPED:<26}" + "".join(f"{c:>12,}" for c in dropped) + f"{sum(dropped):>12,}")
    print(f"\nusable clips: {usable:,} / {len(df):,} ({100 * usable / max(len(df), 1):.1f}%)")


# --------------------------------------------------------------------------
# 3-4. case geometry
# --------------------------------------------------------------------------
def report_cases(cases: pd.DataFrame, spec: DataSpec, sites: list[str], top: int) -> None:
    rule("3. CASE GEOMETRY  (the split moves whole cases, so this is the grain)")
    for s in sites + ["ALL"]:
        d = cases if s == "ALL" else cases[cases["site"] == s]
        if d.empty:
            continue
        clips = d["clips"]
        top5 = clips.nlargest(5).sum()
        print(f"\n{s}: {len(d)} cases, {clips.sum():,} clips")
        print(f"  clips/case   min {clips.min():,}  p25 {clips.quantile(.25):,.0f}  "
              f"median {clips.median():,.0f}  p75 {clips.quantile(.75):,.0f}  max {clips.max():,}")
        print(f"  concentration: Gini {gini(clips):.2f}, "
              f"top-5 cases hold {pct(top5, clips.sum()).strip()} of the site's clips")
        empty = int((d["usable"] == 0).sum())
        if empty:
            print(f"  [!] {empty} case(s) contribute ZERO usable clips under this config "
                  f"— they occupy a split slot without supervising anything")

    print(f"\nlargest {top} cases")
    cols = ["case_id", "site", "clips", "usable", "dominant"] + [f"n_{a}" for a in spec.activities]
    print(cases[cols].head(top).to_string(index=False))

    rule("4. WHERE EACH CLASS LIVES  (effective sample size is CASES, not clips)")
    print(f"{'class':<20}{'clips':>10}{'cases':>8}{'top case share':>16}  per-site cases")
    for name in [spec.negative_class, *spec.activities]:
        col = f"n_{name}"
        if col not in cases:
            continue
        holders = cases[cases[col] > 0]
        total = int(cases[col].sum())
        if total == 0:
            print(f"{name:<20}{0:>10}{0:>8}{'-':>16}  [!] absent from the corpus")
            continue
        share = pct(holders[col].max(), total).strip()
        per_site = ", ".join(f"{s}:{int((holders['site'] == s).sum())}" for s in sites)
        print(f"{name:<20}{total:>10,}{len(holders):>8}{share:>16}  {per_site}")
    print("\nA class whose clips come from few cases cannot be split finely: every case "
          "\nis all-or-nothing, so its test-set count jumps in whole-case steps.")


# --------------------------------------------------------------------------
# 5. existing splits
# --------------------------------------------------------------------------
def load_splits(paths: list[Path], spec: DataSpec) -> dict[str, pd.DataFrame]:
    out = {}
    for p in paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["case_id"] = d["case_id"].astype(str)
        if "site" not in d.columns:
            d["site"] = "?"
        d["label"] = resolve_labels(d, spec)
        out[p.stem] = d
    return out


def canonical(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Splits with the redundancy removed: `test.csv` is the union of the
    per-site test files, so counting both double-counts every test clip."""
    per_site = [n for n in splits if n.startswith("test_")]
    return {n: d for n, d in splits.items() if not (n == "test" and per_site)}


def report_splits(splits: dict[str, pd.DataFrame], df: pd.DataFrame,
                  spec: DataSpec, sites: list[str]) -> None:
    rule("5. EXISTING SPLITS")
    if not splits:
        print("no split CSVs found — pass --splits-dir or --splits")
        return

    total_clips = len(df)
    print(f"{'split':<20}{'clips':>11}{'share':>8}{'cases':>8}  " +
          "  ".join(f"{s} clips/cases" for s in sites))
    for name, d in splits.items():
        per_site = "  ".join(
            f"{int(d['site'].eq(s).sum()):,}/{d[d['site'].eq(s)]['case_id'].nunique()}"
            for s in sites)
        print(f"{name:<20}{len(d):>11,}{pct(len(d), total_clips):>8}"
              f"{d['case_id'].nunique():>8}  {per_site}")

    # label mix, and how far each split drifts from the corpus
    order = [spec.negative_class, *spec.activities]
    corpus_usable = int((df["label"] != DROPPED).sum())
    corpus_mix = {n: 100 * int(df["label"].eq(n).sum()) / max(corpus_usable, 1) for n in order}
    print("\nlabel mix per split (share of that split's USABLE clips; "
          "Δ = percentage points vs corpus)")
    head = f"{'split':<20}{'usable':>10}" + "".join(f"{n[:11]:>13}" for n in order)
    print(head)
    print(f"{'corpus':<20}{corpus_usable:>10,}" +
          "".join(f"{corpus_mix[n]:>12.1f}%" for n in order))
    for name, d in splits.items():
        usable = int((d["label"] != DROPPED).sum())
        cells = ""
        for n in order:
            c = int(d["label"].eq(n).sum())
            share = 100 * c / max(usable, 1)
            cells += f"{share:>7.1f}% {share - corpus_mix[n]:>+5.1f}"
        print(f"{name:<20}{usable:>10,}{cells}")

    print("\nabsolute class counts per split (what the metrics are computed on)")
    print(f"{'split':<20}" + "".join(f"{n[:12]:>14}" for n in order))
    for name, d in splits.items():
        cells = ""
        for n in order:
            sub = d[d["label"].eq(n)]
            cells += f"{len(sub):>9,}/{sub['case_id'].nunique():<4}"
        print(f"{name:<20}{cells}")
    print("(clips / distinct cases)")

    warn_splits(splits, spec)


def warn_splits(splits: dict[str, pd.DataFrame], spec: DataSpec) -> None:
    print("\nfindings")
    found = False

    # leakage: a case must live in exactly one split
    owners = defaultdict(set)
    for name, d in splits.items():
        for c in d["case_id"].unique():
            owners[c].add(name)
    leaked = {c: s for c, s in owners.items() if len(s) > 1}
    # test.csv is expected to be the union of the per-site test files
    combined = {n for n in splits if n.startswith("test")}
    for case, names in sorted(leaked.items()):
        if names <= combined:
            continue
        found = True
        print(f"  [LEAK] case {case} appears in {sorted(names)} — clips from one "
              f"episode on both sides of the split")

    dupes = Counter()
    for d in canonical(splits).values():
        dupes.update(d["video_path"])
    n_dupe = sum(1 for v in dupes.values() if v > 1)
    if n_dupe:
        found = True
        print(f"  [DUPE] {n_dupe} clip paths appear in more than one split")

    for name, d in splits.items():
        if name == "train":
            continue
        for n in [spec.negative_class, *spec.activities]:
            sub = d[d["label"].eq(n)]
            if len(sub) == 0:
                found = True
                print(f"  [EMPTY] '{n}' has NO clips in {name} — every metric for it "
                      f"is undefined there")
                continue
            if len(sub) < MIN_EVAL_CLIPS:
                found = True
                print(f"  [THIN]  '{n}' has {len(sub)} clips in {name} "
                      f"(<{MIN_EVAL_CLIPS}): one clip moves its F1 by "
                      f"~{100 / len(sub):.1f} points")
            if sub["case_id"].nunique() < MIN_EVAL_CASES:
                found = True
                print(f"  [NARROW] '{n}' in {name} comes from only "
                      f"{sub['case_id'].nunique()} case(s) — that score measures "
                      f"those episodes, not the class")
    all_sites = sorted({s for d in splits.values() for s in d["site"].unique()})
    for name, d in splits.items():
        if name.startswith("test") and name != "test":
            continue  # a per-site test file is single-site by construction
        absent = [s for s in all_sites if not d["site"].eq(s).any()]
        if absent:
            found = True
            print(f"  [SITE]  {name} contains no clips from {absent} — nothing "
                  f"trained or measured there transfers to that hospital")

    if not found:
        print("  none — every split carries every class with a usable sample size")


# --------------------------------------------------------------------------
# 6. what a re-split would have to move
# --------------------------------------------------------------------------
def report_target(cases: pd.DataFrame, splits: dict[str, pd.DataFrame],
                  sites: list[str], target: float, spec: DataSpec) -> None:
    rule(f"6. TARGET: {target:.0%} of each site's clips held out for test")
    current = {}
    for name, d in canonical(splits).items():
        if not name.startswith("test"):
            continue
        for s in sites:
            current[s] = current.get(s, 0) + int(d["site"].eq(s).sum())

    print(f"{'site':<12}{'clips':>12}{'target test':>14}{'current test':>14}"
          f"{'gap':>10}{'cases to add*':>15}")
    for s in sites:
        d = cases[cases["site"] == s]
        site_clips = int(d["clips"].sum())
        want = int(round(target * site_clips))
        have = current.get(s, 0)
        median = float(d["clips"].median() or 1)
        need = max(0, want - have)
        print(f"{s:<12}{site_clips:>12,}{want:>14,}{have:>14,}{want - have:>+10,}"
              f"{need / max(median, 1):>15.1f}")
    print("* at this site's median case size — indicative only; the splitter picks "
          "cases\n  by best fit, not by size.")

    print("\nSuggested re-split (per-site test sets, thesis cases kept inside them):\n")
    print(f"  python -m src.data.split_cases \\\n"
          f"      --manifest data/clips_all.csv --out-dir data \\\n"
          f"      --data-config {spec.source} \\\n"
          f"      --test-ratio {target} --train-ratio 0.8 --seed 2025\n")
    print("Then re-run this audit against the new CSVs to confirm the class counts "
          "moved\nthe way you wanted.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("data/clips_all.csv"),
                   help="Combined manifest from build_manifest.py")
    p.add_argument("--splits-dir", type=Path, default=None,
                   help="Directory holding train.csv / validation.csv / test*.csv")
    p.add_argument("--splits", type=Path, nargs="*", default=None,
                   help="Explicit split CSVs (overrides --splits-dir)")
    p.add_argument("--data-config", default=None,
                   help="Data/label config YAML (default: configs/data.yaml)")
    p.add_argument("--target-test-ratio", type=float, default=0.20,
                   help="Share of each site's clips you want held out (section 6)")
    p.add_argument("--top", type=int, default=20, help="Rows in the largest-cases table")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Also write per_case.csv (the full case table) here")
    args = p.parse_args()

    spec = DataSpec.load(args.data_config)
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest} — run build_manifest.py first")

    df = pd.read_csv(args.manifest)
    df["case_id"] = df["case_id"].astype(str)
    missing = [c for c in ["video_path", "case_id", "site", "bucket"] + spec.frac_columns()
               if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.manifest} is missing columns {missing}; rebuild it "
                         f"with the current activity list.")
    df["label"] = resolve_labels(df, spec)
    sites = sorted(df["site"].unique())

    print(spec.describe())
    report_corpus(df, spec, sites)

    cases = case_table(df, spec)
    report_cases(cases, spec, sites, args.top)

    if args.splits is not None:
        paths = list(args.splits)
    elif args.splits_dir is not None:
        paths = sorted(args.splits_dir.glob("train.csv")) + \
                sorted(args.splits_dir.glob("validation.csv")) + \
                sorted(args.splits_dir.glob("test*.csv"))
    else:
        paths = []
    splits = load_splits(paths, spec)
    report_splits(splits, df, spec, sites)
    report_target(cases, splits, sites, args.target_test_ratio, spec)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / "per_case.csv"
        cases.to_csv(out, index=False)
        print(f"\n[written] {out}  ({len(cases)} cases)")


if __name__ == "__main__":
    main()
