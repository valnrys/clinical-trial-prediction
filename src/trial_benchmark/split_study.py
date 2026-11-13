"""How different is a model's score on a randomly split test set from its score on later trials?

    uv run python -m trial_benchmark.split_study                 # core models -> results/dev_split_study.md
    uv run python -m trial_benchmark.split_study --models all    # every honest model (slow: an hour or more)

Development trials only. The study asks two plain questions.

1. How much higher (or lower) is a model's score when the holdout is a random slice of the same
   trials than when it is the latest trials? (AUROC points and calibration gap.)
2. Does the model that wins on one kind of holdout still win on the other? ("Cross-over cost":
   the AUROC you give up by choosing on one split and being judged on the other.)

Three holdouts of the same size:

  chronological   train on the older pool, score the later trials (the real test's shape)
  random by trial the older and later trials shuffled together, a random slice held out
  random by drug  the same, but every lead drug stays wholly on one side

Each random holdout is drawn 20 times (seeds); a random-split score is the mean over seeds.

`run_study(train_pool, holdout, ...)` takes both frames as arguments, with every feature table
already joined (see `validate.load_development_with`) and a `lead_drug` column. The owner can later
call it with the real test trials as `holdout`; the module itself never opens held-out files and
`main` only builds the mock split from development data. Scoring goes through
`validate.predict_all` and `metrics`, the same path as everything else.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from trial_benchmark import baselines as bl
from trial_benchmark import metrics as mt
from trial_benchmark.models.linear import REGISTRATION_NAME
from trial_benchmark.models.registry import ModelSpec, all_specs
from trial_benchmark.validate import (
    ROOT,
    completion_year,
    drug_groups,
    load_development_with,
    predict_all,
    split_specs,
)

REPORT_PATH = ROOT / "results" / "dev_split_study.md"
CACHE_DIR = ROOT / "outputs" / "split_study_cache"
CUT_YEAR = 2019  # the later trials are those that completed in this year or after
N_SEEDS = 20
N_DRAWS = 1000  # bootstrap resamples (whole lead drugs at a time)
BASE_SEED = 20261112
CHRONO, BY_TRIAL, BY_DRUG = "chronological", "random by trial", "random by drug"
REGIMES = (CHRONO, BY_TRIAL, BY_DRUG)

FULL_LADDER = "ladder: registration + endpoint + history + genetics + words"
CORE_MODELS = (
    REGISTRATION_NAME,
    FULL_LADDER,
    "memory: lead drug",
    "text: drug names only",
    "lookup: phase",
)
# The models `--models all` runs, fixed by name: the 41 behind the committed results. A family
# added to the registry later does not change which model wins or the rank agreement here. Ask
# for a new model by naming it in `run_study(model_names=...)`.
ALL_MODELS = (
    "biology: ladder, all four tables + science",
    "biology: registration + DepMap",
    "biology: registration + all sources",
    "biology: registration + annotations",
    "biology: registration + chemistry",
    "biology: registration + expression",
    "biology: registration + networks",
    "boosted trees: registration",
    "boosted trees: registration, tuned",
    "embeddings: biomedical vector alone",
    "embeddings: biomedical vector plus registration",
    "embeddings: biomedical vector plus registration plus words",
    "embeddings: general vector alone",
    "embeddings: general vector plus registration",
    "embeddings: general vector plus registration plus words",
    "ladder: boosted trees, endpoint + history + genetics",
    "ladder: registration + endpoint",
    "ladder: registration + endpoint + history + genetics + words",
    "ladder: registration + genetics",
    "ladder: registration + history",
    "ladder: registration + words",
    "logistic: last 4 years only",
    "logistic: last 8 years only",
    "logistic: recency-weighted",
    "logistic: registration",
    "logistic: registration + interactions",
    "logistic: registration + pooled sponsor",
    "logistic: registration, tuned penalty",
    "logistic: separate by phase",
    "logistic: separate for company and other sponsors",
    "logistic: trend-shifted intercept",
    "lookup: constant",
    "lookup: phase",
    "lookup: phase x sponsor class",
    "memory: lead drug",
    "memory: lead sponsor",
    "neural network: registration",
    "random forest: registration",
    "text: drug names only",
    "text: words",
    "text: words without drug names",
)


# ---------------------------------------------------------------- which models


def choose_models(names: tuple[str, ...] | None) -> list[ModelSpec]:
    """The named models; `None` means `ALL_MODELS`. A name that is unknown or lacks a table is refused."""
    wanted = set(ALL_MODELS if names is None else names)
    usable, missing = split_specs(all_specs())
    absent = sorted(wanted & set(missing))
    if absent:
        raise ValueError(f"Missing feature tables for: {absent}")
    unknown = wanted - {s.name for s in all_specs()}
    if unknown:
        raise ValueError(f"No such model: {sorted(unknown)}")
    return [s for s in usable if s.name in wanted]


# ---------------------------------------------------------------- the three holdouts


def chronological_split(
    frame: pl.DataFrame, cut_year: int = CUT_YEAR
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(older pool, later holdout): the pool completed before `cut_year`, the holdout in or after it."""
    years = completion_year(frame)
    return frame.filter(years < cut_year), frame.filter(years >= cut_year)


def trial_holdout(n_total: int, n_hold: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted row positions of `n_hold` trials drawn at random from `n_total`."""
    return np.sort(rng.permutation(n_total)[:n_hold])


def drug_holdout(groups: np.ndarray, n_hold: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted row positions of a holdout of whole drugs, as close to `n_hold` trials as it gets.

    Drugs are taken in random order; one that would overshoot is skipped. A trial with no known
    drug (None or NaN) counts as a drug of its own. The holdout can fall a
    few trials short of `n_hold` when no remaining drug fits, never above it.
    """
    inverse = mt.cluster_codes(np.asarray(groups, dtype=object), len(groups))
    sizes = np.bincount(inverse)  # a trial with no drug (None or NaN) is a drug of its own
    chosen = np.zeros(len(sizes), dtype=bool)
    total = 0
    for i in rng.permutation(len(sizes)):
        if total + sizes[i] <= n_hold:
            chosen[i] = True
            total += sizes[i]
        if total == n_hold:
            break
    return np.flatnonzero(chosen[inverse])


# ---------------------------------------------------------------- fitting


def _fit_predict(
    names: list[str], train: pl.DataFrame, test: pl.DataFrame
) -> dict[str, np.ndarray]:
    warnings.filterwarnings("ignore")
    wanted = set(names)
    return predict_all([s for s in all_specs() if s.name in wanted], train, test)


def fingerprint(combined: pl.DataFrame, train_rows: np.ndarray, test_rows: np.ndarray) -> str:
    """A hash of exactly which trials a fit trains on and which it predicts."""
    ids = combined["nct_id"].to_numpy()
    parts = (sorted(ids[train_rows]), "|", sorted(ids[test_rows]))
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:20]


def _run_fits(
    plan: dict, names: list[str], combined: pl.DataFrame, jobs: int, cache_file: Path | None
) -> dict:
    """Predictions for every (train rows, test rows) in `plan`, reusing a cache file if given.

    A cache entry is keyed by the model name and a hash of the training and scored trials' ids
    (`fingerprint`), never by the seed or the regime, so a different holdout can not pick up an
    earlier holdout's predictions. Entries for models not asked for now are kept.
    """
    cache = pickle.loads(cache_file.read_bytes()) if cache_file and cache_file.exists() else {}
    prints = {k: fingerprint(combined, *plan[k]) for k in plan}
    todo = [k for k in plan if any((prints[k], n) not in cache for n in names)]
    if todo:
        results = Parallel(n_jobs=jobs)(
            delayed(_fit_predict)(names, combined[plan[k][0]], combined[plan[k][1]]) for k in todo
        )
        for key, result in zip(todo, results, strict=True):
            for name, predictions in result.items():
                cache[(prints[key], name)] = predictions
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(pickle.dumps(cache))
    return {k: {n: cache[(prints[k], n)] for n in names} for k in plan}


# ---------------------------------------------------------------- numbers


@dataclass
class Run:
    """One holdout: outcomes, lead drugs and each model's predictions."""

    outcomes: np.ndarray
    groups: np.ndarray
    predictions: dict[str, np.ndarray]


def interval(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if not len(values):
        return (float("nan"), float("nan"))
    low, high = np.percentile(values, (2.5, 97.5))
    return (float(low), float(high))


def bootstrap_draws(runs: list[Run], names: list[str], rng: np.random.Generator) -> np.ndarray:
    """AUROC and calibration gap per draw: shape (2, n_models, N_DRAWS).

    Each draw picks one run (one seed) and resamples its trials a whole lead drug at a time
    (`metrics.cluster_members`: a trial with no lead drug is its own cluster), so the interval
    covers both seed-to-seed spread and sampling. Every model sees the same draws.

    Not `metrics.bootstrap(clusters=...)` on purpose: that returns an interval from one set of
    predictions, while this study needs raw draws, mixed over the seeds.
    """
    members = [
        mt.cluster_members(np.asarray(r.groups, dtype=object), len(r.outcomes)) for r in runs
    ]
    out = np.full((2, len(names), N_DRAWS), np.nan)
    for i in range(N_DRAWS):
        k = int(rng.integers(len(runs)))
        rows = members[k]
        index = np.concatenate([rows[j] for j in rng.integers(0, len(rows), len(rows))])
        y = runs[k].outcomes[index]
        for m, name in enumerate(names):
            p = runs[k].predictions[name][index]
            out[0, m, i], out[1, m, i] = mt.auroc(y, p), mt.calibration_gap(y, p)
    return out


def per_run(runs: list[Run], name: str, metric) -> np.ndarray:
    return np.array([metric(r.outcomes, r.predictions[name]) for r in runs])


def mean_over_runs(runs: list[Run], name: str, metric) -> float:
    return float(np.nanmean(per_run(runs, name, metric)))


@dataclass
class StudyResult:
    names: list[str]
    honest: dict[str, bool]
    n_pool: int
    n_holdout: int
    n_seeds: int
    auroc: dict[str, dict[str, float]] = field(default_factory=dict)  # regime -> model -> value
    # What is shown beside each score. Chronological: a 95% interval from resampling whole lead
    # drugs. Random regimes: the lowest and highest of the per-seed scores (the score is their mean).
    auroc_ci: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    gap: dict[str, dict[str, float]] = field(default_factory=dict)  # per 100 trials
    flattery: dict[str, dict[str, tuple[float, tuple[float, float]]]] = field(default_factory=dict)
    winners: dict[str, str] = field(default_factory=dict)  # regime -> best honest model
    crossover: dict[str, dict[str, float]] = field(default_factory=dict)  # chosen on -> judged on
    spearman: dict[str, float] = field(default_factory=dict)  # regime pair -> rank agreement
    composition: dict[str, dict[str, float]] = field(default_factory=dict)


def mean_or_nan(values: list[float]) -> float:
    kept = [v for v in values if not np.isnan(v)]
    return float(np.mean(kept)) if kept else float("nan")


def composition(runs: dict[str, list[Run]], years: dict[str, list[np.ndarray]], cut: int) -> dict:
    """Who is in each holdout: share completed before the cut, success rate, and how well the
    registration baseline ranks the older trials against the later ones within it."""
    out: dict[str, dict[str, float]] = {}
    for regime, regime_runs in runs.items():
        old_share, rate, old_auc, new_auc = [], [], [], []
        for run, year in zip(regime_runs, years[regime], strict=True):
            old = year < cut
            p = run.predictions[REGISTRATION_NAME]
            old_share.append(old.mean())
            rate.append(run.outcomes.mean())
            old_auc.append(
                mt.auroc(run.outcomes[old], p[old]) if len(set(run.outcomes[old])) == 2 else np.nan
            )
            new_auc.append(
                mt.auroc(run.outcomes[~old], p[~old])
                if len(set(run.outcomes[~old])) == 2
                else np.nan
            )
        out[regime] = {
            "share_older": float(np.mean(old_share)),
            "success_rate": float(np.mean(rate)),
            "baseline_auroc_older": float(mean_or_nan(old_auc)),
            "baseline_auroc_later": float(mean_or_nan(new_auc)),
        }
    return out


def build_plan(
    n_pool: int, n_hold: int, groups: np.ndarray, n_seeds: int
) -> tuple[dict, dict[str, list[np.ndarray]]]:
    """Every fit as (train rows, test rows), and the random holdouts by regime.

    The chronological fit trains on the first `n_pool` rows and scores the rest; each seed then
    gives a random-by-trial and a random-by-drug fit trained on everything but its holdout.
    """
    n_all = n_pool + n_hold
    plan: dict = {CHRONO: (np.arange(n_pool), np.arange(n_pool, n_all))}
    holds: dict[str, list[np.ndarray]] = {BY_TRIAL: [], BY_DRUG: []}
    for seed in range(n_seeds):
        drawn = {
            BY_TRIAL: trial_holdout(n_all, n_hold, np.random.default_rng(BASE_SEED + seed)),
            BY_DRUG: drug_holdout(groups, n_hold, np.random.default_rng(BASE_SEED + 1000 + seed)),
        }
        for regime, held in drawn.items():
            holds[regime].append(held)
            plan[(regime, seed)] = (np.setdiff1d(np.arange(n_all), held), held)
    return plan, holds


def run_study(
    train_pool: pl.DataFrame,
    holdout: pl.DataFrame,
    model_names: tuple[str, ...] | None = CORE_MODELS,
    n_seeds: int = N_SEEDS,
    cut_year: int = CUT_YEAR,
    jobs: int = 3,
    cache_file: Path | None = None,
) -> StudyResult:
    """Score the models on a chronological holdout and on random holdouts of the same size.

    `train_pool` is the older, labelled pool; `holdout` the later labelled trials. Both need every
    feature table the models read, plus `lead_drug`. Pass development data only unless the owner
    has opened the held-out set through the logged guard. `model_names=None` runs every honest
    ranked model (rank agreement is only meaningful over many models).
    """
    specs = choose_models(model_names)
    names = [s.name for s in specs]
    if REGISTRATION_NAME not in names:
        raise ValueError(f"The study needs {REGISTRATION_NAME!r} as its baseline.")
    candidates = [s.name for s in specs if s.honest]
    combined = pl.concat([train_pool, holdout], how="vertical_relaxed")
    n_pool, n_hold = train_pool.height, holdout.height
    outcomes = bl.training_outcomes(combined)
    groups = drug_groups(combined)
    years = completion_year(combined)

    plan, holds = build_plan(n_pool, n_hold, groups, n_seeds)
    fits = _run_fits(plan, names, combined, jobs, cache_file)

    def run_for(held: np.ndarray, predictions: dict[str, np.ndarray]) -> Run:
        return Run(outcomes[held], groups[held], predictions)

    runs = {CHRONO: [run_for(plan[CHRONO][1], fits[CHRONO])]}
    year_lists = {CHRONO: [years[plan[CHRONO][1]]]}
    for regime in (BY_TRIAL, BY_DRUG):
        runs[regime] = [run_for(h, fits[(regime, s)]) for s, h in enumerate(holds[regime])]
        year_lists[regime] = [years[h] for h in holds[regime]]

    result = StudyResult(names, {s.name: s.honest for s in specs}, n_pool, n_hold, n_seeds)
    draws = {}
    for i, regime in enumerate(REGIMES):
        draws[regime] = bootstrap_draws(runs[regime], names, np.random.default_rng(BASE_SEED + i))
        result.auroc[regime] = {n: mean_over_runs(runs[regime], n, mt.auroc) for n in names}
        result.gap[regime] = {n: mean_over_runs(runs[regime], n, mt.calibration_gap) for n in names}
        if regime == CHRONO:
            result.auroc_ci[regime] = {
                n: interval(draws[regime][0, m]) for m, n in enumerate(names)
            }
        else:
            seed_scores = {n: per_run(runs[regime], n, mt.auroc) for n in names}
            result.auroc_ci[regime] = {
                n: (float(np.nanmin(v)), float(np.nanmax(v))) for n, v in seed_scores.items()
            }
    for regime in (BY_TRIAL, BY_DRUG):
        result.flattery[regime] = {
            n: (
                result.auroc[regime][n] - result.auroc[CHRONO][n],
                interval(draws[regime][0, m] - draws[CHRONO][0, m]),
            )
            for m, n in enumerate(names)
        }

    for regime in REGIMES:
        result.winners[regime] = max(candidates, key=lambda n: result.auroc[regime][n])
    for chosen_on in REGIMES:
        winner = result.winners[chosen_on]
        result.crossover[chosen_on] = {
            judged_on: result.auroc[judged_on][result.winners[judged_on]]
            - result.auroc[judged_on][winner]
            for judged_on in REGIMES
            if judged_on != chosen_on
        }
    for a, b in ((CHRONO, BY_TRIAL), (CHRONO, BY_DRUG), (BY_TRIAL, BY_DRUG)):
        result.spearman[f"{a} vs {b}"] = float(
            spearmanr(
                [result.auroc[a][n] for n in candidates], [result.auroc[b][n] for n in candidates]
            ).statistic
        )
    result.composition = composition(runs, year_lists, cut_year)
    return result


# ---------------------------------------------------------------- the report


def short(name: str) -> str:
    return "full ladder (all four extra tables + words)" if name == FULL_LADDER else name


def excludes_zero(interval_: tuple[float, float]) -> bool:
    return interval_[0] > 0 or interval_[1] < 0


def write_report(result: StudyResult, cut_year: int = CUT_YEAR, path: Path = REPORT_PATH) -> str:
    """Write the one-table summary with a few plain bullets, and return its text."""
    shown = [n for n in CORE_MODELS if n in result.names]
    rows = [
        (
            "| Model | Later trials | Random by trial | Random by drug | Random by trial minus later | "
            "Random by drug minus later | Gap per 100: later / by trial / by drug |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for n in shown:
        cells = []
        for regime in REGIMES:
            low, high = result.auroc_ci[regime][n]
            cells.append(f"{result.auroc[regime][n]:.3f} ({low:.3f} to {high:.3f})")
        for regime in (BY_TRIAL, BY_DRUG):
            gain, (low, high) = result.flattery[regime][n]
            cells.append(f"{gain:+.3f} ({low:+.3f} to {high:+.3f})")
        gaps = " / ".join(f"{result.gap[r][n]:+.1f}" for r in REGIMES)
        rows.append(f"| {short(n)} | {' | '.join(cells)} | {gaps} |")

    base = REGISTRATION_NAME
    comp = result.composition
    spear = result.spearman[f"{CHRONO} vs {BY_TRIAL}"]
    n_models = len(result.names)
    higher = [n for n in result.names if result.flattery[BY_TRIAL][n][0] > 0]
    clear = [n for n in higher if excludes_zero(result.flattery[BY_TRIAL][n][1])]
    memory = [n for n in ("memory: lead drug", "text: drug names only") if n in result.names]
    memory_text = (
        "; ".join(
            f"{n}: {result.flattery[BY_TRIAL][n][0]:+.3f} by trial, "
            f"{result.flattery[BY_DRUG][n][0]:+.3f} by drug"
            for n in memory
        )
        if memory
        else "no drug-memory model was run"
    )
    cross = result.crossover
    worst = max(v for d in cross.values() for v in d.values())
    same = len(set(result.winners.values())) == 1
    winners = (
        f"The same model wins under all three splits ({result.winners[CHRONO]})."
        if same
        else "The winner changes with the split: "
        + "; ".join(f"{r}: {w}" for r, w in result.winners.items())
        + "."
    )
    bullets = [
        (
            f"- **Scores are higher under random-by-trial.** {len(higher)} of {n_models} models score higher "
            f"than on the later trials; for {len(clear)} of them the interval of the difference excludes zero. "
            f"The baseline scores {result.auroc[CHRONO][base]:.3f} on the {result.n_holdout} later trials and "
            f"{result.auroc[BY_TRIAL][base]:.3f} / {result.auroc[BY_DRUG][base]:.3f} on random holdouts of the "
            f"same size (by trial / by drug). Its calibration gap goes from {result.gap[CHRONO][base]:+.1f} to "
            f"{result.gap[BY_TRIAL][base]:+.1f} / {result.gap[BY_DRUG][base]:+.1f} per 100 trials."
        ),
        (
            "- **Drug-name memory is helped by random-by-trial and hurt by random-by-drug** "
            f"(difference from the later-trials score): {memory_text}. A random-by-drug holdout keeps every "
            "lead drug on one side, but a combination trial can still share a non-lead drug across the two sides."
        ),
        (
            f"- **Random holdouts are older and easier.** {100 * comp[BY_TRIAL]['share_older']:.0f}% of a random-by-trial "
            f"holdout finished before {cut_year}, against 0% of the later trials; the baseline ranks those older trials "
            f"at {comp[BY_TRIAL]['baseline_auroc_older']:.3f} but the later ones in the same holdout at "
            f"{comp[BY_TRIAL]['baseline_auroc_later']:.3f}. The overall success rate is {100 * comp[BY_TRIAL]['success_rate']:.0f} "
            f"versus {100 * comp[CHRONO]['success_rate']:.0f} per 100 trials."
        ),
        (
            f"- **Choosing on one split and being judged on the other.** {winners} Rank agreement between the "
            f"later-trials and random-by-trial splits is {spear:.2f} (Spearman; 1 means the same order; over {len([n for n in result.names if result.honest[n]])} honest "
            f"models). The most AUROC lost by picking on one split and being judged on another is {worst:.3f}."
        ),
    ]
    text = "\n".join(
        [
            "# Does a random test split change a model's score? (development data only)",
            "",
            (
                f"Pool: {result.n_pool} trials finished before {cut_year}. Later holdout: {result.n_holdout} trials "
                f"finished in or after {cut_year}. Random holdouts have the same size ({result.n_seeds} seeds each). "
                "Later trials: AUROC with a 95% interval from resampling whole lead drugs. Random splits: the mean "
                "AUROC over seeds, with the lowest to highest seed in brackets. Each difference is the random score "
                "minus the later-trials score; its 95% interval comes from 1,000 draws, each resampling whole "
                "lead drugs in one randomly chosen seed's holdout and, separately, in the later trials, so it covers seed-to-seed "
                "spread and sampling together. A gap above 0 means the model is too optimistic. No held-out test "
                "trial is used."
            ),
            "",
            *rows,
            "",
            *bullets,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", choices=("core", "all"), default="core")
    parser.add_argument("--cut-year", type=int, default=CUT_YEAR)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    names = CORE_MODELS if args.models == "core" else None
    tables = {t for s in choose_models(names) for t in s.tables}
    dev = load_development_with(tables)  # development trials only
    pool, later = chronological_split(dev, args.cut_year)
    print(f"pool {pool.height}, later {later.height}", file=sys.stderr)
    result = run_study(
        pool,
        later,
        names,
        args.seeds,
        args.cut_year,
        args.jobs,
        CACHE_DIR / f"fits_{args.models}_{args.cut_year}.pkl",
    )
    print(write_report(result, args.cut_year))


if __name__ == "__main__":
    main()
