"""Compare every model on the development trials only. Nothing here reads a held-out label.

    uv run python -m elman_exercise.validate                  # all models -> results/dev_models.md
    uv run python -m elman_exercise.validate --only trees     # one family or model name; prints only

Two ways of scoring, because they answer different questions.

Time-ordered (the main check). Three rounds, each training on everything that
completed before a year and scoring the next two years: train before 2015 and
score 2015-2016, train before 2017 and score 2017-2018, train before 2019 and
score 2019-2020. That is what the real test is: learn from the past, predict the
future.

Grouped cross-validation (second column). Ten folds in which a drug never sits on
both sides: every trial of a drug is in the same fold. The drug is the trial's
lead drug (genetics.parquet), or the trial itself when that is unknown. This
removes drug memory: no model can recognise a validation drug. But the split
ignores time, so a model may train on trials that finished AFTER the ones it
predicts, which a forecaster cannot do. Read it beside the time-ordered check,
not instead of it.

Seen and unseen drugs. In the time-ordered rounds, the validation trials whose
lead drug appears in that round's training trials are "seen"; the rest are
"unseen". Trials with no known lead drug are in neither group.

How the numbers are made. The headline AUROC is the mean of the AUROCs within each
round (or fold). Each round has its own trained model, whose scores are on their
own scale, so pooling the rounds before ranking would let a shift of scale between
rounds change the result. Brier score and calibration gap are averages over trials,
so they are pooled. Intervals are bootstrap intervals that resample lead drugs (all of a
drug's trials together) inside each round, and every difference from "logistic: registration"
is taken on the same resamples. Trials that share a sponsor are still treated as
independent, so the intervals are somewhat too narrow. With a few thousand validation trials, an AUROC difference
under about 0.01 is a tie.

`predict_all` is the one place models meet validation trials: it fits on a training
frame and predicts a frame whose label columns have been removed first. The final
test run can use it on the real split.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from elman_exercise import baselines as bl
from elman_exercise import dev_summary, metrics
from elman_exercise.dataset import LABEL_COLUMNS
from elman_exercise.models.linear import REGISTRATION_NAME
from elman_exercise.models.registry import ModelSpec, all_specs

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "results" / "dev_models.md"
ROUNDS = (
    (2015, 2016),
    (2017, 2018),
    (2019, 2020),
)  # (first, last) year scored; train is before first
FOLDS = 10
KEY = ["nct_id"]
DRUG_TABLE = "genetics"  # holds lead_drug; optional: without it every trial is its own drug


# ---------------------------------------------------------------- data


def table_path(name: str) -> Path:
    return INTERIM / f"{name}.parquet"


def split_specs(specs: list[ModelSpec]) -> tuple[list[ModelSpec], dict[str, list[str]]]:
    """Specs whose tables all exist, and for each other spec the tables it is missing."""
    usable, missing = [], {}
    for spec in specs:
        absent = [t for t in (*spec.tables, *spec.own_tables) if not table_path(t).exists()]
        if absent:
            missing[spec.name] = absent
        else:
            usable.append(spec)
    return usable, missing


def join_tables(frame: pl.DataFrame, tables: set[str]) -> pl.DataFrame:
    """Left-join each feature table onto `frame` by nct_id, without its `split` column."""
    for name in sorted(tables):
        extra = pl.read_parquet(table_path(name)).drop("split", strict=False)
        clash = (set(extra.columns) & set(frame.columns)) - set(KEY)
        if clash:
            raise ValueError(
                f"Table {name!r} repeats columns already in the frame: {sorted(clash)}"
            )
        frame = frame.join(extra, on=KEY, how="left", validate="1:1")
    return frame


def load_development_with(tables: set[str]) -> pl.DataFrame:
    """The development trials (checked to be development only) with the named tables joined.

    The lead-drug table is added when it exists. Without it `lead_drug` is empty, so the
    grouped folds treat every trial as its own drug and the seen/unseen table is left out.
    """
    dev = dev_summary.load_dev()
    if table_path(DRUG_TABLE).exists():
        return join_tables(dev, tables | {DRUG_TABLE})
    return join_tables(dev, tables).with_columns(pl.lit(None, pl.String).alias("lead_drug"))


def completion_year(frame: pl.DataFrame) -> np.ndarray:
    return frame["primary_completion_date"].dt.year().to_numpy()


def drug_groups(frame: pl.DataFrame) -> np.ndarray:
    """The lead drug of each trial, or the trial's own id when the drug is unknown."""
    return frame.select(pl.coalesce("lead_drug", "nct_id"))[:, 0].to_numpy()


def lead_drug_clusters(frame: pl.DataFrame) -> np.ndarray:
    """Bootstrap cluster of each trial: its lead drug, or its own id when it has none.

    Uses the frame's `lead_drug` column when there is one, else the lead-drug table (`DRUG_TABLE`)
    joined by trial id, else (or with no trial id) every trial is its own cluster. That table holds features known at
    registration, so it is safe to read for any split.
    """
    if "lead_drug" not in frame.columns and KEY[0] not in frame.columns:
        return np.array([f"trial {i}" for i in range(frame.height)], dtype=object)
    if "lead_drug" not in frame.columns and table_path(DRUG_TABLE).exists():
        drugs = pl.read_parquet(table_path(DRUG_TABLE), columns=[*KEY, "lead_drug"])
        frame = frame.select(KEY).join(drugs, on=KEY, how="left", validate="m:1")
    elif "lead_drug" not in frame.columns:
        frame = frame.select(KEY).with_columns(pl.lit(None, pl.String).alias("lead_drug"))
    return drug_groups(frame)


# ---------------------------------------------------------------- splits


def time_round(frame: pl.DataFrame, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
    """Row positions to train on (completed before `first`) and to score (`first` to `last`)."""
    years = completion_year(frame)
    return np.flatnonzero(years < first), np.flatnonzero((years >= first) & (years <= last))


def grouped_folds(groups: np.ndarray, n_folds: int = FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train rows, test rows) per fold; every group is wholly on one side."""
    splitter = GroupKFold(n_splits=n_folds)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


# ---------------------------------------------------------------- the AUROC averaged within rounds


@dataclass(frozen=True)
class MeanAuroc:
    """The mean of the AUROCs within each round (or fold), with a bootstrap interval."""

    value: float
    interval: metrics.Interval
    per_group: tuple[float, ...]  # the AUROC within each round, in order


@dataclass(frozen=True)
class AurocDifference:
    value: float
    interval: metrics.Interval


def percentile_interval(values: np.ndarray) -> metrics.Interval:
    """Percentile interval of the non-NaN values; NaN if there are none."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return metrics.Interval(float("nan"), float("nan"))
    low, high = np.percentile(values, metrics.INTERVAL_PERCENTILES)
    return metrics.Interval(float(low), float(high))


def mean_auroc_bootstrap(
    outcomes: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    pairs: list[tuple[str, str]],
    n_resamples: int = metrics.N_RESAMPLES,
    seed: int = metrics.BOOTSTRAP_SEED,
    clusters: np.ndarray | None = None,
) -> tuple[dict[str, MeanAuroc], dict[str, AurocDifference]]:
    """Mean within-group AUROC per model, and its difference for each (better, baseline) pair.

    Each resample draws trials with replacement inside every group, and every model and
    pair is scored on the same draw. A resample with one class only in some group has no
    AUROC and is left out of that interval.

    With `clusters` (one label per trial, for example the lead drug), each resample instead draws
    whole clusters, as many as there are, with replacement, ONCE for all groups; a group then
    scores the trials of the drawn clusters that fall in it, as often as each was drawn. A
    cluster with trials in two rounds is therefore in or out of both. A trial with a missing
    label is its own cluster.
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    rows_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    codes = None
    if clusters is not None:
        codes = metrics.cluster_codes(np.asarray(clusters, dtype=object), len(outcomes))
    n_clusters = 0 if codes is None else int(codes.max()) + 1
    names = list(predictions)
    rng = np.random.default_rng(seed)
    draws = np.empty((len(names), n_resamples, len(rows_by_group)))
    for i in range(n_resamples):
        if codes is not None:
            # One draw of clusters for the whole resample: a drug with trials in two rounds
            # is in or out of both. Each round then takes its own share of the drawn trials.
            times = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        for j, rows in enumerate(rows_by_group):
            if codes is None:
                chosen = rows[rng.integers(0, len(rows), size=len(rows))]
            else:
                chosen = np.repeat(rows, times[codes[rows]])
            for k, name in enumerate(names):
                draws[k, i, j] = metrics.auroc(outcomes[chosen], predictions[name][chosen])
    resampled = dict(zip(names, draws.mean(axis=2), strict=True))

    means = {}
    for name in names:
        per_group = tuple(metrics.auroc(outcomes[r], predictions[name][r]) for r in rows_by_group)
        means[name] = MeanAuroc(
            float(np.mean(per_group)), percentile_interval(resampled[name]), per_group
        )
    differences = {
        better: AurocDifference(
            means[better].value - means[baseline].value,
            percentile_interval(resampled[better] - resampled[baseline]),
        )
        for better, baseline in pairs
    }
    return means, differences


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Evaluation:
    outcomes: np.ndarray
    predictions: dict[str, np.ndarray]
    scores: dict[str, metrics.Scores]  # pooled over trials: Brier score, calibration gap
    aurocs: dict[str, MeanAuroc]  # averaged within rounds (or folds)
    differences: dict[str, AurocDifference]  # by model name, against the registration logistic


def predict_all(
    specs: list[ModelSpec], train: pl.DataFrame, evaluate_frame: pl.DataFrame
) -> dict[str, np.ndarray]:
    """Fit each spec on `train` and predict `evaluate_frame` with its label columns removed.

    A model that returns anything other than one finite number per trial is refused by name.
    """
    blind = evaluate_frame.drop([c for c in LABEL_COLUMNS if c in evaluate_frame.columns])
    predictions = {}
    for spec in specs:
        p = np.asarray(spec.fit(train).predict(blind), dtype=float)
        if p.shape != (blind.height,) or not np.all(np.isfinite(p)):
            raise ValueError(f"Model {spec.name!r} gave missing or non-finite predictions.")
        predictions[spec.name] = p
    return predictions


def summarise(
    outcomes: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    baseline: str = REGISTRATION_NAME,
    clusters: np.ndarray | None = None,
) -> Evaluation:
    """Pooled Brier and calibration, mean within-group AUROC, and differences from `baseline`.

    `groups` says which round (or fold) each trial belongs to. `clusters` (one label per trial,
    such as the lead drug) makes every interval resample whole clusters; see
    `mean_auroc_bootstrap` and `metrics.bootstrap`. Without it trials are resampled singly.
    """
    pairs = (
        [(name, baseline) for name in predictions if name != baseline]
        if baseline in predictions
        else []
    )
    # `metrics.bootstrap` wants a majority class for the accuracy it holds; the report shows none.
    scores, _ = metrics.bootstrap(
        outcomes, predictions, majority_class=bool(np.mean(outcomes) > 0.5), clusters=clusters
    )
    aurocs, differences = mean_auroc_bootstrap(
        outcomes, predictions, groups, pairs, clusters=clusters
    )
    return Evaluation(outcomes, predictions, scores, aurocs, differences)


@dataclass(frozen=True)
class TimeOrdered:
    outcomes: np.ndarray
    predictions: dict[str, np.ndarray]  # pooled over the rounds
    round_of: np.ndarray  # 0, 1, 2 for each pooled trial
    seen_drug: np.ndarray  # 1.0 seen, 0.0 unseen, NaN lead drug unknown
    drug: np.ndarray | None = None  # lead drug of each pooled trial (its own id if unknown)


def run_time_ordered(specs: list[ModelSpec], frame: pl.DataFrame) -> TimeOrdered:
    outcomes, predictions, rounds, seen = [], {spec.name: [] for spec in specs}, [], []
    drugs = []
    for number, (first, last) in enumerate(ROUNDS):
        train_rows, score_rows = time_round(frame, first, last)
        train, scored = frame[train_rows], frame[score_rows]
        print(f"round {number + 1}: train {train.height}, score {scored.height}", file=sys.stderr)
        for name, p in predict_all(specs, train, scored).items():
            predictions[name].append(p)
        outcomes.append(bl.training_outcomes(scored))
        rounds.append(np.full(scored.height, number))
        drugs.append(drug_groups(scored))
        known = set(train["lead_drug"].drop_nulls())
        drug = scored["lead_drug"]
        seen.append(
            np.where(drug.is_null().to_numpy(), np.nan, drug.is_in(known).to_numpy().astype(float))
        )
    return TimeOrdered(
        np.concatenate(outcomes),
        {name: np.concatenate(parts) for name, parts in predictions.items()},
        np.concatenate(rounds),
        np.concatenate(seen),
        np.concatenate(drugs),
    )


@dataclass(frozen=True)
class Grouped:
    outcomes: np.ndarray
    predictions: dict[str, np.ndarray]  # out of fold
    fold_of: np.ndarray
    drug: np.ndarray | None = None  # lead drug of each trial (its own id if unknown)


def run_grouped(specs: list[ModelSpec], frame: pl.DataFrame) -> Grouped:
    """Out-of-fold predictions for every dev trial, with no drug on both sides of any fold."""
    predictions = {spec.name: np.empty(frame.height) for spec in specs}
    fold_of = np.empty(frame.height, dtype=int)
    for number, (train_rows, test_rows) in enumerate(grouped_folds(drug_groups(frame))):
        print(f"fold {number + 1}/{FOLDS}", file=sys.stderr)
        for name, p in predict_all(specs, frame[train_rows], frame[test_rows]).items():
            predictions[name][test_rows] = p
        fold_of[test_rows] = number
    return Grouped(bl.training_outcomes(frame), predictions, fold_of, drug_groups(frame))


# ---------------------------------------------------------------- report


def plus_minus(value: float, places: int = 3) -> str:
    return f"{value:+.{places}f}".replace("-", "−")


def interval_text(
    value: float, interval: metrics.Interval, signed: bool = False, places: int = 3
) -> str:
    def show(v: float) -> str:
        return plus_minus(v, places) if signed else f"{v:.{places}f}"

    return f"{show(value)} ({show(interval.low)} to {show(interval.high)})"


def auroc_cell(evaluation: Evaluation, name: str) -> str:
    mean = evaluation.aurocs[name]
    return interval_text(mean.value, mean.interval)


def difference_cell(name: str, evaluation: Evaluation) -> str:
    difference = evaluation.differences.get(name)
    if difference is None:
        return "(reference)" if name == REGISTRATION_NAME else ""
    return interval_text(difference.value, difference.interval, signed=True)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def by_auroc(evaluation: Evaluation) -> list[str]:
    return sorted(evaluation.aurocs, key=lambda name: -evaluation.aurocs[name].value)


def honest_split(specs: list[ModelSpec], order: list[str]) -> tuple[list[str], list[str]]:
    honest = {spec.name for spec in specs if spec.honest}
    return [n for n in order if n in honest], [n for n in order if n not in honest]


def not_ranked(specs: list[ModelSpec]) -> dict[str, list[str]]:
    """The names of the specs marked `ranked=False`, grouped by their stated reason.

    These models predict one constant for trials they are not defined on, so their score on all
    trials compares a constant with real predictions. They are left out of the all-trial
    rankings here and scored on the trials they cover in their own report.
    """
    groups: dict[str, list[str]] = {}
    for spec in specs:
        if not spec.ranked:
            groups.setdefault(spec.not_ranked_reason, []).append(spec.name)
    return groups


def under_headings(
    specs: list[ModelSpec],
    order: list[str],
    make_table: Callable[[list[str]], list[str]],
    level: str = "###",
) -> list[str]:
    """The table for the honest models, then the table for the deliberately leaky ones.

    Models that are not ranked are left out. `not_ranked_lines` names them and says why, once.
    """
    left_out = {name for names in not_ranked(specs).values() for name in names}
    honest, leaky = honest_split(specs, [n for n in order if n not in left_out])
    lines = []
    for title, names in (("Honest models", honest), ("Not honest (deliberately leaky)", leaky)):
        if names:
            lines += [f"{level} {title}", "", *make_table(names), ""]
    return lines


def not_ranked_lines(specs: list[ModelSpec]) -> list[str]:
    """One line per reason naming the models left out of every ranking below, and why."""
    lines = []
    for reason, names in not_ranked(specs).items():
        lines += [f"Not ranked in any table below ({reason}): " + ", ".join(names) + ".", ""]
    return lines


def headline_rows(evaluation: Evaluation, names: list[str]) -> list[list[str]]:
    return [
        [n, auroc_cell(evaluation, n), difference_cell(n, evaluation),
         f"{evaluation.scores[n].brier:.3f}", plus_minus(evaluation.scores[n].calibration_gap, 1)]
        for n in names
    ]  # fmt: skip


HEADLINE_HEADER = ["model", "mean AUROC (95% interval)", "AUROC minus registration",
                   "Brier", "calibration gap (pts)"]  # fmt: skip


def headline_table(specs, evaluation: Evaluation) -> list[str]:
    return under_headings(
        specs,
        by_auroc(evaluation),
        lambda names: table(HEADLINE_HEADER, headline_rows(evaluation, names)),
    )


def per_round_table(specs, evaluation: Evaluation) -> list[str]:
    header = ["model"] + [f"{a}-{b}" for a, b in ROUNDS]
    return under_headings(
        specs,
        by_auroc(evaluation),
        lambda names: table(
            header, [[n, *(f"{a:.3f}" for a in evaluation.aurocs[n].per_group)] for n in names]
        ),
        level="####",
    )


def seen_unseen_evaluations(result: TimeOrdered) -> dict[str, tuple[int, Evaluation]]:
    parts = {}
    for label, wanted in (("unseen", 0.0), ("seen", 1.0)):
        mask = result.seen_drug == wanted
        predictions = {name: p[mask] for name, p in result.predictions.items()}
        parts[label] = (
            int(mask.sum()),
            summarise(
                result.outcomes[mask],
                predictions,
                result.round_of[mask],
                clusters=None if result.drug is None else result.drug[mask],
            ),
        )
    return parts


def seen_unseen_table(specs, result: TimeOrdered) -> list[str]:
    parts = seen_unseen_evaluations(result)
    (n_unseen, unseen), (n_seen, seen) = parts["unseen"], parts["seen"]
    unknown = int(np.isnan(result.seen_drug).sum())
    counts = (
        f"Validation trials: {n_unseen} with an unseen drug, {n_seen} with a seen drug, "
        f"{unknown} with no known lead drug (left out)."
    )
    header = ["model", "unseen: AUROC (95% interval)", "unseen: minus registration",
              "seen: AUROC (95% interval)", "seen: minus registration"]  # fmt: skip

    def rows(names: list[str]) -> list[str]:
        return table(
            header,
            [
                [n, auroc_cell(unseen, n), difference_cell(n, unseen),
                 auroc_cell(seen, n), difference_cell(n, seen)]
                for n in names
            ],
        )  # fmt: skip

    return [counts, "", *under_headings(specs, by_auroc(unseen), rows)]


def legend(specs: list[ModelSpec]) -> list[str]:
    """What each model is, in the order of the registry, with whether it is honest."""
    lines = []
    for spec in specs:
        flags = ["honest" if spec.honest else "NOT honest"]
        if spec.pretrained:
            flags.append("uses a pretrained model")
        lines.append(f"- **{spec.name}** ({', '.join(flags)}): {spec.description}")
    if not any(spec.pretrained for spec in specs):
        lines += ["", "No model here uses a pretrained model."]
    return lines


READ_FIRST = (
    "**Read this first.** With about {n:,} validation trials, an AUROC difference under about "
    "0.01 is a tie. Intervals are bootstrap intervals that resample whole lead drugs; trials that "
    "share a sponsor are not independent, so they are somewhat too narrow. Every difference is against "
    '"{reference}", on the same resamples. Calibration gap: mean predicted minus observed '
    "success rate, in points; positive means too optimistic."
)
AUROC_CHOICE = (
    "The AUROC shown is the mean of the AUROCs within each round (or fold), and its interval "
    "resamples lead drugs inside each round. Each round's model scores on its own scale, so "
    "ranking all rounds together would reward or punish a model for how its scores shift "
    "between rounds rather than for how well it ranks trials. Brier score and calibration "
    "gap are averages over trials, so they are pooled."
)
GROUPED_NOTE = (
    "No drug is on both sides of a fold, so every validation drug is unseen and a model "
    "that memorises drugs gets nothing from remembering. But the split ignores time: a "
    "model may be trained on trials that finished after the ones it predicts, which a "
    "forecaster cannot do. That helps any model that learns from later trials of the same "
    "condition or sponsor, so scores here can run higher than in the time-ordered check. "
    'The drift models are not meaningful here: with no time cut, their "most recent years" '
    "are just the newest years of the training folds."
)
SEEN_NOTE = (
    "A drug is seen if it is the lead drug of at least one trial in that round's training trials."
)


def write_report(specs, missing, time_result, time_eval, grouped_eval) -> str:
    n = len(time_result.outcomes)
    lines = [
        "# Models on the development trials",
        "",
        "Development trials only (completed before 2021). No test or late trial is used.",
        "",
        READ_FIRST.format(n=n, reference=REGISTRATION_NAME),
        "",
        AUROC_CHOICE,
        "",
        "## The models",
        "",
        *legend(specs),
        "",
        *not_ranked_lines(specs),
        f"## Time-ordered check ({n} validation trials, three rounds)",
        "",
        "Rounds: " + "; ".join(f"train before {a}, score {a}-{b}" for a, b in ROUNDS) + ".",
        "",
        *headline_table(specs, time_eval),
        "### AUROC within each round",
        "",
        *per_round_table(specs, time_eval),
        f"## Grouped cross-validation ({len(grouped_eval.outcomes)} dev trials, {FOLDS} folds)",
        "",
        GROUPED_NOTE,
        "",
        *headline_table(specs, grouped_eval),
        "## Seen versus unseen drugs (time-ordered rounds)",
        "",
        SEEN_NOTE,
        "",
        *seen_unseen_lines(specs, time_result),
        "",
        "## Skipped: table missing",
        "",
    ]
    if missing:
        lines += [
            f"- {name}: needs {', '.join(tables)}" for name, tables in sorted(missing.items())
        ]
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def seen_unseen_lines(specs, result: TimeOrdered) -> list[str]:
    if np.isnan(result.seen_drug).all():
        return ["Not available: the lead-drug table (genetics.parquet) is missing."]
    return seen_unseen_table(specs, result)


# ---------------------------------------------------------------- command line


def choose(specs: list[ModelSpec], only: list[str] | None) -> list[ModelSpec]:
    """Specs matching a family or a model name. The reference model is always kept.

    With no `only`, every spec given is kept (`main` leaves the opt-in families out of it).
    """
    if not only:
        return specs
    wanted = [s for s in specs if s.family in only or s.name in only]
    if not wanted:
        raise SystemExit(
            f"No family or model named {only}. Families: {sorted({s.family for s in specs})}"
        )
    reference = [s for s in specs if s.name == REGISTRATION_NAME and s not in wanted]
    return reference + wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", help="family or model name(s); prints, writes no file")
    args = parser.parse_args()

    usable, missing = split_specs(choose(all_specs(include_opt_in=bool(args.only)), args.only))
    frame = load_development_with({t for spec in usable for t in spec.tables})
    time_result = run_time_ordered(usable, frame)
    time_eval = summarise(
        time_result.outcomes,
        time_result.predictions,
        time_result.round_of,
        clusters=time_result.drug,
    )
    grouped_result = run_grouped(usable, frame)
    grouped_eval = summarise(
        grouped_result.outcomes,
        grouped_result.predictions,
        grouped_result.fold_of,
        clusters=grouped_result.drug,
    )
    report = write_report(usable, missing, time_result, time_eval, grouped_eval)
    if args.only:
        print(report)
    else:
        REPORT_PATH.write_text(report)
        print(f"wrote {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
