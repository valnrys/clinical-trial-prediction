"""Run the baseline ladder and write the results as markdown.

    uv run python -m elman_exercise.evaluate dev
    uv run python -m elman_exercise.evaluate test --note "why" [--extended]

`dev` uses data/interim/dev.parquet only and writes results/dev_baselines.md:

- A. time-forward validation: train on trials that finished before 2017, score
  trials that finished in 2017-2020. This mimics the real split (older trials
  train, newer ones test).
- B. five-fold cross-validation over all development trials, shuffled.
- C. A and B on the same trials: the cross-validated predictions for the 2017-2020
  trials, set beside the time-forward ones. B scores more trials than A, so only
  this comparison shows what a random split hides (over-optimism, not ranking).
- D. what the registration model leans on: time-forward AUROC with each design
  column dropped in turn.

`test` fits every rung on all development trials and scores the held-out test
trials. It cannot run until the `prereg` tag exists, because the labels come
only through `heldout.open_heldout`, which refuses until then and logs every
access. Until the tag exists, this module is exercised only on synthetic data.

Subsets slice the evaluated trials by a column. Some slicing columns are model
inputs (phase, sponsor class, modality); control kind (from the results section)
and enrollment (the actual figure, recorded after the trial) are not.
"""

from __future__ import annotations

import argparse
import subprocess
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from elman_exercise import baselines as bl
from elman_exercise import metrics as mt
from elman_exercise.dataset import LABEL_COLUMNS, ROOT, TEST_START
from elman_exercise.dev_summary import (
    ENROLLMENT_ORDER,
    check_is_development,
    enrollment_band,
    load_dev,
)
from elman_exercise.heldout import LOG_FILE, access_log, open_heldout, repo_root
from elman_exercise.validate import lead_drug_clusters

RESULTS_DIR = Path("results")
DEV_REPORT = RESULTS_DIR / "dev_baselines.md"
TEST_REPORT = RESULTS_DIR / "test_baselines.md"

VALIDATION_START = date(2017, 1, 1)  # train before this date; validate from it to TEST_START
FOLDS = 5
FOLD_SEED = 0
MIN_SUBSET = 30  # no AUROC for a subset with fewer trials than this

# (better, baseline): the three differences the pre-registered predictions are about.
PAIRS = (
    (bl.REGISTRATION, bl.PHASE_LOOKUP),
    (bl.LEAKY, bl.REGISTRATION),
    (bl.REGISTRATION, bl.NO_MODALITY),
)


# ---------------------------------------------------------------- evaluating the ladder


@dataclass
class Evaluation:
    """Every rung scored on one set of trials."""

    table: pl.DataFrame
    outcomes: np.ndarray
    predictions: dict[str, np.ndarray]
    majority_class: bool
    scores: dict[str, mt.Scores]
    differences: list[mt.Difference]


@contextmanager
def convergence_is_an_error() -> Iterator[None]:
    """A logistic fit that does not converge would be a quiet error in every number: stop instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        yield


def score_ladder(
    table: pl.DataFrame, predictions: dict[str, np.ndarray], majority_class: bool
) -> Evaluation:
    outcomes = bl.training_outcomes(table)
    scores, differences = mt.bootstrap(
        outcomes,
        predictions,
        majority_class,
        PAIRS,
        n_resamples=mt.N_RESAMPLES,
        clusters=lead_drug_clusters(table),
    )
    return Evaluation(table, outcomes, predictions, majority_class, scores, differences)


def evaluate_fitted(train: pl.DataFrame, evaluated: pl.DataFrame) -> Evaluation:
    """Fit every rung on `train`, and score each on `evaluated` (which must have labels)."""
    models = bl.fit_ladder(train)
    predictions = {name: model.predict_probability(evaluated) for name, model in models.items()}
    majority = mt.majority_class_of(bl.training_outcomes(train))
    return score_ladder(evaluated, predictions, majority)


def time_forward_split(dev: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Trials finished before 2017 to train on; 2017 to 2020 to validate on."""
    check_is_development(dev)
    finished = pl.col("primary_completion_date")
    return dev.filter(finished < VALIDATION_START), dev.filter(finished >= VALIDATION_START)


def evaluate_time_forward(dev: pl.DataFrame) -> Evaluation:
    train, validation = time_forward_split(dev)
    return evaluate_fitted(train, validation)


def evaluate_cross_validated(dev: pl.DataFrame) -> Evaluation:
    """Five stratified, shuffled folds; each trial is predicted by models that never saw it."""
    check_is_development(dev)
    outcomes = bl.training_outcomes(dev)
    predictions = {name: np.empty(dev.height) for name in bl.RUNGS}
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=FOLD_SEED)
    for train_rows, held_rows in folds.split(np.zeros(dev.height), outcomes):
        models = bl.fit_ladder(dev[train_rows])
        held = dev[held_rows]
        for name, model in models.items():
            predictions[name][held_rows] = model.predict_probability(held)
    # The majority class is the same in every fold's training data (success), so use the whole.
    return score_ladder(dev, predictions, mt.majority_class_of(outcomes))


# ---------------------------------------------------------------- comparisons on the dev set

FORWARD = "time-forward"
CROSS_VALIDATED = "cross-validated"
ALL_COLUMNS = "(all columns)"


@dataclass(frozen=True)
class SameTrials:
    """One rung scored on the 2017-2020 trials two ways, with the paired differences."""

    model: str
    forward: mt.Scores
    cross_validated: mt.Scores
    difference: mt.Difference  # cross-validated minus time-forward


def same_trials_comparison(forward: Evaluation, shuffled: Evaluation) -> list[SameTrials]:
    """Cross-validated predictions restricted to the time-forward validation trials.

    Both are scored on the same trials with the same resamples, so a difference is not
    a difference of cohort or training size. `forward.table` is the validation trials in
    the order they have in `shuffled.table`.
    """
    mask = (shuffled.table["primary_completion_date"] >= VALIDATION_START).to_numpy()
    if shuffled.table.filter(mask)["nct_id"].to_list() != forward.table["nct_id"].to_list():
        raise ValueError("The validation trials are not the same in both evaluations.")
    rows = []
    for name in bl.RUNGS:
        predictions = {
            FORWARD: forward.predictions[name],
            CROSS_VALIDATED: shuffled.predictions[name][mask],
        }
        scores, (difference,) = mt.bootstrap(
            forward.outcomes,
            predictions,
            forward.majority_class,
            [(CROSS_VALIDATED, FORWARD)],
            n_resamples=mt.N_RESAMPLES,
            clusters=lead_drug_clusters(forward.table),
        )
        rows.append(SameTrials(name, scores[FORWARD], scores[CROSS_VALIDATED], difference))
    return rows


@dataclass(frozen=True)
class DropOne:
    column: str
    auroc_without: float
    loss: float  # AUROC with every column minus AUROC without this one
    interval: mt.Interval


def drop_one_column(train: pl.DataFrame, validation: pl.DataFrame) -> list[DropOne]:
    """Time-forward AUROC of the registration model without each design column, biggest loss first."""
    outcomes = bl.training_outcomes(validation)
    predictions = {
        ALL_COLUMNS: bl.fit_logistic(train, bl.FEATURES[bl.REGISTRATION]).predict_probability(
            validation
        )
    }
    for column in bl.REGISTRATION_COLUMNS:
        features = bl.registration_features_without(column)
        predictions[column] = bl.fit_logistic(train, features).predict_probability(validation)
    pairs = [(ALL_COLUMNS, column) for column in bl.REGISTRATION_COLUMNS]
    scores, differences = mt.bootstrap(
        outcomes,
        predictions,
        True,
        pairs,
        n_resamples=mt.N_RESAMPLES,
        clusters=lead_drug_clusters(validation),
    )
    rows = [
        DropOne(d.baseline, scores[d.baseline].auroc, d.difference, d.interval) for d in differences
    ]
    return sorted(rows, key=lambda row: -row.loss)


# ---------------------------------------------------------------- subsets


class Grouping(NamedTuple):
    name: str
    values: Callable[[pl.DataFrame], list[str | None]]
    shown: tuple[str, ...]  # the subsets reported, in order; other values are left out
    is_input: bool  # whether the registration model reads this column
    why_not_input: str = ""


def _column(name: str) -> Callable[[pl.DataFrame], list[str | None]]:
    return lambda table: table[name].cast(pl.String).to_list()


def _sponsor(table: pl.DataFrame) -> list[str | None]:
    return [None if s is None else ("industry" if s == "INDUSTRY" else "not industry")
            for s in table["sponsor_class"]]  # fmt: skip


CONTROL_KIND = Grouping(
    "control kind", _column("control_kind"), ("placebo", "active"), False,
    "it comes from the results section",
)  # fmt: skip
ENROLLMENT = Grouping(
    "enrollment (actual, post-hoc)",
    lambda table: [None if n is None else enrollment_band(n) for n in table["enrollment"]],
    ENROLLMENT_ORDER,
    False,
    "it is the actual figure, recorded after the trial",
)
MODALITY = Grouping(
    "modality",
    _column("modality"),
    ("small molecule", "biologic", "advanced", "mixed", "vaccine"),
    True,
)
PHASE = Grouping("phase", _column("phase"), ("phase2", "phase3"), True)
SPONSOR = Grouping("sponsor class", _sponsor, ("industry", "not industry"), True)

LEAD_GROUPINGS = (CONTROL_KIND, ENROLLMENT, MODALITY)
DEV_GROUPINGS = (*LEAD_GROUPINGS, PHASE, SPONSOR)


@dataclass(frozen=True)
class SubsetRow:
    group: str
    subset: str
    n: int
    observed_rate: float
    scores: mt.Scores | None  # None for fewer than MIN_SUBSET trials
    reason: str  # why there is no AUROC (or no scores at all), when there is none


def subset_rows(
    evaluation: Evaluation, model: str, groupings: Sequence[Grouping]
) -> list[SubsetRow]:
    """How one model does within each subset of the evaluated trials.

    Nothing is computed for a subset with fewer than MIN_SUBSET trials, and a subset with
    one class has no AUROC.
    """
    probabilities = evaluation.predictions[model]
    clusters = lead_drug_clusters(evaluation.table)
    rows = []
    for grouping in groupings:
        labels = np.array(grouping.values(evaluation.table), dtype=object)
        for subset in grouping.shown:
            chosen = labels == subset
            outcomes = evaluation.outcomes[chosen]
            n = len(outcomes)
            if n == 0:
                rows.append(SubsetRow(grouping.name, subset, 0, float("nan"), None, "no trials"))
                continue
            observed = float(outcomes.mean())
            if n < MIN_SUBSET:
                rows.append(
                    SubsetRow(grouping.name, subset, n, observed, None, f"n < {MIN_SUBSET}")
                )
                continue
            scores = mt.bootstrap(
                outcomes,
                {model: probabilities[chosen]},
                evaluation.majority_class,
                n_resamples=mt.N_RESAMPLES,
                clusters=clusters[chosen],
            )[0][model]
            reason = "one class only" if outcomes.all() or not outcomes.any() else ""
            rows.append(SubsetRow(grouping.name, subset, n, observed, scores, reason))
    return rows


def not_shown(evaluation: Evaluation, groupings: Sequence[Grouping]) -> list[str]:
    """For each grouping, the values of its column that are not in a subset row, with counts."""
    notes = []
    for grouping in groupings:
        values = grouping.values(evaluation.table)
        left_out: dict[str, int] = {}
        for value in values:
            if value not in grouping.shown:
                key = "missing" if value is None else value
                left_out[key] = left_out.get(key, 0) + 1
        if left_out:
            listed = ", ".join(f"{k} ({v})" for k, v in sorted(left_out.items()))
            notes.append(f"{grouping.name}: {listed}")
    return notes


# ---------------------------------------------------------------- markdown


def _rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _interval(low: float, high: float, digits: int = 3) -> str:
    return f"({low:.{digits}f} to {high:.{digits}f})"


def _auroc(value: float, interval: mt.Interval) -> str:
    return f"{value:.3f} {_interval(interval.low, interval.high)}"


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def _signed_gap(value: float, interval: mt.Interval) -> str:
    return f"{_signed(value)} {_interval(interval.low, interval.high, 1)}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def ladder_table(evaluation: Evaluation) -> str:
    rows = []
    for name, s in evaluation.scores.items():
        accuracy = f"{_rate(s.accuracy)} {_interval(s.accuracy_interval.low * 100, s.accuracy_interval.high * 100, 1)}"
        rows.append(
            [name, _rate(s.mean_predicted), _signed_gap(s.calibration_gap, s.gap_interval),
             _auroc(s.auroc, s.auroc_interval), f"{s.brier:.3f}", accuracy,
             _rate(s.majority_accuracy)]
        )  # fmt: skip
    header = ["Model", "Mean predicted", "Calibration gap (points)", "AUROC", "Brier",
              "Accuracy at 0.5", "Always-majority accuracy"]  # fmt: skip
    return _table(header, rows)


def differences_table(evaluation: Evaluation) -> str:
    rows = [
        [
            f"{d.better} minus {d.baseline}",
            f"{d.difference:+.3f} {_interval(d.interval.low, d.interval.high)}",
        ]
        for d in evaluation.differences
    ]
    return _table(["AUROC difference", "Estimate (95% interval)"], rows)


def subsets_table(rows: Sequence[SubsetRow]) -> str:
    cells = []
    for r in rows:
        if r.scores is None:
            auroc = gap = f"not computed ({r.reason})"
            if r.n == 0:
                gap = "-"
        else:
            auroc = (
                f"not computed ({r.reason})"
                if r.reason
                else _auroc(r.scores.auroc, r.scores.auroc_interval)
            )
            gap = _signed_gap(r.scores.calibration_gap, r.scores.gap_interval)
        observed = _rate(r.observed_rate) if r.n else "-"
        cells.append([r.group, r.subset, str(r.n), observed, auroc, gap])
    header = [
        "Grouping",
        "Subset",
        "n",
        "Observed success",
        "AUROC (95% interval)",
        "Calibration gap (points, 95% interval)",
    ]
    return _table(header, cells)


def _skipped(where: str, counts: Sequence[int]) -> str:
    return (
        f"Bootstrap resamples skipped for holding one class only, {where}: at most "
        f"{max(counts, default=0)} of {mt.N_RESAMPLES} in any row."
    )


def _skipped_note(evaluation: Evaluation, rows: Sequence[SubsetRow] = ()) -> str:
    """Skipped resamples, the headline tables and the subset table counted separately."""
    headline = [s.skipped_resamples for s in evaluation.scores.values()]
    headline += [d.skipped_resamples for d in evaluation.differences]
    note = _skipped("headline tables", headline)
    if rows:
        note += " " + _skipped(
            "subset table", [r.scores.skipped_resamples for r in rows if r.scores]
        )
    return note


BOOTSTRAP_CAVEAT = (
    "Intervals are 95% percentile intervals from 1,000 bootstrap resamples of lead drugs: each "
    "resample draws whole drugs, with all their trials (a trial with no known lead drug counts "
    "as its own). Trials that share a sponsor are still treated as independent, so the "
    "intervals are somewhat too narrow."
)
READING_THE_LADDER = (
    "Calibration gap is mean predicted minus observed success, in points (positive: too "
    "optimistic). Accuracy should be read against the always-majority column. The leaky and "
    "results-structure models are deliberately wrong: they read fields that exist only after "
    "the trial ended or are taken from posted results."
)


def _counts(evaluation: Evaluation) -> str:
    return (
        f"{evaluation.table.height:,} trials; observed success rate "
        f"{_rate(float(evaluation.outcomes.mean()))}."
    )


def _subset_caption(groupings: Sequence[Grouping]) -> str:
    inputs = [g.name for g in groupings if g.is_input]
    others = [f"{g.name} ({g.why_not_input})" for g in groupings if not g.is_input]
    parts = ["The registration model within subsets of these trials."]
    if inputs:
        parts.append(
            f"{', '.join(inputs)} {'is an input' if len(inputs) == 1 else 'are inputs'} to the "
            "model, so within-subset AUROC is lower "
            "partly because the model's strongest input is held fixed."
        )
    if others:
        parts.append(f"Not model inputs: {'; '.join(others)}.")
    return "_" + " ".join(parts) + "_"


CONTROL_KIND_NOTE = (
    "Control kinds other than placebo and active are not shown: 'no treatment' and 'unspecified' "
    "are too few to compare and are not the placebo-against-active contrast."
)


def ladder_section(
    evaluation: Evaluation,
    heading: str,
    groupings: Sequence[Grouping],
    footnote: str = "",
) -> str:
    """Rungs, paired differences and registration-model subsets for one evaluated set."""
    rows = subset_rows(evaluation, bl.REGISTRATION, groupings)
    parts = [
        heading,
        _counts(evaluation),
        (
            "_Every model on the same trials: how well it ranks trials (AUROC), how far off its "
            "average is, and how it compares with always saying the common answer._"
        ),
        ladder_table(evaluation),
        "_How much better one model ranks trials than another, using the same resamples._",
        differences_table(evaluation),
    ]
    if footnote:
        parts.append(footnote)
    if rows:
        parts += [_subset_caption(groupings), subsets_table(rows)]
        left_out = not_shown(evaluation, groupings)
        if left_out:
            note = "Trials in no subset row (values not shown): " + "; ".join(left_out) + "."
            if CONTROL_KIND in groupings:
                note += " " + CONTROL_KIND_NOTE
            parts.append(note)
    parts.append(_skipped_note(evaluation, rows))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- git and headers


def _git_line(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def provenance(root: Path) -> str:
    """The commit the code was at, and whether other files had uncommitted changes.

    The results folder and the held-out access log are left out of the check: both are
    written by a run, so every re-run would otherwise call itself dirty.
    """
    commit = _git_line(root, "rev-parse", "HEAD")
    dirty = _git_line(
        root,
        "status",
        "--porcelain",
        "--",
        ".",
        f":(exclude){RESULTS_DIR}",
        f":(exclude){LOG_FILE}",
    )
    state = "with uncommitted changes" if dirty else "clean"
    return f"Git commit `{commit}` ({state})."


# ---------------------------------------------------------------- the development report


def comparison_section(forward: Evaluation, shuffled: Evaluation) -> str:
    rows = same_trials_comparison(forward, shuffled)
    cells = [
        [
            r.model,
            f"{r.forward.auroc:.3f}",
            f"{r.cross_validated.auroc:.3f}",
            f"{r.difference.difference:+.3f} "
            + _interval(r.difference.interval.low, r.difference.interval.high),
            _signed(r.forward.calibration_gap),
            _signed(r.cross_validated.calibration_gap),
            _signed_gap(r.difference.gap_difference, r.difference.gap_difference_interval),
        ]
        for r in rows
    ]
    registration = next(r for r in rows if r.model == bl.REGISTRATION)
    return "\n\n".join(
        [
            "## C. Time-forward against cross-validated, on the same trials",
            (
                f"_Both columns score the same {forward.table.height:,} trials (finished "
                f"{VALIDATION_START.year} to {TEST_START.year - 1}). Time-forward models were "
                "trained on earlier trials only. Cross-validated models were trained on a random "
                "80% of all development trials, so they had seen trials from the same years. "
                "Differences are cross-validated minus time-forward, from shared bootstrap "
                "resamples. Section B scores more trials than A, so its numbers cannot be "
                "compared with A's._"
            ),
            _table(
                [
                    "Model",
                    "AUROC time-forward",
                    "AUROC cross-validated",
                    "AUROC difference (95% interval)",
                    "Gap time-forward (points)",
                    "Gap cross-validated (points)",
                    "Gap difference (points, 95% interval)",
                ],
                cells,
            ),
            (
                f"For the registration model, ranking is {registration.cross_validated.auroc:.3f} "
                f"cross-validated against {registration.forward.auroc:.3f} time-forward, but the "
                f"calibration gap is {_signed(registration.cross_validated.calibration_gap)} points "
                f"against {_signed(registration.forward.calibration_gap)}. A random split hides how "
                "over-optimistic the model is; it does not flatter its ranking."
            ),
        ]
    )


def drop_one_section(train: pl.DataFrame, validation: pl.DataFrame) -> str:
    rows = [
        [
            r.column,
            f"{r.auroc_without:.3f}",
            f"{r.loss:+.3f} {_interval(r.interval.low, r.interval.high)}",
        ]
        for r in drop_one_column(train, validation)
    ]
    return "\n\n".join(
        [
            "## D. What the registration model leans on",
            (
                "_Each row refits the registration model without that one column (trained before "
                f"{VALIDATION_START.year}, scored on the same trials as A) and shows how much "
                "AUROC it loses; a loss near zero means the other columns cover for it, not that "
                "the column says nothing, since related columns (phase and sponsor, modality and "
                "vaccine) can stand in for each other._"
            ),
            _table(["Column dropped", "AUROC without it", "AUROC lost (95% interval)"], rows),
        ]
    )


def dev_report(dev: pl.DataFrame, header_note: str) -> str:
    """The whole development-set markdown. `dev` must be development data only."""
    check_is_development(dev)
    train, validation = time_forward_split(dev)
    with convergence_is_an_error():
        forward = evaluate_time_forward(dev)
        shuffled = evaluate_cross_validated(dev)
        comparison = comparison_section(forward, shuffled)
        drop_one = drop_one_section(train, validation)
    last_validation_day = TEST_START - timedelta(days=1)
    constant_auroc = shuffled.scores[bl.CONSTANT].auroc
    parts = [
        "# Baselines on the development set",
        header_note,
        (
            f"Development trials: {dev.height:,} (primary completion before {TEST_START}). "
            "No test-set trial is read."
        ),
        (
            "_Plain predictors of the headline label (any treatment arm beats control), to show what "
            "the benchmark reads. Nothing here is tuned. Start year is deliberately not an input to "
            "any model._"
        ),
        READING_THE_LADDER,
        BOOTSTRAP_CAVEAT,
        ladder_section(
            forward,
            "## A. Time-forward validation\n\n"
            f"Trained on {train.height:,} trials finishing before {VALIDATION_START}; scored on "
            f"{validation.height:,} finishing from {VALIDATION_START} to {last_validation_day}.",
            DEV_GROUPINGS,
        ),
        ladder_section(
            shuffled,
            f"## B. Five-fold cross-validation\n\nAll development trials, {FOLDS} stratified "
            f"shuffled folds (seed {FOLD_SEED}); each trial is predicted by models that did not "
            "see it. Subsets are not repeated here.",
            (),
            footnote=(
                f"The constant model's AUROC is {constant_auroc:.4f} rather than 0.5 because each "
                "fold predicts a slightly different constant (its own training rate) and the "
                "out-of-fold predictions are pooled."
            ),
        ),
        comparison,
        drop_one,
        (
            "Logistic regression converged in every fit; a fit that failed to converge would have "
            "stopped the run."
        ),
    ]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------- the test evaluation


class IdsDisagree(RuntimeError):
    """The trial ids in the features, labels and committed trials.csv do not match."""


def _pairs(table: pl.DataFrame) -> set[tuple[str, str]]:
    return set(table.select("nct_id", "split").iter_rows())


def check_unique_ids(name: str, table: pl.DataFrame) -> None:
    """Stop unless every trial id appears once in `table`."""
    repeated = table.height - table["nct_id"].n_unique()
    if repeated:
        raise IdsDisagree(f"{name} lists {repeated} trial ids more than once.")


def check_ids_agree(
    name: str, table: pl.DataFrame, trials: pl.DataFrame, splits: Sequence[str]
) -> None:
    """Stop unless `table` holds exactly the (trial, split) pairs trials.csv lists for `splits`."""
    expected = _pairs(trials.filter(pl.col("split").is_in(list(splits))))
    found = _pairs(table)
    if found != expected or table.height != len(expected):
        raise IdsDisagree(
            f"{name} and trials.csv disagree: {len(found - expected)} trial/split pairs only in "
            f"{name}, {len(expected - found)} only in trials.csv, {table.height} rows "
            f"against {len(expected)} expected."
        )


def join_labels(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Attach the headline label to the test features: on trial and split, one row per trial."""
    unlabelled = features.drop([c for c in LABEL_COLUMNS if c in features.columns])
    joined = unlabelled.join(labels.select("nct_id", "split", bl.LABEL), on=["nct_id", "split"])
    if joined.height != features.height or joined["nct_id"].n_unique() != joined.height:
        raise IdsDisagree(
            f"Joining labels to features gave {joined.height} rows for {features.height} trials."
        )
    return joined


def _slices(splits: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    """The test sets to report: the primary one and, when late is included, late alone and both."""
    if "test" not in splits:
        raise ValueError("The primary test set (split 'test') must be included.")
    unknown = set(splits) - {"test", "late"}
    if unknown:
        raise ValueError(f"Unknown splits {sorted(unknown)}; choose from 'test' and 'late'.")
    sets = [("Primary test set (finished 2021 to 2023)", ("test",))]
    if "late" in splits:
        sets.append(("Late test set (finished 2024 to 2025)", ("late",)))
        sets.append(("Extended test set (finished 2021 to 2025)", ("test", "late")))
    return sets


def _access_section(root: Path) -> str:
    log = access_log(root)
    rows = [[entry["at"], f"`{entry['commit'][:12]}`", entry["note"]] for entry in log]
    return "\n\n".join(
        [
            (
                f"## Access to the held-out labels\n\nThe test labels have been opened {len(log)} "
                "time(s), counting this run. Every opening is listed, so a re-run cannot quietly "
                "replace an earlier result."
            ),
            _table(["Time (UTC)", "Commit", "Note"], rows),
        ]
    )


def evaluate_on_test(note: str, splits: Sequence[str] = ("test",), root: Path | None = None) -> str:
    """Fit on all development trials, score the held-out test trials, write and print the report.

    `splits` is ("test",) for the primary set, or ("test", "late") to add the late set on
    its own and the extended set. Labels come only through `open_heldout`, which refuses
    before the `prereg` tag exists and logs the access. Before any score is computed, the
    trial ids in the features, the labels and trials.csv must agree exactly, and be unique.
    """
    root = root or repo_root()
    sets = _slices(splits)
    code_state = provenance(root)  # before the labels are opened: opening writes to the log
    dev = pl.read_parquet(root / "data" / "interim" / "dev.parquet")
    check_is_development(dev)
    features = pl.read_parquet(root / "data" / "interim" / "test_features.parquet")
    trials = pl.read_csv(root / "trials.csv")
    # Features against trials.csv first: no label is opened for a mismatch we can already see.
    for name, table in (("trials.csv", trials), ("dev.parquet", dev), ("test_features", features)):
        check_unique_ids(name, table)
    check_ids_agree("dev.parquet", dev, trials, ["dev"])
    check_ids_agree("test_features.parquet", features, trials, ["test", "late"])
    labels = pl.read_parquet(open_heldout("test_labels.parquet", note, root=root))
    check_unique_ids("test_labels.parquet", labels)
    check_ids_agree("test_labels.parquet", labels, trials, ["test", "late"])

    labelled = join_labels(features, labels).filter(pl.col("split").is_in(list(splits)))
    with convergence_is_an_error():
        models = bl.fit_ladder(dev)
    majority = mt.majority_class_of(bl.training_outcomes(dev))
    scored = ", ".join(f"{labelled.filter(pl.col('split') == s).height:,} {s}" for s in splits)
    parts = [
        "# Baselines on the held-out test set",
        code_state,
        f"Fit on all {dev.height:,} development trials. Trials scored: {scored}.",
        f"Access note: {note}",
        _access_section(root),
        READING_THE_LADDER,
        BOOTSTRAP_CAVEAT,
    ]
    for heading, members in sets:
        chosen = labelled.filter(pl.col("split").is_in(list(members)))
        predictions = {name: model.predict_probability(chosen) for name, model in models.items()}
        evaluation = score_ladder(chosen, predictions, majority)
        parts.append(ladder_section(evaluation, f"## {heading}", LEAD_GROUPINGS))
    text = "\n\n".join(parts) + "\n"
    out = root / TEST_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    return text


# ---------------------------------------------------------------- entry point


def run_dev(root: Path = ROOT) -> str:
    dev = load_dev()
    header = f"{provenance(root)} Dataset rows: {dev.height:,} development trials."
    text = dev_report(dev, header)
    out = root / DEV_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "dev", help="time-forward and cross-validated results, development set only"
    )
    test = commands.add_parser("test", help="score the held-out test set (needs the prereg tag)")
    test.add_argument("--note", required=True, help="why the test labels are being opened")
    test.add_argument(
        "--extended", action="store_true", help="also report the late set, and 2021 to 2025"
    )
    args = parser.parse_args()
    if args.command == "dev":
        run_dev()
    else:
        evaluate_on_test(args.note, ("test", "late") if args.extended else ("test",))


if __name__ == "__main__":
    main()
