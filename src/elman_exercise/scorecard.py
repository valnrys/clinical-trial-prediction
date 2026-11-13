"""Score the seven pre-registered predictions against the test set.

    uv run python -m elman_exercise.scorecard rehearse
    uv run python -m elman_exercise.scorecard test --note "why"

`rehearse` runs the identical code on development data only. Dev trials that
finished 2017 to 2020 stand in for the test set, and the models are trained on dev
trials that finished before 2017. It writes results/rehearsal_scorecard.md, with a
banner saying so. It cannot reach the held-out labels: nothing on its path calls
`open_heldout`.

`test` is the real run, made once by the project lead. It trains on all development
trials, opens the test labels through `heldout.open_heldout` (which logs the access),
and writes results/test_scorecard.md. It refuses to run on a dirty working tree, so the
log records the code that made the numbers. `load_test_inputs` is the only function
that touches the test set. Everything that needs no labels is checked first, so a
failure cannot burn a logged opening: the trial ids, the columns, the genetics table,
the Open Targets and Orphanet files, and a dry fit of every model on the dev trials.

Every prediction is one row: the claim in the email's words, the predicted range, the
observed value with a 95% interval, and a verdict from one fixed rule (RULE below). A
claim with no stated range ("about 1.2 times", "a gap like 63% vs 58%") gets no verdict.
Each quantity is computed the way the matching development figure was, by calling the
same code (`evaluate`, `genetics_summary`, `failures_summary`, `adjusted`).

Two things need the test period's own records, and so only run in the real run:

- the labels (through `open_heldout`);
- the stop reasons of every finished test-period trial, for the recruitment-stop ratio
  (prediction 7). Being stopped for recruitment is an outcome. As in development, this
  is counted over all finished drug trials in the period, not only the labelled ones.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple

import polars as pl

from elman_exercise import baselines as bl
from elman_exercise import download
from elman_exercise import evaluate as ev
from elman_exercise import failures_summary as fs
from elman_exercise import metrics as mt
from elman_exercise.dataset import LATE_START, ROOT, TEST_START
from elman_exercise.dev_summary import check_is_development, wilson_interval
from elman_exercise.genetics import OUT_PATH as GENETICS_PATH
from elman_exercise.genetics_summary import ESTABLISHED_AFTER, md_table, newcombe_difference
from elman_exercise.heldout import HELDOUT_DIR, access_log, open_heldout, repo_root
from elman_exercise.rare_disease import UNMAPPED, rare_flags
from elman_exercise.validate import lead_drug_clusters

REHEARSAL_REPORT = Path("results") / "rehearsal_scorecard.md"
TEST_REPORT = Path("results") / "test_scorecard.md"
LABELS_FILE = "test_labels.parquet"

REHEARSAL_START = ev.VALIDATION_START  # 2017-01-01: first day of the stand-in test period
EXPECTED_TOP_TWO = ("phase", "sponsor_class")
LATER_ERA = "2021 or later"  # the era stratum for test-period trials
RECRUITMENT_STRATA = ["phase", "sponsor", "era"]  # as in failures_summary section 4.3
SMALL_TRIAL, LARGE_TRIAL = 100, 300  # enrollment cut points of prediction 5
DEV_RECRUITMENT_RATIO = "1.19 (1.00 to 1.40)"  # the training-set figure, results/dev_failures.md
GENETIC_COLUMNS = ["nct_id", "mapped", "supported_today", "lead_drug_earlier_trials"]
SCORECARD_COLUMNS = ("modality", "control_kind", "enrollment", "sponsor_class", "split")

RULE = (
    "**How each row is judged.** The observed value is compared, unrounded, with the "
    'predicted range (both ends count as inside, unless the range says "below"). *Inside '
    "the predicted range*: the value is in it. *Outside, but the interval overlaps the "
    "range*: the value is out, but the 95% interval reaches into the range. *Outside*: the "
    "value and the whole interval are out. Where the email gives no range the row is "
    "reported without a verdict. Prediction 2b is not judged by this rule; it has its own, "
    "stated under it."
)
INTERVAL_CAVEAT = (
    "Intervals: bootstrap over lead drugs for AUROCs and the over-prediction (as in "
    "results/dev_baselines.md); Wilson for a rate; Newcombe for a difference of rates; "
    "sponsor-resampled for the adjusted recruitment ratio (as in results/dev_failures.md). "
    "The Wilson and Newcombe intervals treat trials as independent, which they are not, so "
    "they are somewhat too narrow."
)

INSIDE = "inside the predicted range"
OVERLAPS = "outside, but the interval overlaps the range"
OUTSIDE = "outside"
NO_RANGE = "no range in the email"
NOT_COMPUTED = "not computed"


# ---------------------------------------------------------------- the verdict rule


class Range(NamedTuple):
    """A predicted range. None for an end means open on that side ("under 0.01").

    `high_included` False makes the upper end strict: the value must be below it.
    """

    low: float | None
    high: float | None
    high_included: bool = True


def _within(value: float, predicted: Range) -> bool:
    above = predicted.low is None or value >= predicted.low
    if predicted.high is None:
        return above
    return above and (
        value <= predicted.high if predicted.high_included else value < predicted.high
    )


def verdict(value: float, low: float, high: float, predicted: Range | None) -> str:
    """One of the three verdicts of RULE (or a note when there is no range or no value)."""
    if predicted is None:
        return NO_RANGE
    if math.isnan(value):
        return NOT_COMPUTED
    if _within(value, predicted):
        return INSIDE
    if math.isnan(low) or math.isnan(high):
        return OUTSIDE
    reaches_up = predicted.low is None or high >= predicted.low
    reaches_down = predicted.high is None or _within(
        low, Range(None, predicted.high, predicted.high_included)
    )
    return OVERLAPS if reaches_up and reaches_down else OUTSIDE


# ---------------------------------------------------------------- how a value is shown


class Style(NamedTuple):
    """How a row's numbers are printed: a format, a mark after each, a unit after the last."""

    spec: str
    mark: str = ""
    unit: str = ""

    def show(self, value: float) -> str:
        return "n/a" if math.isnan(value) else format(value, self.spec) + self.mark


PERCENT = Style(".1f", mark="%")
POINTS = Style("+.1f", unit=" points")
AUROC = Style(".3f")
AUROC_DIFFERENCE = Style("+.3f")
RATIO = Style(".2f")


def show_range(style: Style, predicted: Range | None) -> str:
    if predicted is None:
        return "no range stated"
    low = style.show(predicted.low) if predicted.low is not None else None
    high = style.show(predicted.high) if predicted.high is not None else None
    if low is None:
        word = "up to" if predicted.high_included else "below"
        return f"{word} {high}{style.unit}"
    if high is None:
        return f"{low} or more{style.unit}"
    return f"{low} to {high}{style.unit}"


# ---------------------------------------------------------------- rows


@dataclass(frozen=True)
class Row:
    number: str
    claim: str  # the prediction in the email's words, short; says which group is subtracted
    style: Style
    predicted: Range | None
    value: float
    low: float
    high: float
    note: str = ""

    @property
    def verdict(self) -> str:
        return verdict(self.value, self.low, self.high, self.predicted)


class Contrast(NamedTuple):
    """First group's success rate minus the second's, in points, with Newcombe's interval."""

    difference: float
    low: float
    high: float
    n_first: int
    n_second: int


def contrast(first: pl.DataFrame, second: pl.DataFrame) -> Contrast:
    """Success rate of `first` minus `second` in points, as `genetics_summary` computes it.

    Both tables must hold only trials with a known label: a missing label would count
    as a failure in the denominator.
    """
    if first[bl.LABEL].null_count() or second[bl.LABEL].null_count():
        raise ValueError("A contrast needs trials with a known label.")
    successes = [int(t[bl.LABEL].sum() or 0) for t in (first, second)]  # empty sums to None
    difference, low, high = newcombe_difference(
        successes[0], first.height, successes[1], second.height
    )
    return Contrast(difference * 100, low * 100, high * 100, first.height, second.height)


def _counts(c: Contrast, first: str, second: str) -> str:
    return f"{c.n_first:,} {first} against {c.n_second:,} {second}"


def _contrast_row(number, claim, predicted, c: Contrast, first: str, second: str, note="") -> Row:
    text = "; ".join(part for part in (_counts(c, first, second), note) if part)
    return Row(number, claim, POINTS, predicted, c.difference, c.low, c.high, text)


def _labelled(scored: pl.DataFrame) -> pl.DataFrame:
    return scored.filter(pl.col(bl.LABEL).is_not_null())


# ---------------------------------------------------------------- the inputs


class Sources(NamedTuple):
    """Where the genetics, the raw records and the rare-disease groups come from.

    The defaults are the real ones. A test passes small fixtures instead.
    """

    genetics: Path = GENETICS_PATH
    studies: Callable[[], Iterable[dict]] = download.iter_studies
    flags: Callable[[list[str]], pl.DataFrame] = rare_flags


REAL_SOURCES = Sources()


@dataclass
class Inputs:
    """Everything the scorecard needs, with no outcome-reading left to do.

    `train` and `scored` are labelled trials. `scored` also carries the genetic columns
    and `group` (the rare-disease group). `population` is every finished drug trial of the
    scored period, with stop reasons and groups, for the recruitment ratio.
    """

    train: pl.DataFrame
    scored: pl.DataFrame
    population: pl.DataFrame


def build_population(
    studies: Iterable[dict],
    flags: Callable[[list[str]], pl.DataFrame],
    first_day: date,
    end_day: date,
) -> pl.DataFrame:
    """Every finished drug trial that finished in the window, with stop reasons and groups."""
    population = fs.build_population(studies, first_day, end_day)
    found = flags(population["nct_id"].to_list())
    if found["nct_id"].is_duplicated().any():
        raise ValueError("The rare-disease flags list a trial more than once.")
    grouped = fs.add_groups(population, found.select("nct_id", "group"), later_era=LATER_ERA)
    return grouped.with_columns(pl.col("group").fill_null(UNMAPPED))


def check_genetics(genetics: pl.DataFrame, trials: pl.DataFrame) -> None:
    """Stop unless the genetics table has the columns used and a known `mapped` for every trial."""
    missing = set(GENETIC_COLUMNS) - set(genetics.columns)
    if missing:
        raise ValueError(f"The genetics table lacks columns {sorted(missing)}.")
    known = genetics.filter(pl.col("mapped").is_not_null())
    absent = set(trials["nct_id"]) - set(known["nct_id"])
    if absent:
        raise ValueError(f"{len(absent)} scored trials have no row in the genetics table.")


def assemble(
    train: pl.DataFrame, scored: pl.DataFrame, genetics: pl.DataFrame, population: pl.DataFrame
) -> Inputs:
    """Attach genetic features and rare-disease groups to the scored trials.

    The group comes from the population table, as `failures_summary.with_flags` does: a
    labelled trial that is not in the population counts as unmapped.
    """
    check_genetics(genetics, scored)
    with_genetics = scored.join(
        genetics.select(GENETIC_COLUMNS), on="nct_id", how="left", validate="1:1"
    )
    groups = population.select("nct_id", "group")
    full = with_genetics.join(groups, on="nct_id", how="left", validate="1:1")
    return Inputs(train, full.with_columns(pl.col("group").fill_null(UNMAPPED)), population)


# ---------------------------------------------------------------- the seven predictions


def prediction_1_and_2(evaluation: ev.Evaluation) -> list[Row]:
    registration = evaluation.scores[bl.REGISTRATION]
    rate = float(evaluation.outcomes.mean())
    low, high = wilson_interval(int(evaluation.outcomes.sum()), evaluation.table.height)
    return [
        Row(
            "1a",
            "Only 45 to 50% of test trials succeed (down from 58% in training)",
            PERCENT,
            Range(45, 50),
            rate * 100,
            low * 100,
            high * 100,
            f"{evaluation.table.height:,} trials",
        ),
        Row(
            "1b",
            "The registration model over-predicts by 5 to 9 per 100 "
            "(mean predicted minus observed)",
            POINTS,
            Range(5, 9),
            registration.calibration_gap,
            registration.gap_interval.low,
            registration.gap_interval.high,
            f"mean predicted {registration.mean_predicted * 100:.1f}%",
        ),
        Row(
            "2a",
            "Registration model AUROC is 0.66 to 0.72",
            AUROC,
            Range(0.66, 0.72),
            registration.auroc,
            registration.auroc_interval.low,
            registration.auroc_interval.high,
        ),
    ]


def _difference(evaluation: ev.Evaluation, better: str, baseline: str):
    return next(d for d in evaluation.differences if d.better == better and d.baseline == baseline)


def prediction_3_and_4_models(evaluation: ev.Evaluation) -> tuple[Row, Row]:
    leaky = _difference(evaluation, bl.LEAKY, bl.REGISTRATION)
    modality = _difference(evaluation, bl.REGISTRATION, bl.NO_MODALITY)
    note = "" if modality.difference >= 0 else "the model did no better with modality"
    return (
        Row(
            "3",
            "Leaked fields raise AUROC by 0.02 to 0.05 (leaky minus registration)",
            AUROC_DIFFERENCE,
            Range(0.02, 0.05),
            leaky.difference,
            leaky.interval.low,
            leaky.interval.high,
        ),
        Row(
            "4b",
            "Telling the model the drug type barely improves AUROC, by less than 0.01 "
            "(registration minus registration without modality)",
            AUROC_DIFFERENCE,
            Range(None, 0.01, high_included=False),
            modality.difference,
            modality.interval.low,
            modality.interval.high,
            note,
        ),
    )


def prediction_4_gap(scored: pl.DataFrame) -> Row:
    """Biologic minus small molecule, among all scored trials with a known label."""
    known = _labelled(scored)
    c = contrast(
        known.filter(pl.col("modality") == "biologic"),
        known.filter(pl.col("modality") == "small molecule"),
    )
    excludes = c.low > 0 or c.high < 0  # False when the interval is missing
    note = "the interval excludes zero" if excludes else "the interval does not exclude zero"
    return _contrast_row(
        "4a",
        "Biologics succeed a little more than small molecules, a gap like 63% vs 58% "
        "(biologic minus small molecule)",
        None,
        c,
        "biologic",
        "small molecule",
        note,
    )


def prediction_5(scored: pl.DataFrame) -> list[Row]:
    known = _labelled(scored)
    placebo = known.filter(pl.col("control_kind") == "placebo")
    active = known.filter(pl.col("control_kind") == "active")
    sized = known.filter(pl.col("enrollment").is_not_null())
    small = sized.filter(pl.col("enrollment") < SMALL_TRIAL)
    large = sized.filter(pl.col("enrollment") >= LARGE_TRIAL)
    return [
        _contrast_row(
            "5a",
            "Active-comparator trials succeed 3 to 10 per 100 less than placebo-controlled "
            "ones (active minus placebo)",
            Range(-10, -3),
            contrast(active, placebo),
            "active",
            "placebo",
        ),
        _contrast_row(
            "5b",
            "Trials under 100 participants succeed 18 to 28 per 100 less than those with 300 "
            "or more (under 100 minus 300 or more)",
            Range(-28, -18),
            contrast(small, large),
            "under 100",
            "300 or more",
        ),
    ]


def _genetic_contrast(scored: pl.DataFrame, earlier_trials: pl.Expr | None = None) -> Contrast:
    """Supported minus not supported (evidence today), among mapped trials with a label.

    `genetics_summary` section 2 for everyone; section 12 for the groups split by how many
    earlier trials the lead drug had (`earlier_trials` is the condition that picks a group).
    """
    known = scored.filter(pl.col(bl.LABEL).is_not_null() & pl.col("supported_today").is_not_null())
    if earlier_trials is not None:
        known = known.filter(pl.col("lead_drug_earlier_trials").is_not_null()).filter(
            earlier_trials
        )
    supported = known.filter(pl.col("supported_today"))
    return contrast(supported, known.filter(~pl.col("supported_today")))


def _overlap_text(first: Contrast, second: Contrast) -> str:
    ends = (first.low, first.high, second.low, second.high)
    if any(math.isnan(x) for x in ends):
        return "an interval is missing"
    separate = first.high < second.low or second.high < first.low
    return "the intervals are separate" if separate else "the intervals overlap"


def prediction_6(scored: pl.DataFrame) -> list[Row]:
    """6b mirrors genetics_summary section 12: 5 or fewer earlier trials against more than 5."""
    earlier = pl.col("lead_drug_earlier_trials")
    few = _genetic_contrast(scored, earlier <= ESTABLISHED_AFTER)
    established = _genetic_contrast(scored, earlier > ESTABLISHED_AFTER)
    lower = (
        "point estimate lower"
        if few.difference < established.difference
        else "point estimate not lower"
    )
    comparison = (
        f"{lower} than the {established.difference:+.1f} for drugs with more than "
        f"{ESTABLISHED_AFTER} earlier trials ({_counts(established, 'supported', 'not supported')}; "
        f"interval {established.low:+.1f} to {established.high:+.1f}); "
        f"{_overlap_text(few, established)}"
    )
    return [
        _contrast_row(
            "6a",
            "Genetic support adds 5 to 13 successes per 100 trials, among trials mapped to a "
            "target (supported minus not supported)",
            Range(5, 13),
            _genetic_contrast(scored),
            "supported",
            "not supported",
        ),
        _contrast_row(
            "6b",
            f"Weaker for drugs with {ESTABLISHED_AFTER} or fewer earlier trials than for those "
            "with more (supported minus not supported, within each)",
            None,
            few,
            "supported",
            "not supported",
            comparison,
        ),
    ]


def prediction_7(scored: pl.DataFrame, population: pl.DataFrame) -> list[Row]:
    outside = population.filter(~pl.col("oncology"))
    estimate, low, high = fs.adjusted_ratio(outside, "recruitment", RECRUITMENT_STRATA)
    rare, other = fs.split_groups(outside)
    crude = fs.ratio_text(
        fs.count(rare, "recruitment"), rare.height, fs.count(other, "recruitment"), other.height
    )
    labelled = fs.success_frame(scored)
    c = contrast(labelled.filter(pl.col("is_rare")), labelled.filter(~pl.col("is_rare")))
    return [
        Row(
            "7a",
            "Outside oncology, rare-disease trials stop for recruitment about 1.2 times as "
            "often (rare over not rare, adjusted for phase, sponsor type and era)",
            RATIO,
            None,
            estimate,
            low,
            high,
            f"email: about 1.2; training-set figure {DEV_RECRUITMENT_RATIO}. All finished trials "
            f"in the period: {rare.height:,} rare, {other.height:,} not rare, outside oncology; "
            f"crude ratio {crude}",
        ),
        _contrast_row(
            "7b",
            "Among trials that report a result, rare-disease trials succeed 8 to 18 per 100 "
            "less (rare minus not rare)",
            Range(-18, -8),
            c,
            "rare-disease",
            "not rare",
        ),
    ]


def build_rows(inputs: Inputs, evaluation: ev.Evaluation) -> list[Row]:
    leaky, no_modality = prediction_3_and_4_models(evaluation)
    rows = [
        *prediction_1_and_2(evaluation),
        leaky,
        prediction_4_gap(inputs.scored),
        no_modality,
        *prediction_5(inputs.scored),
        *prediction_6(inputs.scored),
        *prediction_7(inputs.scored, inputs.population),
    ]
    return sorted(rows, key=lambda row: (int(row.number[0]), row.number))


# ---------------------------------------------------------------- prediction 2b


class Fitted(NamedTuple):
    """The model results: the ladder, the drop-one table (biggest loss first), and 2b's test."""

    evaluation: ev.Evaluation
    drops: list[ev.DropOne]
    second_over_third: mt.Interval | None  # paired interval of (second's loss minus third's)


def second_over_third(train: pl.DataFrame, scored: pl.DataFrame, ranked: list[ev.DropOne]):
    """Interval of the second-ranked column's loss minus the third's, on the same resamples.

    The loss of column c is AUROC(all) minus AUROC(without c), so second's loss minus
    third's is AUROC(without third) minus AUROC(without second). None if there is no third.
    """
    if len(ranked) < 3:
        return None
    second, third = ranked[1].column, ranked[2].column
    predictions = {
        column: bl.fit_logistic(
            train, bl.registration_features_without(column)
        ).predict_probability(scored)
        for column in (second, third)
    }
    outcomes = bl.training_outcomes(scored)
    _, (paired,) = mt.bootstrap(
        outcomes,
        predictions,
        True,
        [(third, second)],
        n_resamples=mt.N_RESAMPLES,
        clusters=lead_drug_clusters(scored),
    )
    return paired.interval


def score(inputs: Inputs) -> Fitted:
    """Fit on `inputs.train`, score `inputs.scored`, and drop one column at a time."""
    with ev.convergence_is_an_error():
        evaluation = ev.evaluate_fitted(inputs.train, inputs.scored)
        drops = sorted(ev.drop_one_column(inputs.train, inputs.scored), key=lambda d: -d.loss)
        separation = second_over_third(inputs.train, inputs.scored, drops)
    return Fitted(evaluation, drops, separation)


def top_two_table(drops: list[ev.DropOne], separation: mt.Interval | None) -> str:
    """Prediction 2b: do phase and sponsor class rank first and second by AUROC lost?

    Its own rule: the two named columns are the two biggest losses, and both losses are
    positive. Separately, whether the second loss is clearly above the third.
    """
    ranked = sorted(drops, key=lambda d: -d.loss)
    top_two = [d.column for d in ranked[:2]]
    matches = set(top_two) == set(EXPECTED_TOP_TWO) and all(d.loss > 0 for d in ranked[:2])
    if separation is None or math.isnan(separation.low):
        apart = "Whether the second loss is above the third could not be tested."
    elif separation.low > 0:
        apart = (
            "The second loss is above the third (paired interval of the difference excludes zero)."
        )
    else:
        apart = "The second loss is not clearly above the third (paired interval of the difference includes zero)."
    rows = [
        [
            str(i + 1),
            d.column,
            f"{d.auroc_without:.3f}",
            f"{d.loss:+.3f} ({d.interval.low:.3f} to {d.interval.high:.3f})",
        ]
        for i, d in enumerate(ranked)
    ]
    answer = (
        f"**2b. Predicted: {EXPECTED_TOP_TWO[0]} and {EXPECTED_TOP_TWO[1]} matter most. "
        f"Observed: {' and '.join(top_two)}. {'Matches' if matches else 'Does not match'}.** "
        "Rule for this row only: the two named columns must be the two biggest losses, and "
        f"both must be positive. {apart}"
    )
    caption = (
        "_Same method as results/dev_baselines.md section D: refit the registration model "
        "without one column, score the same trials, and rank the columns by the AUROC lost._"
    )
    table = md_table(
        ["Rank", "Column dropped", "AUROC without it", "AUROC lost (95% interval)"], rows
    )
    return f"{answer}\n\n{caption}\n\n{table}"


# ---------------------------------------------------------------- the report


def rows_table(rows: list[Row]) -> str:
    def observed(row: Row) -> str:
        s = row.style
        return f"{s.show(row.value)}{s.unit} ({s.show(row.low)} to {s.show(row.high)})"

    cells = [
        [r.number, r.claim, show_range(r.style, r.predicted), observed(r), r.verdict, r.note]
        for r in rows
    ]
    header = ["#", "Prediction", "Predicted", "Observed (95% interval)", "Verdict", "Notes"]
    return md_table(header, cells)


def scorecard_text(banner: str, header: list[str], inputs: Inputs, fitted: Fitted) -> str:
    parts = [
        banner,
        *header,
        RULE,
        INTERVAL_CAVEAT,
        "## The predictions",
        rows_table(build_rows(inputs, fitted.evaluation)),
        "## Prediction 2b: which columns matter most",
        top_two_table(fitted.drops, fitted.second_over_third),
    ]
    return "\n\n".join(parts) + "\n"


def _header(code_state: str, inputs: Inputs, trained: str, period: str) -> list[str]:
    return [
        code_state,
        (
            f"{trained}; scored {inputs.scored.height:,} trials finishing {period}. "
            f"Recruitment-stop population: {inputs.population.height:,} finished drug trials "
            "in the same period."
        ),
    ]


def _write_report(path: Path, banner: str, header: list[str], inputs: Inputs) -> str:
    text = scorecard_text(banner, header, inputs, score(inputs))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ---------------------------------------------------------------- the rehearsal (dev only)

REHEARSAL_BANNER = (
    "# REHEARSAL on development data. This is NOT the test set.\n\n"
    "**Every number below is from development trials.** Dev trials that finished "
    f"{REHEARSAL_START.year} to {TEST_START.year - 1} stand in for the test set, and the models "
    f"were trained on dev trials that finished before {REHEARSAL_START.year}. The held-out "
    "labels were not opened. It runs the same code as the real scorecard, to check the code, "
    "not to say anything about the predictions."
)


def rehearse(root: Path | None = None, sources: Sources = REAL_SOURCES) -> str:
    """Train on dev trials before 2017, score those finishing 2017 to 2020. Reads no test data."""
    root = root or ROOT
    dev = pl.read_parquet(root / "data" / "interim" / "dev.parquet")
    check_is_development(dev)
    train, scored = ev.time_forward_split(dev)
    population = build_population(sources.studies(), sources.flags, REHEARSAL_START, TEST_START)
    inputs = assemble(train, scored, pl.read_parquet(sources.genetics), population)
    header = _header(
        ev.provenance(root),
        inputs,
        f"Trained on {train.height:,} dev trials",
        f"from {REHEARSAL_START} up to {TEST_START}",
    )
    return _write_report(root / REHEARSAL_REPORT, REHEARSAL_BANNER, header, inputs)


# ---------------------------------------------------------------- the real run (test set)


def _check_without_labels(
    dev: pl.DataFrame, features: pl.DataFrame, trials: pl.DataFrame, sources: Sources
) -> pl.DataFrame:
    """Everything that can fail without the labels. Returns the genetics table.

    Runs before `open_heldout`, so a failure here costs no logged opening: ids, columns,
    genetics coverage, the rare-disease files, and a dry fit of every model on the dev trials
    (a fit that does not converge stops the run) that predicts the unlabelled test features.
    """
    check_is_development(dev)
    for name, table in (("trials.csv", trials), ("dev.parquet", dev), ("test_features", features)):
        ev.check_unique_ids(name, table)
    ev.check_ids_agree("dev.parquet", dev, trials, ["dev"])
    ev.check_ids_agree("test_features.parquet", features, trials, ["test", "late"])
    test_features = features.filter(pl.col("split") == "test")
    needed = {c for f in bl.FEATURES.values() for c in f.columns} | set(SCORECARD_COLUMNS)
    if missing := needed - set(test_features.columns):
        raise ValueError(f"The test features lack columns {sorted(missing)}.")
    genetics = pl.read_parquet(sources.genetics)
    check_genetics(genetics, test_features)
    sources.flags([])  # reads the Open Targets and Orphanet files, for trials of none
    with ev.convergence_is_an_error():
        for model in bl.fit_ladder(dev).values():
            model.predict_probability(test_features)
        for column in bl.REGISTRATION_COLUMNS:
            bl.fit_logistic(dev, bl.registration_features_without(column)).predict_probability(
                test_features
            )
    return genetics


def load_test_inputs(note: str, root: Path, sources: Sources = REAL_SOURCES) -> Inputs:
    """Train on all development trials; read the test labels and the test period's records.

    The only function that touches the test set. All label-free checks come first. The
    labels come through `open_heldout` (refused before the `prereg` tag exists, and logged),
    and only the test rows are read from the file, so late-split labels are never loaded.
    The raw records of test-period trials are read only after the log entry is written.
    """
    dev = pl.read_parquet(root / "data" / "interim" / "dev.parquet")
    features = pl.read_parquet(root / "data" / "interim" / "test_features.parquet")
    trials = pl.read_csv(root / "trials.csv")
    genetics = _check_without_labels(dev, features, trials, sources)

    path = open_heldout(LABELS_FILE, note, root=root)
    labels = (
        pl.scan_parquet(path)
        .select("nct_id", "split", bl.LABEL)
        .filter(pl.col("split") == "test")
        .collect()
    )
    test_features = features.filter(pl.col("split") == "test")
    ev.check_unique_ids(LABELS_FILE, labels)
    ev.check_ids_agree(LABELS_FILE, labels, trials, ["test"])
    scored = ev.join_labels(test_features, labels)

    population = build_population(sources.studies(), sources.flags, TEST_START, LATE_START)
    return assemble(dev, scored, genetics, population)


def access_section(root: Path) -> str:
    """Every opening of the held-out folder so far, and how many were of the test labels."""
    log = access_log(root)
    labels = str(HELDOUT_DIR / LABELS_FILE)
    opened = sum(entry["path"] == labels for entry in log)
    rows = [[e["at"], f"`{e['commit'][:12]}`", e["path"], e["note"]] for e in log]
    return "\n\n".join(
        [
            (
                f"The test labels have been opened {opened} time(s), counting this run. Every "
                "opening of the held-out folder is listed (also in evals/heldout_log.jsonl)."
            ),
            md_table(["Time (UTC)", "Commit", "File", "Note"], rows),
        ]
    )


def run_test(note: str, root: Path | None = None, sources: Sources = REAL_SOURCES) -> str:
    """The real run. Refuses on a dirty working tree, before anything is opened."""
    root = root or repo_root()
    code_state = ev.provenance(root)  # before the labels are opened: opening writes to the log
    if "uncommitted changes" in code_state:
        raise RuntimeError("The working tree has uncommitted changes. Commit them, then run again.")
    inputs = load_test_inputs(note, root, sources)
    header = _header(
        code_state,
        inputs,
        f"Trained on all {inputs.train.height:,} development trials",
        f"{TEST_START.year} to {LATE_START.year - 1}",
    )
    header += [f"Access note: {note}", access_section(root)]
    banner = "# Scorecard on the held-out test set (trials finished 2021 to 2023)"
    return _write_report(root / TEST_REPORT, banner, header, inputs)


# ---------------------------------------------------------------- entry point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rehearse", help="the same code on development data only")
    test = commands.add_parser("test", help="the real run on the test set (needs the prereg tag)")
    test.add_argument("--note", required=True, help="why the test labels are being opened")
    args = parser.parse_args()
    print(rehearse() if args.command == "rehearse" else run_test(args.note))


if __name__ == "__main__":
    main()
