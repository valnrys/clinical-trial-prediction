"""The final run: score every model and every analysis in FINAL_PLAN.md, once.

    uv run python -m trial_benchmark.final rehearse
    uv run python -m trial_benchmark.final check
    uv run python -m trial_benchmark.final test --note "why"

`rehearse` runs the identical code on development data only. Dev trials that finished
2017 to 2020 stand in for the test set, and the models are trained on dev trials that finished
before 2017. The 300 sampled development trials stand in for the sampled test and late trials
of the language models. There is no stand-in for the late set: that path is exercised by the
synthetic test only. It writes results/rehearsal_final.md under a banner saying so. Nothing on
its path calls `open_heldout`.

`check` runs every step of the real run that needs no labels, and stops. It opens nothing and
logs nothing; it is for finding a problem before the night it matters. It includes the dress
rehearsal described below.

`test` is the real run, made once by the project lead. It trains every registered model on all
development trials, opens the test and late labels through `heldout.open_heldout` (one logged
access for both splits), and writes results/test_final.md. Everything that needs no labels
happens first, so a failure cannot burn the access: FINAL_PLAN.md must be committed, unchanged
since and free of `<<` markers, the working tree must be clean (checked with `git status`; only
the access log and the reports this session writes are left out, and the header names any of
those that were uncommitted), every feature table must cover every test and late trial, every
registered model must fit on all development trials without a convergence failure and predict
the unlabelled test and late features without a missing value, and the language-model inputs
must be complete, finite and in range for every sampled test and late trial (the "named" flags,
which need no labels, are worked out here too). The fitted models predict before the labels are
opened. After that only arithmetic is left.

The dress rehearsal is why nothing after the opening is new code on new data. Before the labels
are opened, `make_periods` and the whole of `run_plan` run once on the real test and late
features and predictions with made-up labels (a seeded random vector, then again with the late
set all one class) and a few bootstrap resamples. Any plan item that raises or fails refuses the
run. The made-up results are thrown away and never written. What first runs after the opening is
the real labels' own checks (ids agree, none missing); if they fail, a report says so.

The held-out log's own `dirty` flag: `open_heldout` ignores only the log, so for whichever
of `final test` and `scorecard test` runs second it reads true, unless the first one's report was
committed in between. The report header says which files were uncommitted.

What is computed is the plan and nothing else. Each plan item is one function that takes frames
and returns rows, so the rehearsal and the real run share all of it:

    R1  reference check: the registry's registration model against the pre-registered one
    P1  the best honest model against the pre-registered baseline    (primary)
    S1  every other registered model, once, in one table
    S2  the information ladder, rung by rung
    S3  calibration: the trend-shifted intercept against the baseline (arithmetic check)
    S4  the public science sources, and a positive control
    S5  the same primary and ladder comparisons on the late set
    E1  endpoint type          E2  Phase 3 after a labelled Phase 2
    E3  broadly expressed single targets              (exploratory)
    E4  the language models, on the sampled trials
    E5  the seven pre-registered predictions: not here, see `scorecard`

Stored language-model predictions for test and late trials are features. They meet outcomes in
exactly one place: `language_rows`, which only receives a `Period` whose split matches. The only
function that reads the predictions file is `read_sampled_predictions`, and the rehearsal asks
it for development trials only (`tests/test_final.py` checks both).

Intervals are 95% percentile intervals from 1,000 bootstrap resamples of lead drugs (a trial
with no known lead drug counts as its own; seed in `metrics.BOOTSTRAP_SEED`), shared by every
model in a table, so a difference is taken within each resample. Trials that share a sponsor
are still treated as independent, so they are somewhat too narrow. The adjusted differences of E1 and E3 resample 2,000 lead drugs instead.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark import evaluate as ev
from trial_benchmark import llm_predictor as lp
from trial_benchmark import llm_report as lr
from trial_benchmark import metrics as mt
from trial_benchmark import scorecard as sc
from trial_benchmark.adjusted import N_RESAMPLES as DRUG_RESAMPLES
from trial_benchmark.adjusted import adjusted
from trial_benchmark.dataset import LABEL_COLUMNS, ROOT
from trial_benchmark.dev_summary import check_is_development
from trial_benchmark.endpoint import BIOMARKER, SURVIVAL
from trial_benchmark.genetics_summary import md_table
from trial_benchmark.heldout import (
    LOG_FILE,
    HeldoutLocked,
    open_heldout,
    prereg_locked,
    prereg_unchanged,
    repo_root,
)
from trial_benchmark.models.linear import REGISTRATION_NAME
from trial_benchmark.models.registry import ModelSpec, all_specs
from trial_benchmark.science import report as science
from trial_benchmark.scorecard import Style
from trial_benchmark.validate import lead_drug_clusters

PLAN_FILE = "FINAL_PLAN.md"
PLAN_PLACEHOLDER = "<<"  # a plan with any marker like <<FILL IN left in is not finished
REHEARSAL_REPORT = Path("results") / "rehearsal_final.md"
TEST_REPORT = Path("results") / "test_final.md"
# What this session legitimately writes. The tree check leaves these out and names any that were
# uncommitted in the report header.
GENERATED = (LOG_FILE, REHEARSAL_REPORT, TEST_REPORT, sc.REHEARSAL_REPORT, sc.TEST_REPORT)
DEV_MODELS_REPORT = Path("results") / "dev_models.md"
LABELS_FILE = sc.LABELS_FILE

REHEARSAL, TEST, LATE, EXTENDED = "rehearsal", "test", "late", "extended"
SPLIT_OF = {REHEARSAL: "dev", TEST: "test", LATE: "late"}  # the sets a language-model sample has
SPLITS_OF = {REHEARSAL: {"dev"}, TEST: {"test"}, LATE: {"late"}, EXTENDED: {"test", "late"}}
SET_TITLE = {
    REHEARSAL: "Development trials finishing 2017 to 2020 (stand-in for the test set)",
    TEST: "Test set (finished 2021 to 2023)",
    LATE: "Late set (finished 2024 to 2025)",
    EXTENDED: "Test and late sets together (2021 to 2025)",
}
# The registry's "logistic: registration" is the pre-registered model, so its AUROC must be the
# one the baseline run printed. If it is not, something has drifted.
REFERENCE_AUROC = {TEST: 0.730, LATE: 0.733}
DRESS_RESAMPLES = 20  # the dress rehearsal checks that things compute, not what they say
DRESS_SEED = 20261111

PRIMARY_MODEL = "ladder: registration + endpoint + history + genetics + words"
WORDS_MODEL = lr.LADDER_NAME  # "ladder: registration + words"
TREND_MODEL = "logistic: trend-shifted intercept"
SCIENCE_MODEL = "biology: registration + all sources"
LADDER_SCIENCE_MODEL = science.LADDER_SCIENCE
CONTROL_MODEL = science.CONTROL_SPEC
RUNGS = (  # S2: each is reported minus registration
    ("+ endpoint", "ladder: registration + endpoint"),
    ("+ history", "ladder: registration + history"),
    ("+ genetics", "ladder: registration + genetics"),
    ("+ words", WORDS_MODEL),
    ("all four tables", PRIMARY_MODEL),
)
REQUIRED_MODELS = tuple(  # every model the plan names, scored whatever its `ranked` flag
    dict.fromkeys(
        [
            REGISTRATION_NAME,
            PRIMARY_MODEL,
            WORDS_MODEL,
            TREND_MODEL,
            SCIENCE_MODEL,
            LADDER_SCIENCE_MODEL,
            CONTROL_MODEL,
            *(name for _, name in RUNGS),
        ]
    )
)
REQUIRED_TABLES = ("endpoint", "history", "genetics", "text", "science")
DRUG_TABLE = "genetics"  # holds lead_drug, the cluster of E1 and E3
NOT_FITTED_FAMILY = "llm"  # scored in E4 on the sampled trials only
STRATA = science.STRATA  # phase, sponsor class, therapeutic area, drug type
BRIER = Style(".3f")
NAN = float("nan")

ITEMS = {
    "R1": ("reference", "Does the registry's registration model reproduce the pre-registered one?"),
    "P1": ("primary", "Does the best honest model built here beat the pre-registered baseline?"),
    "S1": ("secondary", "How does every other registered model do, scored once?"),
    "S2": ("secondary", "What does each table of information add to registration details?"),
    "S3": ("secondary", "Does the trend-shifted intercept reduce the over-prediction?"),
    "S4": ("secondary", "Do the public science sources help, and does the positive control show?"),
    "S5": ("secondary", "Do the primary and ladder comparisons hold on the late set?"),
    "E1": ("exploratory", "Do biomarker endpoints succeed more, and survival endpoints less?"),
    "E2": ("exploratory", "Do Phase 3 trials succeed more after a successful Phase 2?"),
    "E3": ("exploratory", "Do single-target drugs with a broadly expressed target succeed less?"),
    "E4": ("exploratory", "How do the language models do, and how much of it is recall?"),
    "E5": ("exploratory", "The seven pre-registered predictions"),
}


class Refused(RuntimeError):
    """A check that needs no labels failed, so the labels were not opened."""


# ---------------------------------------------------------------- git: the plan and the tree


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    if result.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed: {result.stderr.decode().strip()}")
    return result.stdout


def plan_commit(root: Path) -> str:
    """The commit that last changed the plan, if the plan on disk is that committed text.

    Refuses when the plan is missing, was never committed, still has a `<<` marker, or differs
    from the committed copy (a change that is staged or not).
    """
    plan = root / PLAN_FILE
    if not plan.is_file():
        raise Refused(f"{PLAN_FILE} is missing. Write it and commit it before the run.")
    commit = _git(root, "log", "-1", "--format=%H", "--", PLAN_FILE).decode().strip()
    if not commit:
        raise Refused(f"{PLAN_FILE} has never been committed. Commit it before the run.")
    if PLAN_PLACEHOLDER in plan.read_text():
        raise Refused(f"{PLAN_FILE} still contains a '{PLAN_PLACEHOLDER}' marker. Finish it first.")
    committed = _git(root, "show", f"HEAD:{PLAN_FILE}")
    if plan.read_bytes() != committed:
        raise Refused(f"{PLAN_FILE} has changed since its commit {commit[:12]}. Restore it.")
    return commit


def tree_state(root: Path) -> tuple[list[str], list[str]]:
    """(files changed that nothing here may change, generated files that are uncommitted).

    Straight from `git status`, untracked files included. The first list leaves out only the
    access log and the reports this session writes; the second is those, so the report can say.
    """
    leave_out = [f":(exclude){path}" for path in GENERATED]
    stray = _git(root, "status", "--porcelain", "--", ".", *leave_out).decode().splitlines()
    pending = _git(root, "status", "--porcelain", "--", *map(str, GENERATED)).decode().splitlines()
    return stray, pending


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- where things are read from


class Sources(NamedTuple):
    """The registry, the language-model files and the raw-record reader. The defaults are real."""

    specs: Callable[[], list[ModelSpec]] = all_specs
    required_tables: tuple[str, ...] = REQUIRED_TABLES
    sample: Path = lp.SAMPLE_PATH
    predictions: Path = lp.PREDICTIONS_PATH
    records: Callable[[Iterable[str]], dict[str, dict]] = lp.load_records


REAL_SOURCES = Sources()


def interim(root: Path) -> Path:
    return root / "data" / "interim"


def read_table(root: Path, name: str) -> pl.DataFrame:
    """A feature table without its `split` column (the trial files carry it)."""
    return pl.read_parquet(interim(root) / f"{name}.parquet").drop("split", strict=False)


def join_tables(frame: pl.DataFrame, root: Path, names: Iterable[str]) -> pl.DataFrame:
    """Left-join each feature table by trial id. A repeated column is an error."""
    for name in sorted(set(names)):
        extra = read_table(root, name)
        if clash := (set(extra.columns) & set(frame.columns)) - {"nct_id"}:
            raise ValueError(f"Table {name!r} repeats columns already there: {sorted(clash)}")
        frame = frame.join(extra, on="nct_id", how="left", validate="1:1", maintain_order="left")
    return frame


def check_tables_cover(root: Path, names: Iterable[str], trials: pl.DataFrame) -> None:
    """Every table must list every trial, so a missing row cannot pass as 'unknown'."""
    wanted = set(trials["nct_id"])
    for name in sorted(set(names)):
        table = read_table(root, name)
        if table["nct_id"].n_unique() != table.height:
            raise Refused(f"The {name} table lists a trial more than once.")
        if absent := wanted - set(table["nct_id"]):
            raise Refused(f"The {name} table lacks {len(absent)} of the {len(wanted)} trials.")


def with_control_column(frame: pl.DataFrame) -> pl.DataFrame:
    """The positive control: genetic support at the start, blank where the trial has no target."""
    return frame.with_columns(
        pl.when(pl.col("has_target")).then(pl.col("supported_at_start")).alias(science.CONTROL)
    )


def usable_specs(specs: list[ModelSpec], root: Path) -> tuple[list[ModelSpec], dict[str, str]]:
    """The specs to fit, and for each other spec why not. Nothing is dropped without a reason."""
    usable, skipped = [], {}
    for spec in specs:
        absent = [
            t
            for t in (*spec.tables, *spec.own_tables)
            if not (interim(root) / f"{t}.parquet").exists()
        ]
        if spec.family == NOT_FITTED_FAMILY:
            skipped[spec.name] = "scored on the sampled trials only: see E4"
        elif absent:
            skipped[spec.name] = f"table missing: {', '.join(absent)}"
        else:
            usable.append(spec)
    return usable, skipped


def table_names(usable: list[ModelSpec], sources: Sources, own: bool = False) -> set[str]:
    """The feature tables the usable models and the plan read. `own`: and the models' own tables."""
    names = {t for s in usable for t in s.tables} | {DRUG_TABLE, *sources.required_tables}
    return names | {t for s in usable for t in s.own_tables} if own else names


# ---------------------------------------------------------------- the frames and the fitting


@dataclass
class Fitted:
    """Every model fitted on the training trials and its predictions for each unlabelled frame."""

    specs: list[ModelSpec]
    predictions: dict[str, dict[str, np.ndarray]]  # period -> model -> probabilities
    skipped: dict[str, str]
    seconds: float


def checked(name: str, probabilities, height: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (height,) or not np.all(np.isfinite(p)):
        raise Refused(f"Model {name!r} gave missing or non-finite predictions.")
    return p


def fit_and_predict(
    specs: list[ModelSpec],
    skipped: dict[str, str],
    train: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
) -> Fitted:
    """Fit each spec once on `train` and predict every frame with its label columns removed.

    A fit that does not converge stops the run. This is the slow step, and it needs no labels.
    """
    blind = {k: f.drop([c for c in LABEL_COLUMNS if c in f.columns]) for k, f in frames.items()}
    predictions: dict[str, dict[str, np.ndarray]] = {key: {} for key in frames}
    started = time.perf_counter()
    with ev.convergence_is_an_error():
        for number, spec in enumerate(specs, start=1):
            one = time.perf_counter()
            predictor = spec.fit(train)
            for key, frame in blind.items():
                predictions[key][spec.name] = checked(
                    spec.name, predictor.predict(frame), frame.height
                )
            print(
                f"[{number}/{len(specs)}] {spec.name}: {time.perf_counter() - one:.1f}s",
                file=sys.stderr,
            )
    return Fitted(specs, predictions, skipped, time.perf_counter() - started)


def check_required_models(fitted: Fitted) -> None:
    fitted_names = {spec.name for spec in fitted.specs}
    if missing := [name for name in REQUIRED_MODELS if name not in fitted_names]:
        raise Refused(f"The plan needs models that were not fitted: {missing}.")


def check_columns(frames: dict[str, pl.DataFrame]) -> None:
    """The columns the plan's analyses read, in every frame that will be scored."""
    needed = {"endpoint_type", "prior_phase2_labelled", "prior_phase2_succeeded", "lead_drug",
              "n_target_genes", "expr_n_tissues", "has_target", "supported_at_start",
              *STRATA}  # fmt: skip
    for key, frame in frames.items():
        if missing := needed - set(frame.columns):
            raise Refused(f"The {key} features lack columns {sorted(missing)}.")


# ---------------------------------------------------------------- the language-model files


class LanguageInputs(NamedTuple):
    """Stored predictions of the sampled trials of each period, and what needs no labels."""

    wide: dict[str, pl.DataFrame]  # period -> one row per sampled trial, by id, a column per field
    named: dict[str, dict[str, dict[str, np.ndarray]]]  # period -> definition -> reply set -> bool


def read_sampled_predictions(
    path: Path, sample: pl.DataFrame, splits: Iterable[str]
) -> pl.DataFrame:
    """Stored predictions of the sampled trials of `splits`. The only reader of the file.

    The rehearsal asks for ("dev",). Predictions of other trials are never returned, so they
    cannot meet an outcome.
    """
    ids = set(sample.filter(pl.col("split").is_in(list(splits)))["nct_id"])
    return lp.read_predictions(path).filter(pl.col("nct_id").is_in(list(ids)))


def check_language_columns(key: str, found: pl.DataFrame) -> None:
    """Every model and prompt present for every sampled trial, finite and a probability."""
    for name in lr.COMBOS:
        if name not in found.columns or f"{name} recognised" not in found.columns:
            raise Refused(f"The {key} language-model predictions lack the column for {name}.")
        p = checked(f"{name} ({key})", found[name].to_numpy(), found.height)
        if p.min() < 0 or p.max() > 1:
            raise Refused(f"{name} gave a probability outside 0 to 1 for a sampled {key} trial.")
        if found[f"{name} recognised"].null_count():
            raise Refused(f"{name} left 'recognised' blank for a sampled {key} trial.")


def language_inputs(sources: Sources, periods: dict[str, pl.DataFrame]) -> LanguageInputs | None:
    """The complete stored predictions for the sampled trials of each period, and their flags.

    `periods` maps a period to its unlabelled features. None if the sample or the predictions
    file does not exist. Raises if a sampled trial is not among the trials scored, lacks a
    prediction from any model and prompt, has a missing, non-finite or out-of-range value, or has
    no protocol record. The named / not-named flags need no labels, so they are made here.
    """
    if not sources.sample.exists() or not sources.predictions.exists():
        return None
    sample = lp.read_sample(sources.sample)
    splits = [SPLIT_OF[key] for key in periods]
    predictions = read_sampled_predictions(sources.predictions, sample, splits)
    wide, named = {}, {}
    for key, frame in periods.items():
        ids = sample.filter(pl.col("split") == SPLIT_OF[key])["nct_id"].to_list()
        if not ids:
            raise Refused(f"No {key} trials are sampled for the language models.")
        if absent := set(ids) - set(frame["nct_id"]):
            raise Refused(f"{len(absent)} sampled {key} trials are not among the trials scored.")
        found = lr.wide_predictions(predictions, ids)
        if found.height != len(ids):
            raise Refused(
                f"{len(ids) - found.height} of {len(ids)} sampled {key} trials lack a stored "
                "prediction from some model or prompt."
            )
        check_language_columns(key, found)
        wide[key] = found.sort("nct_id")  # the order llm_report scores in, so the resamples match
    records = sources.records([i for w in wide.values() for i in w["nct_id"]])
    for key, found in wide.items():
        kept = found["nct_id"].to_list()
        if missing := [i for i in kept if i not in records]:
            raise Refused(f"{len(missing)} sampled {key} trials have no protocol record.")
        try:
            named[key] = lr.named_arrays(found, kept, records)
        except Exception as error:
            raise Refused(f"The named / not-named flags failed for {key}: {error!r}") from error
    return LanguageInputs(wide, named)


# ---------------------------------------------------------------- periods: trials with labels


@dataclass(frozen=True)
class Period:
    """One set of scored trials: labelled, with every model's predictions, in the same order.

    A period holds trials of its own split only (the extended set: test and late, both), so
    predictions of one period cannot be scored against the outcomes of another.
    """

    key: str
    table: pl.DataFrame
    predictions: dict[str, np.ndarray]

    def __post_init__(self):
        if set(self.table["split"].unique()) != SPLITS_OF[self.key]:
            raise ValueError(f"The {self.key} period holds trials of another split.")
        for name, p in self.predictions.items():
            if len(p) != self.table.height:
                raise ValueError(f"{name}: {len(p)} predictions for {self.table.height} trials.")

    @property
    def outcomes(self) -> np.ndarray:
        return bl.training_outcomes(self.table)


def attach_labels(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """The features with the headline label, in the order of `features` (predictions line up)."""
    joined = ev.join_labels(features, labels)
    ordered = features.select("nct_id").join(joined, on="nct_id", maintain_order="left")
    if ordered["nct_id"].to_list() != features["nct_id"].to_list():
        raise ValueError("Attaching labels changed the order or the number of trials.")
    return ordered


def make_periods(fitted: Fitted, features: dict[str, pl.DataFrame], labels: pl.DataFrame):
    """A labelled period for each frame, and the test and late sets together if both are there."""
    periods = {
        key: Period(key, attach_labels(frame, labels), fitted.predictions[key])
        for key, frame in features.items()
    }
    if TEST in periods and LATE in periods:
        test, late = periods[TEST], periods[LATE]
        both = {
            n: np.concatenate([test.predictions[n], late.predictions[n]]) for n in test.predictions
        }
        periods[EXTENDED] = Period(EXTENDED, pl.concat([test.table, late.table]), both)
    return periods


# ---------------------------------------------------------------- rows


@dataclass(frozen=True)
class Row:
    item: str
    what: str  # says which group or model comes off which
    style: Style
    value: float
    low: float
    high: float
    note: str = ""

    def estimate(self) -> str:
        s = self.style
        if np.isnan(self.value) or not np.isnan(self.low):
            return f"{s.show(self.value)}{s.unit} ({s.show(self.low)} to {s.show(self.high)})"
        return f"{s.show(self.value)}{s.unit} (no interval)"


def rows_table(rows: list[Row]) -> str:
    cells = [[r.item, r.what, r.estimate(), r.note] for r in rows]
    return md_table(["Item", "Comparison", "Estimate (95% interval)", "Trials and notes"], cells)


def success_text(trials: pl.DataFrame) -> str:
    return f"{trials[bl.LABEL].mean() * 100:.1f}%" if trials.height else "n/a"


def contrast_row(item: str, what: str, first: pl.DataFrame, second: pl.DataFrame, note="") -> Row:
    """First group's success rate minus the second's, in points, with Newcombe's interval."""
    c = sc.contrast(first, second)
    counts = (
        f"{c.n_first:,} against {c.n_second:,}; "
        f"success {success_text(first)} against {success_text(second)}"
    )
    return Row(
        item, what, sc.POINTS, c.difference, c.low, c.high, "; ".join(filter(None, [counts, note]))
    )


def adjusted_row(item: str, what: str, table: pl.DataFrame, exposed: pl.Expr, note="") -> Row:
    """Adjusted (Mantel-Haenszel) difference in points, exposed minus not, resampling lead drugs."""
    frame = table.with_columns(exposed.alias("exposed"))
    estimate, low, high = adjusted(
        frame, "exposed", bl.LABEL, STRATA, science.CLUSTER, measure="difference",
        n_resamples=DRUG_RESAMPLES,
    )  # fmt: skip
    yes, no = frame.filter(pl.col("exposed")), frame.filter(~pl.col("exposed"))
    counts = (
        f"{yes.height:,} against {no.height:,}; "
        f"success {success_text(yes)} against {success_text(no)}"
    )
    return Row(item, what, sc.POINTS, estimate * 100, low * 100, high * 100,
               "; ".join(filter(None, [counts, note])))  # fmt: skip


# ---------------------------------------------------------------- P1, S1, S2, S3, S5


def evaluate_period(spec_names: list[str], period: Period, majority: bool) -> ev.Evaluation:
    """Every named model scored on the period, each paired with the registration baseline."""
    predictions = {name: period.predictions[name] for name in spec_names}
    pairs = [(n, REGISTRATION_NAME) for n in spec_names if n != REGISTRATION_NAME]
    scores, differences = mt.bootstrap(
        period.outcomes,
        predictions,
        majority,
        pairs,
        n_resamples=mt.N_RESAMPLES,
        clusters=lead_drug_clusters(period.table),
    )
    return ev.Evaluation(period.table, period.outcomes, predictions, majority, scores, differences)


def against_registration(evaluation: ev.Evaluation, name: str) -> mt.Difference:
    return next(
        d for d in evaluation.differences if d.better == name and d.baseline == REGISTRATION_NAME
    )


def trials_note(evaluation: ev.Evaluation) -> str:
    rate = next(iter(evaluation.scores.values())).observed_rate * 100
    return f"{evaluation.table.height:,} trials, success {rate:.1f}%"


def auroc_gain_row(item: str, evaluation: ev.Evaluation, name: str, label: str = "") -> Row:
    d = against_registration(evaluation, name)
    return Row(item, f"{label or name} minus {REGISTRATION_NAME}", sc.AUROC_DIFFERENCE,
               d.difference, d.interval.low, d.interval.high,
               f"AUROC, {trials_note(evaluation)}")  # fmt: skip


def primary_rows(evaluation: ev.Evaluation, item: str = "P1") -> list[Row]:
    """The gain over registration, then each model's AUROC, Brier score and calibration gap."""
    rows = [auroc_gain_row(item, evaluation, PRIMARY_MODEL)]
    for name in (PRIMARY_MODEL, REGISTRATION_NAME):
        s = evaluation.scores[name]
        note = trials_note(evaluation)
        rows += [
            Row(item, f"AUROC of {name}", sc.AUROC, s.auroc, s.auroc_interval.low,
                s.auroc_interval.high, note),
            Row(item, f"Brier score of {name} (lower is better)", BRIER, s.brier, NAN, NAN, note),
            Row(item, f"calibration gap of {name}: mean predicted minus observed", sc.POINTS,
                s.calibration_gap, s.gap_interval.low, s.gap_interval.high, note),
        ]  # fmt: skip
    return rows


def ladder_rows(evaluation: ev.Evaluation, item: str = "S2") -> list[Row]:
    return [auroc_gain_row(item, evaluation, name, label) for label, name in RUNGS]


S3_NOTE = (
    "This is an arithmetic check, not a test. The trend-shifted model moves only the intercept of "
    "the registration model, so its mean prediction sits a fixed distance below the registration "
    "model's (that distance needs no labels) and its gap is the registration model's gap plus that "
    "shift, whatever the labels say."
)


def calibration_rows(evaluation: ev.Evaluation, item: str = "S3") -> list[Row]:
    """Mean predicted minus observed success, in points: positive is over-prediction."""
    rows = []
    for name in (REGISTRATION_NAME, TREND_MODEL):
        s = evaluation.scores[name]
        rows.append(Row(item, f"{name}: mean predicted minus observed", sc.POINTS,
                        s.calibration_gap, s.gap_interval.low, s.gap_interval.high,
                        f"observed success {s.observed_rate * 100:.1f}%"))  # fmt: skip
    shift = (
        evaluation.scores[TREND_MODEL].mean_predicted
        - evaluation.scores[REGISTRATION_NAME].mean_predicted
    ) * 100
    rows.append(Row(item, f"{TREND_MODEL} minus {REGISTRATION_NAME}: mean prediction, no labels "
                    "needed", sc.POINTS, shift, NAN, NAN, trials_note(evaluation)))  # fmt: skip
    rows.append(auroc_gain_row(item, evaluation, TREND_MODEL))
    return rows


class ModelRow(NamedTuple):
    spec: ModelSpec
    scores: mt.Scores
    difference: mt.Difference | None  # None for the baseline itself


def model_rows(specs: list[ModelSpec], evaluation: ev.Evaluation) -> list[ModelRow]:
    rows = []
    for spec in specs:
        if spec.name not in evaluation.scores:
            continue
        d = None if spec.name == REGISTRATION_NAME else against_registration(evaluation, spec.name)
        rows.append(ModelRow(spec, evaluation.scores[spec.name], d))
    return sorted(rows, key=lambda r: -r.scores.auroc)


def model_table(rows: list[ModelRow]) -> str:
    def cells(r: ModelRow) -> list[str]:
        gain = (
            "(baseline)"
            if r.difference is None
            else (
                f"{sc.AUROC_DIFFERENCE.show(r.difference.difference)} "
                f"({sc.AUROC_DIFFERENCE.show(r.difference.interval.low)} to "
                f"{sc.AUROC_DIFFERENCE.show(r.difference.interval.high)})"
            )
        )
        s = r.scores
        auroc = f"{s.auroc:.3f} ({s.auroc_interval.low:.3f} to {s.auroc_interval.high:.3f})"
        flags = " (uses a pretrained model)" if r.spec.pretrained else ""
        return [r.spec.name + flags, auroc, gain, f"{s.brier:.3f}", f"{s.calibration_gap:+.1f}"]

    header = ["Model", "AUROC (95% interval)", "AUROC minus registration", "Brier", "Gap (points)"]
    return md_table(header, [cells(r) for r in rows])


def all_models_text(specs: list[ModelSpec], skipped: dict[str, str], evaluation) -> str:
    """S1: honest and not honest apart, ranked as the registry says, nothing left out unsaid."""
    ranked = [s for s in specs if s.ranked]
    parts = []
    for title, honest in (("Honest models", True), ("Not honest (deliberately leaky)", False)):
        rows = model_rows([s for s in ranked if s.honest is honest], evaluation)
        if rows:
            parts += [f"### {title}", model_table(rows)]
    by_reason: dict[str, list[str]] = {}
    for spec in specs:
        if not spec.ranked:
            by_reason.setdefault(spec.not_ranked_reason, []).append(spec.name)
    for reason, names in by_reason.items():
        parts.append(f"Not ranked here ({reason}): " + ", ".join(names) + ".")
    why_skipped: dict[str, list[str]] = {}
    for name, reason in skipped.items():
        why_skipped.setdefault(reason, []).append(name)
    for reason, names in why_skipped.items():
        parts.append(f"Skipped ({reason}): " + ", ".join(names) + ".")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- S4: the public science sources


def science_rows(period: Period, majority: bool, item: str = "S4") -> list[Row]:
    """Registration plus the sources, the positive control, and the ladder plus the sources.

    The first two are scored on the trials with a mapped target and on all trials; the ladder
    plus the sources on mapped trials only. Each is a paired AUROC difference from the model
    without the addition.
    """
    first_two = [(SCIENCE_MODEL, REGISTRATION_NAME), (CONTROL_MODEL, REGISTRATION_NAME)]
    third = (LADDER_SCIENCE_MODEL, PRIMARY_MODEL)
    mapped = period.table["has_target"].fill_null(False).to_numpy()
    clusters = lead_drug_clusters(period.table)
    rows = []
    for label, keep, pairs in (
        ("mapped trials", mapped, [*first_two, third]),
        ("all trials", np.ones_like(mapped), first_two),
    ):
        names = sorted({n for pair in pairs for n in pair})
        outcomes = period.outcomes[keep]
        _, differences = mt.bootstrap(
            outcomes, {n: period.predictions[n][keep] for n in names}, majority, pairs,
            n_resamples=mt.N_RESAMPLES, clusters=clusters[keep],
        )  # fmt: skip
        note = f"AUROC, {int(keep.sum()):,} trials, success {outcomes.mean() * 100:.1f}%"
        for d in differences:
            rows.append(Row(item, f"{d.better} minus {d.baseline}, {label}", sc.AUROC_DIFFERENCE,
                            d.difference, d.interval.low, d.interval.high, note))  # fmt: skip
    return rows


# ---------------------------------------------------------------- E1, E2, E3: hypotheses


def endpoint_rows(table: pl.DataFrame, item: str = "E1") -> list[Row]:
    """Biomarker endpoints against the rest, and survival endpoints against the rest."""
    known = table.filter(pl.col("endpoint_type").is_not_null())
    rows = []
    for kind, short in ((BIOMARKER, "biomarker"), (SURVIVAL, "survival")):
        mine = pl.col("endpoint_type") == kind
        what = f"{short} first primary endpoint minus all other endpoint types"
        rows.append(contrast_row(item, f"{what}, crude", known.filter(mine), known.filter(~mine)))
        rows.append(adjusted_row(item, f"{what}, adjusted", known, mine,
                                 "adjusted for phase, sponsor type, area, drug type"))  # fmt: skip
    return rows


def history_rows(table: pl.DataFrame, item: str = "E2") -> list[Row]:
    """Phase 3 trials that follow a labelled Phase 2: success after a success minus after a failure."""
    follows = table.filter(
        (pl.col("phase") == "phase3")
        & pl.col("prior_phase2_labelled").fill_null(False)
        & pl.col("prior_phase2_succeeded").is_not_null()
    )
    after_success = follows.filter(pl.col("prior_phase2_succeeded"))
    after_failure = follows.filter(~pl.col("prior_phase2_succeeded"))
    return [
        contrast_row(
            item,
            "Phase 3 after a successful Phase 2 minus after a failed Phase 2",
            after_success,
            after_failure,
        )
    ]


def expression_rows(train: pl.DataFrame, table: pl.DataFrame, item: str = "E3") -> list[Row]:
    """The candidate of `science/report.py`: top third minus bottom third of `expr_n_tissues`.

    Single-target drugs only, adjusted for phase, sponsor type, area and drug type, with the
    cut points taken from the training trials. The same code that measured it in development.
    """
    contrast = science.Contrast("expr_n_tissues", "top third minus bottom third")
    _, cuts = science.groups(science.subsets(train)["single"], contrast)
    result, _ = science.estimate(
        science.subsets(table)["single"], contrast, cuts, resamples=DRUG_RESAMPLES
    )
    what = "expr_n_tissues, top third minus bottom third, single-target drugs, adjusted"
    note = f"top third is more than {cuts[1]:g} tissues, bottom third is {cuts[0]:g} or fewer"
    if result is None:
        return [Row(item, what, sc.POINTS, float("nan"), float("nan"), float("nan"),
                    f"too few trials in a group (fewer than {science.MIN_GROUP}); {note}")]  # fmt: skip
    counts = (
        f"{result.n_exposed:,} against {result.n_reference:,}; success "
        f"{result.rate_exposed * 100:.1f}% against {result.rate_reference * 100:.1f}%"
    )
    return [Row(item, what, sc.POINTS, result.difference * 100, result.low * 100,
                result.high * 100, f"{counts}; {note}")]  # fmt: skip


# ---------------------------------------------------------------- E4: the language models


def estimate_row(item: str, what: str, style: Style, e: lr.Estimate | None, note: str) -> Row:
    if e is None:
        return Row(item, what, style, float("nan"), float("nan"), float("nan"),
                   f"too few to score; {note}")  # fmt: skip
    return Row(item, what, style, e.value, e.low, e.high, note)


def subset_rows(item: str, where: str, s: lr.Subset) -> list[Row]:
    """The blinded prompt on a subset: its AUROC, and its gain over each baseline, same trials."""
    note = f"{s.n} trials, success {s.rate:.0%}"
    return [
        estimate_row(item, f"AUROC of {where}", sc.AUROC, s.blinded, note),
        estimate_row(item, f"{where}, minus {REGISTRATION_NAME}", sc.AUROC_DIFFERENCE,
                     s.difference, note),
        estimate_row(item, f"{where}, minus {WORDS_MODEL}", sc.AUROC_DIFFERENCE,
                     s.ladder_difference, note),
    ]  # fmt: skip


def language_rows(period: Period, wide: pl.DataFrame, named, item: str = "E4") -> list[Row]:
    """Each model and prompt on the sampled trials of one period, beside the two baselines.

    `wide` is sorted by trial id and `named` holds its flags (both from `language_inputs`). The
    contrasts between prompts, the named / not-named split, the four-cell table and the trials
    neither model named are those of `llm_report`: the same code. The baselines are the period's
    own predictions for the same trials.
    """
    kept = wide["nct_id"].to_list()
    position = {nct: i for i, nct in enumerate(period.table["nct_id"])}
    at = [position[nct] for nct in kept]
    outcomes = period.outcomes[at]
    arrays = {n: wide[n].to_numpy() for n in lr.COMBOS}
    refs = [REGISTRATION_NAME, WORDS_MODEL]
    arrays |= {name: period.predictions[name][at] for name in refs}
    pairs = [(lr.combo(m, a), lr.combo(m, b)) for m in lp.MODELS for a, b in lr.CONTRASTS]
    pairs += [(c, r) for c in lr.COMBOS for r in refs]
    single, paired = lr.paired_bootstrap(
        outcomes, arrays, pairs, clusters=lead_drug_clusters(period.table[at])
    )
    n = f"{len(kept)} sampled trials"
    rows = [
        Row(item, f"AUROC of {name}", sc.AUROC, e.value, e.low, e.high, n)
        for name in [*lr.COMBOS, *refs]
        for e in [single[name]["auroc"]]
    ]
    for a, b in pairs:
        e = paired[(a, b)]["auroc"]
        rows.append(Row(item, f"{a} minus {b}", sc.AUROC_DIFFERENCE, e.value, e.low, e.high, n))
    for (definition, model, label), s in lr.not_named_results(outcomes, arrays, named).items():
        rows += subset_rows(item, f"blinded {model}, {definition}, {label}", s)
    stricter = named[lr.STRICTER]
    recognised = {
        lr.combo(m, "blinded"): wide[f"{lr.combo(m, 'blinded')} recognised"].to_numpy().astype(bool)
        for m in lp.MODELS
    }
    for (model, was_named, knew), s in lr.four_cell_results(
        outcomes, arrays, stricter, recognised
    ).items():
        drug = lr.NAMED if was_named else lr.NOT_NAMED
        said = "said it recognised the trial" if knew else "said it did not recognise the trial"
        rows += subset_rows(item, f"blinded {model}, {lr.STRICTER}, {drug}, {said}", s)
    for model, s in lr.neither_named_results(outcomes, arrays, stricter).items():
        rows += subset_rows(item, f"blinded {model}, {lr.STRICTER}, drug named by neither model", s)
    return rows


# ---------------------------------------------------------------- running the plan


@dataclass(frozen=True)
class Outcome:
    item: str
    text: str  # markdown, or the failure
    failed: bool = False


def attempt(item: str, produce: Callable[[], str]) -> Outcome:
    """Run one item. A failure is written into the report, so one bug cannot lose the others."""
    try:
        return Outcome(item, produce())
    except Exception as error:  # noqa: BLE001 - reported, then the run exits non-zero
        detail = "".join(traceback.format_exception_only(type(error), error)).strip()
        print(traceback.format_exc(), file=sys.stderr)
        return Outcome(item, f"**This item failed and was not computed.** {detail}", failed=True)


class ReferenceDrift(RuntimeError):
    """The registry's registration model no longer reproduces the pre-registered one."""


def reference_text(periods: dict[str, Period], reference: dict[str, float] | None) -> str:
    """R1: the registration model's AUROC against the figure the baseline run printed."""
    if reference is None:
        return "Not checked here: the figures to match are those of the real test and late sets."
    rows, drifted = [], []
    for key, expected in reference.items():
        period = periods[key]
        got = mt.auroc(period.outcomes, period.predictions[REGISTRATION_NAME])
        rows.append([SET_TITLE[key], f"{expected:.3f}", f"{got:.3f}"])
        if f"{got:.3f}" != f"{expected:.3f}":
            drifted.append(f"{key} {got:.3f} against {expected:.3f}")
    if drifted:
        raise ReferenceDrift(
            "the registration model does not reproduce the pre-registered one: "
            + "; ".join(drifted)
            + ". Something has drifted. Trust nothing else in this report."
        )
    table = md_table(["Set", "Printed by the baseline run", "This run"], rows)
    return f"Matches to three decimals. {table}"


def per_set(keys: list[str], periods: dict[str, Period], make) -> str:
    """One table for each set in `keys`, each under its own title."""
    parts = []
    for key in keys:
        parts += [f"### {SET_TITLE[key]}", rows_table(make(periods[key]))]
    return "\n\n".join(parts)


def run_plan(
    fitted: Fitted,
    train: pl.DataFrame,
    periods: dict[str, Period],
    language: LanguageInputs | None,
    reference: dict[str, float] | None = None,
) -> list[Outcome]:
    """Every plan item, in order.

    `periods` holds the primary period (the rehearsal's or the test set) and, in the real run,
    the late set and the two together. `reference` maps a period to the registration AUROC it
    must reproduce (R1); the real run passes REFERENCE_AUROC.
    """
    key = REHEARSAL if REHEARSAL in periods else TEST
    primary = periods[key]
    pooled = [key, *(k for k in (EXTENDED,) if k in periods)]
    both = [key, *(k for k in (LATE,) if k in periods)]
    majority = mt.majority_class_of(bl.training_outcomes(train))
    registry = [s for s in fitted.specs if s.name != CONTROL_MODEL]
    needed = [name for name in REQUIRED_MODELS if name != CONTROL_MODEL]  # scored, ranked or not
    every = [s.name for s in registry if s.ranked or s.name in needed]

    cache: dict[str, ev.Evaluation] = {}

    def evaluated(period: Period) -> ev.Evaluation:
        if period.key not in cache:
            names = every if period is primary else needed
            cache[period.key] = evaluate_period(names, period, majority)
        return cache[period.key]

    results: dict[str, Outcome] = {}
    if reference is None and REHEARSAL in periods:
        results["R1"] = Outcome("R1", reference_text(periods, None))
    else:
        results["R1"] = attempt("R1", lambda: reference_text(periods, reference))
    results["P1"] = attempt(
        "P1", lambda: per_set(pooled, periods, lambda p: primary_rows(evaluated(p)))
    )
    results["S1"] = attempt(
        "S1", lambda: all_models_text(registry, fitted.skipped, evaluated(primary))
    )
    results["S2"] = attempt(
        "S2", lambda: per_set(pooled, periods, lambda p: ladder_rows(evaluated(p)))
    )
    results["S3"] = attempt(
        "S3",
        lambda: S3_NOTE + "\n\n" + per_set(both, periods, lambda p: calibration_rows(evaluated(p))),
    )
    results["S4"] = attempt("S4", lambda: rows_table(science_rows(primary, majority)))
    if LATE in periods:
        late = periods[LATE]
        results["S5"] = attempt(
            "S5",
            lambda: rows_table(
                [*primary_rows(evaluated(late), "S5"), *ladder_rows(evaluated(late), "S5")]
            ),
        )
    else:
        results["S5"] = Outcome(
            "S5",
            "Not run in the rehearsal: there is no stand-in for the late set. The code path is "
            "exercised by the synthetic test (`tests/test_final.py`).",
        )
    results["E1"] = attempt("E1", lambda: per_set(both, periods, lambda p: endpoint_rows(p.table)))
    results["E2"] = attempt("E2", lambda: per_set(both, periods, lambda p: history_rows(p.table)))
    results["E3"] = attempt(
        "E3", lambda: per_set(both, periods, lambda p: expression_rows(train, p.table))
    )
    results["E4"] = attempt("E4", lambda: language_text(periods, language))
    results["E5"] = Outcome(
        "E5",
        "Not computed here. The seven pre-registered predictions are scored by "
        "`uv run python -m trial_benchmark.scorecard test`, a separate command in the same session.",
    )
    return [results[item] for item in ITEMS]


def language_text(periods: dict[str, Period], language: LanguageInputs | None) -> str:
    if language is None:
        return "Skipped: the sample or the stored predictions file is not on this machine."
    parts = []
    for key, period in periods.items():
        if key not in language.wide:
            continue
        title = {REHEARSAL: "Sampled development trials", TEST: "Sampled test trials",
                 LATE: "Sampled late trials"}[key]  # fmt: skip
        rows = language_rows(period, language.wide[key], language.named[key])
        parts += [f"### {title}", rows_table(rows)]
    return "\n\n".join(parts)


# ---------------------------------------------------------------- the rehearsal's own check


def per_round_aurocs(text: str, column: str = "2017-2018") -> dict[str, float]:
    """Model -> AUROC in one round, from the 'AUROC within each round' tables of dev_models.md."""
    start = text.index("### AUROC within each round")
    block = text[start:].split("\n## ", 1)[0]
    lines = [line for line in block.splitlines() if line.startswith("| ")]
    found, at = {}, None
    for line in lines:
        cells = [c.strip() for c in line.strip("| ").split("|")]
        if cells[0] == "model":
            at = cells.index(column)
        elif at is not None and re.fullmatch(r"\d\.\d+", cells[at]):
            found[cells[0]] = float(cells[at])
    return found


def consistency_section(period: Period, dev_models_text: str) -> str:
    """Rehearsal AUROCs on the 2017-2018 trials against the round trained before 2017 in dev_models.

    Those rounds fit on the same trials, so the numbers should agree to the last digit.
    """
    expected = per_round_aurocs(dev_models_text)
    years = period.table["primary_completion_date"].dt.year().to_numpy()
    keep = (years >= 2017) & (years <= 2018)
    rows, worst = [], 0.0
    for name, value in expected.items():
        if name not in period.predictions:
            continue
        mine = mt.auroc(period.outcomes[keep], period.predictions[name][keep])
        worst = max(worst, abs(mine - value))
        rows.append([name, f"{value:.3f}", f"{mine:.3f}", f"{mine - value:+.3f}"])
    lead = (
        f"{len(rows)} models compared. The largest difference is {worst:.3f}. `dev_models.md` "
        "scores the round trained before 2017 on trials finishing 2017 to 2018, with the same "
        "training trials as this rehearsal, so agreement to the third decimal is expected."
    )
    return "\n\n".join(
        [lead, md_table(["Model", "dev_models.md", "this rehearsal", "Difference"], rows)]
    )


# ---------------------------------------------------------------- the report


def period_line(period: Period, what: str) -> str:
    rate = float(period.outcomes.mean()) * 100
    return f"- {what}: {period.table.height:,} trials, observed success {rate:.1f}%."


def report_text(banner, header: list[str], outcomes: list[Outcome], extra: list[str]) -> str:
    parts = [banner, *header]
    failed = [o.item for o in outcomes if o.failed]
    if failed:
        parts.insert(1, f"**ITEMS THAT FAILED: {', '.join(failed)}.** The rest are below.")
    for o in outcomes:
        level, question = ITEMS[o.item]
        parts += [f"## {o.item} ({level}): {question}", o.text]
    return "\n\n".join([*parts, *extra]) + "\n"


READING = (
    "Differences are first-named minus second-named. AUROC differences are paired: taken within "
    "each bootstrap resample. Calibration gap is mean predicted minus observed success, in points "
    "(positive means over-prediction). Intervals are 95% percentile intervals from 1,000 "
    "resamples of lead drugs (all of a drug's trials together), except the adjusted differences "
    "of E1 and E3, which use 2,000 resamples of lead drugs; trials that share a sponsor are "
    "not independent, so intervals are somewhat too narrow. The expectation for each item is in FINAL_PLAN.md, written before this run."
)
REHEARSAL_BANNER = (
    "# REHEARSAL on development data. This is NOT the final run.\n\n"
    "**Every number below is from development trials.** Dev trials that finished 2017 to 2020 "
    "stand in for the test set, and every model was trained on dev trials that finished before "
    "2017. The 300 sampled development trials stand in for the sampled test and late trials of "
    "the language models. There is no stand-in for the late set (S5). The held-out labels were "
    "not opened. This runs the same code as the final run, to check the code, not to say "
    "anything about the plan."
)


# ---------------------------------------------------------------- the rehearsal (dev only)


def development_frames(root: Path, sources: Sources):
    """The development trials with every table the usable models and the plan read."""
    dev = pl.read_parquet(interim(root) / "dev.parquet")
    check_is_development(dev)
    specs = [*sources.specs(), science.control_spec()]
    usable, skipped = usable_specs(specs, root)
    names = table_names(usable, sources)
    return with_control_column(join_tables(dev, root, names)), usable, skipped


def rehearse(root: Path | None = None, sources: Sources = REAL_SOURCES) -> str:
    """Train on dev trials before 2017, score those finishing 2017 to 2020. Reads no test data."""
    root = root or ROOT
    dev, usable, skipped = development_frames(root, sources)
    train, scored = ev.time_forward_split(dev)
    check_columns({REHEARSAL: scored})
    fitted = fit_and_predict(usable, skipped, train, {REHEARSAL: scored})
    check_required_models(fitted)
    language = language_inputs(sources, {REHEARSAL: scored})
    labels = scored.select("nct_id", "split", bl.LABEL)
    periods = make_periods(fitted, {REHEARSAL: scored}, labels)
    outcomes = run_plan(fitted, train, periods, language)
    header = [
        ev.provenance(root),
        (
            f"Trained {len(fitted.specs)} models on {train.height:,} dev trials (finishing before "
            f"2017) in {fitted.seconds:.0f} seconds. Scored:"
        ),
        period_line(periods[REHEARSAL], "development trials finishing 2017 to 2020"),
        READING,
    ]
    dev_models = root / DEV_MODELS_REPORT
    agreement = (
        consistency_section(periods[REHEARSAL], dev_models.read_text())
        if dev_models.exists()
        else f"{DEV_MODELS_REPORT} is not on this machine, so there is nothing to compare with."
    )
    extra = ["## Check against results/dev_models.md", agreement]
    text = report_text(REHEARSAL_BANNER, header, outcomes, extra)
    (root / REHEARSAL_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / REHEARSAL_REPORT).write_text(text)
    return text


# ---------------------------------------------------------------- the real run (test and late)

LABELS_FAILED = "THE LABELS WERE OPENED BUT FAILED THEIR OWN CHECKS"
DIRTY_NOTE = (
    "The access log's own `dirty` flag ignores only the log, so it reads true for whichever of "
    "`final test` and `scorecard test` ran second unless the first one's report was committed "
    "in between; the line above names the files that were uncommitted."
)


def check_before_opening(root: Path) -> tuple[str, str]:
    """The checks that need no labels and no models. Returns the code state and the plan commit.

    The plan is checked first, then the tree, straight from `git status`.
    """
    commit = plan_commit(root)
    stray, pending = tree_state(root)
    if stray:
        raise Refused(
            "The working tree has uncommitted changes. Commit them, then run again:\n"
            + "\n".join(stray)
        )
    if not prereg_locked(root) or not prereg_unchanged(root):
        raise HeldoutLocked("The pre-registration is not locked, or has changed since its tag.")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    left = (
        "Uncommitted at the start (the files this session writes): "
        + ", ".join(f"`{line[3:]}`" for line in pending)
        if pending
        else "Uncommitted at the start: nothing."
    )
    return f"Git commit `{head}`, clean tree. {left}", commit


def load_real_frames(root: Path, sources: Sources):
    """Development and held-out features, with every table joined and covering every trial."""
    dev = pl.read_parquet(interim(root) / "dev.parquet")
    features = pl.read_parquet(interim(root) / "test_features.parquet")
    trials = pl.read_csv(root / "trials.csv")
    check_is_development(dev)
    for name, table in (("trials.csv", trials), ("dev.parquet", dev), ("test_features", features)):
        ev.check_unique_ids(name, table)
    ev.check_ids_agree("dev.parquet", dev, trials, ["dev"])
    ev.check_ids_agree("test_features.parquet", features, trials, ["test", "late"])
    specs = [*sources.specs(), science.control_spec()]
    usable, skipped = usable_specs(specs, root)
    names = table_names(usable, sources)
    check_tables_cover(root, table_names(usable, sources, own=True), trials)
    dev = with_control_column(join_tables(dev, root, names))
    held = with_control_column(join_tables(features, root, names))
    frames = {
        TEST: held.filter(pl.col("split") == "test"),
        LATE: held.filter(pl.col("split") == "late"),
    }
    return dev, frames, usable, skipped, trials


def input_hashes(root: Path, usable: list[ModelSpec], sources: Sources) -> list[str]:
    """The sha256 of every data/interim table read and of the sample file. Git ignores them."""
    paths = [interim(root) / f"{name}.parquet" for name in ("dev", "test_features")]
    paths += [interim(root) / f"{name}.parquet" for name in sorted(table_names(usable, sources))]
    paths += [sources.sample, sources.predictions]
    return [
        f"- `{path.relative_to(root) if path.is_relative_to(root) else path}` sha256 "
        f"`{file_sha256(path)}`"
        for path in paths
    ]


def read_labels(path: Path, trials: pl.DataFrame) -> pl.DataFrame:
    """The test and late rows of the opened labels file, and the checks on them. Fits nothing."""
    labels = (
        pl.scan_parquet(path)
        .select("nct_id", "split", bl.LABEL)
        .filter(pl.col("split").is_in(["test", "late"]))
        .collect()
    )
    ev.check_unique_ids(LABELS_FILE, labels)
    ev.check_ids_agree(LABELS_FILE, labels, trials, ["test", "late"])
    if labels[bl.LABEL].null_count():
        raise ValueError(f"{LABELS_FILE} has a missing {bl.LABEL} for some test or late trial.")
    return labels


def made_up_labels(frames: dict[str, pl.DataFrame], late_one_class: bool) -> pl.DataFrame:
    """Random 0/1 labels for the real trials (seeded), for the dress rehearsal. Mean nothing."""
    ids = pl.concat([f.select("nct_id", "split") for f in frames.values()])
    random = np.random.default_rng(DRESS_SEED).random(ids.height) < 0.5
    values = np.where(ids["split"].to_numpy() == "late", True, random) if late_one_class else random
    return ids.with_columns(pl.Series(bl.LABEL, values))


@contextlib.contextmanager
def few_resamples(count: int):
    """Fewer bootstrap resamples, for code that is only being checked to run."""
    global DRUG_RESAMPLES
    before, drugs = mt.N_RESAMPLES, DRUG_RESAMPLES
    mt.N_RESAMPLES = DRUG_RESAMPLES = count
    try:
        yield
    finally:
        mt.N_RESAMPLES, DRUG_RESAMPLES = before, drugs


def dress_rehearsal(
    fitted: Fitted, train: pl.DataFrame, frames: dict[str, pl.DataFrame], language: LanguageInputs
) -> float:
    """Run `make_periods` and all of `run_plan` on the real features with made-up labels.

    Twice: random labels, then random labels with the late set all one class. Nothing is kept.
    Refuses if any plan item fails. Returns the seconds taken.
    """
    started = time.perf_counter()
    for late_one_class in (False, True):
        labels = made_up_labels(frames, late_one_class)
        with few_resamples(DRESS_RESAMPLES):
            outcomes = run_plan(fitted, train, make_periods(fitted, frames, labels), language)
        if failed := [f"{o.item}: {o.text}" for o in outcomes if o.failed]:
            raise Refused("The dress rehearsal on made-up labels failed:\n" + "\n".join(failed))
    return time.perf_counter() - started


class Prepared(NamedTuple):
    """Everything the real run settles before the labels are opened."""

    code_state: str
    commit: str  # the plan's
    inputs: list[str]  # sha256 of every input file
    train: pl.DataFrame
    frames: dict[str, pl.DataFrame]
    trials: pl.DataFrame
    fitted: Fitted
    language: LanguageInputs
    dress_seconds: float


def prepare_real(root: Path, sources: Sources) -> Prepared:
    """Every check that needs no labels, the fitting, and the dress rehearsal. Opens nothing."""
    code_state, commit = check_before_opening(root)
    train, frames, usable, skipped, trials = load_real_frames(root, sources)
    check_columns({"train": train, **frames})
    fitted = fit_and_predict(usable, skipped, train, frames)
    check_required_models(fitted)
    language = language_inputs(sources, frames)
    if language is None:
        raise Refused("The language-model sample or predictions are missing.")
    seconds = dress_rehearsal(fitted, train, frames, language)
    inputs = input_hashes(root, usable, sources)
    return Prepared(code_state, commit, inputs, train, frames, trials, fitted, language, seconds)


def check_real(root: Path | None = None, sources: Sources = REAL_SOURCES) -> str:
    """Run every label-free step of the real run and stop. Opens nothing, logs nothing."""
    p = prepare_real(root or repo_root(), sources)
    return (
        f"All label-free checks passed. {len(p.fitted.specs)} models fitted on {p.train.height:,} "
        f"development trials in {p.fitted.seconds:.0f} seconds and predicted "
        f"{p.frames[TEST].height:,} test and {p.frames[LATE].height:,} late trials with no "
        f"missing value; the language-model inputs are complete; the dress rehearsal on made-up "
        f"labels ran every plan item twice in {p.dress_seconds:.0f} seconds; plan at "
        f"commit {p.commit[:12]}. The labels were not opened."
    )


def run_test(note: str, root: Path | None = None, sources: Sources = REAL_SOURCES) -> str:
    """The real run. Refuses before anything is opened if any label-free check fails."""
    root = root or repo_root()
    if not note.strip():
        raise ValueError("Give a note saying why the held-out data is being opened.")
    p = prepare_real(root, sources)
    path = open_heldout(LABELS_FILE, note, root=root)  # the one logged access
    header = [
        p.code_state,
        f"Plan: {PLAN_FILE} at commit `{p.commit}`, unchanged since.",
        DIRTY_NOTE,
        "Input files (git ignores them, so their hashes are recorded here):",
        "\n".join(p.inputs),
        (
            f"Trained {len(p.fitted.specs)} models on all {p.train.height:,} development trials "
            f"in {p.fitted.seconds:.0f} seconds (before the labels were opened); the dress "
            f"rehearsal on made-up labels took {p.dress_seconds:.0f} seconds. Scored:"
        ),
    ]
    try:  # from here the real labels are in hand: only their own checks are new
        labels = read_labels(path, p.trials)
        periods = make_periods(p.fitted, p.frames, labels)
    except Exception as error:  # noqa: BLE001 - written into the report, not lost in a traceback
        detail = "".join(traceback.format_exception_only(type(error), error)).strip()
        banner = f"# {LABELS_FAILED}\n\n**{LABELS_FAILED}: {detail}** Nothing was scored."
        return write_test_report(
            root, report_text(banner, [*header, f"Access note: {note}"], [], [])
        )
    outcomes = run_plan(p.fitted, p.train, periods, p.language, REFERENCE_AUROC)
    header += [
        period_line(periods[TEST], "test set, finishing 2021 to 2023"),
        period_line(periods[LATE], "late set, finishing 2024 to 2025"),
        f"Access note: {note}",
        sc.access_section(root),
        READING,
    ]
    banner = "# The final run on the held-out test and late sets"
    return write_test_report(root, report_text(banner, header, outcomes, []))


def write_test_report(root: Path, text: str) -> str:
    (root / TEST_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / TEST_REPORT).write_text(text)
    return text


# ---------------------------------------------------------------- entry point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rehearse", help="the same code on development data only")
    commands.add_parser("check", help="every label-free step of the real run; opens nothing")
    test = commands.add_parser("test", help="the real run (needs the prereg tag and a clean tree)")
    test.add_argument("--note", required=True, help="why the held-out labels are being opened")
    args = parser.parse_args()
    if args.command == "check":
        print(check_real())
        return
    text = rehearse() if args.command == "rehearse" else run_test(args.note)
    print(text)
    if "ITEMS THAT FAILED" in text or LABELS_FAILED in text:
        sys.exit(1)


if __name__ == "__main__":
    main()
