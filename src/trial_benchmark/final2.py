"""The second scoring of the test set: the eight items of TEST_PLAN_2.md, once.

    uv run --extra hierarchical python -m trial_benchmark.final2 --dry-run [--hierarchy full full_pg]
    uv run --extra hierarchical python -m trial_benchmark.final2 --real --note "why"
    uv run --extra hierarchical python -m trial_benchmark.final2 --real --note "why" --rerun "reason"

`--real` is the one run on the test set (trials finished 2021 to 2023), made once by the project
lead. In order:

1. Checks that need nothing fitted: TEST_PLAN_2.md is committed and unchanged since, the working
   tree is clean (only the access log and this run's own outputs are left out), the
   pre-registration is locked, this is the first run (no earlier opening named `PURPOSE` in the
   access log, and none of the run's output files exists), every file the run writes under
   outputs/final2/ is git-ignored, and the plan's table of expectations can be read.
2. Every model in `MODELS` (a constant: nothing here chooses, tunes or drops a model) is fitted on
   ALL development trials and predicts the test trials with no outcome column in sight
   (`fit_and_predict` and `check_blind` refuse a frame with one, before and after the feature
   tables are joined, a training frame holding a scored trial, or a scored trial with a
   results-posting date). Item 8's biology score is fitted the same way. The earlier results'
   models are fitted through `final.prepare_real`, with the families added since the first run
   left out.
3. The dress rehearsal: the whole shared path (`shared_tail`) on made-up labels, writing nothing.
4. The plan and the tree are checked again, then ONE call to `open_heldout`. That one read gives
   the test labels and the late-set labels (2024 to 2025), which item 7 needs to regenerate the
   first run's report.
5. `shared_tail` computes every item and writes results/test_final2.md and .json (with the
   SHA-256 of every input read), the regenerated earlier reports, and the per-trial predictions
   to outputs/final2/ (git-ignored: the file holds outcomes and must never be committed).

The run refuses a second time. `--rerun "reason"` allows another opening only after an earlier one
named `PURPOSE`, with that run's output files moved aside; the log note and the report say it is a
re-run and why.

`--dry-run` runs the same code with the last development round standing in for the test set:
trained on development trials finished before 2019, scored on those finished 2019 to 2020. The
stand-in is made to look like test rows before the shared code sees it (its labels are taken off,
its split reads "test", and its development-only results-posting date is blanked as the tables are
read). Everything from the labels in hand to the last file written is `shared_tail`, the same
function as the real run. What still differs is set out in `Setting` and in the report: where the
frames and labels come from, the guard and the git checks, item 7's stand-in (the earlier code's
own rehearsal split, trained before 2017 and scored 2017 to 2020) and its scorecard window, and
where files go (outputs/final2_dry/). It may restrict the hierarchical variants (`--hierarchy`);
the real run refuses that flag.

What each item scores (expectations are copied verbatim from TEST_PLAN_2.md into the report):

    1  hierarchical variants minus the registration baseline   (headline: full hierarchy, today)
    2  registration + precedent flag, 3 link versions, plain and strict
                                       (headline: strict, "2020 names and links")
    3  HOW, WHAT, WHAT + phase, HOW + WHAT per link version, and own-lead-drug-excluded variants;
       "both beats HOW" is HOW + WHAT minus HOW
    4  Phase 2 and Phase 3 success per 100 by third of the WHAT score (thirds cut within the
       scored set and phase), and the share of the gap that survives within phase and sponsor
    5  the split study with the scored set as the later holdout
    6  per-trial predictions of every model and item 8's per-trial biology score, for curves
    7  the earlier results' intervals regenerated with lead-drug resampling, compared row by row
       with the same code resampling single trials (which reproduces the earlier intervals)
    8  the biology-only score within phase: (a) two or more risk flags against none, (b) top
       third against bottom third of a biology-only logistic (headline: today's links)

Verdicts, as "How each item is judged" in the plan: met when the 95% interval lies wholly on the
expected side of zero; where a range is given, also inside or outside it (`scorecard.verdict`);
where the plan says "about", the interval rule only. Intervals resample lead drugs.

Item 5 is the one place test labels reach a fit: the split study's random holdouts mix the test
trials into training by design. Nothing from them is used in any other item.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark import biology_score as bs
from trial_benchmark import evaluate as ev
from trial_benchmark import final, validate
from trial_benchmark import metrics as mt
from trial_benchmark import scorecard as sc
from trial_benchmark import split_study as sp
from trial_benchmark import whathow_report as wr
from trial_benchmark.dataset import LABEL_COLUMNS, LATE_START, ROOT, TEST_START
from trial_benchmark.dev_summary import check_is_development
from trial_benchmark.genetics_summary import md_table
from trial_benchmark.heldout import (
    LOG_FILE,
    HeldoutLocked,
    access_log,
    open_heldout,
    prereg_locked,
    prereg_unchanged,
    repo_root,
)
from trial_benchmark.models import hierarchical as hm
from trial_benchmark.models import linear
from trial_benchmark.models import precedent as pm
from trial_benchmark.models import whathow as wm
from trial_benchmark.models.registry import ModelSpec, all_specs

# ---------------------------------------------------------------- the fixed model list

REGISTRATION = linear.REGISTRATION_NAME  # "logistic: registration", the baseline of every item
LINK_VERSIONS = ("today", "2020", "2020_names")
LINK_TEXT = {"today": "today's links", "2020": "2020 links", "2020_names": "2020 names and links"}
HIERARCHY_VERSIONS = ("today", "2020")  # the hierarchical model has no "2020 names" version
HIERARCHY_MODELS = (  # (variant key, link version, model name)
    ("registration", "today", "hierarchical: registration only"),
    ("target", "today", "hierarchical: + target"),
    ("condition", "today", "hierarchical: + condition"),
    ("sponsor", "today", "hierarchical: + sponsor"),
    ("full", "today", "hierarchical: full hierarchy"),
    ("full_pg", "today", "hierarchical: full + precedent + genetics"),
    ("registration", "2020", "hierarchical: registration only [2020 links]"),
    ("target", "2020", "hierarchical: + target [2020 links]"),
    ("condition", "2020", "hierarchical: + condition [2020 links]"),
    ("sponsor", "2020", "hierarchical: + sponsor [2020 links]"),
    ("full", "2020", "hierarchical: full hierarchy [2020 links]"),
    ("full_pg", "2020", "hierarchical: full + precedent + genetics [2020 links]"),
)
PRECEDENT_MODELS = tuple(
    f"logistic: registration + precedent ({LINK_TEXT[v]}{', experimental arms only' if strict else ''})"
    for strict in (False, True)
    for v in LINK_VERSIONS
)
BLOCKS = ("HOW", "WHAT", "WHAT + phase", "HOW + WHAT")
EXCLUDED_BLOCKS = ("WHAT", "WHAT + phase", "HOW + WHAT")  # HOW has no group rates to exclude from


def what_name(block: str, version: str, excluded: bool = False) -> str:
    """The name of one WHAT / HOW model, as `models.whathow.spec_name` gives it."""
    return (
        f"logistic: {block} ({LINK_TEXT[version]}{', own lead drug excluded' if excluded else ''})"
    )


WHATHOW_MODELS = tuple(
    name
    for v in LINK_VERSIONS
    for name in (
        *(what_name(b, v) for b in BLOCKS),
        *(what_name(b, v, excluded=True) for b in EXCLUDED_BLOCKS),
    )
)
MODELS: tuple[str, ...] = (
    REGISTRATION,
    *(name for _, _, name in HIERARCHY_MODELS),
    *PRECEDENT_MODELS,
    *WHATHOW_MODELS,
)

HEADLINE_HIERARCHY = "hierarchical: full hierarchy"
HEADLINE_LINKS = "2020_names"
HEADLINE_PRECEDENT = (
    "logistic: registration + precedent (2020 names and links, experimental arms only)"
)
SPLIT_STUDY_MODELS = sp.CORE_MODELS  # the split study's own pinned list

# ---------------------------------------------------------------- files and settings

PLAN_FILE = "TEST_PLAN_2.md"
ITEMS = tuple(range(1, 9))  # the rows of the plan's table
PLAN_PLACEHOLDER = "<<"
PURPOSE = (
    "TEST_PLAN_2.md: the second scoring of the test set (items 1 to 8), one opening, which also "
    "supplies the late-set labels (2024 to 2025) that item 7 needs"
)
LABELS_FILE = sc.LABELS_FILE
REPORT = Path("results") / "test_final2.md"
JSON_REPORT = Path("results") / "test_final2.json"
ITEM7_FINAL = Path("results") / "test_final2_item7_final.md"
ITEM7_SCORECARD = Path("results") / "test_final2_item7_scorecard.md"
OUTPUT_DIR = Path("outputs") / "final2"
DRY_DIR = Path("outputs") / "final2_dry"
PREDICTIONS_NAME = "predictions.parquet"
SPLIT_CACHE_NAME = "split_study_cache.pkl"  # never the development rehearsal's cache file
REAL_OUTPUTS = (REPORT, JSON_REPORT, ITEM7_FINAL, ITEM7_SCORECARD, OUTPUT_DIR)
GENERATED = (LOG_FILE, *REAL_OUTPUTS[:-1])  # left out of the tree check (outputs/ is ignored)
IGNORED = (OUTPUT_DIR / PREDICTIONS_NAME, OUTPUT_DIR / SPLIT_CACHE_NAME)  # hold outcomes
DRY_CUT_YEAR = 2019  # the dry run scores development trials finished 2019 to 2020
JOBS = 6
DRESS_RESAMPLES = 20
DRESS_SEED = 20261112
DRESS_SPLIT_SEEDS = 1
PHASES = wr.PHASES  # ("phase2", "phase3")
NAN = float("nan")

LABELS_FAILED = "THE LABELS WERE OPENED BUT FAILED THEIR OWN CHECKS"
# Exact column names that mean an outcome. Any of them in a scored frame is a leak.
OUTCOME_COLUMNS = (*LABEL_COLUMNS, "outcome", "label", "success", "y")
MET = "met"
NOT_MET = "not met"
NOT_MET_OPPOSITE = "not met: the interval is wholly on the other side of zero"
NOT_COMPUTED = "not computed"
NO_RANGE = "no range stated in the plan"

# What each item reads as its range, where the plan states one (`scorecard.Range`).
RANGES = {
    2: sc.Range(0.005, 0.01),  # "strict version +0.005 to +0.01"
    3: sc.Range(0.015, 0.03),  # "both beats HOW by +0.015 to +0.03"
    4: sc.Range(15.0, 25.0),  # "top third beats bottom third by 15 to 25 per 100 trials"
}


class Refused(RuntimeError):
    """A check that needs no labels failed, so the test labels were not opened."""


class LabelLeak(RuntimeError):
    """An outcome, or a trial of the scored set, reached a fit or a scored frame."""


# ---------------------------------------------------------------- the plan and the tree


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    if result.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed: {result.stderr.decode().strip()}")
    return result.stdout


def plan_commit(root: Path) -> str:
    """The commit that last changed TEST_PLAN_2.md, if the file on disk is that committed text.

    The same rule as `final.plan_commit` (which is fixed to FINAL_PLAN.md).
    """
    plan = root / PLAN_FILE
    if not plan.is_file():
        raise Refused(f"{PLAN_FILE} is missing.")
    commit = _git(root, "log", "-1", "--format=%H", "--", PLAN_FILE).decode().strip()
    if not commit:
        raise Refused(f"{PLAN_FILE} has never been committed. Commit it before the run.")
    if PLAN_PLACEHOLDER in plan.read_text():
        raise Refused(f"{PLAN_FILE} still contains a '{PLAN_PLACEHOLDER}' marker.")
    if plan.read_bytes() != _git(root, "show", f"HEAD:{PLAN_FILE}"):
        raise Refused(f"{PLAN_FILE} has changed since its commit {commit[:12]}. Restore it.")
    return commit


def stray_changes(root: Path) -> list[str]:
    """`git status` lines, untracked files included, leaving out the log and this run's outputs."""
    leave_out = [f":(exclude){path}" for path in GENERATED]
    return _git(root, "status", "--porcelain", "--", ".", *leave_out).decode().splitlines()


def is_ignored(root: Path, path: Path) -> bool:
    found = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], cwd=root, capture_output=True, check=False
    )
    return found.returncode == 0


def check_ignored(root: Path, paths=IGNORED) -> None:
    """Refuse unless git ignores every file the run writes under outputs/ (they hold outcomes)."""
    if missing := [str(p) for p in paths if not is_ignored(root, p)]:
        raise Refused(
            f"Not git-ignored, and they will hold outcomes: {missing}. Ignore them first."
        )


def earlier_openings(root: Path) -> list[dict]:
    """Every logged opening made by this script (its note starts with `PURPOSE`)."""
    return [e for e in access_log(root) if e.get("note", "").startswith(PURPOSE)]


def check_first_run(root: Path, rerun: str) -> list[dict]:
    """Refuse a second run. Returns the earlier openings (empty on a first run).

    Every output file must be absent (a stale split-study cache would be reused silently). A
    declared re-run (`rerun`, the reason) is allowed only after an earlier opening by this script.
    """
    prior = earlier_openings(root)
    if prior and not rerun.strip():
        raise Refused(
            f"This script has already opened the labels {len(prior)} time(s). It runs once; a "
            "re-run must be declared with --rerun and a reason, and is reported as such."
        )
    if existing := [str(p) for p in REAL_OUTPUTS if (root / p).exists()]:
        raise Refused(
            f"Output files from an earlier run exist: {existing}. A run happens once; move them "
            "aside (keep them) before a declared re-run."
        )
    if rerun.strip() and not prior:
        raise Refused("--rerun was given but there is no earlier run to re-run.")
    return prior


def expectations(text: str) -> dict[int, tuple[str, str]]:
    """Item number -> (what is scored, expectation), verbatim from the plan's table."""
    found = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].isdigit():
            found[int(cells[0])] = (cells[1], cells[2])
    if sorted(found) != list(ITEMS):
        raise Refused(f"Could not read items 1 to {ITEMS[-1]} from the table in {PLAN_FILE}.")
    return found


# ---------------------------------------------------------------- the models


@dataclass(frozen=True)
class Group:
    """Models that share one frame: the same feature tables joined."""

    key: str
    tables: tuple[str, ...]  # joined to every trial, by id
    own_tables: tuple[str, ...]  # read by the models themselves; must exist
    specs: tuple[ModelSpec, ...]


def make_group(key: str, specs: list[ModelSpec]) -> Group:
    tables = tuple(sorted({t for s in specs for t in s.tables} | {"genetics"}))  # lead_drug
    own = tuple(sorted({t for s in specs for t in s.own_tables}))
    return Group(key, tables, own, tuple(specs))


def expected_names(hierarchy: tuple[str, ...] | None) -> list[str]:
    """`MODELS`, less the hierarchical variants a dry run left out."""
    dropped = {
        n for key, _, n in HIERARCHY_MODELS if hierarchy is not None and key not in hierarchy
    }
    return [n for n in MODELS if n not in dropped]


def model_groups(hierarchy: tuple[str, ...] | None = None) -> list[Group]:
    """The specs behind `MODELS`, from their own modules. `hierarchy`: dry run only."""
    registration = next(s for s in linear.SPECS if s.name == REGISTRATION)
    hierarchical = [
        spec
        for version in HIERARCHY_VERSIONS
        for variant, spec in zip(hm.VARIANTS, hm.make_specs(version), strict=True)
        if hierarchy is None or variant.key in hierarchy
    ]
    groups = [make_group("base", [registration, *hierarchical, *pm.SPECS])]
    for version in LINK_VERSIONS:
        specs = [*wm.make_specs(version), *wm.make_specs(version, exclude="lead")]
        groups.append(make_group(f"whathow_{version}", specs))
    return groups


def check_model_names(groups: list[Group], hierarchy: tuple[str, ...] | None) -> None:
    """The specs must be exactly the fixed list, in order. Nothing is added or dropped."""
    names = [s.name for g in groups for s in g.specs]
    if names != expected_names(hierarchy):
        extra = sorted(set(names) - set(MODELS))
        absent = sorted(set(expected_names(hierarchy)) - set(names))
        raise Refused(
            f"The models built are not the fixed list. Not in the list: {extra}. Missing: {absent}."
        )


def study_tables() -> set[str]:
    return {t for s in sp.choose_models(SPLIT_STUDY_MODELS) for t in s.tables} | {"genetics"}


# ---------------------------------------------------------------- the label guard


def check_training(train: pl.DataFrame, scored: pl.DataFrame) -> None:
    """A training frame holds development trials only, none of them a scored trial."""
    if "split" not in train.columns or set(train["split"].unique()) != {"dev"}:
        raise LabelLeak("A training frame holds trials that are not development trials.")
    if shared := set(train["nct_id"]) & set(scored["nct_id"]):
        raise LabelLeak(f"{len(shared)} scored trials are in a training frame.")


def check_blind(frame: pl.DataFrame) -> None:
    """A frame about to be predicted has no outcome column and no results-posting date.

    Fails loudly rather than dropping: a label column here means something upstream is wrong.
    `wh_posted_last` is a development-only date; a scored trial with one would mean its own
    result is known.
    """
    if present := [c for c in OUTCOME_COLUMNS if c in frame.columns]:
        raise LabelLeak(f"Label or outcome columns reached a scored frame: {present}.")
    posted = wm.whathow.column("posted_last")
    if posted in frame.columns and frame[posted].null_count() != frame.height:
        raise LabelLeak(f"Scored trials carry a results-posting date ({posted}).")


@dataclass
class Fitted:
    predictions: dict[str, np.ndarray]  # model -> probabilities, in the order of the scored trials
    seconds: dict[str, float]


def fit_and_predict(groups: list[Group], train: dict, scored: dict) -> Fitted:
    """Fit every spec on its group's training frame and predict its group's scored frame.

    `train` and `scored` map a group key to a frame. Every scored frame must list the same trials
    in the same order. A fit that does not converge stops the run (as in `final`).
    """
    order = None
    predictions, seconds = {}, {}
    for group in groups:
        fit_on, predict = train[group.key], scored[group.key]
        check_training(fit_on, predict)
        check_blind(predict)
        if order is None:
            order = predict["nct_id"].to_list()
        elif predict["nct_id"].to_list() != order:
            raise ValueError(f"The {group.key} frame lists the scored trials in another order.")
        with ev.convergence_is_an_error():
            for spec in group.specs:
                started = time.perf_counter()
                predictor = spec.fit(fit_on)
                predictions[spec.name] = final.checked(
                    spec.name, predictor.predict(predict), predict.height
                )
                seconds[spec.name] = time.perf_counter() - started
                print(f"{spec.name}: {seconds[spec.name]:.1f}s", file=sys.stderr)
    return Fitted(predictions, seconds)


# ---------------------------------------------------------------- the frames


def interim(root: Path) -> Path:
    return root / "data" / "interim"


def check_tables(root: Path, groups: list[Group], extra: set[str], ids: set[str]) -> list[Path]:
    """Every table exists, and every joined one lists every trial once. Returns the paths read."""
    paths = []
    joined = {t for g in groups for t in g.tables} | extra
    for name in sorted(joined | {t for g in groups for t in g.own_tables}):
        path = interim(root) / f"{name}.parquet"
        if not path.exists():
            raise Refused(f"The table {name} is missing ({path}).")
        paths.append(path)
    for name in sorted(joined):
        table = final.read_table(root, name)
        if table["nct_id"].n_unique() != table.height:
            raise Refused(f"The {name} table lists a trial more than once.")
        if absent := ids - set(table["nct_id"]):
            raise Refused(f"The {name} table lacks {len(absent)} of the trials.")
    return paths


class Frames(NamedTuple):
    train: dict[str, pl.DataFrame]  # group key -> labelled development trials
    scored: dict[str, pl.DataFrame]  # group key -> scored trials, no labels
    base: pl.DataFrame  # the scored trials, no labels: ids and what the items group by
    pool: pl.DataFrame  # the split study's pool: development trials, its tables joined
    scored_for_study: pl.DataFrame  # the scored trials with the split study's tables, no labels


Reader = Callable[[str], pl.DataFrame]


@dataclass
class Setting:
    """Everything the dry and the real run differ on. The code after it is shared."""

    dry: bool
    train: pl.DataFrame  # labelled development trials the models learn from
    held: pl.DataFrame  # the scored trials as the shared code sees them: no labels, split "test"
    reader: Reader  # a feature table by name, without its split column
    out: Path  # where predictions and the split cache go
    report: Path
    numbers: Path
    item7: tuple[Path, Path]
    cut_year: int  # the split study's "later" begins here
    what: str  # the scored trials, in words
    stand_in_labels: pl.DataFrame | None = None  # dry run only


def real_setting(root: Path) -> Setting:
    dev = pl.read_parquet(interim(root) / "dev.parquet")
    check_is_development(dev)
    held = pl.read_parquet(interim(root) / "test_features.parquet").filter(
        pl.col("split") == "test"
    )
    return Setting(
        False, dev, held, lambda name: final.read_table(root, name), root / OUTPUT_DIR,
        root / REPORT, root / JSON_REPORT, (ITEM7_FINAL, ITEM7_SCORECARD), TEST_START.year,
        "test trials (2021 to 2023)",
    )  # fmt: skip


def dry_setting(root: Path) -> Setting:
    """The last development round as a stand-in, made to look like test rows.

    Its labels are taken off (kept apart, as the guard would hand them over), its split reads
    "test", and the whathow tables' development-only results-posting date is blank for it as the
    tables are read: a real test row has none.
    """
    dev = pl.read_parquet(interim(root) / "dev.parquet")
    check_is_development(dev)
    first, last = validate.ROUNDS[-1]
    train_rows, scored_rows = validate.time_round(dev, first, last)
    scored = dev[scored_rows].with_columns(pl.lit("test").alias("split"))
    labels = scored.select("nct_id", "split", bl.LABEL)
    held = scored.drop([c for c in OUTCOME_COLUMNS if c in scored.columns])
    stand_in = held["nct_id"].implode()
    posted = wm.whathow.column("posted_last")

    def reader(name: str) -> pl.DataFrame:
        table = final.read_table(root, name)
        if posted in table.columns:
            table = table.with_columns(
                pl.when(pl.col("nct_id").is_in(stand_in)).then(None).otherwise(pl.col(posted))
                .alias(posted)
            )  # fmt: skip
        return table

    return Setting(
        True, dev[train_rows], held, reader, root / DRY_DIR, root / DRY_DIR / "report.md",
        root / DRY_DIR / "results.json",
        (DRY_DIR / "item7_final.md", DRY_DIR / "item7_scorecard.md"), DRY_CUT_YEAR,
        "development trials finished 2019 to 2020 (stand-in)", labels,
    )  # fmt: skip


def join(frame: pl.DataFrame, names, reader: Reader) -> pl.DataFrame:
    """Left-join each feature table by trial id, as `final.join_tables`, through `reader`."""
    for name in sorted(set(names)):
        extra = reader(name)
        if clash := (set(extra.columns) & set(frame.columns)) - {"nct_id"}:
            raise ValueError(f"Table {name!r} repeats columns already there: {sorted(clash)}")
        frame = frame.join(extra, on="nct_id", how="left", validate="1:1", maintain_order="left")
    return frame


def load_frames(root: Path, groups: list[Group], study: set[str], setting: Setting) -> Frames:
    train, held, reader = setting.train, setting.held, setting.reader
    check_tables(root, groups, study, set(train["nct_id"]) | set(held["nct_id"]))
    check_blind(held)
    train_frames, scored_frames = {}, {}
    for group in groups:
        train_frames[group.key] = join(train, group.tables, reader)
        scored_frames[group.key] = join(held, group.tables, reader)
        check_blind(scored_frames[group.key])
    base = join(held, ["genetics"], reader).select(
        "nct_id", "split", "primary_completion_date", "phase", "sponsor_class", "lead_drug"
    )
    for_study = join(held, study, reader)
    check_blind(for_study)
    return Frames(train_frames, scored_frames, base, join(train, study, reader), for_study)


def input_hashes(root: Path, extra=()) -> dict[str, str]:
    """SHA-256 of every file directly in data/interim (all the run may read) and of `extra`."""
    paths = sorted(p for p in interim(root).iterdir() if p.is_file()) + list(extra)
    return {
        str(path.relative_to(root) if path.is_relative_to(root) else path): final.file_sha256(path)
        for path in paths
        if path.exists()
    }


def labelled(frame: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """`frame` with the headline label, in the order of `frame`."""
    return final.attach_labels(frame, labels.select("nct_id", "split", bl.LABEL))


# ---------------------------------------------------------------- verdicts


def rule_verdict(low: float, high: float, sign: int = +1) -> str:
    """The plan's rule: met when the 95% interval lies wholly on the expected side of zero.

    `sign` is the side the plan expects: +1 for a gain, -1 for "fewer"."""
    if math.isnan(low) or math.isnan(high):
        return NOT_COMPUTED
    if (low > 0 and sign > 0) or (high < 0 and sign < 0):
        return MET
    if low > 0 or high < 0:
        return NOT_MET_OPPOSITE
    return NOT_MET


def range_verdict(value: float, low: float, high: float, predicted: sc.Range | None) -> str:
    """Inside or outside the plan's range (`scorecard.verdict`), or that the plan gives none."""
    return NO_RANGE if predicted is None else sc.verdict(value, low, high, predicted)


# ---------------------------------------------------------------- items 1 to 4


@dataclass
class Scored:
    """The scored trials with their outcomes, every model's predictions, and the shared draws."""

    table: pl.DataFrame  # base columns plus the label
    outcomes: np.ndarray
    predictions: dict[str, np.ndarray]
    clusters: np.ndarray
    majority: bool


def triple(value: float, interval: mt.Interval) -> dict:
    return {"value": value, "low": interval.low, "high": interval.high}


def pairs_to_score() -> list[tuple[str, str]]:
    """Every model minus the baseline, then the WHAT models minus HOW of the same links."""
    pairs = [(n, REGISTRATION) for n in MODELS if n != REGISTRATION]
    for v in LINK_VERSIONS:
        how = what_name("HOW", v)
        pairs += [(what_name(b, v), how) for b in BLOCKS if b != "HOW"]
        pairs += [(what_name(b, v, True), how) for b in EXCLUDED_BLOCKS]
    return pairs


def model_scores(s: Scored, n_resamples: int) -> dict:
    """AUROC, calibration gap (per 100), Brier and mean prediction of each model, and every
    paired difference of `pairs_to_score`, all on one set of lead-drug resamples."""
    names = [n for n in MODELS if n in s.predictions]
    pairs = [(a, b) for a, b in pairs_to_score() if a in s.predictions and b in s.predictions]
    scores, differences = mt.bootstrap(
        s.outcomes, {n: s.predictions[n] for n in names}, s.majority, pairs,
        n_resamples=n_resamples, clusters=s.clusters,
    )  # fmt: skip
    models = {
        n: {
            "auroc": triple(x.auroc, x.auroc_interval),
            "gap_per_100": triple(x.calibration_gap, x.gap_interval),
            "brier": x.brier,
            "mean_predicted": x.mean_predicted,
            "skipped_resamples": x.skipped_resamples,
        }
        for n, x in scores.items()
    }
    diffs = {
        f"{d.better} minus {d.baseline}": {
            "better": d.better,
            "baseline": d.baseline,
            "auroc": triple(d.difference, d.interval),
            "gap_per_100": triple(d.gap_difference, d.gap_difference_interval),
        }
        for d in differences
    }
    return {"models": models, "differences": diffs}


def difference(scores: dict, better: str, baseline: str = REGISTRATION) -> dict:
    return scores["differences"][f"{better} minus {baseline}"]["auroc"]


def headline(item: int, scores: dict, better: str, baseline: str = REGISTRATION) -> dict:
    d = difference(scores, better, baseline)
    return {
        "comparison": f"{better} minus {baseline}, AUROC",
        **d,
        "rule": rule_verdict(d["low"], d["high"]),
        "range": range_verdict(d["value"], d["low"], d["high"], RANGES.get(item)),
    }


def item_1(scores: dict) -> dict:
    names = [n for _, _, n in HIERARCHY_MODELS if n in scores["models"]]
    return {"headline": headline(1, scores, HEADLINE_HIERARCHY), "models": names}


def item_2(scores: dict) -> dict:
    return {"headline": headline(2, scores, HEADLINE_PRECEDENT), "models": list(PRECEDENT_MODELS)}


def item_3(scores: dict) -> dict:
    v = HEADLINE_LINKS
    what = scores["models"][what_name("WHAT", v)]["auroc"]
    return {
        "headline": headline(3, scores, what_name("HOW + WHAT", v), what_name("HOW", v)),
        "what_alone": {**what, "range": NO_RANGE},
        "models": list(WHATHOW_MODELS),
    }


def thirds_with_intervals(
    outcomes: np.ndarray, score: np.ndarray, phase: np.ndarray, weights: np.ndarray
) -> dict:
    """Success per 100 by third of `score` within each phase, top minus bottom, with intervals.

    `whathow_report.success_by_thirds` and `thirds_intervals`: thirds cut once, within the scored
    set and phase (as in development), held fixed; `weights` are lead-drug resample counts
    (`whathow_report.cluster_weights`).
    """
    parts = np.zeros(len(outcomes), dtype=int)
    point = wr.success_by_thirds(outcomes, score, parts, phase)
    found = wr.thirds_intervals(outcomes, score, parts, phase, None, weights=weights)
    out = {}
    for ph in PHASES:
        rates, intervals, (gap_low, gap_high) = found[ph]
        counts = point[ph][1]
        cells = [
            {"per_100": rates[t], "low": intervals[t][0], "high": intervals[t][1], "n": counts[t]}
            for t in range(3)
        ]
        out[ph] = {
            "thirds": cells,
            "top_minus_bottom": {"value": rates[2] - rates[0], "low": gap_low, "high": gap_high},
        }
    return out


def item_4(s: Scored, n_resamples: int) -> dict:
    phase = s.table["phase"].to_numpy()
    sponsor = s.table["sponsor_class"].to_numpy()
    parts = np.zeros(s.table.height, dtype=int)
    weights = wr.cluster_weights(s.clusters, n_resamples)  # one set of draws for every row
    by_version = {}
    for v in LINK_VERSIONS:
        how = s.predictions[what_name("HOW", v)]
        for excluded in (False, True):
            what = s.predictions[what_name("WHAT", v, excluded)]
            survived = wr.survival(s.outcomes, how, what, parts, phase, sponsor)
            shares, gaps = wr.survival_draws(
                s.outcomes, how, what, parts, s.clusters, phase, sponsor, weights=weights
            )
            by_version[what_name("WHAT", v, excluded)] = {
                "by_third": thirds_with_intervals(s.outcomes, what, phase, weights),
                "survival": {
                    **asdict(survived),
                    "share_interval": wr.percentile_pair(shares),
                    "within_strata_interval": wr.percentile_pair(gaps),
                },
            }
    lead = by_version[what_name("WHAT", HEADLINE_LINKS)]["by_third"]["phase2"]["top_minus_bottom"]
    return {
        "headline": {
            "comparison": f"Phase 2: top third minus bottom third of {what_name('WHAT', HEADLINE_LINKS)}, per 100",
            **lead,
            "rule": rule_verdict(lead["low"], lead["high"]),
            "range": range_verdict(lead["value"], lead["low"], lead["high"], RANGES[4]),
        },
        "by_model": by_version,
    }


# ---------------------------------------------------------------- item 5: the split study


def study_dict(result: sp.StudyResult) -> dict:
    """The study result with tuple keys and values made JSON-friendly."""

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        return value

    return clean(asdict(result))


def item_5(result: sp.StudyResult) -> dict:
    flattery, (low, high) = result.flattery[sp.BY_TRIAL][REGISTRATION]
    by_drug, (drug_low, drug_high) = result.flattery[sp.BY_DRUG][REGISTRATION]
    later_gap = result.gap[sp.CHRONO][REGISTRATION]
    random_gap = result.gap[sp.BY_TRIAL][REGISTRATION]
    hidden = later_gap > 0 and random_gap < later_gap
    return {
        "headline": {
            "comparison": f"{REGISTRATION}: random-by-trial holdout minus the later trials, AUROC",
            "value": flattery,
            "low": low,
            "high": high,
            "rule": rule_verdict(low, high),
            "range": NO_RANGE,
        },
        "by_drug": {"value": by_drug, "low": drug_low, "high": drug_high},
        "over_prediction": {
            "later_gap_per_100": later_gap,
            "random_by_trial_gap_per_100": random_gap,
            "random_by_drug_gap_per_100": result.gap[sp.BY_DRUG][REGISTRATION],
            "hidden": hidden,
            "verdict": (MET if hidden else NOT_MET) + " (by direction only: no interval)",
        },
        "study": study_dict(result),
    }


def run_split_study(
    study: Callable[..., sp.StudyResult],
    pool: pl.DataFrame,
    holdout: pl.DataFrame,
    cut_year: int,
    cache_file: Path | None,
    n_seeds: int = sp.N_SEEDS,
) -> sp.StudyResult:
    """The split study, development trials as the pool and the labelled scored set as the later
    holdout. The two frames are cut to their shared columns so they stack."""
    shared = [c for c in pool.columns if c in holdout.columns]
    holdout = holdout.select(shared).cast({c: pool.schema[c] for c in shared})
    return study(
        pool.select(shared),
        holdout,
        model_names=SPLIT_STUDY_MODELS,
        n_seeds=n_seeds,
        cut_year=cut_year,
        jobs=JOBS,
        cache_file=cache_file,
    )


# ---------------------------------------------------------------- item 7: the earlier results


class Earlier(Protocol):
    def prepare(self, root: Path) -> None:
        """Everything that needs no labels: the earlier models fitted, their checks passed."""
        ...

    def label_ids(self) -> pl.DataFrame:
        """(nct_id, split) of every trial whose label `finish` needs."""
        ...

    def finish(self, root: Path, labels: pl.DataFrame, rehearse: bool = False) -> dict:
        """The earlier reports with lead-drug intervals and their comparison with single-trial
        ones. `labels` holds at least the labels of `label_ids`. `rehearse`: the dress rehearsal,
        which builds no scorecard (it would read the scored period's raw records)."""
        ...


@contextlib.contextmanager
def each_trial_its_own_cluster():
    """Run the earlier code with every trial its own cluster: single-trial resampling.

    `metrics.bootstrap` and `llm_report.paired_bootstrap` with one cluster per trial draw exactly
    the indices they drew with no clusters (cluster k is trial k), so this reproduces the
    earlier, single-trial intervals with the same code. Every clustered bootstrap item 7 reaches
    takes its clusters from `lead_drug_clusters` as imported into `final`, `evaluate` or
    `scorecard` (the tests check this on made-up trials); those three are switched, and only for
    the duration. The adjusted differences resample lead drugs or sponsors in both, as the first
    run did.
    """
    modules = [m for m in (final, ev, sc) if hasattr(m, "lead_drug_clusters")]
    saved = {m: m.lead_drug_clusters for m in modules}

    def singletons(frame: pl.DataFrame) -> np.ndarray:
        return np.array([f"trial {i}" for i in range(frame.height)], dtype=object)

    for module in modules:
        module.lead_drug_clusters = singletons
    try:
        yield
    finally:
        for module, function in saved.items():
            module.lead_drug_clusters = function


INTERVAL = re.compile(r"\(([^()]*?) to ([^()]*?)\)")
NUMBER = re.compile(r"[+\-−]?\d+(?:\.\d+)?")


def _number(text: str) -> float:
    found = NUMBER.search(text.replace(",", "").replace("−", "-"))
    return float(found.group()) if found else NAN


def row_label(line: str) -> str:
    """The first cells of a table row (what the row is about), or the start of a text line."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return " | ".join(c for c in cells[:2] if not INTERVAL.search(c))[:160] or line[:160]


def compare_reports(before: str, after: str) -> dict:
    """Interval by interval, line by line: wider or narrower, and any verdict that changed.

    A changed verdict is an interval that excluded zero in one and not in the other, or a line
    whose words (intervals removed) differ, such as a scorecard verdict cell.
    """
    a, b = before.splitlines(), after.splitlines()
    if len(a) != len(b):
        return {"comparable": False, "reason": "the two reports have different lengths"}
    wider = narrower = same = 0
    ratios, flips, reworded = [], [], []
    for x, y in zip(a, b, strict=True):
        if x == y:
            same += len(INTERVAL.findall(x))
            continue
        first, second = INTERVAL.findall(x), INTERVAL.findall(y)
        if INTERVAL.sub("()", x) != INTERVAL.sub("()", y) or len(first) != len(second):
            reworded.append({"row": row_label(y), "before": x[:300], "after": y[:300]})
        for (l1, h1), (l2, h2) in zip(first, second, strict=False):
            lo1, hi1, lo2, hi2 = map(_number, (l1, h1, l2, h2))
            if any(math.isnan(v) for v in (lo1, hi1, lo2, hi2)):
                continue
            w1, w2 = hi1 - lo1, hi2 - lo2
            wider += w2 > w1
            narrower += w2 < w1
            same += w2 == w1
            if w1 > 0:
                ratios.append(w2 / w1)
            if (lo1 > 0 or hi1 < 0) != (lo2 > 0 or hi2 < 0):
                change = (
                    "excluded zero, now includes it"
                    if lo1 > 0 or hi1 < 0
                    else "included zero, now excludes it"
                )
                flips.append({"row": row_label(y), "single_trial": f"{l1} to {h1}",
                              "lead_drug": f"{l2} to {h2}", "change": change})  # fmt: skip
    return {
        "comparable": True,
        "wider": int(wider),
        "narrower": int(narrower),
        "unchanged": int(same),
        "median_width_ratio": float(np.median(ratios)) if ratios else NAN,
        "zero_exclusion_changed": flips,
        "wording_changed": reworded,
    }


def item_7_verdict(comparisons: dict) -> dict:
    """The plan's rule for item 7. Reports that cannot be compared stop the item, loudly."""
    if bad := {k: c.get("reason") for k, c in comparisons.items() if not c.get("comparable")}:
        raise ValueError(f"The single-trial and lead-drug reports cannot be compared: {bad}.")
    usable = list(comparisons.values())
    wider = sum(c["wider"] for c in usable)
    narrower = sum(c["narrower"] for c in usable)
    changed = sum(len(c["zero_exclusion_changed"]) + len(c["wording_changed"]) for c in usable)
    return {
        "wider": MET if wider > narrower else NOT_MET,
        "no_verdict_changes": MET if changed == 0 else NOT_MET,
        "intervals_wider": wider,
        "intervals_narrower": narrower,
        "lines_with_a_changed_verdict": changed,
    }


# Families added to the registry after the first run. Item 7 compares only the first run's rows.
EARLIER_LEFT_OUT = ("precedent", "hierarchical", "whathow")
EARLIER_BANNER = (
    "# Earlier results regenerated with lead-drug intervals (TEST_PLAN_2.md item 7)\n\n"
    "The first run's code and models: the model families added since "
    f"({', '.join(EARLIER_LEFT_OUT)}) are left out, so every row is a row of the first run. Only "
    "the bootstrap differs: each resample draws whole lead drugs instead of single trials."
)
SCORECARD_NOT_REHEARSED = (
    "(not built in the dress rehearsal: the scorecard reads the scored period's raw records, "
    "which it does only after the labels are opened)"
)


def earlier_specs() -> list[ModelSpec]:
    """The registry as the first run had it: every family added since is left out."""
    return [s for s in all_specs() if s.family not in EARLIER_LEFT_OUT]


@dataclass
class EarlierState:
    """Everything item 7 settles before any label: fits, frames, the scorecard's inputs."""

    fitted: final.Fitted
    train: pl.DataFrame
    frames: dict[str, pl.DataFrame]  # period -> unlabelled features
    language: final.LanguageInputs | None
    reference: dict[str, float] | None  # R1's figures (real run only)
    sc_train: pl.DataFrame
    sc_features: pl.DataFrame  # the scorecard's scored trials, unlabelled
    genetics: pl.DataFrame
    window: tuple  # the scorecard's population window


def unlabelled(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop([c for c in LABEL_COLUMNS if c in frame.columns])


class EarlierResults:
    """Item 7 through `final` and `scorecard`'s own functions. Nothing is refitted differently.

    `prepare` differs between the dry and the real run (where the frames come from); `finish` is
    one function for both.
    """

    def __init__(self, dry: bool, final_sources=None, sc_sources=sc.REAL_SOURCES):
        self.dry = dry
        self.final_sources = final_sources or final.REAL_SOURCES._replace(specs=earlier_specs)
        self.sc_sources = sc_sources

    def prepare(self, root: Path) -> None:
        if self.dry:  # the earlier rehearsal's own stand-in (`final.rehearse`), writing nothing
            dev, usable, skipped = final.development_frames(root, self.final_sources)
            train, scored = ev.time_forward_split(dev)
            frames = {final.REHEARSAL: unlabelled(scored)}
            final.check_columns(frames)
            fitted = final.fit_and_predict(usable, skipped, train, frames)
            final.check_required_models(fitted)
            language = final.language_inputs(self.final_sources, frames)
            raw = pl.read_parquet(interim(root) / "dev.parquet")
            sc_train, sc_scored = ev.time_forward_split(raw)
            genetics = pl.read_parquet(self.sc_sources.genetics)
            self.state = EarlierState(fitted, train, frames, language, None, sc_train,
                                      unlabelled(sc_scored), genetics,
                                      (sc.REHEARSAL_START, TEST_START))  # fmt: skip
            return
        p = final.prepare_real(root, self.final_sources)  # fits, checks and its dress rehearsal
        dev = pl.read_parquet(interim(root) / "dev.parquet")
        features = pl.read_parquet(interim(root) / "test_features.parquet")
        trials = pl.read_csv(root / "trials.csv")
        genetics = sc._check_without_labels(dev, features, trials, self.sc_sources)
        self.state = EarlierState(p.fitted, p.train, p.frames, p.language, final.REFERENCE_AUROC,
                                  dev, features.filter(pl.col("split") == "test"), genetics,
                                  (TEST_START, LATE_START))  # fmt: skip

    def label_ids(self) -> pl.DataFrame:
        s = self.state
        parts = [f.select("nct_id", "split") for f in s.frames.values()]
        return pl.concat([*parts, s.sc_features.select("nct_id", "split")]).unique()

    def finish(self, root: Path, labels: pl.DataFrame, rehearse: bool = False) -> dict:
        s = self.state
        periods = final.make_periods(s.fitted, s.frames, labels)
        reference = None if rehearse else s.reference
        header = [
            f"Trained {len(s.fitted.specs)} models on {s.train.height:,} development trials.",
            final.READING,
        ]

        def final_report() -> str:
            outcomes = final.run_plan(s.fitted, s.train, periods, s.language, reference)
            return final.report_text(EARLIER_BANNER, header, outcomes, [])

        clustered = {"final": final_report()}
        with each_trial_its_own_cluster():
            single = {"final": final_report()}
        if rehearse:
            clustered["scorecard"] = single["scorecard"] = SCORECARD_NOT_REHEARSED
        else:
            scored = ev.join_labels(s.sc_features, labels)
            population = sc.build_population(
                self.sc_sources.studies(), self.sc_sources.flags, *s.window
            )
            inputs = sc.assemble(s.sc_train, scored, s.genetics, population)
            clustered["scorecard"] = sc.scorecard_text(EARLIER_BANNER, [], inputs, sc.score(inputs))
            with each_trial_its_own_cluster():
                single["scorecard"] = sc.scorecard_text(
                    EARLIER_BANNER, [], inputs, sc.score(inputs)
                )
        comparisons = {k: compare_reports(single[k], clustered[k]) for k in clustered}
        return {"texts": clustered, "comparisons": comparisons}


def item_7(earlier: Earlier, root: Path, labels: pl.DataFrame, files, write: bool,
           rehearse: bool) -> dict:  # fmt: skip
    """Regenerate, write the reports, then judge. A report with a failed item, or two that cannot
    be compared, fails item 7 loudly (the reports are still written, to be read)."""
    found = earlier.finish(root, labels, rehearse=rehearse)
    if write:
        for path, text in zip(files, found["texts"].values(), strict=True):
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
    if failed := [k for k, text in found["texts"].items() if "ITEMS THAT FAILED" in text]:
        raise ValueError(f"The regenerated {failed} report has failed items; read the file.")
    verdict = item_7_verdict(found["comparisons"])
    return {"comparisons": found["comparisons"], "verdict": verdict,
            "files": [str(f) for f in files]}  # fmt: skip


# ---------------------------------------------------------------- item 8: the biology-only score

# (a) 2+ flags minus 0 flags: "2 to 10 fewer successes per 100"; (b) top third minus bottom third
# of the biology-only logistic: "5 to 15 more in the top third". Both pooled over phase.
BIOLOGY_RANGES = {"flags": sc.Range(-10.0, -2.0), "thirds": sc.Range(5.0, 15.0)}
BIOLOGY_SIGN = {"flags": -1, "thirds": +1}  # the direction the plan expects
BIOLOGY_HEADLINE = "today"  # the other two link versions are sensitivity rows
SAME_DIRECTION = "same direction"
OPPOSITE_DIRECTION = "opposite direction"
TOO_FEW = "too few trials to say"


def direction_verdict(estimate: float, sign: int) -> str:
    """The unseen-targets rows: the plan claims only the direction."""
    if math.isnan(estimate):
        return TOO_FEW
    return SAME_DIRECTION if estimate * sign > 0 else OPPOSITE_DIRECTION


@dataclass
class Biology:
    """One link version's biology score, prepared before any scored outcome is read."""

    version: str
    cuts: tuple[float, float]  # the development fitted-score tertiles
    table: pl.DataFrame  # scored trials with a mapped target: nct_id, score, n_flags, unseen
    n_train: int
    flag_coefficients: dict[str, float]


def prepare_biology(
    features: Callable[[str], pl.DataFrame],
    unseen: Callable[..., set[str]],
    train: pl.DataFrame,
    scored_ids: list[str],
) -> dict[str, Biology]:
    """Fit the biology-only logistic on every training trial with a target and score the rest.

    `train` is the labelled development frame (its label is the only outcome used); `features`
    is `biology_score.feature_frame`. The scored trials' frame carries no outcome (checked).
    """
    out = {}
    wanted = set(scored_ids)
    for version in LINK_VERSIONS:
        frame = features(version)
        check_blind(frame)
        fit_on = frame.join(train.select("nct_id", bl.LABEL), on="nct_id", how="inner")
        if wanted & set(fit_on["nct_id"]):
            raise LabelLeak("A scored trial is among the biology score's training rows.")
        with ev.convergence_is_an_error():
            fitted = bs.fit_logistic(fit_on.drop(bl.LABEL), bl.training_outcomes(fit_on))
        scored = frame.filter(pl.col("nct_id").is_in(list(wanted)))
        check_blind(scored)
        flags = bs.flag_frame(scored)
        # unseen: no target shared with ANY development training trial, not only those scored
        hidden = unseen(scored["nct_id"].to_list(), train["nct_id"].to_list(), version)
        table = scored.select("nct_id").with_columns(
            pl.Series("score", fitted.score(scored)),
            flags["n_flags"],
            pl.col("nct_id").is_in(list(hidden)).alias("unseen"),
        )
        out[version] = Biology(
            version, fitted.tertile_cuts, table, fit_on.height, fitted.flag_coefficients()
        )
    return out


def contrast_dict(c: bs.Contrast, kind: str, unseen: bool = False) -> dict:
    sign = BIOLOGY_SIGN[kind]
    found = {"value": c.estimate, "low": c.low, "high": c.high,
             "n_exposed": c.n_exposed, "n_reference": c.n_reference}  # fmt: skip
    if unseen:
        return {**found, "direction": direction_verdict(c.estimate, sign)}
    return {
        **found,
        "rule": rule_verdict(c.low, c.high, sign),
        "range": range_verdict(c.estimate, c.low, c.high, BIOLOGY_RANGES[kind]),
    }


def both_contrasts(
    y: np.ndarray, n_flags: np.ndarray, score: np.ndarray, cuts: tuple[float, float],
    phase: np.ndarray, lead: list, keep: np.ndarray, unseen: bool = False,
) -> dict:  # fmt: skip
    """(a) and (b) over the trials in `keep`, pooled over phase, with lead-drug intervals."""
    lead_kept = [v for v, k in zip(lead, keep, strict=True) if k]
    flags = bs.flag_contrast(y[keep], n_flags[keep], phase[keep], lead_kept)
    thirds = bs.thirds_contrast(y[keep], score[keep], cuts, phase[keep], lead_kept)
    return {
        "flags": contrast_dict(flags, "flags", unseen),
        "thirds": contrast_dict(thirds, "thirds", unseen),
    }


def item_8(biology: dict[str, Biology], s: Scored) -> dict:
    """(a) the flag contrast and (b) the thirds contrast, pooled over phase, for each link
    version; the within-phase group tables; the same contrasts on unseen-target trials."""
    base = s.table.select("nct_id", "phase", "lead_drug", bl.LABEL)
    versions = {}
    for version, b in biology.items():
        t = b.table.join(base, on="nct_id", how="inner", maintain_order="left")
        y = t[bl.LABEL].to_numpy().astype(bool)
        phase = t["phase"].to_numpy().astype(str)
        lead = t["lead_drug"].to_list()
        n_flags, score = t["n_flags"].to_numpy(), t["score"].to_numpy()
        unseen = t["unseen"].to_numpy()

        everyone = np.ones(len(y), dtype=bool)
        versions[version] = {
            "n_trials": len(y),
            "n_train": b.n_train,
            "cuts": list(b.cuts),
            "flag_coefficients": b.flag_coefficients,
            **both_contrasts(y, n_flags, score, b.cuts, phase, lead, everyone),
            "by_flags": bs.by_group(y, bs.flag_groups(n_flags), phase, lead).to_dicts(),
            "by_thirds": bs.by_group(y, bs.thirds(score, b.cuts), phase, lead).to_dicts(),
            "unseen": {
                "n_trials": int(unseen.sum()),
                **both_contrasts(y, n_flags, score, b.cuts, phase, lead, unseen, unseen=True),
            },
        }
    return {"headline_links": BIOLOGY_HEADLINE, "versions": versions}


# ---------------------------------------------------------------- running the items


@dataclass
class Item:
    number: int
    result: dict = field(default_factory=dict)
    failed: str = ""


def attempt(number: int, produce: Callable[[], dict]) -> Item:
    """One item. A failure is written into the report, so one bug cannot lose the others."""
    try:
        return Item(number, produce())
    except Exception as error:  # noqa: BLE001 - reported, then the run exits non-zero
        print(traceback.format_exc(), file=sys.stderr)
        detail = "".join(traceback.format_exception_only(type(error), error)).strip()
        return Item(number, failed=detail)


class Sources(NamedTuple):
    """The model list, the split study and the earlier results. The defaults are real."""

    groups: Callable[[tuple[str, ...] | None], list[Group]] = model_groups
    study: Callable[..., sp.StudyResult] = sp.run_study
    study_tables: Callable[[], set[str]] = study_tables
    earlier: Callable[[bool], Earlier] = EarlierResults
    biology_features: Callable[[str], pl.DataFrame] = bs.feature_frame
    unseen_targets: Callable[..., set[str]] = bs.unseen_targets


REAL_SOURCES = Sources()


def scored_set(frames: Frames, fitted: Fitted, labels: pl.DataFrame, majority: bool) -> Scored:
    table = labelled(frames.base, labels)
    return Scored(
        table,
        bl.training_outcomes(table),
        fitted.predictions,
        validate.lead_drug_clusters(table),
        majority,
    )


def compute_items(
    s: Scored, frames: Frames, labels: pl.DataFrame, study: Callable, cut_year: int,
    cache_file: Path | None, n_resamples: int, n_seeds: int, biology: dict[str, Biology],
) -> dict[int, Item]:  # fmt: skip
    """Items 1 to 5 and 8 from the outcomes and the predictions made before them."""
    items: dict[int, Item] = {}
    scores_item = attempt(0, lambda: model_scores(s, n_resamples))
    scores = scores_item.result
    for number, make in ((1, item_1), (2, item_2), (3, item_3)):
        if scores_item.failed:
            items[number] = Item(number, failed=f"the shared scores failed: {scores_item.failed}")
        else:
            items[number] = attempt(number, lambda make=make: make(scores))
    items[4] = attempt(4, lambda: item_4(s, n_resamples))
    holdout = labelled(frames.scored_for_study, labels)
    items[5] = attempt(
        5,
        lambda: item_5(run_split_study(study, frames.pool, holdout, cut_year, cache_file, n_seeds)),
    )
    items[8] = attempt(8, lambda: item_8(biology, s))
    items[0] = scores_item
    return items


# ---------------------------------------------------------------- the report


def show(value: float, places: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"


def cell(t: dict, places: int = 3, signed: bool = False) -> str:
    return (
        f"{show(t['value'], places, signed)} "
        f"({show(t['low'], places, signed)} to {show(t['high'], places, signed)})"
    )


def models_table(scores: dict, names: list[str], baseline: str = REGISTRATION) -> str:
    rows = []
    for n in names:
        m = scores["models"][n]
        key = f"{n} minus {baseline}"
        gain = (
            cell(scores["differences"][key]["auroc"], signed=True)
            if key in scores["differences"]
            else "(baseline)"
        )
        rows.append([n, cell(m["auroc"]), gain, cell(m["gap_per_100"], 1, signed=True),
                     show(m["brier"])])  # fmt: skip
    header = ["Model", "AUROC (95% interval)", f"AUROC minus {baseline}",
              "Calibration gap per 100 trials", "Brier"]  # fmt: skip
    return md_table(header, rows)


def verdict_block(expected: str, h: dict, places: int = 3, unit: str = "") -> str:
    signed = True
    return "\n\n".join(
        [
            f"**Expected (TEST_PLAN_2.md, verbatim):** {expected}",
            f"**Result:** {h['comparison']}: {cell(h, places, signed)}{unit}.",
            (
                f"**Verdict:** by the plan's rule (interval wholly on the expected side of "
                f"zero), **{h['rule']}**; "
                f"against the stated range, **{h['range']}**."
            ),
        ]
    )


def item_1_text(r: dict, scores: dict, expected: str) -> str:
    return "\n\n".join(
        [verdict_block(expected, r["headline"]), models_table(scores, [REGISTRATION, *r["models"]])]
    )


def item_2_text(r: dict, scores: dict, expected: str) -> str:
    return "\n\n".join(
        [verdict_block(expected, r["headline"]), models_table(scores, [REGISTRATION, *r["models"]])]
    )


def item_3_text(r: dict, scores: dict, expected: str) -> str:
    what = r["what_alone"]
    parts = [
        verdict_block(expected, r["headline"]),
        (
            f"**WHAT alone ({LINK_TEXT[HEADLINE_LINKS]}):** AUROC {cell(what)}; the plan states no "
            'range ("about 0.66"), so there is no verdict.'
        ),
    ]
    for v in LINK_VERSIONS:
        names = [what_name(b, v) for b in BLOCKS] + [what_name(b, v, True) for b in EXCLUDED_BLOCKS]
        how = what_name("HOW", v)
        rows = [
            [n, cell(scores["differences"][f"{n} minus {how}"]["auroc"], signed=True)]
            for n in names
            if f"{n} minus {how}" in scores["differences"]
        ]
        parts += [
            f"### {LINK_TEXT[v].capitalize()}",
            models_table(scores, [REGISTRATION, *names]),
            "Paired differences from HOW (same resamples):",
            md_table(["Model", f"AUROC minus {how}"], rows),
        ]
    return "\n\n".join(parts)


def item_4_text(r: dict, expected: str) -> str:
    parts = [verdict_block(expected, r["headline"], 1, " per 100 trials")]
    for name, found in r["by_model"].items():
        rows = []
        for ph in PHASES:
            p = found["by_third"][ph]
            thirds = [f"{show(c['per_100'], 1)} ({show(c['low'], 1)} to {show(c['high'], 1)}), "
                      f"n={c['n']}" for c in p["thirds"]]  # fmt: skip
            rows.append(
                [ph.replace("phase", "Phase "), *thirds, cell(p["top_minus_bottom"], 1, True)]
            )
        s = found["survival"]
        parts += [
            f"### {name}",
            md_table(["", "Bottom third", "Middle third", "Top third", "Top minus bottom"], rows),
            (
                f"Survival (`whathow_report.survival`): top third minus bottom third "
                f"{show(s['top_minus_bottom'], 1, True)} per 100; among trials alike in phase, "
                f"sponsor class and HOW third ({s['n_strata']} of {s['n_strata_total']} strata "
                f"with both tails) {show(s['within_strata'], 1, True)} per 100 "
                f"({show(s['within_strata_interval'][0], 1, True)} to "
                f"{show(s['within_strata_interval'][1], 1, True)}), so "
                f"{show(100 * s['share'], 0)}% survives ({show(100 * s['share_interval'][0], 0)}% "
                f"to {show(100 * s['share_interval'][1], 0)}%, lead-drug resamples)."
            ),
        ]
    return "\n\n".join(parts)


def item_5_text(r: dict, expected: str) -> str:
    study = r["study"]
    o = r["over_prediction"]
    rows = []
    for name in study["names"]:
        cells = [f"{show(study['auroc'][k][name])}" for k in sp.REGIMES]
        flat = [study["flattery"][k][name] for k in (sp.BY_TRIAL, sp.BY_DRUG)]
        cells += [f"{show(f[0], signed=True)} ({show(f[1][0], signed=True)} to "
                  f"{show(f[1][1], signed=True)})" for f in flat]  # fmt: skip
        cells.append(" / ".join(show(study["gap"][k][name], 1, True) for k in sp.REGIMES))
        rows.append([name, *cells])
    header = ["Model", "Later trials", "Random by trial", "Random by drug",
              "By trial minus later", "By drug minus later",
              "Gap per 100: later / by trial / by drug"]  # fmt: skip
    return "\n\n".join(
        [
            verdict_block(expected, r["headline"]),
            (
                f"**Over-prediction:** the baseline's calibration gap is "
                f"{show(o['later_gap_per_100'], 1, True)} per 100 on the later trials and "
                f"{show(o['random_by_trial_gap_per_100'], 1, True)} on a random-by-trial holdout "
                f"({show(o['random_by_drug_gap_per_100'], 1, True)} by drug). Hidden: "
                f"**{o['verdict']}**."
            ),
            (
                f"Pool {study['n_pool']:,} trials; later holdout {study['n_holdout']:,}; "
                f"{study['n_seeds']} random seeds each."
            ),
            md_table(header, rows),
        ]
    )


def item_6_text(r: dict, expected: str) -> str:
    return "\n\n".join(
        [
            f"**Expected (TEST_PLAN_2.md, verbatim):** {expected}",
            (
                f"Saved `{r['path']}`: {r['rows']:,} trials, {r['models']} models, with outcomes. "
                f"Git ignores it: {r['ignored']}. It must never be committed."
            ),
        ]
    )


def item_7_text(r: dict, expected: str) -> str:
    v = r["verdict"]
    lines = [
        f"**Expected (TEST_PLAN_2.md, verbatim):** {expected}",
        (
            "**Result:** the earlier reports regenerated by `final` and `scorecard` with lead-drug "
            "resampling, against the same code with single-trial resampling (which reproduces "
            f"the earlier intervals). Wider: {v.get('intervals_wider', 'n/a')} intervals; narrower: "
            f"{v.get('intervals_narrower', 'n/a')}; lines with a changed verdict: "
            f"{v.get('lines_with_a_changed_verdict', 'n/a')}."
        ),
        f"**Verdict:** wider, **{v['wider']}**; no verdict changes, **{v['no_verdict_changes']}**.",
        f"Full reports: {', '.join(f'`{p}`' for p in r['files'])}.",
        (
            "Only rows of the first run are compared: the model families added since "
            f"({', '.join(EARLIER_LEFT_OUT)}) are left out of the regenerated reports. The guard's "
            "one read also supplied the late-set labels (2024 to 2025) that regenerating the first "
            "run's report needs."
        ),
    ]
    for name, c in r["comparisons"].items():
        if not c.get("comparable"):
            lines.append(f"- {name}: not compared ({c.get('reason')}).")
            continue
        lines.append(
            f"- {name}: {c['wider']} wider, {c['narrower']} narrower, {c['unchanged']} unchanged; "
            f"median width ratio {show(c['median_width_ratio'], 2)}."
        )
    flips = [
        (name, f)
        for name, c in r["comparisons"].items()
        for f in c.get("zero_exclusion_changed", [])
    ]
    words = [
        (name, w) for name, c in r["comparisons"].items() for w in c.get("wording_changed", [])
    ]
    if flips or words:
        lines.append(
            f"**Which verdicts changed ({len(flips) + len(words)}).** The plan expected none; "
            "these are the rows where it did not hold."
        )
    if flips:
        lines.append(
            md_table(
                [
                    "Report",
                    "Row",
                    "Single-trial interval (earlier)",
                    "Lead-drug interval",
                    "Change",
                ],
                [[n, f["row"], f["single_trial"], f["lead_drug"], f["change"]] for n, f in flips],
            )
        )
    for name, w in words:
        lines.append(
            f"- {name}, wording changed in `{w['row']}`: before `{w['before']}`; after `{w['after']}`."
        )
    if not (flips or words):
        lines.append("No verdict changed: no interval crossed zero and no verdict wording differs.")
    return "\n\n".join(lines)


def biology_cell(c: dict) -> str:
    n = f"n = {c['n_exposed']:,} against {c['n_reference']:,}"
    if c["value"] is None or math.isnan(c["value"]):
        return f"not estimated: fewer than {bs.MIN_GROUP} trials on a side ({n})"
    return f"{cell(c, 1, True)} per 100 ({n})"


NAMES_NOTE = (
    "Within phase: each difference is computed inside Phase 2 and inside Phase 3 and pooled, so "
    "phase cannot explain it; the phases are also shown apart below. Under the 2020 names and "
    "links version there are no chemistry columns, so two of the six flags (drug activates its "
    "target; low drug-likeness) cannot fire."
)


def item_8_text(r: dict, expected: str) -> str:
    head = r["versions"][r["headline_links"]]
    names = {
        "flags": "(a) two or more flags minus none",
        "thirds": "(b) top third minus bottom third",
    }
    parts = [
        f"**Expected (TEST_PLAN_2.md, verbatim):** {expected}",
        NAMES_NOTE,
    ]
    for kind, name in names.items():
        c = head[kind]
        parts.append(
            f"**{name}, {LINK_TEXT[r['headline_links']]}, pooled over phase:** {biology_cell(c)}. "
            f"**Verdict:** by the plan's rule (interval wholly on the expected side of zero), "
            f"**{c['rule']}**; against the stated range, **{c['range']}**."
        )
    rows = []
    for version, v in r["versions"].items():
        for kind, name in names.items():
            rows.append([LINK_TEXT[version], name, biology_cell(v[kind]), v[kind]["rule"],
                         v[kind]["range"]])  # fmt: skip
    parts += [
        "Every link version (today's is the headline; the other two are sensitivity rows):",
        md_table(["Links", "Contrast", "Estimate (95% interval)", "Interval", "Range"], rows),
    ]
    unseen_rows = []
    for version, v in r["versions"].items():
        u = v["unseen"]
        for kind, name in names.items():
            unseen_rows.append([LINK_TEXT[version], name, str(u["n_trials"]),
                                biology_cell(u[kind]), u[kind]["direction"]])  # fmt: skip
    parts += [
        "Trials whose targets appear in no development trial (the plan claims only the direction):",
        md_table(
            ["Links", "Contrast", "Trials", "Estimate (95% interval)", "Direction"], unseen_rows
        ),
    ]
    for version, v in r["versions"].items():
        for key, title in (("by_flags", "flag count"), ("by_thirds", "third of the score")):
            table = [[g["phase"], g["group"], f"{g['n']:,}",
                      f"{show(g['per_100'], 1)} ({show(g['low'], 1)} to {show(g['high'], 1)})"]
                     for g in v[key]]  # fmt: skip
            parts += [
                f"### {LINK_TEXT[version].capitalize()}: success per 100 by {title}, within phase",
                md_table(["Phase", "Group", "Trials", "Success per 100 (95% interval)"], table),
            ]
        cuts = v["cuts"]
        parts.append(
            f"{v['n_trials']:,} scored trials with a mapped target; logistic fitted on "
            f"{v['n_train']:,} development trials; thirds cut at {cuts[0]:.3f} and {cuts[1]:.3f}."
        )
    return "\n\n".join(parts)


def report_text(banner: str, header: list[str], items: dict[int, Item], plan: dict) -> str:
    scores = items[0].result
    texts = {
        1: lambda r: item_1_text(r, scores, plan[1][1]),
        2: lambda r: item_2_text(r, scores, plan[2][1]),
        3: lambda r: item_3_text(r, scores, plan[3][1]),
        4: lambda r: item_4_text(r, plan[4][1]),
        5: lambda r: item_5_text(r, plan[5][1]),
        6: lambda r: item_6_text(r, plan[6][1]),
        7: lambda r: item_7_text(r, plan[7][1]),
        8: lambda r: item_8_text(r, plan[8][1]),
    }
    parts = [banner, *header]
    failed = [str(n) for n in ITEMS if items[n].failed]
    if failed:
        parts.insert(1, f"**ITEMS THAT FAILED: {', '.join(failed)}.** The rest are below.")
    for number in ITEMS:
        parts.append(f"## Item {number}: {plan[number][0]}")
        item = items[number]
        if item.failed:
            parts.append(f"**This item failed and was not computed.** {item.failed}")
            continue
        try:
            parts.append(texts[number](item.result))
        except Exception as error:  # noqa: BLE001 - the numbers are still in the JSON
            parts.append(f"**This item's text failed; its numbers are in the JSON.** {error!r}")
    return "\n\n".join(parts) + "\n"


READING = (
    "How to read this. Differences are first-named minus second-named, AUROC to three places, "
    "taken within each resample (paired). Calibration gap is mean predicted minus observed "
    "success, per 100 trials (positive: too optimistic). Intervals are 95% percentile intervals "
    "from 1,000 resamples of lead drugs (all of a drug's trials together; a trial with no known "
    "lead drug is its own); trials sharing a sponsor are still treated as independent, so they "
    "are somewhat too narrow. A result is met when its interval is above zero."
)
DRY_BANNER = (
    "# DRY RUN on development data. This is NOT the test set.\n\n"
    "**Every number below is from development trials.** Trials finished 2019 to 2020 stand in "
    "for the test set; every model was trained on development trials finished before 2019. "
    "Item 7 uses the earlier rehearsal's stand-in (trained before 2017, scored 2017 to 2020). "
    "The guard was not called. This checks the code; it says nothing about the plan."
)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------- the run

DRY_DIFFERENCES = (
    "What differs from the real run: the frames and labels (a development stand-in, made to look "
    "like test rows), no guard and no git checks, item 7's stand-in (the earlier rehearsal's "
    "split, trained before 2017 and scored 2017 to 2020) and its scorecard window, R1's reference "
    "figures (not checked), and where files go. Everything from the labels in hand to the last "
    "file written is the same function (`shared_tail`)."
)


@dataclass
class Prepared:
    groups: list[Group]
    frames: Frames
    fitted: Fitted
    majority: bool
    earlier: Earlier
    biology: dict[str, Biology]
    seconds: float


@dataclass
class Context:
    """What the run knows before the labels: where it is, what it must say about itself."""

    root: Path
    setting: Setting
    sources: Sources
    plan: dict
    head: str = ""
    commit: str = ""
    note: str = ""
    rerun: str = ""
    prior: list = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


def prepare(root: Path, sources: Sources, setting: Setting, hierarchy) -> Prepared:
    """Everything that needs no labels: frames, every fit and every prediction."""
    started = time.perf_counter()
    groups = sources.groups(hierarchy)
    check_model_names(groups, hierarchy)
    frames = load_frames(root, groups, sources.study_tables(), setting)
    fitted = fit_and_predict(groups, frames.train, frames.scored)
    majority = mt.majority_class_of(bl.training_outcomes(frames.train["base"]))
    biology = prepare_biology(
        sources.biology_features,
        sources.unseen_targets,
        frames.train["base"],
        frames.base["nct_id"].to_list(),
    )
    earlier = sources.earlier(setting.dry)
    earlier.prepare(root)
    return Prepared(
        groups, frames, fitted, majority, earlier, biology, time.perf_counter() - started
    )


def check_before(root: Path, note: str, hierarchy, rerun: str) -> tuple[str, str, dict, list]:
    """The checks of the real run that need no fit: HEAD, the plan commit, the plan, earlier runs."""
    if hierarchy is not None:
        raise Refused("The real run fits every model in MODELS; --hierarchy is for the dry run.")
    if not note.strip():
        raise ValueError("Give a note saying why the test labels are being opened.")
    commit = plan_commit(root)
    if stray := stray_changes(root):
        raise Refused("The working tree has uncommitted changes:\n" + "\n".join(stray))
    if not prereg_locked(root) or not prereg_unchanged(root):
        raise HeldoutLocked("The pre-registration is not locked, or has changed since its tag.")
    prior = check_first_run(root, rerun)
    check_ignored(root)
    plan = expectations((root / PLAN_FILE).read_text())
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    return head, commit, plan, prior


def check_again(ctx: Context) -> None:
    """Just before the opening: the plan, the tree, the pre-registration and the outputs again."""
    if plan_commit(ctx.root) != ctx.commit:
        raise Refused(f"{PLAN_FILE} was committed again while the models were fitted.")
    if stray := stray_changes(ctx.root):
        raise Refused("The working tree has uncommitted changes:\n" + "\n".join(stray))
    if _git(ctx.root, "rev-parse", "HEAD").decode().strip() != ctx.head:
        raise Refused("HEAD moved while the models were fitted.")
    if not prereg_locked(ctx.root) or not prereg_unchanged(ctx.root):
        raise HeldoutLocked("The pre-registration is not locked, or has changed since its tag.")
    check_first_run(ctx.root, ctx.rerun)


def opening_note(note: str, rerun: str) -> str:
    return f"{PURPOSE}. " + (f"RE-RUN: {rerun.strip()}. " if rerun.strip() else "") + note


def made_up_labels(ids: pl.DataFrame) -> pl.DataFrame:
    """Seeded random labels for the given trials, for the dress rehearsal. They mean nothing."""
    values = np.random.default_rng(DRESS_SEED).random(ids.height) < 0.5
    return ids.select("nct_id", "split").with_columns(pl.Series(bl.LABEL, values))


def predictions_frame(s: Scored, biology: dict[str, Biology]) -> pl.DataFrame:
    """Item 6: one row per scored trial, its outcome, every model's prediction, and item 8's
    biology score, flag count, flag group and third for each link version (blank: no target)."""
    columns = {n: s.predictions[n] for n in MODELS if n in s.predictions}
    frame = s.table.with_columns(
        pl.Series("outcome", s.outcomes), *(pl.Series(n, p) for n, p in columns.items())
    )
    for version, b in biology.items():
        t = b.table
        extra = t.select(
            "nct_id",
            pl.col("score").alias(f"biology {version}: score"),
            pl.col("n_flags").alias(f"biology {version}: flags"),
            pl.Series(f"biology {version}: flag group", bs.flag_groups(t["n_flags"].to_numpy())),
            pl.Series(f"biology {version}: third", bs.thirds(t["score"].to_numpy(), b.cuts)),
            pl.col("unseen").alias(f"biology {version}: unseen targets"),
        )
        frame = frame.join(extra, on="nct_id", how="left", maintain_order="left")
    return frame


def header_lines(ctx: Context, p: Prepared, s: Scored, dress: float) -> list[str]:
    st = ctx.setting
    lines = [
        "Plan: the copy of TEST_PLAN_2.md on disk (dry run)."
        if st.dry
        else (
            f"Plan: `{PLAN_FILE}` at commit `{ctx.commit}`, unchanged since. Code: commit "
            f"`{ctx.head}`, clean tree, checked at the start and again just before the opening."
        )
    ]
    if ctx.prior:
        earlier = "; ".join(f"{e['at']} ({e['note']})" for e in ctx.prior)
        lines.append(
            f"**This is a re-run.** Reason: {ctx.rerun.strip()}. Earlier openings by this "
            f"script: {earlier}. Their output files were moved aside and are kept."
        )
    rate = 100 * float(s.outcomes.mean())
    lines += [
        (
            f"Scored: {s.table.height:,} {st.what}, observed success {rate:.1f} per 100. Trained "
            f"on {p.frames.train['base'].height:,} development trials."
        ),
        (
            f"Models: {len(s.predictions)} (the fixed list in `final2.MODELS`). Fitting and "
            f"predicting took {p.seconds:.0f} seconds, before any label was read; the dress "
            f"rehearsal of the whole shared path on made-up labels {dress:.0f} seconds."
        ),
    ]
    if not st.dry:
        lines.append(f"Access note: {opening_note(ctx.note, ctx.rerun)}")
    lines += [sc.access_section(ctx.root), READING]
    if st.dry:
        lines.append(DRY_DIFFERENCES)
    lines.append("The SHA-256 of every input file is in the JSON (`input_sha256`).")
    return lines


def shared_tail(
    ctx: Context, p: Prepared, labels: pl.DataFrame, *, write: bool, n_resamples: int,
    n_seeds: int, cache_file: Path | None, rehearse: bool, dress: float = 0.0,
    extra_inputs=(),
) -> tuple[str, dict]:  # fmt: skip
    """From the labels in hand to the last file written: one function for the dry run, the real
    run and the dress rehearsal (`write=False`, which writes nothing)."""
    st = ctx.setting
    try:
        s = scored_set(p.frames, p.fitted, labels, p.majority)
    except Exception as error:
        if write:
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            st.report.parent.mkdir(parents=True, exist_ok=True)
            st.report.write_text(f"# {LABELS_FAILED}\n\n**{LABELS_FAILED}: {detail}** "
                                 "Nothing was scored.\n")  # fmt: skip
        raise
    items = compute_items(s, p.frames, labels, ctx.sources.study, st.cut_year, cache_file,
                          n_resamples, n_seeds, p.biology)  # fmt: skip

    def save_predictions() -> dict:
        frame = predictions_frame(s, p.biology)
        target = st.out / PREDICTIONS_NAME
        if write:
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target)
        relative = target.relative_to(ctx.root)
        return {"path": str(relative), "rows": frame.height, "models": len(s.predictions),
                "biology_columns": [c for c in frame.columns if c.startswith("biology ")],
                "ignored": is_ignored(ctx.root, relative)}  # fmt: skip

    items[6] = attempt(6, save_predictions)
    items[7] = attempt(7, lambda: item_7(p.earlier, ctx.root, labels, st.item7, write, rehearse))
    header = header_lines(ctx, p, s, dress)
    banner = DRY_BANNER if st.dry else "# The second scoring of the test set (TEST_PLAN_2.md)"
    text = report_text(banner, header, items, ctx.plan)
    numbers = {
        "dry_run": st.dry,
        "rehearsal_on_made_up_labels": not write,
        "plan_commit": ctx.commit,
        "code_commit": ctx.head,
        "re_run": {"reason": ctx.rerun, "earlier_openings": ctx.prior} if ctx.prior else None,
        "purpose": PURPOSE,
        "scored": {"n": s.table.height, "success_per_100": 100 * float(s.outcomes.mean()),
                   "what": st.what},
        "n_train": p.frames.train["base"].height,
        "models": list(s.predictions),
        "seconds": {"fit_and_predict": p.seconds, "dress": dress,
                    "total": time.perf_counter() - ctx.started, "per_model": p.fitted.seconds},
        "access_log_entries": len(access_log(ctx.root)),
        "input_sha256": input_hashes(ctx.root, extra_inputs),
        "scores": items[0].result,
        "items": {str(k): {"failed": i.failed, **i.result} for k, i in items.items() if k},
        "plan": {str(k): {"what": w, "expectation": e} for k, (w, e) in ctx.plan.items()},
    }  # fmt: skip
    blob = json.dumps(jsonable(numbers), indent=1)
    if write:
        st.report.parent.mkdir(parents=True, exist_ok=True)
        st.report.write_text(text)
        st.numbers.write_text(blob)
    return text, numbers


def dress_rehearsal(ctx: Context, p: Prepared) -> float:
    """`shared_tail` once on made-up labels, few resamples, one split-study seed, no cache,
    writing nothing. Refuses if any item fails."""
    started = time.perf_counter()
    ids = pl.concat([p.frames.base.select("nct_id", "split"), p.earlier.label_ids()]).unique(
        maintain_order=True
    )
    with final.few_resamples(DRESS_RESAMPLES):
        _, numbers = shared_tail(
            ctx, p, made_up_labels(ids), write=False, n_resamples=DRESS_RESAMPLES,
            n_seeds=DRESS_SPLIT_SEEDS, cache_file=None, rehearse=True,
        )  # fmt: skip
    if failed := [f"item {k}: {v['failed']}" for k, v in numbers["items"].items() if v["failed"]]:
        raise Refused("The dress rehearsal on made-up labels failed:\n" + "\n".join(failed))
    return time.perf_counter() - started


def development_labels(root: Path) -> pl.DataFrame:
    return pl.read_parquet(interim(root) / "dev.parquet").select("nct_id", "split", bl.LABEL)


def run(
    root: Path | None = None,
    note: str = "",
    dry: bool = False,
    hierarchy: tuple[str, ...] | None = None,
    sources: Sources = REAL_SOURCES,
    rerun: str = "",
) -> str:
    """The whole run. `dry`: development stand-in, no guard, outputs under outputs/final2_dry/."""
    root = root or (ROOT if dry else repo_root())
    if dry:
        if (root / DRY_DIR).exists():  # this script's own dry outputs: a stale cache is not reused
            shutil.rmtree(root / DRY_DIR)
        ctx = Context(
            root, dry_setting(root), sources, expectations((root / PLAN_FILE).read_text())
        )
    else:
        head, commit, plan, prior = check_before(root, note, hierarchy, rerun)
        ctx = Context(root, real_setting(root), sources, plan, head, commit, note, rerun, prior)
    p = prepare(root, sources, ctx.setting, hierarchy)
    dress = dress_rehearsal(ctx, p)
    if dry:
        labels = pl.concat([ctx.setting.stand_in_labels, development_labels(root)])
        extra = []
    else:
        check_again(ctx)
        path = open_heldout(LABELS_FILE, opening_note(note, rerun), root=root)  # THE one read
        extra = [path, root / "trials.csv"]
        try:  # the test and the late labels, from the one read
            labels = final.read_labels(path, pl.read_csv(root / "trials.csv"))
        except Exception as error:
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            ctx.setting.report.parent.mkdir(parents=True, exist_ok=True)
            ctx.setting.report.write_text(
                f"# {LABELS_FAILED}\n\n**{LABELS_FAILED}: {detail}** Nothing was scored.\n"
            )
            raise
    text, _ = shared_tail(
        ctx, p, labels, write=True, n_resamples=mt.N_RESAMPLES, n_seeds=sp.N_SEEDS,
        cache_file=ctx.setting.out / SPLIT_CACHE_NAME, rehearse=False, dress=dress,
        extra_inputs=extra,
    )  # fmt: skip
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="development stand-in; no guard")
    mode.add_argument("--real", action="store_true", help="THE run on the test set (logged)")
    parser.add_argument("--note", default="", help="why the test labels are opened (real run)")
    parser.add_argument("--rerun", default="", help="real run only: a declared re-run, and why")
    parser.add_argument("--hierarchy", nargs="+", choices=list(hm.VARIANT),
                        help="dry run only: the hierarchical variants to fit")  # fmt: skip
    args = parser.parse_args()
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    hierarchy = tuple(args.hierarchy) if args.hierarchy else None
    text = run(note=args.note, dry=args.dry_run, hierarchy=hierarchy, rerun=args.rerun)
    print(text)
    if "ITEMS THAT FAILED" in text:
        sys.exit(1)


if __name__ == "__main__":
    main()
