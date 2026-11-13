import ast
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials
from test_evaluate import git, lock, write_synthetic_project

from elman_exercise import baselines as bl
from elman_exercise import evaluate as ev
from elman_exercise import final, heldout
from elman_exercise import llm_predictor as lp
from elman_exercise import llm_report as lr
from elman_exercise import metrics as mt
from elman_exercise.endpoint import BIOMARKER, SURVIVAL
from elman_exercise.models import drift, linear, llm, trees
from elman_exercise.models.registry import ModelSpec

PLAN_TEXT = "# Final plan\n\nP1 and the rest, written before the run.\n"
ITEMS = list(final.ITEMS)


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    monkeypatch.setattr(mt, "N_RESAMPLES", 30)


# ---------------------------------------------------------------- a small made-up project


def raw_study(nct: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"A trial {nct}"},
            "designModule": {"phases": ["PHASE2"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme", "class": "INDUSTRY"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "Zorbitrex"}],
                "armGroups": [
                    {
                        "label": "Zorbitrex",
                        "type": "EXPERIMENTAL",
                        "interventionNames": ["Drug: Zorbitrex"],
                    }
                ],
            },
        }
    }


def fake_records(ids) -> dict[str, dict]:
    return {i: lp.extract_record(raw_study(i)) for i in ids}


def fake_specs() -> list[ModelSpec]:
    """The registry's role, small: every name the plan reads, plus one of each kind of oddity."""
    base = linear.SPECS[0]
    boosted = trees.SPECS[0]
    trend = next(s for s in drift.SPECS if s.name == final.TREND_MODEL)
    plain = [
        replace(base, name=n, family="fake")
        for n in final.REQUIRED_MODELS
        if n
        not in (
            final.REGISTRATION_NAME,
            final.TREND_MODEL,
            final.PRIMARY_MODEL,
            final.CONTROL_MODEL,
        )
    ]
    return [
        base,
        trend,
        replace(boosted, name=final.PRIMARY_MODEL, family="fake"),
        *plain,
        replace(base, name="fake: leaky", family="fake", honest=False),
        replace(base, name="fake: partial", family="fake", ranked=False, not_ranked_reason="r"),
        replace(base, name="fake: needs a table", family="fake", tables=("nonexistent",)),
        llm.SPECS[0],
    ]


class FinalFake:
    """A temporary repo with made-up trials, feature tables, a plan and language-model files."""

    def __init__(self, root: Path):
        write_synthetic_project(root)
        self.root = root
        self.interim = root / "data" / "interim"
        self.trials = pl.read_csv(root / "trials.csv")
        self.labels = pl.read_parquet(root / "data" / "heldout" / "test_labels.parquet")
        self.write_tables()
        self.write_language_files()
        self.sources = final.Sources(
            specs=fake_specs,
            sample=self.interim / "llm_sample.csv",
            predictions=self.interim / "llm_predictions.parquet",
            records=fake_records,
        )

    def write_tables(self) -> None:
        rng = np.random.default_rng(5)
        ids, split = self.trials["nct_id"], self.trials["split"]
        n = self.trials.height

        def table(name: str, **columns) -> None:
            pl.DataFrame({"nct_id": ids, "split": split, **columns}).write_parquet(
                self.interim / f"{name}.parquet"
            )

        kinds = [BIOMARKER, SURVIVAL, "symptom or function scale"]
        table("endpoint", endpoint_type=rng.choice(kinds, n).tolist())
        table(
            "history",
            prior_phase2_labelled=(rng.random(n) < 0.4).tolist(),
            prior_phase2_succeeded=[None if rng.random() < 0.3 else bool(rng.random() < 0.6)
                                    for _ in range(n)],
        )  # fmt: skip
        table(
            "genetics",
            lead_drug=[None if rng.random() < 0.2 else f"D{rng.integers(0, 40)}" for _ in range(n)],
            supported_at_start=[None if rng.random() < 0.3 else bool(rng.random() < 0.4)
                                for _ in range(n)],
        )  # fmt: skip
        table("text", summary=["words"] * n)
        table(
            "science",
            n_target_genes=np.where(rng.random(n) < 0.9, 1, 3).tolist(),
            has_target=(rng.random(n) < 0.7).tolist(),
            expr_n_tissues=rng.integers(1, 52, n).tolist(),
        )

    def write_language_files(self) -> None:
        rng = np.random.default_rng(6)
        outcomes = dict(zip(self.labels["nct_id"], self.labels[bl.LABEL], strict=True))
        dev = synthetic_trials(500, seed=1)
        outcomes |= dict(zip(dev["nct_id"], dev[bl.LABEL], strict=True))
        late_dev = dev.filter(pl.col("primary_completion_date").dt.year() >= 2017)
        chosen = {
            "dev": late_dev["nct_id"].to_list()[:40],
            "test": self.trials.filter(pl.col("split") == "test")["nct_id"].to_list()[:40],
            "late": self.trials.filter(pl.col("split") == "late")["nct_id"].to_list()[:30],
        }
        sample, rows = [], []
        for split, ids in chosen.items():
            for position, nct in enumerate(ids, start=1):
                sample.append({"nct_id": nct, "split": split, "period": "p", "post_hoc": False,
                               "batch": 1, "position": position})  # fmt: skip
                for model in lp.MODELS:
                    for variant in lp.VARIANTS:
                        p = 0.5 + (0.3 if outcomes[nct] else -0.3) * rng.random()
                        rows.append({
                            "nct_id": nct, "model": model, "variant": variant, "run": 0,
                            "probability": float(p), "recognised": bool(rng.random() < 0.5),
                            "drug_guess": "Zorbitrex" if rng.random() < 0.4 else None,
                            "reason": "a reason", "model_id": "m", "batch": 1,
                            "position": position, "prompt_sha256": "x", "error": None,
                        })  # fmt: skip
        pl.DataFrame(sample).write_csv(self.interim / "llm_sample.csv")
        pl.DataFrame(rows, schema=lp.TABLE_SCHEMA).write_parquet(
            self.interim / "llm_predictions.parquet"
        )
        self.chosen = chosen

    def write_plan(self) -> None:
        (self.root / "FINAL_PLAN.md").write_text(PLAN_TEXT)

    def commit_plan(self) -> None:
        self.write_plan()
        git(self.root, "add", "FINAL_PLAN.md")
        git(self.root, "commit", "-q", "-m", "the plan")


@pytest.fixture
def fake(tmp_path):
    return FinalFake(tmp_path)


@pytest.fixture
def ready(fake, monkeypatch):
    """Locked, with the plan committed: the real run may go ahead.

    The reference figures belong to the real data, so here R1 has nothing to compare with; the
    tests of R1 say what they use.
    """
    monkeypatch.setattr(final, "REFERENCE_AUROC", {})
    lock(fake.root)
    fake.commit_plan()
    return fake


def assert_nothing_opened(fake) -> None:
    assert heldout.access_log(fake.root) == []
    assert not (fake.root / final.TEST_REPORT).exists()


# ---------------------------------------------------------------- the real run


def test_the_real_run_makes_one_logged_access_reads_both_splits_and_covers_every_plan_item(
    ready, monkeypatch
):
    opened = []

    def counting(*args, **kwargs):
        opened.append(args)
        return heldout.open_heldout(*args, **kwargs)

    monkeypatch.setattr(final, "open_heldout", counting)
    text = final.run_test("final run", ready.root, ready.sources)
    assert len(opened) == 1
    (entry,) = heldout.access_log(ready.root)
    assert entry["note"] == "final run"
    assert entry["path"] == "data/heldout/test_labels.parquet"
    assert (ready.root / final.TEST_REPORT).read_text() == text
    for item in ITEMS:
        assert f"## {item} (" in text
    assert "ITEMS THAT FAILED" not in text
    assert "150 trials, observed success" in text  # the test split
    assert "60 trials, observed success" in text  # and the late split, from the same opening
    assert "### Test and late sets together" in text and "210 trials, success" in text
    assert "Access note: final run" in text
    assert "opened 1 time(s), counting this run" in text
    plan = final.plan_commit(ready.root)
    assert f"`{plan}`" in text
    assert "Trained" in text and "before the labels were opened" in text


def test_the_late_set_gets_the_primary_and_ladder_comparisons(ready):
    text = final.run_test("late", ready.root, ready.sources)
    late = text.split("## S5 (secondary)")[1].split("## E1")[0]
    assert late.count("| S5 |") == 7 + len(
        final.RUNGS
    )  # gain, then AUROC, Brier, gap of two models
    assert f"{final.PRIMARY_MODEL} minus {final.REGISTRATION_NAME}" in late
    assert "Brier score of" in late and "calibration gap of" in late
    assert "492" not in late and "60 trials" in late


def test_the_language_models_are_scored_on_the_sampled_test_and_late_trials(ready):
    text = final.run_test("llm", ready.root, ready.sources)
    section = text.split("## E4 (exploratory)")[1].split("## E5")[0]
    assert "### Sampled test trials" in section and "### Sampled late trials" in section
    assert "40 sampled trials" in section and "30 sampled trials" in section
    assert "AUROC of sonnet (blinded)" in section
    assert "sonnet (identified) minus sonnet (blinded)" in section
    assert "guess names the drug" in section


def test_a_missing_model_row_or_fit_failure_is_reported_not_hidden(ready):
    text = final.run_test("skips", ready.root, ready.sources)
    assert "Skipped (table missing: nonexistent): fake: needs a table." in text
    assert "scored on the sampled trials only: see E4" in text
    assert "Not ranked here (r): fake: partial." in text


def test_a_dirty_working_tree_stops_the_run_before_anything_is_opened(ready):
    (ready.root / "stray.py").write_text("print('changed')\n")
    with pytest.raises(final.Refused, match="uncommitted"):
        final.run_test("dirty", ready.root, ready.sources)
    assert_nothing_opened(ready)


def test_a_missing_plan_stops_the_run(fake):
    lock(fake.root)
    with pytest.raises(final.Refused, match="missing"):
        final.run_test("no plan", fake.root, fake.sources)
    assert_nothing_opened(fake)


def test_a_plan_that_was_never_committed_stops_the_run(fake):
    lock(fake.root)
    fake.write_plan()  # on disk, untracked
    with pytest.raises(final.Refused, match="never been committed"):
        final.plan_commit(fake.root)
    with pytest.raises(final.Refused, match="never been committed"):
        final.run_test("untracked plan", fake.root, fake.sources)
    assert_nothing_opened(fake)
    git(fake.root, "add", "FINAL_PLAN.md")
    with pytest.raises(final.Refused, match="never been committed"):  # staged is not committed
        final.plan_commit(fake.root)
    with pytest.raises(final.Refused, match="never been committed"):
        final.run_test("staged plan", fake.root, fake.sources)
    assert_nothing_opened(fake)


def test_a_plan_changed_since_its_commit_stops_the_run(ready):
    (ready.root / "FINAL_PLAN.md").write_text(PLAN_TEXT + "\nAn edit after the commit.\n")
    with pytest.raises(final.Refused, match="changed since its commit"):
        final.run_test("changed plan", ready.root, ready.sources)
    assert_nothing_opened(ready)


def test_the_plan_check_says_which_problem_it_found(ready):
    assert len(final.plan_commit(ready.root)) == 40
    (ready.root / "FINAL_PLAN.md").write_text("different\n")
    with pytest.raises(final.Refused, match="changed since its commit"):
        final.plan_commit(ready.root)


def test_before_the_tag_the_real_run_is_refused_and_logs_nothing(fake):
    fake.commit_plan()
    with pytest.raises(heldout.HeldoutLocked):
        final.run_test("too early", fake.root, fake.sources)
    assert_nothing_opened(fake)


def test_an_empty_note_is_refused_before_anything_is_opened(ready):
    with pytest.raises(ValueError, match="note"):
        final.run_test("  ", ready.root, ready.sources)
    assert_nothing_opened(ready)


def drop_a_trial_from_a_table(fake):
    path = fake.interim / "endpoint.parquet"
    table = pl.read_parquet(path)
    first_test = fake.trials.filter(pl.col("split") == "test")["nct_id"][0]
    table.filter(pl.col("nct_id") != first_test).write_parquet(path)


def drop_a_feature_column(fake):
    path = fake.interim / "test_features.parquet"
    pl.read_parquet(path).drop("modality").write_parquet(path)


def a_model_predicts_nan(fake):
    def nan_fit(train):
        class Nan:
            def predict(self, frame):
                return np.full(frame.height, np.nan)

        return Nan()

    broken = replace(linear.SPECS[0], name="fake: nan", family="fake", fit=nan_fit)
    specs = fake.sources.specs
    fake.sources = fake.sources._replace(specs=lambda: [*specs(), broken])


def a_fit_does_not_converge(fake):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    def slow_fit(train):
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
        return linear.SPECS[0].fit(train)

    broken = replace(linear.SPECS[0], name="fake: slow", family="fake", fit=slow_fit)
    specs = fake.sources.specs
    fake.sources = fake.sources._replace(specs=lambda: [*specs(), broken])


def a_required_model_is_missing(fake):
    specs = fake.sources.specs
    fake.sources = fake.sources._replace(
        specs=lambda: [s for s in specs() if s.name != final.TREND_MODEL]
    )


def a_sampled_test_trial_has_no_prediction(fake):
    path = fake.interim / "llm_predictions.parquet"
    table = pl.read_parquet(path)
    table.filter(
        ~((pl.col("nct_id") == fake.chosen["test"][0]) & (pl.col("model") == "haiku"))
    ).write_parquet(path)


def a_sampled_late_trial_is_not_in_the_features(fake):
    sample = pl.read_csv(fake.interim / "llm_sample.csv")
    stray = (
        sample.filter(pl.col("split") == "late").head(1).with_columns(nct_id=pl.lit("NCT9999999"))
    )
    pl.concat([sample, stray]).write_csv(fake.interim / "llm_sample.csv")


def the_records_are_not_in_the_raw_download(fake):
    def refuse(ids):
        raise ValueError("sampled trials are not in the raw download")

    fake.sources = fake.sources._replace(records=refuse)


def a_table_is_missing(fake):
    (fake.interim / "history.parquet").unlink()


def a_probability_is_out_of_range(fake):
    path = fake.interim / "llm_predictions.parquet"
    table = pl.read_parquet(path)
    first = fake.chosen["late"][0]
    table.with_columns(
        pl.when(pl.col("nct_id") == first)
        .then(1.5)
        .otherwise(pl.col("probability"))
        .alias("probability")
    ).write_parquet(path)


def a_probability_is_not_finite(fake):
    path = fake.interim / "llm_predictions.parquet"
    table = pl.read_parquet(path)
    first = fake.chosen["test"][0]
    table.with_columns(
        pl.when(pl.col("nct_id") == first)
        .then(float("nan"))
        .otherwise(pl.col("probability"))
        .alias("probability")
    ).write_parquet(path)


def a_sampled_trial_has_no_protocol_record(fake):
    keep = fake.sources.records
    fake.sources = fake.sources._replace(
        records=lambda ids: {k: v for k, v in keep(ids).items() if k != fake.chosen["test"][0]}
    )


BREAKAGES = [
    (drop_a_trial_from_a_table, final.Refused, "table lacks"),
    (drop_a_feature_column, final.Refused, "lack columns"),
    (a_model_predicts_nan, final.Refused, "non-finite"),
    (a_fit_does_not_converge, Warning, "did not converge"),
    (a_required_model_is_missing, final.Refused, "not fitted"),
    (a_sampled_test_trial_has_no_prediction, final.Refused, "lack a stored prediction"),
    (a_sampled_late_trial_is_not_in_the_features, final.Refused, "not among the trials scored"),
    (the_records_are_not_in_the_raw_download, ValueError, "not in the raw download"),
    (a_sampled_trial_has_no_protocol_record, final.Refused, "no protocol record"),
    (a_probability_is_out_of_range, final.Refused, "outside 0 to 1"),
    (a_probability_is_not_finite, final.Refused, "non-finite"),
    (a_table_is_missing, FileNotFoundError, "No such file"),
]


@pytest.mark.parametrize(("breakage", "error", "message"), BREAKAGES)
def test_a_label_free_failure_cannot_burn_the_logged_access(ready, breakage, error, message):
    breakage(ready)
    with pytest.raises(error, match=message):
        final.run_test("must not open", ready.root, ready.sources)
    assert_nothing_opened(ready)
    with pytest.raises(error, match=message):  # `check` stops at the same place
        final.check_real(ready.root, ready.sources)


@pytest.mark.parametrize("missing", ["llm_predictions.parquet", "llm_sample.csv"])
def test_a_missing_language_model_file_stops_the_real_run_but_not_the_rehearsal(ready, missing):
    (ready.interim / missing).unlink()
    with pytest.raises(final.Refused, match="language-model"):
        final.run_test("no language files", ready.root, ready.sources)
    assert_nothing_opened(ready)
    text = final.rehearse(ready.root, ready.sources)
    assert "Skipped: the sample or the stored predictions file is not on this machine" in text


def test_one_failing_item_is_written_into_the_report_and_the_rest_still_run(ready, monkeypatch):
    def broken(*args, **kwargs):  # the dress rehearsal runs first, so fail only on the real run
        if heldout.access_log(ready.root):
            raise ZeroDivisionError("an arithmetic bug")
        return []

    monkeypatch.setattr(final, "endpoint_rows", broken)
    text = final.run_test("one bug", ready.root, ready.sources)
    assert "ITEMS THAT FAILED: E1." in text
    assert "ZeroDivisionError: an arithmetic bug" in text
    assert "| P1 |" in text and "| E2 |" in text


# ---------------------------------------------------------------- the rehearsal


def test_the_rehearsal_runs_the_same_code_and_cannot_reach_the_held_out_labels(fake, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the rehearsal reached the held-out labels")

    def guarded(read):
        def wrapper(path, *args, **kwargs):
            if "heldout" in str(path):
                refuse()
            return read(path, *args, **kwargs)

        return wrapper

    for module in (final, heldout, ev):
        monkeypatch.setattr(module, "open_heldout", refuse, raising=False)
    monkeypatch.setattr(pl, "read_parquet", guarded(pl.read_parquet))
    monkeypatch.setattr(pl, "scan_parquet", guarded(pl.scan_parquet))
    (fake.root / "results").mkdir()
    (fake.root / "results" / "dev_models.md").write_text(
        "### AUROC within each round\n\n#### Honest models\n\n| model | 2015-2016 | 2017-2018 |\n"
        "|---|---|---|\n| logistic: registration | 0.600 | 0.700 |\n\n## Next\n"
    )
    text = final.rehearse(fake.root, fake.sources)
    assert text.startswith("# REHEARSAL on development data. This is NOT the final run.")
    assert "There is no stand-in for the late set" in text
    assert (fake.root / final.REHEARSAL_REPORT).read_text() == text
    assert heldout.access_log(fake.root) == []
    for item in ITEMS:
        assert f"## {item} (" in text
    assert "ITEMS THAT FAILED" not in text
    assert "Not run in the rehearsal: there is no stand-in for the late set" in text
    assert "## Check against results/dev_models.md" in text


def test_the_rehearsal_asks_for_development_predictions_only(fake, monkeypatch):
    asked = []
    real = final.read_sampled_predictions

    def watching(path, sample, splits):
        asked.append(list(splits))
        return real(path, sample, splits)

    monkeypatch.setattr(final, "read_sampled_predictions", watching)
    (fake.root / "results").mkdir()
    (fake.root / "results" / "dev_models.md").write_text(
        "### AUROC within each round\n\n| model | 2017-2018 |\n|---|---|\n"
    )
    final.rehearse(fake.root, fake.sources)
    assert asked == [["dev"]]


def test_the_agreement_check_reads_the_round_that_shares_its_training_trials():
    text = (
        "### AUROC within each round\n\n#### Honest models\n\n"
        "| model | 2015-2016 | 2017-2018 | 2019-2020 |\n|---|---|---|---|\n"
        "| a | 0.600 | 0.710 | 0.800 |\n| b | 0.500 | 0.500 | 0.500 |\n\n"
        "#### Not honest (deliberately leaky)\n\n| model | 2015-2016 | 2017-2018 | 2019-2020 |\n"
        "|---|---|---|---|\n| c | 0.1 | 0.9 | 0.3 |\n\n## Grouped\n\n| model | 2017-2018 |\n|---|---|\n| z | 0.1 |\n"
    )
    assert final.per_round_aurocs(text) == {"a": 0.71, "b": 0.5, "c": 0.9}


# ---------------------------------------------------------------- stored predictions meet outcomes in one place


def functions_naming(source: str, names: set[str]) -> set[str]:
    """The functions of a module whose bodies mention any of `names`."""
    tree = ast.parse(source)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            used |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            if used & names:
                found.add(node.name)
    return found


READS = {"read_parquet", "scan_parquet", "read_predictions", "read_bytes", "read_text", "open"}


def functions_reading_predictions(source: str) -> set[str]:
    """Functions that call a reader with the predictions path (or `read_predictions` itself)."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            name = getattr(call.func, "attr", getattr(call.func, "id", ""))
            mentioned = {n.id for a in call.args for n in ast.walk(a) if isinstance(n, ast.Name)}
            mentioned |= {
                n.attr for a in call.args for n in ast.walk(a) if isinstance(n, ast.Attribute)
            }
            if name == "read_predictions" or (
                name in READS and mentioned & {"predictions", "PREDICTIONS_PATH"}
            ):
                found.add(node.name)
    return found


def test_only_one_function_reads_the_stored_predictions_file():
    source = Path(final.__file__).read_text()
    assert functions_reading_predictions(source) == {"read_sampled_predictions"}
    assert functions_naming(source, {"PREDICTIONS_PATH"}) == set()
    callers = functions_naming(source, {"read_sampled_predictions"})
    assert callers == {"language_inputs"}


def test_the_reader_check_catches_a_direct_read_of_the_predictions_path():
    sneaky = (
        "def peek(sources):\n    return pl.read_parquet(sources.predictions)\n"
        "def also(root):\n    return pl.scan_parquet(PREDICTIONS_PATH)\n"
        "def harmless(root, name):\n    return pl.read_parquet(root / name)\n"
    )
    assert functions_reading_predictions(sneaky) == {"peek", "also"}


def test_stored_predictions_only_reach_scoring_through_a_period_of_the_matching_split(fake):
    features = pl.read_parquet(fake.interim / "test_features.parquet")
    labelled = final.attach_labels(features.filter(pl.col("split") == "test"), fake.labels)
    with pytest.raises(ValueError, match="another split"):
        final.Period(final.REHEARSAL, labelled, {})  # test trials cannot stand in for dev trials
    with pytest.raises(ValueError, match="another split"):
        final.Period(final.LATE, labelled, {})
    period = final.Period(final.TEST, labelled, {})
    assert period.table.height == 150


def test_the_predictions_reader_never_returns_a_trial_outside_the_asked_splits(fake):
    sample = pl.read_csv(fake.interim / "llm_sample.csv")
    table = final.read_sampled_predictions(
        fake.interim / "llm_predictions.parquet", sample, ["dev"]
    )
    assert set(table["nct_id"]) == set(fake.chosen["dev"])
    assert not set(table["nct_id"]) & (set(fake.chosen["test"]) | set(fake.chosen["late"]))


def test_labels_are_attached_in_the_order_of_the_features_so_predictions_line_up(fake):
    features = pl.read_parquet(fake.interim / "test_features.parquet").filter(
        pl.col("split") == "test"
    )
    shuffled = features.sample(fraction=1.0, shuffle=True, seed=3)
    labelled = final.attach_labels(shuffled, fake.labels)
    assert labelled["nct_id"].to_list() == shuffled["nct_id"].to_list()
    expected = dict(zip(fake.labels["nct_id"], fake.labels[bl.LABEL], strict=True))
    assert labelled[bl.LABEL].to_list() == [expected[i] for i in shuffled["nct_id"]]


# ---------------------------------------------------------------- signs and arithmetic


def labelled(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def many(count: int, **columns) -> list[dict]:
    return [dict(columns) for _ in range(count)]


def test_a_phase_3_after_a_success_is_compared_with_after_a_failure_as_success_minus_failure():
    rows = many(10, phase="phase3", prior_phase2_labelled=True, prior_phase2_succeeded=True,
                any_arm=True)  # fmt: skip
    rows += many(10, phase="phase3", prior_phase2_labelled=True, prior_phase2_succeeded=True,
                 any_arm=False)  # fmt: skip
    rows += many(4, phase="phase3", prior_phase2_labelled=True, prior_phase2_succeeded=False,
                 any_arm=True)  # fmt: skip
    rows += many(16, phase="phase3", prior_phase2_labelled=True, prior_phase2_succeeded=False,
                 any_arm=False)  # fmt: skip
    rows += many(30, phase="phase3", prior_phase2_labelled=False, prior_phase2_succeeded=None,
                 any_arm=True)  # fmt: skip
    rows += many(30, phase="phase2", prior_phase2_labelled=True, prior_phase2_succeeded=True,
                 any_arm=False)  # fmt: skip
    (row,) = final.history_rows(labelled(rows))
    assert row.value == pytest.approx(100 * (0.5 - 0.2))
    assert row.note.startswith("20 against 20")
    assert "minus" in row.what and row.what.startswith("Phase 3 after a successful")


def test_endpoint_rows_are_each_type_minus_all_the_others_crude_then_adjusted():
    def trials(kind, n, successes):
        return many(
            n, endpoint_type=kind, phase="phase2", sponsor_class="INDUSTRY",
            therapeutic_area="x", modality="small molecule", lead_drug=None,
        )  # fmt: skip

    rows = []
    for kind, n, successes in ((BIOMARKER, 40, 30), (SURVIVAL, 40, 10), ("symptom", 80, 40)):
        rows += [r | {"any_arm": i < successes} for i, r in enumerate(trials(kind, n, successes))]
    found = final.endpoint_rows(labelled(rows))
    by_what = {r.what: r for r in found}
    biomarker = by_what["biomarker first primary endpoint minus all other endpoint types, crude"]
    survival = by_what["survival first primary endpoint minus all other endpoint types, crude"]
    assert biomarker.value == pytest.approx(100 * (30 / 40 - 50 / 120))
    assert survival.value == pytest.approx(100 * (10 / 40 - 70 / 120))
    adjusted_biomarker = by_what[
        "biomarker first primary endpoint minus all other endpoint types, adjusted"
    ]
    assert adjusted_biomarker.value == pytest.approx(biomarker.value)  # one stratum: same figure
    assert len(found) == 4


def test_a_better_model_has_a_positive_auroc_difference_and_the_trend_shift_is_the_gap_difference():
    rng = np.random.default_rng(0)
    outcomes = rng.random(300) < 0.5
    table = labelled([{"any_arm": bool(o), "split": "test"} for o in outcomes])
    good = np.where(outcomes, 0.7, 0.3) + rng.normal(0, 0.1, 300)
    poor = rng.random(300)
    predictions = {
        final.REGISTRATION_NAME: poor,
        final.PRIMARY_MODEL: good,
        final.TREND_MODEL: poor - 0.2,
        **{name: poor for _, name in final.RUNGS if name != final.PRIMARY_MODEL},
    }
    period = final.Period(final.TEST, table, predictions)
    evaluation = final.evaluate_period(list(predictions), period, True)
    primary = final.primary_rows(evaluation)
    assert primary[0].value > 0.2 and primary[0].low > 0
    assert primary[0].what == f"{final.PRIMARY_MODEL} minus {final.REGISTRATION_NAME}"
    wanted = {"AUROC of", "Brier score of", "calibration gap of"}
    assert {r.what.split(final.PRIMARY_MODEL)[0].strip() for r in primary[1:]} >= wanted
    gap_rows = final.calibration_rows(evaluation)
    shift = next(r for r in gap_rows if "mean prediction" in r.what)
    assert shift.value == pytest.approx(-20, abs=0.01)  # trend minus registration, in points
    assert shift.estimate().endswith("(no interval)")
    assert gap_rows[0].value == pytest.approx(100 * (poor.mean() - outcomes.mean()))
    assert gap_rows[1].value == pytest.approx(gap_rows[0].value + shift.value)  # the arithmetic
    auroc = next(r for r in gap_rows if r.style is final.sc.AUROC_DIFFERENCE)
    assert auroc.value == pytest.approx(0)  # shifting the intercept cannot change the ranking


def test_the_expression_cut_points_come_from_the_training_trials_not_the_scored_ones():
    def single(tissues, successes, n):
        return many(n, n_target_genes=1, expr_n_tissues=tissues, phase="phase2",
                    sponsor_class="INDUSTRY", therapeutic_area="x", modality="small molecule",
                    lead_drug=None)  # fmt: skip

    def with_success(rows, k):
        return [r | {"any_arm": i < k} for i, r in enumerate(rows)]

    train = labelled(
        with_success(single(5, 0, 60), 30) + with_success(single(20, 0, 60), 30)
        + with_success(single(50, 0, 60), 30)
    )  # fmt: skip
    scored = labelled(
        with_success(single(5, 0, 40), 30) + with_success(single(30, 0, 40), 20)
        + with_success(single(50, 0, 40), 10)
    )  # fmt: skip
    (row,) = final.expression_rows(train, scored)
    # training thirds: bottom 20 or fewer, top above 20; so 30 tissues counts as "top"
    assert row.value == pytest.approx(100 * (30 / 80 - 30 / 40))
    assert "top third is more than 20 tissues" in row.note


def test_a_group_too_small_for_the_expression_comparison_is_reported_as_such():
    train = labelled(
        [{"n_target_genes": 1, "expr_n_tissues": i, "any_arm": True, "phase": "phase2",
          "sponsor_class": "INDUSTRY", "therapeutic_area": "x", "modality": "m",
          "lead_drug": None} for i in range(90)]
    )  # fmt: skip
    (row,) = final.expression_rows(train, train.head(12))
    assert np.isnan(row.value) and "too few trials" in row.note


def test_the_language_rows_score_each_model_on_its_own_sampled_trials_against_the_baselines(ready):
    features = pl.read_parquet(ready.interim / "test_features.parquet").filter(
        pl.col("split") == "test"
    )
    table = final.attach_labels(features, ready.labels)
    rng = np.random.default_rng(1)
    predictions = {
        final.REGISTRATION_NAME: rng.random(table.height),
        final.WORDS_MODEL: rng.random(table.height),
    }
    period = final.Period(final.TEST, table, predictions)
    wide = lr.wide_predictions(
        pl.read_parquet(ready.interim / "llm_predictions.parquet"), ready.chosen["test"]
    )
    wide = wide.sort("nct_id")
    named = lr.named_arrays(wide, wide["nct_id"].to_list(), fake_records(wide["nct_id"]))
    rows = final.language_rows(period, wide, named)
    sonnet = next(r for r in rows if r.what == "AUROC of sonnet (identified)")
    ids = sorted(ready.chosen["test"])
    outcomes = table.filter(pl.col("nct_id").is_in(ids)).sort("nct_id")[bl.LABEL].to_numpy()
    expected = mt.auroc(outcomes, wide.sort("nct_id")["sonnet (identified)"].to_numpy())
    assert sonnet.value == pytest.approx(expected)
    assert sonnet.value > 0.7  # the fixture's probabilities follow the outcome
    gain = next(r for r in rows if r.what == f"sonnet (identified) minus {final.REGISTRATION_NAME}")
    assert gain.value > 0
    contrast = next(r for r in rows if r.what == "sonnet (identified) minus sonnet (blinded)")
    assert contrast.style is final.sc.AUROC_DIFFERENCE
    whats = " | ".join(r.what for r in rows)
    assert "said it recognised the trial" in whats  # the four-cell table
    assert "drug named by neither model" in whats


def plan_item_ids(plan: str) -> set[str]:
    """The ids of the plan's `### P1. ...` sections."""
    return set(re.findall(r"^### ([A-Z]\d)\.", plan, re.MULTILINE))


def test_the_items_in_the_report_are_exactly_the_items_in_the_plan_file(ready):
    plan = (Path(final.__file__).resolve().parents[2] / final.PLAN_FILE).read_text()
    text = final.run_test("items", ready.root, ready.sources)
    in_report = set(re.findall(r"^## ([A-Z]\d) \(", text, re.MULTILINE))
    assert in_report == plan_item_ids(plan) == set(final.ITEMS)
    for level in ("PRIMARY", "SECONDARY", "EXPLORATORY"):
        assert re.search(rf"^## {level}", plan, re.MULTILINE)


def test_check_runs_every_label_free_step_and_opens_nothing(ready, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the check reached the held-out labels")

    monkeypatch.setattr(final, "open_heldout", refuse)
    text = final.check_real(ready.root, ready.sources)
    assert text.startswith("All label-free checks passed.")
    assert "The labels were not opened." in text
    assert_nothing_opened(ready)


def test_check_refuses_what_the_real_run_refuses(ready):
    (ready.root / "FINAL_PLAN.md").write_text("changed\n")
    with pytest.raises(final.Refused, match="changed since its commit"):
        final.check_real(ready.root, ready.sources)


@pytest.mark.parametrize("marker", ["<<FILL IN: the numbers>>", "<<TBD>>", "<< later"])
def test_a_plan_with_any_double_less_than_marker_left_in_is_refused(ready, marker):
    (ready.root / "FINAL_PLAN.md").write_text(PLAN_TEXT + marker + "\n")
    git(ready.root, "commit", "-q", "-am", "an unfinished plan")
    with pytest.raises(final.Refused, match="marker"):
        final.plan_commit(ready.root)
    with pytest.raises(final.Refused, match="marker"):
        final.run_test("unfinished", ready.root, ready.sources)
    assert_nothing_opened(ready)


# ---------------------------------------------------------------- the clean-tree check


def test_the_sample_file_is_an_input_and_an_edit_to_it_stops_the_run(ready):
    sample = ready.root / "inputs" / "llm_sample.csv"
    sample.parent.mkdir()
    pl.read_csv(ready.interim / "llm_sample.csv").write_csv(sample)
    git(ready.root, "add", "inputs")
    git(ready.root, "commit", "-q", "-m", "the sample")
    sources = ready.sources._replace(sample=sample)
    sample.write_text(sample.read_text() + "\n")
    with pytest.raises(final.Refused, match="uncommitted"):
        final.run_test("edited sample", ready.root, sources)
    assert_nothing_opened(ready)


@pytest.fixture
def tracked_reports(monkeypatch):
    """The made-up repo ignores results/, so let the reports sit where git can see them."""
    paths = (final.LOG_FILE, Path("out") / "a.md", Path("out") / "b.md")
    monkeypatch.setattr(final, "GENERATED", paths)
    return paths


def test_the_tree_check_leaves_out_only_the_log_and_the_reports_and_names_them(
    ready, tracked_reports
):
    for path in final.GENERATED:
        (ready.root / path).parent.mkdir(parents=True, exist_ok=True)
        (ready.root / path).write_text("a leftover from an earlier command\n")
    stray, pending = final.tree_state(ready.root)
    assert stray == []
    assert sorted(line[3:] for line in pending) == sorted(str(p) for p in final.GENERATED)
    (ready.root / "src.py").write_text("x = 1\n")
    assert [line[3:] for line in final.tree_state(ready.root)[0]] == ["src.py"]


def test_the_header_names_the_uncommitted_reports_and_the_hash_of_every_input(
    ready, tracked_reports
):
    previous = ready.root / tracked_reports[1]
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text("the scorecard, not yet committed\n")
    text = final.run_test("header", ready.root, ready.sources)
    assert (
        f"Uncommitted at the start (the files this session writes): `{tracked_reports[1]}`" in text
    )
    assert "dirty` flag" in text
    for name in ("dev", "test_features", "endpoint", "text", "llm_predictions"):
        path = ready.interim / f"{name}.parquet"
        assert f"`data/interim/{name}.parquet` sha256 `{final.file_sha256(path)}`" in text
    sample = ready.interim / "llm_sample.csv"
    assert final.file_sha256(sample) in text


# ---------------------------------------------------------------- the dress rehearsal


def test_the_dress_rehearsal_runs_every_item_before_the_labels_are_opened_and_writes_nothing(
    ready, monkeypatch
):
    seen = []
    real = final.run_plan

    def watching(fitted, train, periods, language, reference=None):
        seen.append((heldout.access_log(ready.root), sorted(periods), reference))
        return real(fitted, train, periods, language, reference)

    monkeypatch.setattr(final, "run_plan", watching)
    final.check_real(ready.root, ready.sources)
    assert len(seen) == 2  # random labels, then the late set all one class
    for log, periods, reference in seen:
        assert log == [] and reference is None
        assert periods == sorted([final.TEST, final.LATE, final.EXTENDED])
    assert_nothing_opened(ready)
    assert not (ready.root / final.REHEARSAL_REPORT).exists()


def test_the_made_up_labels_are_seeded_and_the_second_pass_has_a_one_class_late_set(ready):
    frames = {
        "test": pl.read_parquet(ready.interim / "test_features.parquet").filter(
            pl.col("split") == "test"
        ),
        "late": pl.read_parquet(ready.interim / "test_features.parquet").filter(
            pl.col("split") == "late"
        ),
    }
    first = final.made_up_labels(frames, False)
    assert first[bl.LABEL].to_list() == final.made_up_labels(frames, False)[bl.LABEL].to_list()
    late = lambda labels: labels.filter(pl.col("split") == "late")[bl.LABEL]
    assert late(first).n_unique() == 2
    assert late(final.made_up_labels(frames, True)).n_unique() == 1


def test_an_item_that_fails_in_the_dress_rehearsal_refuses_the_run_before_the_opening(
    ready, monkeypatch
):
    def broken(*args, **kwargs):
        raise ZeroDivisionError("a bug the real labels would have hit")

    monkeypatch.setattr(final, "history_rows", broken)
    with pytest.raises(final.Refused, match="(?s)dress rehearsal.*E2.*ZeroDivisionError"):
        final.run_test("must not open", ready.root, ready.sources)
    assert_nothing_opened(ready)


def test_the_dress_rehearsal_uses_few_resamples_and_puts_them_back(ready):
    before = (mt.N_RESAMPLES, final.DRUG_RESAMPLES)
    with final.few_resamples(7):
        assert (mt.N_RESAMPLES, final.DRUG_RESAMPLES) == (7, 7)
    assert (mt.N_RESAMPLES, final.DRUG_RESAMPLES) == before


# ---------------------------------------------------------------- after the opening


def test_nothing_is_fitted_after_the_labels_are_opened(ready, monkeypatch):
    def guarded(spec):
        def fit(train):
            assert not heldout.access_log(ready.root), "a model was fitted after the opening"
            return spec.fit(train)

        return replace(spec, fit=fit)

    specs = ready.sources.specs
    ready.sources = ready.sources._replace(specs=lambda: [guarded(s) for s in specs()])
    control = final.science.control_spec
    monkeypatch.setattr(final.science, "control_spec", lambda: guarded(control()))
    text = final.run_test("no fitting", ready.root, ready.sources)
    assert len(heldout.access_log(ready.root)) == 1
    assert "ITEMS THAT FAILED" not in text


def test_labels_that_fail_their_own_checks_write_a_report_saying_so(ready):
    path = ready.root / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    labels.filter(
        pl.col("nct_id") != labels.filter(pl.col("split") == "test")["nct_id"][0]
    ).write_parquet(path)
    text = final.run_test("bad labels", ready.root, ready.sources)
    assert text.startswith(f"# {final.LABELS_FAILED}")
    assert (ready.root / final.TEST_REPORT).read_text() == text
    assert len(heldout.access_log(ready.root)) == 1  # it was opened, and says so
    assert "Nothing was scored." in text and "Access note: bad labels" in text


def test_a_missing_outcome_in_the_labels_is_caught_by_the_labels_own_check(ready):
    path = ready.root / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    labels.with_columns(
        pl.when(pl.col("nct_id") == labels["nct_id"][0])
        .then(None)
        .otherwise(pl.col(bl.LABEL))
        .alias(bl.LABEL)
    ).write_parquet(path)
    text = final.run_test("null label", ready.root, ready.sources)
    assert text.startswith(f"# {final.LABELS_FAILED}")


# ---------------------------------------------------------------- R1: the reference check


def test_the_reference_check_fails_loudly_on_drift_and_passes_when_the_figures_match(
    ready, monkeypatch
):
    monkeypatch.setattr(final, "REFERENCE_AUROC", {final.TEST: 0.5, final.LATE: 0.5})
    drifted = final.run_test("drift", ready.root, ready.sources)
    assert "ITEMS THAT FAILED: R1." in drifted and "Trust nothing else" in drifted
    found = dict(re.findall(r"(test|late) (0\.\d{3}) against", drifted))
    monkeypatch.setattr(final, "REFERENCE_AUROC", {k: float(v) for k, v in found.items()})
    matched = final.run_test("match", ready.root, ready.sources)
    assert "Matches to three decimals" in matched and "ITEMS THAT FAILED" not in matched


def test_the_real_reference_figures_are_the_ones_the_baseline_run_printed():
    assert final.REFERENCE_AUROC == {final.TEST: 0.730, final.LATE: 0.733}
