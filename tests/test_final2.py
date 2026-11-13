"""The second test-set run (final2), on made-up trials only. The guard is never reached for real."""

import ast
import contextlib
import inspect
import json
import re
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_evaluate import git, lock, write_synthetic_project

from trial_benchmark import baselines as bl
from trial_benchmark import biology_score as bs
from trial_benchmark import evaluate as ev
from trial_benchmark import final, final2, heldout
from trial_benchmark import metrics as mt
from trial_benchmark import scorecard as sc
from trial_benchmark import split_study as sp
from trial_benchmark.dataset import LABEL_COLUMNS
from trial_benchmark.models import linear
from trial_benchmark.models.registry import all_specs

PLAN = Path(final2.__file__).resolve().parents[2] / final2.PLAN_FILE


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    monkeypatch.setattr(mt, "N_RESAMPLES", 30)


# ---------------------------------------------------------------- a small made-up project


def fake_groups(hierarchy=None):
    """Every name of the fixed list, each a plain registration logistic, in two frames."""
    base = next(s for s in linear.SPECS if s.name == final2.REGISTRATION)
    specs = [replace(base, name=n, tables=("genetics",)) for n in final2.expected_names(hierarchy)]
    return [
        final2.Group("base", ("genetics",), (), tuple(specs[:20])),
        final2.Group("rest", ("genetics",), (), tuple(specs[20:])),
    ]


class FakeStudy:
    """Stands in for `split_study.run_study`; records what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, pool, holdout, model_names, n_seeds, cut_year, jobs, cache_file):
        self.calls.append(
            {
                "pool": pool,
                "holdout": holdout,
                "cut_year": cut_year,
                "cache": cache_file,
                "seeds": n_seeds,
                "jobs": jobs,
            }
        )
        names = list(model_names)
        rate = float(holdout[bl.LABEL].mean())
        result = sp.StudyResult(names, dict.fromkeys(names, True), pool.height, holdout.height, 2)
        for regime in sp.REGIMES:
            result.auroc[regime] = dict.fromkeys(names, rate)
            result.gap[regime] = dict.fromkeys(names, 3.0 if regime == sp.CHRONO else 1.0)
            result.auroc_ci[regime] = dict.fromkeys(names, (rate - 0.1, rate + 0.1))
        for regime in (sp.BY_TRIAL, sp.BY_DRUG):
            result.flattery[regime] = dict.fromkeys(names, (0.03, (0.01, 0.05)))
        return result


class FakeEarlier:
    """Stands in for item 7 (its real code runs un-faked in its own test below)."""

    def __init__(self, dry):
        self.dry = dry
        self.finished = []

    def prepare(self, root):
        self.prepared = True

    def label_ids(self):
        return pl.DataFrame(schema={"nct_id": pl.String, "split": pl.String})

    def finish(self, root, labels, rehearse=False):
        assert self.prepared
        self.finished.append((labels, rehearse))
        comparisons = {
            "final": final2.compare_reports("| a | 0.7 (0.6 to 0.8) |", "| a | 0.7 (0.5 to 0.9) |")
        }
        return {"texts": {"final": "final text", "scorecard": "scorecard text"},
                "comparisons": comparisons}  # fmt: skip


def fake_biology_features(root: Path):
    """A `biology_score.feature_frame` stand-in: made-up biology for 80% of the trials."""

    def features(version: str) -> pl.DataFrame:
        trials = pl.read_csv(root / "trials.csv")
        rng = np.random.default_rng({"today": 11, "2020": 12, "2020_names": 13}[version])
        n = trials.height
        columns = {c: rng.normal(size=n) for c in bs.NUMERIC}
        columns["expr_n_tissues"] = rng.integers(1, 52, n).astype(float)
        columns["chem_qed"] = rng.random(n)
        columns |= {c: rng.random(n) < 0.3 for c in bs.BOOLEAN}
        columns["target_class"] = rng.choice(["enzyme", "receptor"], n)
        columns["target_location"] = rng.choice(["intracellular", "membrane"], n)
        columns["chem_action_type"] = rng.choice(["activator", "inhibitor"], n)
        frame = pl.DataFrame({"nct_id": trials["nct_id"], **columns})
        frame = frame.with_columns(
            pl.lit(1).alias("n_target_genes"),
            pl.Series("lead_drug", [f"D{i % 60}" for i in range(n)]),
        )
        return frame.filter(pl.Series(rng.random(n) < 0.8)).select(
            "nct_id", *bs.INPUT_COLUMNS, *bs.EXTRA_COLUMNS
        )

    return features


def fake_unseen(scored_ids, training_ids, version):
    return set(sorted(scored_ids)[::3])


class Project:
    def __init__(self, root: Path, commit_plan: bool = True, ignore_outputs: bool = True):
        write_synthetic_project(root)
        self.root = root
        trials = pl.read_csv(root / "trials.csv")
        rng = np.random.default_rng(7)
        drugs = [None if rng.random() < 0.2 else f"D{rng.integers(0, 60)}" for _ in trials.rows()]
        trials.with_columns(pl.Series("lead_drug", drugs)).write_parquet(
            root / "data" / "interim" / "genetics.parquet"
        )
        if ignore_outputs:
            with (root / ".gitignore").open("a") as handle:
                handle.write("outputs/\n")
            git(root, "commit", "-q", "-am", "ignore outputs")
        lock(root)
        shutil.copy(PLAN, root / final2.PLAN_FILE)
        if commit_plan:
            git(root, "add", final2.PLAN_FILE)
            git(root, "commit", "-q", "-m", "the plan")
        self.study = FakeStudy()
        self.earlier = []

        def earlier(dry):
            made = FakeEarlier(dry)
            self.earlier.append(made)
            return made

        self.sources = final2.Sources(
            groups=fake_groups,
            study=self.study,
            study_tables=lambda: {"genetics"},
            earlier=earlier,
            biology_features=fake_biology_features(root),
            unseen_targets=fake_unseen,
        )

    def run(self, **kwargs):
        return final2.run(self.root, note="test", sources=self.sources, **kwargs)

    def log(self):
        return heldout.access_log(self.root)


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path)


@pytest.fixture
def no_other_openings(monkeypatch):
    """Any opening other than final2's own call fails the test."""

    def refuse(*args, **kwargs):
        raise AssertionError("the held-out labels were opened outside final2's one call")

    for module in (final, sc, ev):
        monkeypatch.setattr(module, "open_heldout", refuse)


def assert_nothing_opened(project):
    assert project.log() == []
    assert not (project.root / final2.REPORT).exists()


# ---------------------------------------------------------------- the guard, once


def test_the_guard_is_called_exactly_once_and_every_item_is_reported(
    project, monkeypatch, no_other_openings
):
    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return heldout.open_heldout(*args, **kwargs)

    monkeypatch.setattr(final2, "open_heldout", counting)
    text = project.run()
    assert len(calls) == 1 and calls[0][0] == final2.LABELS_FILE
    (entry,) = project.log()
    assert "TEST_PLAN_2" in entry["note"]
    assert (project.root / final2.REPORT).read_text() == text
    assert "ITEMS THAT FAILED" not in text
    for number in final2.ITEMS:
        assert f"## Item {number}:" in text
    numbers = json.loads((project.root / final2.JSON_REPORT).read_text())
    assert numbers["plan_commit"] in text and numbers["scored"]["n"] == 150
    assert set(numbers["items"]) == {str(n) for n in final2.ITEMS}
    assert all(not item["failed"] for item in numbers["items"].values())


def test_the_plans_expectations_are_copied_verbatim(project):
    text = project.run()
    plan = final2.expectations(PLAN.read_text())
    assert plan[4][1] == "top third beats bottom third by 15 to 25 per 100 trials"
    for number in final2.ITEMS:
        assert f"**Expected (TEST_PLAN_2.md, verbatim):** {plan[number][1]}" in text


def test_the_split_study_gets_development_as_pool_the_labelled_test_set_and_its_own_cache(project):
    project.run()
    dress, real = project.study.calls  # the dress rehearsal, then the run
    trials = pl.read_csv(project.root / "trials.csv")
    test_ids = set(trials.filter(pl.col("split") == "test")["nct_id"])
    for call in (dress, real):
        assert set(call["pool"]["split"]) == {"dev"}
        assert set(call["holdout"]["nct_id"]) == test_ids
        assert call["cut_year"] == 2021 and call["jobs"] <= 6
    assert dress["cache"] is None and dress["seeds"] == final2.DRESS_SPLIT_SEEDS
    assert real["cache"] == project.root / final2.OUTPUT_DIR / final2.SPLIT_CACHE_NAME
    assert real["cache"].parent != sp.CACHE_DIR
    labels = pl.read_parquet(project.root / "data" / "heldout" / "test_labels.parquet")
    truth = dict(zip(labels["nct_id"], labels[bl.LABEL], strict=True))
    assert all(truth[i] == y for i, y in real["holdout"].select("nct_id", bl.LABEL).rows())


# ---------------------------------------------------------------- refusals before the opening


def test_a_dirty_tree_refuses_before_anything_is_opened(project):
    (project.root / "stray.py").write_text("print('changed')\n")
    with pytest.raises(final2.Refused, match="uncommitted"):
        project.run()
    assert_nothing_opened(project)


def test_an_uncommitted_plan_refuses(tmp_path):
    project = Project(tmp_path, commit_plan=False)
    with pytest.raises(final2.Refused, match="never been committed"):
        project.run()
    assert_nothing_opened(project)
    git(project.root, "add", final2.PLAN_FILE)  # staged is not committed
    with pytest.raises(final2.Refused, match="never been committed"):
        project.run()
    assert_nothing_opened(project)


def test_a_plan_changed_since_its_commit_refuses(project):
    (project.root / final2.PLAN_FILE).write_text(PLAN.read_text() + "\nAn edit.\n")
    with pytest.raises(final2.Refused, match="changed since its commit"):
        project.run()
    assert_nothing_opened(project)


def test_an_output_folder_git_does_not_ignore_refuses(tmp_path):
    project = Project(tmp_path, ignore_outputs=False)
    with pytest.raises(final2.Refused, match="Not git-ignored"):
        project.run()
    assert_nothing_opened(project)


def test_a_failed_dress_rehearsal_refuses_before_the_opening(project, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("a bug")

    monkeypatch.setattr(final2, "item_4", broken)
    with pytest.raises(final2.Refused, match="dress rehearsal"):
        project.run()
    assert_nothing_opened(project)


def test_labels_that_fail_their_own_checks_after_the_opening_leave_a_report_saying_so(project):
    path = project.root / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    labels.with_columns(pl.lit(None, pl.Boolean).alias(bl.LABEL)).write_parquet(path)
    with pytest.raises(ValueError, match="missing"):
        project.run()
    assert len(project.log()) == 1
    assert final2.LABELS_FAILED in (project.root / final2.REPORT).read_text()


# ---------------------------------------------------------------- no label reaches a fit


def frames_for_guard(project):
    dev = pl.read_parquet(project.root / "data" / "interim" / "dev.parquet")
    held = pl.read_parquet(project.root / "data" / "interim" / "test_features.parquet")
    return dev, held.filter(pl.col("split") == "test")


def one_group():
    return [g for g in fake_groups() if g.key == "base"][:1]


@pytest.mark.parametrize(
    "spoil",
    [
        lambda dev, held: (dev, held.with_columns(pl.lit(True).alias(bl.LABEL))),
        lambda dev, held: (dev, held.with_columns(pl.lit(None, pl.Boolean).alias("all_arms"))),
        lambda dev, held: (dev, held.with_columns(
            pl.col("primary_completion_date").alias("wh_posted_last"))),
        lambda dev, held: (pl.concat([dev, held.with_columns(
            pl.lit("dev").alias("split"), *(pl.lit(True).alias(c) for c in LABEL_COLUMNS))],
            how="diagonal_relaxed"), held),
        lambda dev, held: (dev.with_columns(pl.lit("test").alias("split")), held),
    ],
    ids=["label", "null label column", "posting date", "test trial in training", "not dev"],
)  # fmt: skip
def test_a_label_or_a_test_trial_near_a_fit_fails_loudly(project, spoil):
    dev, held = spoil(*frames_for_guard(project))
    groups = one_group()
    with pytest.raises(final2.LabelLeak):
        final2.fit_and_predict(groups, {"base": dev}, {"base": held})


def test_no_label_column_reaches_feature_building_or_fitting_in_a_whole_run(project, monkeypatch):
    seen = {"fit": [], "predict": []}
    opened = []
    real_open = final2.open_heldout

    def counting(*args, **kwargs):
        opened.append(True)
        return real_open(*args, **kwargs)

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def predict(self, frame):
            seen["predict"].append((set(frame.columns), set(frame["split"]), bool(opened)))
            return self.inner.predict(frame)

    def spying(hierarchy=None):
        groups = fake_groups(hierarchy)
        out = []
        for g in groups:
            specs = []
            for s in g.specs:

                def fit(train, s=s):
                    seen["fit"].append((set(train["split"]), bool(opened)))
                    return Spy(s.fit(train))

                specs.append(replace(s, fit=fit))
            out.append(replace(g, specs=tuple(specs)))
        return out

    monkeypatch.setattr(final2, "open_heldout", counting)
    project.sources = project.sources._replace(groups=spying)
    project.run()
    assert len(seen["fit"]) == len(final2.MODELS)
    assert all(splits == {"dev"} and not after for splits, after in seen["fit"])
    assert all(not (columns & set(LABEL_COLUMNS)) for columns, _, _ in seen["predict"])
    assert all(splits == {"test"} and not after for _, splits, after in seen["predict"])


def test_flipping_every_test_label_changes_the_scores_but_not_one_prediction(project):
    project.run()
    first = pl.read_parquet(project.root / final2.OUTPUT_DIR / final2.PREDICTIONS_NAME)
    scores = json.loads((project.root / final2.JSON_REPORT).read_text())["scores"]
    path = project.root / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    flipped = labels.with_columns(
        pl.when(pl.col("split") == "test").then(~pl.col(bl.LABEL)).otherwise(pl.col(bl.LABEL))
    )
    flipped.write_parquet(path)
    for p in (final2.REPORT, final2.JSON_REPORT, final2.ITEM7_FINAL, final2.ITEM7_SCORECARD):
        (project.root / p).unlink()
    shutil.rmtree(project.root / final2.OUTPUT_DIR)
    project.run(rerun="label-flip test")
    second = pl.read_parquet(project.root / final2.OUTPUT_DIR / final2.PREDICTIONS_NAME)
    again = json.loads((project.root / final2.JSON_REPORT).read_text())["scores"]
    assert len(project.log()) == 2  # one opening per run
    for name in final2.MODELS:
        assert first[name].to_list() == second[name].to_list()
    assert (first["outcome"] != second["outcome"]).all()
    auroc = scores["models"][final2.REGISTRATION]["auroc"]["value"]
    assert again["models"][final2.REGISTRATION]["auroc"]["value"] == pytest.approx(1 - auroc)
    gap = scores["models"][final2.REGISTRATION]["gap_per_100"]["value"]
    assert again["models"][final2.REGISTRATION]["gap_per_100"]["value"] != pytest.approx(gap)


# ---------------------------------------------------------------- the fixed model list


def test_the_model_list_is_a_fixed_constant_and_is_exactly_what_is_fitted():
    assert isinstance(final2.MODELS, tuple) and len(set(final2.MODELS)) == len(final2.MODELS)
    assert len(final2.MODELS) == 40
    groups = final2.model_groups()
    final2.check_model_names(groups, None)
    assert [s.name for g in groups for s in g.specs] == list(final2.MODELS)
    for name in (
        final2.HEADLINE_HIERARCHY,
        final2.HEADLINE_PRECEDENT,
        final2.what_name("HOW + WHAT", "2020_names"),
    ):
        assert name in final2.MODELS
    tree = ast.parse(inspect.getsource(final2))
    assigned = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and target.id == "MODELS"
    ]
    assert len(assigned) == 1  # assigned once, at the top, never changed


def test_a_model_added_or_dropped_is_refused():
    groups = fake_groups()
    final2.check_model_names(groups, None)
    dropped = [replace(groups[0], specs=groups[0].specs[1:]), groups[1]]
    with pytest.raises(final2.Refused, match="Missing"):
        final2.check_model_names(dropped, None)
    extra = [groups[0], replace(groups[1], specs=(*groups[1].specs, replace(
        groups[1].specs[0], name="logistic: something new")))]  # fmt: skip
    with pytest.raises(final2.Refused, match="Not in the list"):
        final2.check_model_names(extra, None)


def test_the_real_run_refuses_to_leave_out_hierarchical_variants(project):
    with pytest.raises(final2.Refused, match="dry run"):
        project.run(hierarchy=("full",))
    assert_nothing_opened(project)


# ---------------------------------------------------------------- verdicts


def test_the_plans_rule_met_only_when_the_interval_is_above_zero():
    assert final2.rule_verdict(0.001, 0.05) == final2.MET
    assert final2.rule_verdict(-0.001, 0.05) == final2.NOT_MET
    assert final2.rule_verdict(0.0, 0.05) == final2.NOT_MET
    assert final2.rule_verdict(-0.05, -0.01) == final2.NOT_MET_OPPOSITE
    assert final2.rule_verdict(float("nan"), 0.1) == final2.NOT_COMPUTED


def test_inside_and_outside_the_plans_range():
    r = final2.RANGES[2]
    assert (r.low, r.high) == (0.005, 0.01)
    assert final2.range_verdict(0.007, 0.001, 0.012, r) == sc.INSIDE
    assert final2.range_verdict(0.01, 0.0, 0.02, r) == sc.INSIDE  # the ends count
    assert final2.range_verdict(0.02, 0.015, 0.03, r) == sc.OUTSIDE
    assert final2.range_verdict(0.012, 0.004, 0.02, r) == sc.OVERLAPS
    assert final2.range_verdict(0.03, 0.0, 0.05, None) == final2.NO_RANGE
    assert final2.RANGES[4] == sc.Range(15.0, 25.0)


def test_a_headline_carries_both_verdicts():
    scores = {"differences": {"b minus a": {"auroc": {"value": 0.02, "low": 0.016, "high": 0.025}}}}
    h = final2.headline(3, scores, "b", "a")
    assert h["rule"] == final2.MET and h["range"] == sc.INSIDE
    h = final2.headline(1, scores, "b", "a")
    assert h["range"] == final2.NO_RANGE


def test_item_7_counts_wider_intervals_and_any_verdict_that_changed():
    before = "| P1 | +0.030 (+0.010 to +0.050) | inside |\n| x | 0.700 (0.680 to 0.720) | y |"
    wider = "| P1 | +0.030 (+0.005 to +0.055) | inside |\n| x | 0.700 (0.670 to 0.730) | y |"
    flipped = "| P1 | +0.030 (-0.002 to +0.062) | inside |\n| x | 0.700 (0.670 to 0.730) | y |"
    reworded = "| P1 | +0.030 (+0.005 to +0.055) | outside |\n| x | 0.700 (0.670 to 0.730) | y |"
    c = final2.compare_reports(before, wider)
    assert (c["wider"], c["narrower"], c["zero_exclusion_changed"]) == (2, 0, [])
    assert final2.item_7_verdict({"a": c}) == {
        "wider": final2.MET, "no_verdict_changes": final2.MET, "intervals_wider": 2,
        "intervals_narrower": 0, "lines_with_a_changed_verdict": 0,
    }  # fmt: skip
    assert len(final2.compare_reports(before, flipped)["zero_exclusion_changed"]) == 1
    assert len(final2.compare_reports(before, reworded)["wording_changed"]) == 1
    v = final2.item_7_verdict({"a": final2.compare_reports(before, flipped)})
    assert v["no_verdict_changes"] == final2.NOT_MET
    assert final2.compare_reports("a\nb", "a")["comparable"] is False


def test_single_trial_clusters_reproduce_the_earlier_single_trial_intervals():
    rng = np.random.default_rng(3)
    y = rng.random(80) < 0.5
    p = {"a": rng.random(80), "b": rng.random(80)}
    plain = mt.bootstrap(y, p, True, [("a", "b")], n_resamples=50)
    frame = pl.DataFrame({"nct_id": [f"N{i}" for i in range(80)]})
    with final2.each_trial_its_own_cluster():
        clusters = final.lead_drug_clusters(frame)
        assert sc.lead_drug_clusters is final.lead_drug_clusters
    assert final.lead_drug_clusters is not final2.each_trial_its_own_cluster  # put back
    single = mt.bootstrap(y, p, True, [("a", "b")], n_resamples=50, clusters=clusters)
    assert plain[0]["a"].auroc_interval == single[0]["a"].auroc_interval
    assert plain[1][0].interval == single[1][0].interval


def test_thirds_are_cut_within_phase_and_a_perfect_score_separates_them():
    y = np.array([False] * 30 + [True] * 30)
    score = np.arange(60, dtype=float)
    phase = np.array(["phase2", "phase3"] * 30, dtype=object)
    found = final2.thirds_with_intervals(
        y, score, phase, final2.wr.cluster_weights(np.arange(60).astype(object), 30)
    )
    for ph in final2.PHASES:
        thirds = found[ph]["thirds"]
        assert [c["n"] for c in thirds] == [10, 10, 10]
        assert thirds[0]["per_100"] == 0 and thirds[2]["per_100"] == 100
        assert found[ph]["top_minus_bottom"]["value"] == 100


# ---------------------------------------------------------------- the dry run


def test_the_dry_run_never_calls_the_guard_and_writes_only_to_its_own_folder(project, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the dry run called the guard")

    for module in (final2, final, sc, ev, heldout):
        monkeypatch.setattr(module, "open_heldout", refuse)
    (project.root / "stray.py").write_text("a dirty tree is fine for a dry run\n")
    text = final2.run(project.root, dry=True, hierarchy=("full",), sources=project.sources)
    assert text.startswith("# DRY RUN on development data")
    assert "ITEMS THAT FAILED" not in text
    out = project.root / final2.DRY_DIR
    assert (out / "report.md").exists() and (out / "results.json").exists()
    assert (out / final2.PREDICTIONS_NAME).exists()
    assert not (project.root / final2.REPORT).exists()
    assert project.log() == []
    numbers = json.loads((out / "results.json").read_text())
    assert numbers["dry_run"] and len(numbers["models"]) == len(final2.expected_names(("full",)))
    call = project.study.calls[-1]
    assert (
        call["cut_year"] == final2.DRY_CUT_YEAR and call["cache"] == out / final2.SPLIT_CACHE_NAME
    )


# ---------------------------------------------------------------- item 8: the biology-only score


def test_item_8_verdicts_follow_the_plans_rule_and_ranges():
    down, up = final2.BIOLOGY_SIGN["flags"], final2.BIOLOGY_SIGN["thirds"]
    assert final2.rule_verdict(-12.0, -1.0, down) == final2.MET
    assert final2.rule_verdict(1.0, 12.0, down) == final2.NOT_MET_OPPOSITE
    assert final2.rule_verdict(-5.0, 3.0, down) == final2.NOT_MET
    assert final2.rule_verdict(10.0, 23.0, up) == final2.MET
    assert final2.rule_verdict(float("nan"), 1.0, up) == final2.NOT_COMPUTED
    assert final2.direction_verdict(-3.0, down) == final2.SAME_DIRECTION
    assert final2.direction_verdict(3.0, down) == final2.OPPOSITE_DIRECTION
    assert final2.direction_verdict(float("nan"), up) == final2.TOO_FEW
    flags, thirds = final2.BIOLOGY_RANGES["flags"], final2.BIOLOGY_RANGES["thirds"]
    assert final2.range_verdict(-8.1, -14.6, -1.7, flags) == sc.INSIDE
    assert final2.range_verdict(16.6, 10.0, 23.0, thirds) == sc.OVERLAPS
    assert final2.range_verdict(-1.0, -1.5, -0.5, flags) == sc.OUTSIDE


def test_item_8_is_computed_for_every_link_version_with_its_tables(project):
    text = project.run()
    numbers = json.loads((project.root / final2.JSON_REPORT).read_text())
    found = numbers["items"]["8"]
    assert not found["failed"] and found["headline_links"] == "today"
    assert set(found["versions"]) == set(final2.LINK_VERSIONS)
    for v in found["versions"].values():
        assert v["flags"]["rule"] and v["thirds"]["range"]
        assert v["by_flags"] and v["by_thirds"] and v["unseen"]["n_trials"] > 0
        assert v["n_train"] < 500  # development trials with a target only
    section = text.split("## Item 8:")[1]
    assert "(a) two or more flags minus none" in section and "within phase" in section


def test_the_biology_score_never_trains_on_a_scored_trial_or_reads_a_scored_label(project):
    dev, held = frames_for_guard(project)
    features = fake_biology_features(project.root)
    with pytest.raises(final2.LabelLeak, match="training rows"):
        final2.prepare_biology(features, fake_unseen, pl.concat([dev, dev[:0]]),
                               dev["nct_id"].to_list()[:50])  # fmt: skip

    def labelled_features(version):
        return features(version).with_columns(pl.lit(True).alias(bl.LABEL))

    with pytest.raises(final2.LabelLeak, match="outcome columns"):
        final2.prepare_biology(labelled_features, fake_unseen, dev, held["nct_id"].to_list())


# ---------------------------------------------------------------- round 13: one run, one path


def test_a_second_real_run_is_refused_and_a_declared_re_run_is_reported_as_one(project):
    project.run()
    with pytest.raises(final2.Refused, match="already"):
        project.run()
    assert len(project.log()) == 1
    with pytest.raises(final2.Refused, match="exist"):  # the first run's files are still there
        project.run(rerun="the first run failed part-way")
    for path in (final2.REPORT, final2.JSON_REPORT, final2.ITEM7_FINAL, final2.ITEM7_SCORECARD):
        (project.root / path).rename(project.root / f"{path}.first")
    shutil.rmtree(project.root / final2.OUTPUT_DIR)
    text = project.run(rerun="the first run failed part-way")
    assert len(project.log()) == 2
    assert "RE-RUN: the first run failed part-way" in project.log()[1]["note"]
    assert "This is a re-run" in text


def test_a_declared_re_run_with_no_earlier_run_is_refused(project):
    with pytest.raises(final2.Refused, match="no earlier run"):
        project.run(rerun="why not")
    assert_nothing_opened(project)


def test_a_split_study_cache_left_from_before_refuses_the_real_run(project):
    cache = project.root / final2.OUTPUT_DIR / final2.SPLIT_CACHE_NAME
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"stale fits")
    with pytest.raises(final2.Refused, match="exist"):
        project.run()
    assert_nothing_opened(project)


def test_every_file_the_real_run_writes_under_outputs_must_be_git_ignored(tmp_path):
    project = Project(tmp_path, ignore_outputs=False)
    with (project.root / ".gitignore").open("a") as handle:
        handle.write(f"{final2.OUTPUT_DIR / final2.PREDICTIONS_NAME}\n")
    git(project.root, "commit", "-q", "-am", "ignore the predictions only")
    with pytest.raises(final2.Refused, match="split_study_cache"):
        project.run()
    assert_nothing_opened(project)


def test_a_change_to_the_tree_while_fitting_is_caught_just_before_the_opening(project, monkeypatch):
    real_prepare = final2.prepare

    def prepare_then_edit(*args, **kwargs):
        prepared = real_prepare(*args, **kwargs)
        (project.root / "edited_while_fitting.py").write_text("x = 1\n")
        return prepared

    monkeypatch.setattr(final2, "prepare", prepare_then_edit)
    with pytest.raises(final2.Refused, match="uncommitted"):
        project.run()
    assert_nothing_opened(project)


def test_dry_and_real_runs_share_one_path_from_the_labels_to_the_last_file(project, monkeypatch):
    calls = []
    real_tail = final2.shared_tail

    def watching(*args, **kwargs):
        calls.append(kwargs["write"])
        return real_tail(*args, **kwargs)

    monkeypatch.setattr(final2, "shared_tail", watching)
    project.run()
    final2.run(project.root, dry=True, hierarchy=("full",), sources=project.sources)
    assert calls == [False, True, False, True]  # dress rehearsal, then the run; real, then dry
    real, dry = project.earlier
    for made in (real, dry):
        (_dress_labels, dress), (labels, rehearse) = made.finished
        assert dress and not rehearse and labels is not None and labels.height > 0
    out = project.root / final2.DRY_DIR
    assert {"report.md", "results.json", final2.PREDICTIONS_NAME} <= {p.name for p in out.iterdir()}
    numbers = json.loads((out / "results.json").read_text())
    assert numbers["access_log_entries"] == 1  # the access section runs in the dry run too


def test_the_dry_run_makes_its_stand_in_look_like_test_rows_before_the_shared_code(project):
    interim = project.root / "data" / "interim"
    trials = pl.read_csv(project.root / "trials.csv")
    pl.DataFrame(
        {"nct_id": trials["nct_id"], "wh_posted_last": [date(2020, 1, 1)] * trials.height}
    ).with_columns(pl.col("wh_posted_last").cast(pl.Date)).write_parquet(
        interim / "whathow_x.parquet"
    )
    setting = final2.dry_setting(project.root)
    assert set(setting.held["split"]) == {"test"}
    assert not set(setting.held.columns) & set(LABEL_COLUMNS)
    table = setting.reader("whathow_x")
    stand_in = set(setting.held["nct_id"])
    posted = dict(zip(table["nct_id"], table["wh_posted_last"], strict=True))
    assert all(posted[i] is None for i in stand_in)
    assert all(posted[i] is not None for i in setting.train["nct_id"])


def test_the_split_study_stacks_development_with_a_frame_shaped_like_the_test_features(project):
    real = Path(final2.__file__).resolve().parents[2] / "data" / "interim"
    if not (real / "test_features.parquet").exists():
        pytest.skip("the label-free test features file is not on this machine")
    schema = pl.read_parquet_schema(real / "test_features.parquet")  # shape only; no label in it
    dev, held = frames_for_guard(project)
    held = held.select(list(schema)).cast(dict(schema)).with_columns(pl.lit(True).alias(bl.LABEL))

    def stacking(pool, holdout, **kwargs):
        both = pl.concat([pool, holdout], how="vertical_relaxed")
        assert both.height == pool.height + holdout.height
        return FakeStudy()(pool, holdout, kwargs["model_names"], kwargs["n_seeds"],
                           kwargs["cut_year"], kwargs["jobs"], kwargs["cache_file"])  # fmt: skip

    result = final2.run_split_study(stacking, dev, held, 2021, None, 1)
    assert result.n_holdout == held.height


def test_item_7_that_cannot_be_compared_fails_loudly():
    with pytest.raises(ValueError, match="cannot be compared"):
        final2.item_7_verdict({"final": {"comparable": False, "reason": "different lengths"}})


def test_item_7_compares_only_the_models_of_the_first_run():
    root = Path(final2.__file__).resolve().parents[2]
    report = root / "results" / "test_final.md"
    if not report.exists() or not (root / "data" / "interim" / "dev.parquet").exists():
        pytest.skip("the committed first-run report or the development data is not here")
    text = report.read_text()
    names = [s.name for s in final2.earlier_specs()]
    assert not {s.family for s in final2.earlier_specs()} & set(final2.EARLIER_LEFT_OUT)
    assert all(name in text for name in names)
    assert not [s.name for s in all_specs() if s.family == "precedent" and s.name in text]
    usable, _ = final.usable_specs([*final2.earlier_specs(), final.science.control_spec()], root)
    trained = int(re.search(r"Trained (\d+) models", text).group(1))
    assert len(usable) == trained


def test_the_json_records_the_sha256_of_every_input_it_read(project):
    project.run()
    numbers = json.loads((project.root / final2.JSON_REPORT).read_text())
    hashes = numbers["input_sha256"]
    for name in ("data/interim/dev.parquet", "data/interim/test_features.parquet",
                 "data/interim/genetics.parquet", "data/heldout/test_labels.parquet"):  # fmt: skip
        assert len(hashes[name]) == 64


def test_the_purpose_and_the_report_say_the_one_read_also_supplies_the_late_labels(project):
    assert "late" in final2.PURPOSE
    text = project.run()
    assert "late-set labels" in text.split("## Item 7")[1]


def test_a_reversed_paired_difference_would_be_caught(project):
    def better_and_worse(hierarchy=None):
        groups = fake_groups(hierarchy)
        out = []
        for g in groups:
            specs = []
            for s in g.specs:
                if s.name == final2.REGISTRATION:  # a baseline that knows nothing: AUROC 0.5
                    s = replace(s, fit=lambda train: type("C", (), {
                        "predict": lambda self, f: np.full(f.height, 0.5)})())  # fmt: skip
                specs.append(s)
            out.append(replace(g, specs=tuple(specs)))
        return out

    project.sources = project.sources._replace(groups=better_and_worse)
    project.run()
    scores = json.loads((project.root / final2.JSON_REPORT).read_text())["scores"]
    better = scores["models"][final2.HEADLINE_PRECEDENT]["auroc"]["value"]
    assert scores["models"][final2.REGISTRATION]["auroc"]["value"] == pytest.approx(0.5)
    d = scores["differences"][f"{final2.HEADLINE_PRECEDENT} minus {final2.REGISTRATION}"]
    assert better > 0.55 and d["auroc"]["value"] == pytest.approx(better - 0.5)


def test_thirds_are_cut_within_phase_where_that_differs_from_cutting_all_trials_together():
    phase = np.array(["phase2"] * 30 + ["phase3"] * 30, dtype=object)
    score = np.concatenate([np.arange(30), 100 + np.arange(30)]).astype(float)
    y = np.concatenate([np.arange(30) >= 20, np.arange(30) >= 20])  # top third of each phase wins
    weights = final2.wr.cluster_weights(np.arange(60).astype(object), 30)
    found = final2.thirds_with_intervals(y, score, phase, weights)
    for ph in final2.PHASES:  # cut over all 60 trials, phase 2 would be all bottom and middle
        assert [c["n"] for c in found[ph]["thirds"]] == [10, 10, 10]
        assert found[ph]["top_minus_bottom"]["value"] == 100


def test_flipping_the_labels_changes_no_biology_score_cut_or_cache_path(project):
    project.run()
    first = pl.read_parquet(project.root / final2.OUTPUT_DIR / final2.PREDICTIONS_NAME)
    item_8 = json.loads((project.root / final2.JSON_REPORT).read_text())["items"]["8"]
    path = project.root / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    labels.with_columns(~pl.col(bl.LABEL)).write_parquet(path)
    for p in (final2.REPORT, final2.JSON_REPORT, final2.ITEM7_FINAL, final2.ITEM7_SCORECARD):
        (project.root / p).unlink()
    shutil.rmtree(project.root / final2.OUTPUT_DIR)
    project.run(rerun="label-flip test")
    second = pl.read_parquet(project.root / final2.OUTPUT_DIR / final2.PREDICTIONS_NAME)
    again = json.loads((project.root / final2.JSON_REPORT).read_text())["items"]["8"]
    biology = [c for c in first.columns if c.startswith("biology ")]
    assert len(biology) == 5 * len(final2.LINK_VERSIONS)
    assert first.select(biology).equals(second.select(biology))
    for v in final2.LINK_VERSIONS:
        assert item_8["versions"][v]["cuts"] == again["versions"][v]["cuts"]
    real_runs = [c for c in project.study.calls if c["cache"] is not None]
    assert len({c["cache"] for c in real_runs}) == 1 and len(real_runs) == 2


def test_an_outcome_named_column_in_a_joined_table_is_a_leak(project):
    path = project.root / "data" / "interim" / "genetics.parquet"
    pl.read_parquet(path).with_columns(pl.lit(True).alias("outcome")).write_parquet(path)
    with pytest.raises(final2.LabelLeak, match="outcome"):
        project.run()
    assert_nothing_opened(project)


def test_unseen_targets_are_judged_against_every_development_trial(project):
    seen = []

    def recording(scored_ids, training_ids, version):
        seen.append(set(training_ids))
        return fake_unseen(scored_ids, training_ids, version)

    project.sources = project.sources._replace(unseen_targets=recording)
    project.run()
    dev = set(pl.read_parquet(project.root / "data" / "interim" / "dev.parquet")["nct_id"])
    assert seen and all(ids == dev for ids in seen)


def test_the_docstring_lists_eight_items_and_the_report_notes_the_missing_chemistry(project):
    assert "the eight items" in final2.__doc__
    text = project.run()
    assert "cannot fire" in text.split("## Item 8")[1]


def test_item_8_uses_the_plans_rule_on_its_expected_side():
    down = final2.BIOLOGY_SIGN["flags"]
    assert final2.rule_verdict(-12.0, -1.0, down) == final2.MET
    assert final2.rule_verdict(-5.0, 3.0, down) == final2.NOT_MET
    assert final2.rule_verdict(1.0, 4.0, down) == final2.NOT_MET_OPPOSITE
    assert final2.rule_verdict(1.0, 4.0) == final2.MET


def test_item_7_runs_unfaked_and_the_switch_reaches_every_clustered_bootstrap(
    tmp_path, monkeypatch
):
    """`EarlierResults` on made-up trials with `final`'s and `scorecard`'s own code, unfaked."""
    from test_failures_summary import study
    from test_final import FinalFake

    from trial_benchmark import llm_report as lr

    fake = FinalFake(tmp_path)
    monkeypatch.setattr(final, "REFERENCE_AUROC", {})  # R1 belongs to the real data
    lock(fake.root)
    fake.commit_plan()
    trials = fake.trials
    rng = np.random.default_rng(4)
    genetics = fake.root / "data" / "interim" / "scorecard_genetics.parquet"
    pl.DataFrame(
        {
            "nct_id": trials["nct_id"],
            "mapped": rng.random(trials.height) < 0.7,
            "supported_today": rng.random(trials.height) < 0.4,
            "lead_drug_earlier_trials": rng.integers(0, 12, trials.height),
        }
    ).write_parquet(genetics)
    studies = []
    for split, completed in (("test", "2022-05"), ("late", "2024-03")):
        for i, nct in enumerate(sorted(trials.filter(pl.col("split") == split)["nct_id"])[:40]):
            studies.append(study(nct, completed=completed, status=["COMPLETED", "TERMINATED"][i % 2],
                                 why="Slow accrual" if i % 2 else None))  # fmt: skip

    def flags(ids):
        return pl.DataFrame({"nct_id": ids, "group": ["rare"] * len(ids)},
                            schema={"nct_id": pl.String, "group": pl.String})  # fmt: skip

    earlier = final2.EarlierResults(
        False, final_sources=fake.sources, sc_sources=sc.Sources(genetics, lambda: studies, flags)
    )
    earlier.prepare(fake.root)
    labels = pl.read_parquet(fake.root / "data" / "heldout" / "test_labels.parquet")

    seen = []
    real_bootstrap, real_paired = mt.bootstrap, lr.paired_bootstrap

    def watch_bootstrap(*args, clusters=None, **kwargs):
        seen.append((switched[0], clusters))
        return real_bootstrap(*args, clusters=clusters, **kwargs)

    def watch_paired(*args, clusters=None, **kwargs):
        seen.append((switched[0], clusters))
        return real_paired(*args, clusters=clusters, **kwargs)

    monkeypatch.setattr(mt, "bootstrap", watch_bootstrap)
    monkeypatch.setattr(lr, "paired_bootstrap", watch_paired)
    switched = [False]
    real_switch = final2.each_trial_its_own_cluster

    @contextlib.contextmanager
    def flagged():
        with real_switch():
            switched[0] = True
            yield
            switched[0] = False

    monkeypatch.setattr(final2, "each_trial_its_own_cluster", flagged)
    found = earlier.finish(fake.root, labels)
    assert "ITEMS THAT FAILED" not in found["texts"]["final"]
    assert found["comparisons"]["final"]["comparable"]
    assert found["comparisons"]["scorecard"]["comparable"]
    single = [c for on, c in seen if on]
    clustered = [c for on, c in seen if not on]
    assert single and clustered
    # every switched call resamples single trials: no clusters, or one cluster per trial
    assert all(c is None or len(set(c)) == len(c) for c in single)
    assert any(c is not None and len(set(c)) < len(c) for c in clustered)  # lead drugs shared
    verdict = final2.item_7_verdict(found["comparisons"])
    assert verdict["wider"] in (final2.MET, final2.NOT_MET)
