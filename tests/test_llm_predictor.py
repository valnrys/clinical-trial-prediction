import ast
import hashlib
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from elman_exercise import llm_predictor as lp
from elman_exercise import llm_report as lr
from elman_exercise.models.linear import REGISTRATION_NAME

FORBIDDEN = {
    "results": "MARKER_RESULTS",
    "derived": "MARKER_DERIVED",
    "summary": "MARKER_SUMMARY",
    "status": "MARKER_STATUS",
    "why_stopped": "MARKER_WHYSTOPPED",
    "completion": "2031-12-31",
    "enrolment": "98765",
}


def study(**overrides) -> dict:
    """A raw record with every forbidden field filled with a marker string."""
    protocol = {
        "identificationModule": {
            "nctId": "NCT01234567",
            "briefTitle": "Brief title",
            "officialTitle": "Official Title of Zorbitrex",
        },
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Pharma", "class": "INDUSTRY"}},
        "statusModule": {
            "overallStatus": FORBIDDEN["status"],
            "whyStopped": FORBIDDEN["why_stopped"],
            "startDateStruct": {"date": "2019-03"},
            "primaryCompletionDateStruct": {"date": FORBIDDEN["completion"]},
            "completionDateStruct": {"date": FORBIDDEN["completion"]},
        },
        "descriptionModule": {"briefSummary": FORBIDDEN["summary"]},
        "designModule": {
            "phases": ["PHASE2"],
            "designInfo": {
                "allocation": "RANDOMIZED",
                "interventionModel": "PARALLEL",
                "primaryPurpose": "TREATMENT",
                "maskingInfo": {"masking": "DOUBLE"},
            },
            "enrollmentInfo": {"count": int(FORBIDDEN["enrolment"]), "type": "ACTUAL"},
        },
        "conditionsModule": {"conditions": ["Asthma"]},
        "armsInterventionsModule": {
            "armGroups": [
                {
                    "label": "Zorbitrex",
                    "type": "EXPERIMENTAL",
                    "description": "Acme Pharma drug, see NCT01234567",
                    "interventionNames": ["Drug: Zorbitrex"],
                },
                {"label": "Placebo", "type": "PLACEBO_COMPARATOR"},
            ],
            "interventions": [{"type": "DRUG", "name": "Zorbitrex"}],
        },
        "outcomesModule": {
            "primaryOutcomes": [{"measure": "FEV1", "timeFrame": "12 weeks"}],
            "secondaryOutcomes": [{"measure": "MARKER_SECONDARY"}],
        },
        "eligibilityModule": {
            "sex": "ALL",
            "minimumAge": "18 Years",
            "healthyVolunteers": False,
            "eligibilityCriteria": "MARKER_CRITERIA",
        },
    }
    protocol.update(overrides)
    return {
        "protocolSection": protocol,
        "resultsSection": {"x": FORBIDDEN["results"]},
        "derivedSection": {"x": FORBIDDEN["derived"]},
        "hasResults": True,
    }


# ---------------------------------------------------------------- what the model sees


@pytest.mark.parametrize("variant", lp.VARIANTS)
def test_the_prompt_never_shows_results_summary_status_completion_or_enrolment(variant):
    record = lp.extract_record(study())
    text = lp.render_prompt([record], variant) + json.dumps(record)
    for marker in FORBIDDEN.values():
        assert marker not in text
    assert "MARKER_SECONDARY" not in text and "MARKER_CRITERIA" not in text


def test_the_identified_prompt_shows_id_title_and_sponsor_name():
    text = lp.render_prompt([lp.extract_record(study())], "identified")
    assert "NCT01234567" in text and "Official Title of Zorbitrex" in text and "Acme Pharma" in text


def test_the_masked_prompt_has_no_id_title_or_sponsor_name_even_inside_arm_text():
    text = lp.render_prompt([lp.extract_record(study())], "masked")
    for hidden in ("NCT01234567", "Official Title", "Brief title", "Acme"):
        assert hidden not in text
    assert "industry" in text  # the sponsor class stays
    assert "Zorbitrex" in text  # drug names stay in both variants


def test_trials_get_neutral_sequential_labels():
    records = [lp.extract_record(study()), lp.extract_record(study())]
    text = lp.render_prompt(records, "masked")
    assert "Trial 1\n" in text and "Trial 2\n" in text


def test_missing_modules_render_without_failing():
    bare = {"protocolSection": {"identificationModule": {"nctId": "NCT9"}}}
    for variant in lp.VARIANTS:
        assert "Trial 1" in lp.render_prompt([lp.extract_record(bare)], variant)


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        lp.render_trial(lp.extract_record(study()), "half", 1)


# ---------------------------------------------------------------- the answer


def answer(*items) -> str:
    return json.dumps(
        [
            {
                "trial": i,
                "probability": p,
                "recognised": r,
                "drug_guess": None,
                "reason": "Because.",
            }
            for i, (p, r) in enumerate(items, start=1)
        ]
    )


def test_a_drug_guess_is_kept_and_an_empty_or_unknown_one_is_no_guess():
    reply = json.dumps(
        [
            {"trial": i, "probability": 0.5, "recognised": False, "drug_guess": g, "reason": "x"}
            for i, g in enumerate(["  Pembrolizumab ", None, "unknown", ""], start=1)
        ]
    )
    assert [p["drug_guess"] for p in lp.parse_response(reply, 4)] == [
        "Pembrolizumab",
        None,
        None,
        None,
    ]


def test_every_variant_asks_for_the_drug_guess_in_the_same_words():
    for variant in lp.VARIANTS:
        prompt = lp.render_prompt([lp.extract_record(study())], variant)
        assert (
            "name of the main drug being tested, if you can tell or guess it, otherwise null"
            in prompt
        )
    assert lp.TASK.count("drug_guess") == 2


def test_a_good_answer_is_parsed_and_a_code_fence_is_tolerated():
    good = answer((0.2, False), (1, True))
    assert [p["probability"] for p in lp.parse_response(good, 2)] == [0.2, 1.0]
    assert lp.parse_response(f"```json\n{good}\n```", 2)[1]["recognised"] is True


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        answer((0.2, False)),  # too few
        answer((1.2, False), (0.5, False)),  # above 1
        answer((-0.1, False), (0.5, False)),
        answer((True, False), (0.5, False)),  # a bool is not a probability
        answer(("0.5", False), (0.5, False)),
        answer((0.5, "yes"), (0.5, False)),
        json.dumps([{"trial": 2, "probability": 0.5, "recognised": True, "reason": "x"}] * 2),
        json.dumps([{"trial": 1, "probability": 0.5, "recognised": True}] * 2),
        json.dumps(
            [{"trial": i, "probability": 0.5, "recognised": True, "reason": "x"} for i in (1, 2)]
        ),  # no drug_guess
        json.dumps(
            [
                {"trial": i, "probability": 0.5, "recognised": True, "reason": "x", "drug_guess": 5}
                for i in (1, 2)
            ]
        ),  # a drug guess that is not text
        "Here you go: " + answer((0.5, True), (0.5, True)),  # prose around it is not repaired
    ],
)
def test_parsing_is_strict(bad):
    with pytest.raises(ValueError):
        lp.parse_response(bad, 2)


# ---------------------------------------------------------------- calls


def output(text: str, **extra) -> str:
    return json.dumps(
        {
            "result": text,
            "num_turns": 1,
            "permission_denials": [],
            "modelUsage": {lp.MODELS["haiku"]: {}},
            "usage": {"server_tool_use": {"web_search_requests": 0}},
            **extra,
        }
    )


RECORDS = {f"NCT{i}": lp.extract_record(study()) for i in range(3)}
IDS = list(RECORDS)


def run_batch(calls: list, call):
    return lp.predict_batch(7, IDS, RECORDS, "haiku", "masked", 0, call=lambda p, m: call(p, m))


def test_a_good_call_records_the_exact_model_id_batch_and_position():
    rows, raw = run_batch([], lambda p, m: output(answer((0.1, False), (0.2, True), (0.3, False))))
    assert [r["probability"] for r in rows] == [0.1, 0.2, 0.3]
    assert {r["model_id"] for r in rows} == {lp.MODELS["haiku"]} and rows[0]["error"] is None
    assert [(r["batch"], r["position"]) for r in rows] == [(7, 1), (7, 2), (7, 3)]
    assert len(raw["attempts"]) == 1


def test_the_raw_record_holds_what_was_asked_and_what_came_back():
    seen = []

    def call(prompt, model):
        seen.append((prompt, model))
        return output(answer((0.1, False), (0.2, True), (0.3, False)))

    rows, raw = run_batch([], call)
    assert raw["prompt"] == seen[0][0] and raw["model_requested"] == seen[0][1]
    assert raw["prompt_sha256"] == hashlib.sha256(raw["prompt"].encode()).hexdigest()
    assert {r["prompt_sha256"] for r in rows} == {raw["prompt_sha256"]}
    assert raw["system_prompt"] == lp.SYSTEM_PROMPT
    assert raw["command"] == lp.claude_command(lp.MODELS["haiku"])
    assert raw["model_id_reported"] == lp.MODELS["haiku"] and raw["attempts"][0]["started"]
    assert "stdout" in raw["attempts"][0] and raw["nct_ids"] == IDS


def test_a_call_that_reports_another_model_id_is_a_failure():
    wrong = output(answer((0.1, False), (0.2, True), (0.3, False)), modelUsage={"claude-other": {}})
    rows, raw = run_batch([], lambda p, m: wrong)
    assert all(r["probability"] is None for r in rows)
    assert "claude-other" in raw["attempts"][-1]["error"]
    with pytest.raises(RuntimeError, match="reports"):
        lp.read_output(wrong, lp.MODELS["haiku"])


def test_a_bad_first_answer_is_retried_once_then_recorded_as_a_failure():
    replies = iter(["garbage", answer((0.1, False), (0.2, True), (0.3, False))])
    rows, raw = run_batch([], lambda p, m: output(next(replies)))
    assert len(raw["attempts"]) == 2 and rows[0]["probability"] == 0.1

    rows, raw = run_batch([], lambda p, m: output("garbage"))
    assert len(raw["attempts"]) == 2
    assert all(r["probability"] is None and r["error"] for r in rows)  # no guessing


def test_a_call_that_used_a_tool_is_a_failure_not_a_prediction():
    good = answer((0.1, False), (0.2, True), (0.3, False))
    rows, _ = run_batch([], lambda p, m: output(good, num_turns=3))
    assert all(r["probability"] is None for r in rows)
    rows, _ = run_batch([], lambda p, m: output(good, permission_denials=[{"tool": "WebFetch"}]))
    assert all(r["probability"] is None for r in rows)


def test_the_command_has_no_tools_no_settings_and_our_own_system_prompt():
    command = lp.claude_command("claude-test-1")
    assert command[command.index("--model") + 1] == "claude-test-1"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--system-prompt") + 1] == lp.SYSTEM_PROMPT
    for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in command


def test_each_call_runs_from_an_empty_directory_outside_the_repository(monkeypatch):
    seen = {}

    def fake_run(command, cwd, **kwargs):
        seen["cwd"] = Path(cwd)
        seen["empty"] = not any(Path(cwd).iterdir())
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(lp.subprocess, "run", fake_run)
    lp.call_claude("hello", "claude-test-1")
    assert seen["empty"] and lp.ROOT not in seen["cwd"].parents


def test_the_subprocess_gets_only_a_minimal_environment(monkeypatch):
    seen = {}

    def fake_run(command, env, **kwargs):
        seen["env"] = env
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(lp.subprocess, "run", fake_run)
    monkeypatch.setenv("SOME_SECRET_TOKEN", "x")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "y")
    lp.call_claude("hello", "claude-test-1")
    assert set(seen["env"]) <= set(lp.CALL_ENVIRONMENT) and "PATH" in seen["env"]


def stream(tools) -> str:
    start = {"type": "system", "tools": tools, "mcp_servers": [], "skills": [], "model": "m"}
    return "\n".join(
        json.dumps(e)
        for e in (start, {"type": "result", "result": "I cannot.", "permission_denials": []})
    )


def test_the_isolation_probe_passes_with_no_tools_and_stops_if_any_exist(monkeypatch):
    monkeypatch.setattr(lp, "call_claude", lambda prompt, model, output="json": stream([]))
    assert lp.probe_isolation("m")["tools"] == []
    monkeypatch.setattr(
        lp, "call_claude", lambda prompt, model, output="json": stream(["WebFetch"])
    )
    with pytest.raises(RuntimeError, match="Isolation failed"):
        lp.probe_isolation("m")
    monkeypatch.setattr(lp, "call_claude", lambda prompt, model, output="json": stream([]))
    with pytest.raises(RuntimeError, match="asked for other"):
        lp.probe_isolation("other")


# ---------------------------------------------------------------- the sample


def frame(n_per_year: int = 160) -> pl.DataFrame:
    rows = []
    for split, years in (
        ("dev", range(2015, 2021)),
        ("test", (2021, 2022, 2023)),
        ("late", (2024, 2025)),
    ):
        for year in years:
            for i in range(n_per_year):
                phase = "phase3" if i % 3 else "phase2"
                rows.append((f"NCT{split}{year}{i:03d}", split, date(year, 6, 1), phase))
    return pl.DataFrame(
        rows, schema=["nct_id", "split", "primary_completion_date", "phase"], orient="row"
    )


def test_the_sample_has_300_per_period_in_the_right_years_with_the_phase_mix_kept():
    sample = lp.draw_sample(frame())
    assert sample.group_by("period").len()["len"].to_list() == [300, 300, 300]
    assert sample["nct_id"].n_unique() == 900
    joined = sample.join(frame(), on="nct_id")
    years = joined.group_by("period").agg(
        pl.col("primary_completion_date").dt.year().min().alias("first"),
        pl.col("primary_completion_date").dt.year().max().alias("last"),
    )
    spans = {r["period"]: (r["first"], r["last"]) for r in years.iter_rows(named=True)}
    assert spans == {
        "2017-2020": (2017, 2020),
        "2021-2023": (2021, 2023),
        "2024-2025": (2024, 2025),
    }
    phase2 = joined.filter(pl.col("phase") == "phase2").group_by("period").len()["len"]
    assert all(abs(c - 100) <= 1 for c in phase2)  # a third of every period


def test_the_sample_is_deterministic_and_does_not_depend_on_row_order():
    first = lp.draw_sample(frame())
    assert first.equals(lp.draw_sample(frame()))
    assert first.equals(lp.draw_sample(frame().sample(fraction=1.0, shuffle=True, seed=3)))
    assert not first.equals(lp.draw_sample(frame(), seed=1))


def frozen():
    sample = lp.draw_sample(frame())
    return lp.freeze_batches(sample.with_columns(post_hoc=pl.lit(False)))


def test_the_committed_sample_file_is_well_formed():
    sample = lp.read_sample()
    assert sample.columns == ["nct_id", "split", "period", "post_hoc", "batch", "position"]
    assert sample.group_by("period").len()["len"].to_list() == [300, 300, 300]
    assert set(sample["split"]) == {"dev", "test", "late"}
    assert sample["nct_id"].n_unique() == 900
    assert sorted(sample["batch"].unique()) == list(range(1, 91))
    assert sample.group_by("batch").len()["len"].unique().to_list() == [10]
    assert sorted(sample.filter(pl.col("batch") == 1)["position"]) == list(range(1, 11))


def test_batches_are_frozen_once_mixed_across_periods_and_seeded():
    first = frozen()
    assert first.equals(frozen())
    assert first.group_by("batch").len()["len"].unique().to_list() == [10]
    assert first.group_by("batch").agg(pl.col("period").n_unique())["period"].min() == 3
    assert first.select("batch", "position").unique().height == first.height
    assert not first.equals(lp.freeze_batches(first.drop("batch", "position"), seed=1))


def test_choosing_batches_filters_whole_batches_and_never_rebuilds_them():
    sample = frozen()
    everything = lp.sample_batches(sample)
    assert all(len(ids) == 10 for ids in everything.values())
    assert lp.choose_batches(sample, only={1, 2}) == {1: everything[1], 2: everything[2]}
    assert list(lp.choose_batches(sample, limit=3)) == [1, 2, 3]
    by_split = lp.choose_batches(sample, split="late")
    assert all(any(i.startswith("NCTlate") for i in ids) for ids in by_split.values())
    assert {i for ids in by_split.values() for i in ids} > {
        i for ids in by_split.values() for i in ids if i.startswith("NCTlate")
    }  # the dev and test companions come along: the batch is whole
    assert all(by_split[n] == everything[n] for n in by_split)


def test_batch_numbers_are_parsed():
    assert lp.parse_batch_numbers("1-2") == {1, 2}
    assert lp.parse_batch_numbers("5") == {5}
    assert lp.parse_batch_numbers("1-3,7") == {1, 2, 3, 7}


def test_a_trial_with_no_phase_is_sampled_not_a_crash():
    with_null = frame().with_columns(
        pl.when(pl.col("nct_id").str.ends_with("0"))
        .then(None)
        .otherwise(pl.col("phase"))
        .alias("phase")
    )
    assert lp.draw_sample(with_null).height == 900


# ---------------------------------------------------------------- the table and resuming


def row(nct_id, probability, run=0, error=None):
    return {
        "nct_id": nct_id, "model": "haiku", "variant": "masked", "run": run,
        "probability": probability, "recognised": probability is not None,
        "drug_guess": None, "reason": "r", "model_id": "m", "batch": 1, "position": 1,
        "prompt_sha256": "s", "error": error,
    }  # fmt: skip


def test_a_failed_trial_is_asked_again_and_a_finished_one_is_not():
    table = lp.add_predictions(
        lp.read_predictions(Path("/nonexistent")), [row("A", 0.4), row("B", None, error="x")]
    )
    assert lp.pending(table, ["A", "B", "C"], "haiku", "masked", 0) == ["B", "C"]
    table = lp.add_predictions(table, [row("B", 0.6)])
    assert table.filter(pl.col("nct_id") == "B").height == 1  # replaced, not duplicated
    assert lp.pending(table, ["A", "B", "C"], "haiku", "masked", 0) == ["C"]
    assert lp.pending(table, ["A"], "haiku", "masked", 1) == ["A"]  # a repeat is its own run


def test_a_run_asks_whole_frozen_batches_and_asks_a_failed_one_again_as_itself(
    tmp_path, monkeypatch
):
    sample = frozen()
    records = {i: lp.extract_record(study()) for i in sample["nct_id"]}
    monkeypatch.setattr(lp, "read_sample", lambda: sample)
    monkeypatch.setattr(lp, "load_records", lambda ids: {i: records[i] for i in ids})
    monkeypatch.setattr(lp, "probe_isolation", lambda model_id: {})
    monkeypatch.setattr(lp, "RAW_DIR", tmp_path / "raw")
    prompts = []
    fail_first = {"left": 1}

    def call(prompt, model_id):
        prompts.append(prompt)
        if fail_first["left"] and len(prompts) <= 2:  # both attempts of the first call fail
            fail_first["left"] -= len(prompts) == 2
            return "{}"
        n = int(re.search(r"Below are (\d+)", prompt).group(1))
        return output(answer(*[(0.5, False)] * n), modelUsage={lp.MODELS["haiku"]: {}})

    path = tmp_path / "p.parquet"
    kwargs = {
        "split": None, "limit": None, "only": {1, 2}, "run": 0, "workers": 1, "predictions_path": path,
    }  # fmt: skip
    lp.run_all(["haiku"], ["masked"], call=call, **kwargs)
    table = lp.read_predictions(path)
    assert table.filter(pl.col("probability").is_null()).height == 10  # the failed batch
    assert set(table["batch"]) == {1, 2} and table.height == 20
    lp.run_all(["haiku"], ["masked"], call=call, **kwargs)  # asks only the failed batch again
    table = lp.read_predictions(path)
    assert table["probability"].null_count() == 0 and table.height == 20
    frozen_order = lp.sample_batches(sample)
    for n in (1, 2):
        got = table.filter(pl.col("batch") == n).sort("position")["nct_id"].to_list()
        assert got == frozen_order[n]
    assert len(list((tmp_path / "raw").glob("haiku_masked_run0_b00*.json"))) == 2
    # another variant, a different model run: the same batch, the same order, the same trials
    again = lp.predict_batch(
        1, frozen_order[1], records, "sonnet", "blinded", 0, call=lambda p, m: "{}"
    )[1]
    assert again["nct_ids"] == frozen_order[1]


# ---------------------------------------------------------------- the report cannot reach sealed labels


def dev_frame(n=40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "nct_id": [f"D{i}" for i in range(n)],
            "split": ["dev"] * n,
            "primary_completion_date": [date(2017 + i % 4, 6, 1) for i in range(n)],
            "any_arm": (rng.random(n) > 0.5).tolist(),
            "phase": ["phase3" if i % 3 else "phase2" for i in range(n)],
            "enrollment": [100 + i for i in range(n)],
            "sponsor_class": ["INDUSTRY" if i % 2 else "OTHER" for i in range(n)],
        }
    )


def stored(ids) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    rows = [
        {**row(i, float(rng.random())), "model": m, "variant": v}
        for i in ids
        for m in lp.MODELS
        for v in lp.VARIANTS
    ]
    return pl.DataFrame(rows, schema=lp.TABLE_SCHEMA)


def report_inputs(n=40):
    dev = dev_frame(n)
    flags = [i % 5 == 0 for i in range(n)]
    sample = pl.DataFrame(
        {"nct_id": dev["nct_id"], "split": "dev", "period": "2017-2020", "post_hoc": flags}
    )
    baseline = {
        REGISTRATION_NAME: dict(zip(dev["nct_id"], np.linspace(0.2, 0.8, dev.height), strict=True))
    }
    return dev, sample, baseline


def test_the_report_builds_from_a_development_frame():
    dev, sample, baseline = report_inputs()
    text = lr.build_report(dev, sample, stored(dev["nct_id"]), baseline)
    assert "## What this shows" in text and "haiku (masked)" in text
    # The main contrasts, the recognised rate by variant and the leak check are all reported.
    for pair in ("identified) minus haiku (blinded)", "masked) minus haiku (blinded)"):
        assert pair in text
    assert f"haiku (blinded) minus {REGISTRATION_NAME}" in text
    assert "How often the model said it recognised the trial" in text
    assert "the leak check was not run for this report" in text
    assert "Only 40 trials have a prediction from every model" in text  # a plumbing check
    assert "Not run for this report (the raw records were not given)" in text
    leaks = {"trials": 40, "any": 5}
    with_leaks = lr.build_report(dev, sample, stored(dev["nct_id"]), baseline, leaks)
    assert "5 of 40 sampled development trials (12%)" in with_leaks


def test_the_report_words_the_contrasts_as_what_they_are():
    dev, sample, baseline = report_inputs()
    text = lr.build_report(dev, sample, stored(dev["nct_id"]), baseline)
    assert "not a floor on recall" in text
    assert "recall plus the legitimate drug knowledge" in text
    assert "drug knowledge, known before the start and recalled afterwards, together" in text
    assert "what the drug name alone is worth" not in text
    assert "recall is reduced, not removed" in text
    assert "Name removal cannot blind a model to a trial it has read about" in text
    assert "13 and partly in 5" in text and "about 15" in text
    assert "Nothing here is an out-of-sample estimate of forecasting skill" in text
    assert "June 2026" in text
    for line in (
        "about plus or minus 0.05",
        "about 0.03",
        "0.06 to 0.08",
        "0.08 to 0.10",
        "15 to 20 points",
    ):
        assert line in text


def test_the_report_scores_trials_the_model_did_not_name_and_those_without_a_post_hoc_phrase():
    dev, sample, baseline = report_inputs()
    table = stored(dev["nct_id"]).with_columns(
        drug_guess=pl.when(pl.col("nct_id").str.slice(1).cast(pl.Int64) % 2 == 0)
        .then(pl.lit("Zorbitrex (brand)"))
        .otherwise(None)
    )
    records = {i: lp.extract_record(study()) for i in dev["nct_id"]}
    text = lr.build_report(dev, sample, table, baseline, records=records)
    section = text.split("## The blinded prompt on trials the model could not name")[1].split(
        "## "
    )[0]
    assert (
        f"| haiku | {lr.NOT_NAMED} | 20 |" in section and f"| haiku | {lr.NAMED} | 20 |" in section
    )
    assert "8 of 40 scored trials have a post-hoc phrase" in text
    assert "| 2017-2020 | 40 | 8 |" in text
    assert "Without the trials with a post-hoc phrase" in text


def test_the_report_refuses_a_sampled_trial_from_before_2017():
    dev, sample, baseline = report_inputs()
    early = dev.with_columns(
        primary_completion_date=pl.when(pl.col("nct_id") == "D0")
        .then(pl.lit(date(2016, 6, 1)))
        .otherwise(pl.col("primary_completion_date"))
    )
    with pytest.raises(ValueError, match="outside 2017-2020"):
        lr.build_report(early, sample, stored(dev["nct_id"]), baseline)


def test_a_repeated_prediction_row_is_refused_not_paired_with_the_wrong_trial():
    dev, _, _ = report_inputs()
    table = stored(dev["nct_id"])
    doubled = pl.concat([table, table.head(1)])
    with pytest.raises(pl.exceptions.ComputeError):
        lr.wide_predictions(doubled, dev["nct_id"].to_list())


def test_a_trial_with_no_drug_guess_is_kept_not_dropped():
    dev, _, _ = report_inputs()
    wide = lr.wide_predictions(stored(dev["nct_id"]), dev["nct_id"].to_list())
    assert wide.height == dev.height  # every guess is null in `stored`


def test_an_interval_is_not_given_when_too_many_resamples_have_no_score():
    estimate = lr.interval(np.array([0.5] * 90 + [np.nan] * 10), 0.5)
    assert np.isnan(estimate.low) and np.isnan(estimate.high)
    assert (
        "could not be measured" in estimate.show()
        and lr.reading(estimate) == "could not be measured"
    )
    fine = lr.interval(np.array([0.4, 0.5, 0.6] * 33 + [np.nan]), 0.5)
    assert fine.low < 0.5 < fine.high and "could not" not in fine.show()


def test_a_subset_with_one_outcome_only_gets_no_auroc():
    outcomes = np.array([True] * 30)
    arrays = {"haiku (blinded)": np.linspace(0, 1, 30)}
    assert (
        lr.subset_result(outcomes, arrays, np.ones(30, dtype=bool), "haiku (blinded)").blinded
        is None
    )


def test_a_bootstrap_refuses_predictions_of_the_wrong_length():
    with pytest.raises(ValueError, match="predictions for"):
        lr.paired_bootstrap(np.array([True, False, True]), {"a": np.array([0.1, 0.2])})


def test_the_leak_section_survives_no_trials():
    assert any("no trials" in line for line in lr.leak_section({"trials": 0, "any": 0}, None))


@pytest.mark.parametrize("split, year", [("test", 2022), ("late", 2025), ("dev", 2021)])
def test_the_report_refuses_any_trial_from_2021_on(split, year):
    dev = pl.concat(
        [dev_frame(), dev_frame(1).with_columns(
            nct_id=pl.lit("X"), split=pl.lit(split), primary_completion_date=pl.lit(date(year, 6, 1)))]
    )  # fmt: skip
    sample = pl.DataFrame(
        {"nct_id": ["X"], "split": "dev", "period": "2017-2020", "post_hoc": [False]}
    )
    with pytest.raises(ValueError, match="2021 or later"):
        lr.build_report(dev, sample, stored(["X"]), {})


def test_the_report_module_cannot_reach_the_sealed_labels():
    source = Path(lr.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {node.module or "", *(a.name for a in node.names)}
    assert not any("heldout" in name for name in imported)
    code = ast.dump(ast.parse(source))
    assert "open_heldout" not in code and "test_labels" not in code
    # the only labels it can read come through the development loader
    assert "load_development_with" in source


def test_the_predictor_module_never_reads_a_label_column():
    code = ast.dump(ast.parse(Path(lp.__file__).read_text()))
    assert "any_arm" not in code and "open_heldout" not in code


# ---------------------------------------------------------------- the not-named split and its companions


def named_record():
    return lp.extract_record(study())  # the tested drug is Zorbitrex


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("This is the SPARTAN trial, which met its primary endpoint.", True),
        ("As I recall, TrustTSC did not meet its endpoint.", True),
        ("The FEV1 endpoint was positive in most asthma trials.", False),  # FEV1 is in the prompt
        ("SPARTAN is a trial I know of.", False),  # a name, but no result claimed
        ("Add-on drugs usually beat placebo.", False),
        (None, False),
    ],
)
def test_a_trial_name_the_prompt_does_not_hold_plus_a_result_claim_counts_as_naming(
    reason, expected
):
    assert lr.named_stricter(None, reason, named_record()) is expected


def test_the_stricter_definition_also_counts_an_alias_in_the_reason_or_the_guess():
    rec = named_record()
    assert lr.named_stricter("Zorbitrex", "", rec)  # the guess
    assert lr.named_stricter(None, "A Zorbitrex trial in asthma.", rec)  # the reason only
    assert not lr.named_stricter("a bronchodilator", "Typical asthma trial.", rec)


def test_a_reply_claims_an_outcome_by_its_wording():
    assert lr.claims_an_outcome("I recall this met its primary endpoint")
    assert lr.claims_an_outcome("The trial failed.")
    assert not lr.claims_an_outcome("Likely to succeed given the dose range")
    assert not lr.claims_an_outcome(None)


def test_a_subset_below_the_minimum_gets_no_auroc_and_both_baselines_are_paired():
    rng = np.random.default_rng(3)
    outcomes = rng.random(80) > 0.5
    arrays = {
        "haiku (blinded)": rng.random(80),
        REGISTRATION_NAME: rng.random(80),
        lr.LADDER_NAME: rng.random(80),
    }
    keep = np.ones(80, dtype=bool)
    full = lr.subset_result(outcomes, arrays, keep, "haiku (blinded)")
    assert full.difference is not None and full.ladder_difference is not None
    assert full.ladder is not None and full.registration is not None
    small = lr.subset_result(outcomes, arrays, keep, "haiku (blinded)", minimum=100)
    assert small.n == 80 and small.blinded is None


def test_the_report_gives_both_definitions_both_baselines_the_four_cells_and_the_rest():
    dev, sample, baseline = report_inputs()
    rng = np.random.default_rng(4)
    baseline[lr.LADDER_NAME] = dict(zip(dev["nct_id"], rng.random(dev.height), strict=True))
    table = stored(dev["nct_id"]).with_columns(
        drug_guess=pl.when(pl.col("nct_id").str.slice(1).cast(pl.Int64) % 2 == 0)
        .then(pl.lit("Zorbitrex (brand)"))
        .otherwise(None),
        reason=pl.lit("SPARTAN met its primary endpoint."),
    )
    records = {i: lp.extract_record(study()) for i in dev["nct_id"]}
    text = lr.build_report(dev, sample, table, baseline, records=records)
    assert text.index("## What this shows") < text.index("1. **The scores.**")
    for item in (
        "1. **The scores.**",
        "2. **Most of that",
        "3. **Without the name",
        "4. **What would",
    ):
        assert item in text
    assert "June 2026" in text and "not evidence of reasoning from design" in text
    # The not-named split is exploratory, never the best or cleanest number, and shows both gaps.
    assert "best number" not in text and "cleanest number" not in text
    assert "(exploratory)" in text and "ladder AUROC, same trials" in text
    for definition in (lr.GUESS, lr.STRICTER):
        assert f"| {definition} | haiku | {lr.NOT_NAMED} |" in text
    for heading in (
        "### By whether the model also said it recognised the trial",
        "### Trials neither model named",
        "### How the not-named trials differ from the named ones",
        "### Replies that claim to know an outcome",
        "### Calibration",
    ):
        assert heading in text
    assert "The automatic match misses about one in six" in text
    # every reply claims an outcome, and every reply is named by the stricter definition
    claims = text.split("### Replies that claim to know an outcome")[1].split("\n## ")[0]
    assert "| Haiku | 40 of 40 |" in claims
    # no sentence presents calibration as a point for the language models
    assert "in the language models' favour" not in text
    assert "under-confident at the top" in text
