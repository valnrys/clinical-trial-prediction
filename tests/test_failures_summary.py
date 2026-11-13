import math
from datetime import date

import polars as pl
import pytest
from synthetic import synthetic_trials

from elman_exercise import failures_summary as fs
from elman_exercise import rare_disease as rd


def study(
    nct="NCT01", completed="2019-06", status="TERMINATED", why=None, types=("DRUG",), **extra
):
    protocol = {
        "identificationModule": {"nctId": nct},
        "statusModule": {
            "overallStatus": status,
            "primaryCompletionDateStruct": {"date": completed},
            "whyStopped": why,
        },
        "armsInterventionsModule": {"interventions": [{"type": t, "name": "x"} for t in types]},
        "designModule": {"phases": ["PHASE2"], "enrollmentInfo": {"count": 12}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"class": "INDUSTRY", "name": "Acme"}},
        "conditionsModule": {"conditions": ["  Asthma "]},
    }
    return {"protocolSection": protocol, "hasResults": True} | extra


class NeverReadStatus(dict):
    """A status module that fails if anyone looks at the trial's status."""

    def get(self, key, default=None):
        if key == "overallStatus":
            raise AssertionError("the status of a 2021 trial was read")
        return super().get(key, default)


def test_a_finished_pre_2021_drug_trial_is_in_the_population():
    row = fs.population_row(study(why="Slow accrual"))
    assert row["status"] == "TERMINATED"
    assert row["phase"] == "phase2"
    assert row["industry"] is True
    assert row["enrollment"] == 12
    assert row["sponsor_name"] == "Acme"
    assert row["first_condition"] == "asthma"


@pytest.mark.parametrize("types", [("DEVICE",), ("BEHAVIORAL", "OTHER")])
def test_trials_without_a_drug_biological_or_genetic_intervention_are_left_out(types):
    assert fs.population_row(study(types=types)) is None


@pytest.mark.parametrize("status", ["RECRUITING", "UNKNOWN", "SUSPENDED"])
def test_unfinished_trials_are_left_out(status):
    assert fs.population_row(study(status=status)) is None


def test_the_status_of_a_trial_finishing_in_2021_is_never_read():
    late = study(completed="2021-01")
    late["protocolSection"]["statusModule"] = NeverReadStatus(
        late["protocolSection"]["statusModule"]
    )
    assert fs.population_row(late) is None


def test_a_trial_with_no_completion_date_is_left_out_without_reading_its_status():
    undated = study()
    undated["protocolSection"]["statusModule"] = NeverReadStatus({"overallStatus": "COMPLETED"})
    assert fs.population_row(undated) is None


def test_the_last_day_of_2020_is_in_and_the_first_of_2021_is_out():
    assert fs.population_row(study(completed="2020-12-31")) is not None
    assert fs.population_row(study(completed="2021-01-01")) is None


def test_the_population_holds_one_row_per_qualifying_trial():
    studies = [study("NCT2"), study("NCT1", status="COMPLETED"), study("NCT3", completed="2022-01")]
    population = fs.build_population(studies)
    assert population["nct_id"].to_list() == ["NCT1", "NCT2"]


def test_phases_are_split_into_two_three_and_combined():
    def with_phases(*phases):
        s = study()
        s["protocolSection"]["designModule"]["phases"] = list(phases)
        return fs.phase_of(s)

    assert with_phases("PHASE2") == "phase2"
    assert with_phases("PHASE1", "PHASE2") == "phase2"
    assert with_phases("PHASE3") == "phase3"
    assert with_phases("PHASE2", "PHASE3") == "phase2/3"


def test_a_trial_with_neither_phase_is_unknown_not_combined():
    for phases in ([], ["PHASE1"], ["EARLY_PHASE1"], ["NA"]):
        s = study()
        s["protocolSection"]["designModule"]["phases"] = phases
        assert fs.phase_of(s) == "unknown"


def population_with_groups(rows):
    """rows: (nct, status, why, group, industry, phase). Everything else is filled in."""
    population = pl.DataFrame(
        [
            {
                "nct_id": nct,
                "primary_completion_date": date(2018, 1, 1),
                "status": status,
                "why_stopped": why,
                "phase": phase,
                "industry": industry,
                "enrollment": 20,
                "has_results": status == "COMPLETED",
                "therapeutic_area": "oncology" if i % 2 else "neurology",
                "sponsor_name": f"Sponsor {i % 5}",
                "first_condition": f"condition {i % 4}",
            }
            for i, (nct, status, why, _, industry, phase) in enumerate(rows)
        ],
        schema=fs.POPULATION_SCHEMA,
    )
    flags = pl.DataFrame(
        {
            "nct_id": [r[0] for r in rows],
            "group": [r[3] for r in rows],
            "band": [3 if r[3] == rd.RARE else None for r in rows],
            "band_commonest": [3 if r[3] == rd.RARE else None for r in rows],
            "all_rare": [r[3] == rd.RARE for r in rows],
        }
    )
    return fs.add_groups(population, flags)


def test_stop_reasons_are_given_to_stopped_trials_only():
    table = population_with_groups(
        [
            ("A", "COMPLETED", None, rd.RARE, True, "phase2"),
            ("B", "TERMINATED", "slow accrual", rd.RARE, True, "phase2"),
            ("C", "WITHDRAWN", None, rd.NOT_RARE, False, "phase3"),
        ]
    )
    assert dict(table.select("nct_id", "reason").iter_rows()) == {
        "A": None,
        "B": "recruitment",
        "C": "no reason given",
    }
    assert table["stopped"].to_list() == [False, True, True]


def test_a_ratio_matches_a_hand_calculation():
    # 30/100 against 20/100: ratio 1.5, standard error sqrt(1/30 - 1/100 + 1/20 - 1/100).
    ratio, low, high = fs.ratio_interval(30, 100, 20, 100)
    se = math.sqrt(1 / 30 - 1 / 100 + 1 / 20 - 1 / 100)
    assert ratio == pytest.approx(1.5)
    assert low == pytest.approx(1.5 * math.exp(-1.96 * se))
    assert (low, high) == pytest.approx((0.916, 2.457), abs=1e-3)


def test_a_ratio_with_no_events_in_the_comparison_group_is_not_a_number():
    assert all(math.isnan(x) for x in fs.ratio_interval(5, 10, 0, 10))
    assert all(math.isnan(x) for x in fs.ratio_interval(0, 0, 5, 10))


def test_a_ratio_with_no_events_in_the_first_group_is_zero_with_an_upper_limit():
    ratio, low, high = fs.ratio_interval(0, 50, 5, 50)
    assert (ratio, low) == (0.0, 0.0)
    assert 0.0 < high < 3.0
    assert fs.ratio_text(0, 50, 5, 50).startswith("0.00 (0.00 to ")


def test_the_summary_refuses_a_population_with_a_trial_from_2021():
    table = population_with_groups([("A", "COMPLETED", None, rd.RARE, True, "phase2")])
    late = table.with_columns(primary_completion_date=pl.lit(date(2021, 1, 1)))
    with pytest.raises(ValueError, match="2021 or later"):
        fs.check_pre_2021(late)
    with pytest.raises(ValueError, match="2021 or later"):
        fs.report(late, synthetic_trials(5))


def test_the_summary_refuses_a_trial_with_no_date():
    table = population_with_groups([("A", "COMPLETED", None, rd.RARE, True, "phase2")])
    with pytest.raises(ValueError, match="no primary completion date"):
        fs.check_pre_2021(table.with_columns(primary_completion_date=pl.lit(None, dtype=pl.Date)))


def test_the_summary_refuses_a_development_table_with_a_trial_from_2021():
    table = population_with_groups([("A", "COMPLETED", None, rd.RARE, True, "phase2")])
    with pytest.raises(ValueError, match="2021 or later"):
        fs.report(table, synthetic_trials(5, split="test", first_day=date(2022, 1, 1)))


def test_the_enrolment_table_does_not_fail_on_an_empty_slice():
    table = population_with_groups(
        [
            ("A", "COMPLETED", None, rd.RARE, True, "phase2"),
            ("B", "COMPLETED", None, rd.NOT_RARE, True, "phase2"),
        ]
    )
    text = fs.enrolment_table(table)
    assert "| Terminated | rare | 0 | n/a | n/a |" in text


def mixed_population():
    rows = []
    for i in range(40):
        group = [rd.RARE, rd.NOT_RARE][i % 2]
        rows.append(
            (
                f"N{i:03d}",
                "TERMINATED" if i % 3 == 0 else "COMPLETED",
                "slow accrual",
                group,
                i % 4 < 2,
                "phase2",
            )
        )
    table = population_with_groups(rows)
    # every other rare trial also has a common disease: not all of its diseases are rare
    return table.with_columns(
        all_rare=pl.when(pl.col("group") == rd.RARE)
        .then(pl.col("nct_id").str.slice(1).cast(pl.Int64) % 4 == 0)
        .otherwise(False)
    )


def test_the_every_disease_variant_keeps_mixed_trials_as_not_rare_or_shows_them_left_out():
    table = mixed_population()
    n_rare = table.filter(pl.col("group") == rd.RARE).height
    n_all_rare = table.filter((pl.col("group") == rd.RARE) & pl.col("all_rare")).height
    moved = fs.regrouped(table, "rare only if every disease is rare; the rest not rare", None)
    left_out = fs.regrouped(table, "rare only if every disease is rare; the rest left out", None)
    assert moved.filter(pl.col("group") == rd.RARE).height == n_all_rare
    assert moved.filter(pl.col("group") == rd.NOT_RARE).height == (
        table.filter(pl.col("group") == rd.NOT_RARE).height + n_rare - n_all_rare
    )
    assert left_out.filter(pl.col("group") == "mixed").height == n_rare - n_all_rare
    assert (
        left_out.filter(pl.col("group") == rd.NOT_RARE).height
        == table.filter(pl.col("group") == rd.NOT_RARE).height
    )


def test_the_robustness_table_counts_every_trial_in_the_moved_variant():
    table = mixed_population()
    text = fs.robustness_table(table, None)
    assert "the rest not rare" in text and "the rest left out" in text
    assert "(earlier rule)" not in text  # no alternative flags were given


def test_the_whole_report_is_written_for_a_small_synthetic_population():
    dev = synthetic_trials(60, seed=3)
    groups = [rd.RARE, rd.NOT_RARE, rd.UNCLEAR, rd.UNMAPPED]
    rows = []
    for i, nct in enumerate(dev["nct_id"].to_list()):
        rows.append(
            (
                nct,
                ["COMPLETED", "TERMINATED", "WITHDRAWN"][i % 3],
                ["slow accrual", "toxicity", None][i % 3],
                groups[i % 4],
                i % 2 == 0,
                ["phase2", "phase3", "phase2/3"][i % 3],
            )
        )
    text = fs.report(population_with_groups(rows), dev)
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8."):
        assert heading in text
    assert "All 60 labelled trials are in this population." in text
    assert "### 4.1" in text and "### 4.3" in text
    assert "concentrated in rare" not in text and "clear gap" not in text
    assert (
        text.index("## 3.") < text.index("### 4.1") < text.index("### 4.2") < text.index("### 4.3")
    )


# ---------------------------------------------------------------- other windows and eras


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"why": "Slow accrual"},
        {"status": "COMPLETED"},
        {"status": "WITHDRAWN", "why": "no funding"},
        {"types": ("DEVICE",)},
        {"status": "RECRUITING"},
        {"completed": "2017-01"},
        {"completed": "2020-12-31"},
        {"completed": "2021-01-01"},
    ],
)
def test_the_default_window_is_everything_before_2021(extra):
    s = study(**extra)
    assert fs.population_row(s) == fs.population_row(s, None, date(2021, 1, 1))


def test_a_window_includes_its_first_day_and_excludes_its_end_day():
    first, end = date(2021, 1, 1), date(2024, 1, 1)
    assert fs.population_row(study(completed="2021-01-01"), first, end) is not None
    assert fs.population_row(study(completed="2023-12-31"), first, end) is not None
    assert fs.population_row(study(completed="2020-12-31"), first, end) is None
    assert fs.population_row(study(completed="2024-01-01"), first, end) is None


def test_the_status_of_a_trial_outside_a_window_is_never_read():
    first, end = date(2021, 1, 1), date(2024, 1, 1)
    for completed in ("2020-06", "2024-06"):
        outside = study(completed=completed)
        outside["protocolSection"]["statusModule"] = NeverReadStatus(
            outside["protocolSection"]["statusModule"]
        )
        assert fs.population_row(outside, first, end) is None


def test_later_trials_are_refused_unless_they_are_given_an_era():
    flags = pl.DataFrame({"nct_id": ["NCT9"], "group": ["rare"]})
    later = fs.build_population(
        [study("NCT9", completed="2022-05")], date(2021, 1, 1), date(2024, 1, 1)
    )
    with pytest.raises(ValueError):
        fs.add_groups(later, flags)
    assert fs.add_groups(later, flags, later_era="2021 or later")["era"].to_list() == [
        "2021 or later"
    ]
    early = fs.build_population([study("NCT9", completed="2012-03")])
    assert fs.add_groups(early, flags, later_era="x").equals(fs.add_groups(early, flags))
