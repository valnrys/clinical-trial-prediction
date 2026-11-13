import math
from datetime import date

import polars as pl
import pytest

from trial_benchmark import dataset as ds
from trial_benchmark import dev_summary as dv


def frame(rows):
    """Development rows with only the columns the summary needs; the rest are null."""
    base = {name: None for name in ds.SCHEMA}
    base.update(split="dev", primary_completion_date=date(2019, 1, 1), n_arms=1)
    base.update(has_dose_ladder=False)
    return pl.DataFrame([{**base, **row} for row in rows], schema=ds.SCHEMA)


def test_wilson_interval_matches_a_hand_calculation():
    low, high = dv.wilson_interval(8, 10)
    assert (low, high) == pytest.approx((0.4902, 0.9433), abs=1e-3)


def test_wilson_interval_stays_inside_zero_and_one_at_the_edges():
    assert dv.wilson_interval(0, 5)[0] == 0.0
    assert dv.wilson_interval(5, 5)[1] == 1.0
    assert all(math.isnan(x) for x in dv.wilson_interval(0, 0))


@pytest.mark.parametrize(
    ("n", "band"),
    [(None, "(missing)"), (99, "<100"), (100, "100-299"), (299, "100-299"), (300, "300-999"),
     (999, "300-999"), (1000, "1000+")],
)  # fmt: skip
def test_enrollment_bands(n, band):
    assert dv.enrollment_band(n) == band


def test_arm_bands():
    assert [dv.arms_band(n) for n in (1, 2, 3, 7)] == ["1", "2", "3+", "3+"]
    assert dv.arms_band(None) == "(missing)"


def test_all_four_enrollment_bands_print_in_size_order_with_missing_last():
    sizes = [50, 150, 500, 2000, None]
    rows = [{"enrollment": n, "any_arm": True} for n in sizes]
    data = frame(rows)
    bands = [dv.enrollment_band(n) for n in data["enrollment"].to_list()]
    text = dv.by_group(data, "By enrollment", bands, order=dv.ENROLLMENT_ORDER)
    assert [line.split()[0] for line in text.splitlines()[2:]] == [
        "<100", "100-299", "300-999", "1000+", "(missing)",
    ]  # fmt: skip


def test_a_group_table_counts_successes_and_leaves_out_undetermined_labels():
    rows = [
        {"phase": "phase2", "any_arm": True},
        {"phase": "phase2", "any_arm": False},
        {"phase": "phase2", "any_arm": None},
        {"phase": "phase3", "any_arm": True},
        {"phase": None, "any_arm": False},
    ]
    data = frame(rows)
    text = dv.by_group(data, "By phase", data["phase"].to_list())
    cells = [line.split()[:4] for line in text.splitlines()[2:]]
    assert cells == [
        ["phase2", "2", "1", "50.0%"],
        ["phase3", "1", "1", "100.0%"],
        ["(missing)", "1", "0", "0.0%"],
    ]


def test_the_summary_runs_on_development_rows():
    rows = [
        {"nct_id": "A", "any_arm": True, "all_arms": True, "enrollment": 50, "phase": "phase2"},
        {"nct_id": "B", "any_arm": False, "all_arms": False, "enrollment": 500, "phase": "phase3"},
        {"nct_id": "C", "any_arm": True, "all_arms": False, "highest_dose": True,
         "has_dose_ladder": True, "n_arms": 2},
    ]  # fmt: skip
    text = dv.summary(frame(rows))
    assert "Development set: 3 trials" in text
    assert "By primary completion year" in text and "2019" in text


def test_the_summary_refuses_trials_from_2021_on():
    rows = [{"any_arm": True}, {"any_arm": True, "primary_completion_date": date(2021, 1, 1)}]
    with pytest.raises(ValueError, match="2021 or later"):
        dv.summary(frame(rows))
    with pytest.raises(ValueError, match="2021 or later"):
        dv.summary(frame([{"any_arm": True, "split": "test"}]))


def test_it_reads_only_the_development_file(tmp_path):
    with pytest.raises(ValueError, match="only"):
        dv.load_dev(tmp_path / "test_features.parquet")


def test_null_completion_dates_are_refused_even_when_all_are_null():
    with pytest.raises(ValueError, match="no primary completion date"):
        dv.summary(frame([{"any_arm": True, "primary_completion_date": None}]))
    with pytest.raises(ValueError, match="no primary completion date"):
        dv.check_is_development(frame([{"any_arm": True}, {"primary_completion_date": None}]))


def test_null_arm_counts_and_null_labels_do_not_crash_the_summary():
    rows = [
        {"any_arm": None, "has_dose_ladder": True, "highest_dose": False, "n_arms": None},
        {"any_arm": True, "has_dose_ladder": True, "highest_dose": True, "n_arms": 2},
    ]
    text = dv.summary(frame(rows))
    assert "(missing)" in text


def test_the_printout_marks_post_hoc_and_results_derived_tables():
    text = dv.summary(frame([{"any_arm": True, "enrollment": 50}]))
    assert "enrollment (actual, post-hoc)" in text
    assert "overall status (post-hoc)" in text
    assert "arms compared (from the results section)" in text
