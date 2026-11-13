from datetime import date

import numpy as np
import polars as pl

from trial_benchmark.science import report as rp


def result(difference: float, low: float, high: float) -> rp.Result:
    return rp.Result(100, 100, 0.3, 0.4, difference, low, high)


def test_the_explore_and_confirm_columns_use_the_same_sign_convention():
    # Both periods say the first group did worse. Neither column may be flipped.
    contrast = rp.Contrast("depmap_common_essential", "yes minus no")
    pair = (result(-0.084, -0.15, -0.01), result(-0.060, -0.11, -0.01))
    row = rp.confirm_table([(contrast, pair)])[-1].split(" | ")
    assert row[2] == "−8.4"
    assert row[3].startswith("−6.0 (")
    assert row[4] == "yes" and row[5].startswith("yes")


def test_a_reversed_confirm_is_shown_with_its_raw_sign_and_marked_as_not_the_same_sign():
    contrast = rp.Contrast("depmap_common_essential", "yes minus no")
    pair = (result(-0.084, -0.15, -0.01), result(0.109, 0.05, 0.17))
    row = rp.confirm_table([(contrast, pair)])[-1].split(" | ")
    assert row[3].startswith("+10.9")
    assert row[4] == "no" and row[5].startswith("no")
    assert rp.confirm_summary([(contrast, pair)]) == (1, 0, [])


def test_explore_confirm_returns_raw_differences_for_a_group_that_does_worse_in_both_periods():
    rng = np.random.default_rng(1)
    n = 1600
    exposed = rng.random(n) < 0.5
    year = np.where(np.arange(n) % 2 == 0, 2012, 2017)
    frame = pl.DataFrame(
        {
            "flag": exposed,
            "any_arm": rng.random(n) < np.where(exposed, 0.2, 0.5),
            "primary_completion_date": [date(int(y), 6, 1) for y in year],
            "phase": "Phase 2",
            "sponsor_class": "industry",
            "therapeutic_area": "oncology",
            "modality": "small molecule",
            "lead_drug": [f"drug{i}" for i in range(n)],
        }
    )
    first, second = rp.explore_confirm(frame, rp.Contrast("flag", "yes minus no"))
    assert first.difference < -0.1 and second.difference < -0.1
    assert rp.same_sign(first, second)
