import math

import polars as pl
import pytest

from elman_exercise import adjusted as adj


def frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "exposed": pl.Boolean,
            "event": pl.Boolean,
            "stratum": pl.String,
            "sponsor": pl.String,
        },
        orient="row",
    )


def repeat(exposed, event, stratum, sponsor, times):
    return [(exposed, event, stratum, sponsor) for _ in range(times)]


def test_a_confounded_gap_disappears_inside_strata():
    """Exposed trials are mostly in stratum A, where events are common for everyone."""
    rows = (
        repeat(True, True, "A", "s1", 40)
        + repeat(True, False, "A", "s1", 40)
        + repeat(False, True, "A", "s2", 5)
        + repeat(False, False, "A", "s2", 5)
        + repeat(True, True, "B", "s3", 1)
        + repeat(True, False, "B", "s3", 9)
        + repeat(False, True, "B", "s4", 10)
        + repeat(False, False, "B", "s4", 90)
    )
    data = frame(rows)
    crude, _, _ = adj.adjusted(data, "exposed", "event", [], "sponsor", n_resamples=50)
    pooled, _, _ = adj.adjusted(data, "exposed", "event", ["stratum"], "sponsor", n_resamples=50)
    assert crude == pytest.approx((41 / 90) / (15 / 110))
    assert pooled == pytest.approx(1.0, abs=0.05)


def test_the_risk_difference_pools_strata_by_size():
    rows = (
        repeat(True, True, "A", "s1", 6)
        + repeat(True, False, "A", "s1", 4)
        + repeat(False, True, "A", "s2", 3)
        + repeat(False, False, "A", "s2", 7)
    )
    point, _, _ = adj.adjusted(
        frame(rows), "exposed", "event", ["stratum"], "sponsor", "difference", n_resamples=50
    )
    assert point == pytest.approx(0.3)


def test_a_stratum_with_only_one_group_carries_no_weight():
    rows = (
        repeat(True, True, "A", "s1", 5)
        + repeat(False, True, "B", "s2", 5)
        + (repeat(True, True, "B", "s3", 5) + repeat(False, False, "B", "s2", 5))
    )
    point, _, _ = adj.adjusted(
        frame(rows), "exposed", "event", ["stratum"], "sponsor", "difference", n_resamples=50
    )
    assert point == pytest.approx(0.5)


def test_the_interval_resamples_sponsors_not_trials():
    """One sponsor holds most of the exposed trials, so the interval must be wide."""
    rows = []
    for i in range(20):
        rows += repeat(False, i % 2 == 0, "A", f"o{i}", 5)
    rows += repeat(True, True, "A", "big", 50) + repeat(True, False, "A", "big", 50)
    point, low, high = adj.adjusted(frame(rows), "exposed", "event", [], "sponsor", n_resamples=400)
    assert low < point < high
    assert high - low > 0.1


def test_the_same_seed_gives_the_same_interval():
    rows = (
        repeat(True, True, "A", "s1", 5)
        + repeat(False, True, "A", "s2", 3)
        + (repeat(False, False, "A", "s3", 4) + repeat(True, False, "A", "s4", 2))
    )
    first = adj.adjusted(frame(rows), "exposed", "event", [], "sponsor", n_resamples=100)
    second = adj.adjusted(frame(rows), "exposed", "event", [], "sponsor", n_resamples=100)
    assert first == second


def test_a_ratio_with_no_events_in_the_comparison_group_is_nan_not_an_error():
    rows = repeat(True, True, "A", "s1", 3) + repeat(False, False, "A", "s2", 3)
    point, low, high = adj.adjusted(frame(rows), "exposed", "event", [], "sponsor", n_resamples=20)
    assert math.isnan(point) and math.isnan(low) and math.isnan(high)


def test_an_empty_frame_gives_nan():
    empty = frame([])
    assert all(math.isnan(x) for x in adj.adjusted(empty, "exposed", "event", [], "sponsor"))


def test_a_missing_sponsor_is_its_own_cluster():
    rows = [(True, True, "A", None), (True, False, "A", None), (False, True, "A", None)] * 3
    point, _, _ = adj.adjusted(frame(rows), "exposed", "event", [], "sponsor", n_resamples=20)
    assert point == pytest.approx(0.5 / 1.0)


def test_the_text_form_shows_ratio_and_difference():
    rows = (
        repeat(True, True, "A", "s1", 6)
        + repeat(True, False, "A", "s2", 4)
        + (repeat(False, True, "A", "s3", 3) + repeat(False, False, "A", "s4", 7))
    )
    data = frame(rows)
    assert adj.adjusted_text(data, "exposed", "event", [], "sponsor", n_resamples=50).startswith(
        "2.00"
    )
    text = adj.adjusted_text(
        data, "exposed", "event", [], "sponsor", measure="difference", n_resamples=50
    )
    assert text.startswith("+30.0")
