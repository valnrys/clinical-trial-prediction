import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from elman_exercise.models import ladder


def with_genetics(n: int, seed: int, fill_by_hand: bool) -> pl.DataFrame:
    """Trials with the genetics columns, as the real table has them.

    Unmapped trials have null at-start evidence. Trials with no lead drug have a null earlier-trial
    count, and so do a few with no start date. With `fill_by_hand` the fixture itself writes the
    "no support" and "no earlier trials" values, without calling the code under test.
    """
    rng = np.random.default_rng(seed)
    mapped = rng.random(n) < 0.5
    no_start_date = np.arange(n) < 2
    has_lead_drug = (rng.random(n) < 0.7) | no_start_date
    score = rng.random(n)
    earlier = rng.integers(0, 5, n)
    count = pl.Series("lead_drug_earlier_trials", earlier).cast(pl.Int64)
    count = pl.select(
        pl.when(pl.Series(has_lead_drug) & ~pl.Series(no_start_date))
        .then(count)
        .otherwise(None)
        .alias("lead_drug_earlier_trials")
    ).to_series()
    supported = pl.Series(score > 0.6).zip_with(pl.Series(mapped), pl.Series([None] * n))
    supported_investigational = pl.Series(score > 0.7).zip_with(
        pl.Series(mapped), pl.Series([None] * n)
    )
    frame = synthetic_trials(n, seed=seed).with_columns(
        pl.Series("mapped", mapped),
        pl.Series("lead_drug", np.where(has_lead_drug, "CHEMBL1", None).tolist(), dtype=pl.String),
        pl.Series("genetic_score_at_start", np.where(mapped, score, np.nan)).fill_nan(None),
        supported.alias("supported_at_start"),
        supported_investigational.alias("supported_at_start_investigational"),
        count,
    )
    if not fill_by_hand:
        return frame
    return frame.with_columns(
        pl.col("genetic_score_at_start").fill_null(0.0),
        pl.col("supported_at_start").fill_null(False),
        pl.col("supported_at_start_investigational").fill_null(False),
        pl.when(pl.col("lead_drug").is_null() & pl.col("lead_drug_earlier_trials").is_null())
        .then(0)
        .otherwise(pl.col("lead_drug_earlier_trials"))
        .alias("lead_drug_earlier_trials"),
    )


@pytest.mark.parametrize("boosted_trees", [False, True], ids=["logistic", "boosted trees"])
def test_honest_genetics_rungs_cannot_tell_which_trials_are_unmapped_or_have_no_lead_drug(
    boosted_trees,
):
    sizes = ((600, 1), (150, 2))
    nulls = [with_genetics(n, seed, fill_by_hand=False) for n, seed in sizes]
    filled = [with_genetics(n, seed, fill_by_hand=True) for n, seed in sizes]
    assert nulls[0]["genetic_score_at_start"].null_count() > 0
    assert nulls[0]["lead_drug_earlier_trials"].null_count() > 2
    assert filled[0]["genetic_score_at_start"].null_count() == 0

    from_nulls = ladder.fit_rung(nulls[0], ("genetics",), boosted_trees=boosted_trees)
    from_filled = ladder.fit_rung(filled[0], ("genetics",), boosted_trees=boosted_trees)
    assert from_nulls.predict(nulls[1]) == pytest.approx(from_filled.predict(filled[1]))
    assert from_nulls.predict(nulls[1]) == pytest.approx(from_nulls.predict(filled[1]))


def test_every_null_is_filled_whether_or_not_the_trial_is_mapped():
    frame = with_genetics(200, 3, fill_by_hand=False)
    # One mapped trial whose score is null, which a rule keyed on `mapped` would leave alone.
    row = frame["mapped"].arg_max()
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == row)
        .then(None)
        .otherwise(pl.col("genetic_score_at_start"))
        .alias("genetic_score_at_start")
    )
    assert frame["mapped"][row] and frame["genetic_score_at_start"][row] is None
    out = ladder.fill_unknown_as_no_support(frame)
    for name in ladder.NO_SUPPORT:
        assert out[name].null_count() == 0
    assert out["genetic_score_at_start"][row] == 0.0
    known = frame["genetic_score_at_start"].is_not_null()
    assert out.filter(known)["genetic_score_at_start"].to_list() == pytest.approx(
        frame.filter(known)["genetic_score_at_start"].to_list()
    )


def test_a_trial_with_no_lead_drug_has_no_earlier_trials_and_a_known_count_is_kept():
    frame = with_genetics(200, 4, fill_by_hand=False)
    out = ladder.fill_unknown_as_no_support(frame)
    no_drug = out.filter(pl.col("lead_drug").is_null())
    assert no_drug.height > 0
    assert no_drug["lead_drug_earlier_trials"].to_list() == [0] * no_drug.height
    known = frame["lead_drug_earlier_trials"].is_not_null()
    assert (
        out.filter(known)["lead_drug_earlier_trials"].to_list()
        == frame.filter(known)["lead_drug_earlier_trials"].to_list()
    )


def test_the_not_honest_rung_is_given_the_raw_columns(monkeypatch):
    frame = with_genetics(300, 5, fill_by_hand=False)
    monkeypatch.setattr(
        ladder, "columns_read", lambda tables, allowed: {"genetic_score_at_start": "dated"}
    )
    assert not isinstance(
        ladder.fit_rung(frame, ("genetics",), allowed=ladder.EVERYTHING), ladder.FilledPredictor
    )
    assert isinstance(ladder.fit_rung(frame, ("genetics",)), ladder.FilledPredictor)


def test_no_rung_reads_an_identifier_column():
    read = ladder.columns_read(("genetics",), ladder.EVERYTHING)
    assert not set(read) & ladder.IDENTIFIER_COLUMNS
    assert "lead_drug_earlier_trials" in read


def test_boosted_trees_with_words_is_refused():
    frame = with_genetics(50, 6, fill_by_hand=False)
    with pytest.raises(ValueError, match="Boosted trees"):
        ladder.fit_rung(frame, (ladder.WORD_TABLE,), boosted_trees=True)
