import polars as pl
import pytest

from elman_exercise.baselines import FEATURES as BASELINE_FEATURES
from elman_exercise.baselines import REGISTRATION
from elman_exercise.models import text as tm
from elman_exercise.models.registry import all_specs

TRAIN = pl.DataFrame(
    {
        "title": [
            "asthma inhaler study",
            "asthma pill study",
            "cancer infusion study",
            "cancer pill study",
        ],
        "conditions": ["asthma", "asthma", "cancer", "cancer"],
        "interventions": ["Zorbex", "Zorbex", "Quenlimab", "Quenlimab"],
        "primary_outcomes": ["fev1", "fev1", "survival", "survival"],
        "summary": ["a", "b", "c", "d"],
        "any_arm": [True, True, False, False],
    }
)


def predict_frame(**changes):
    frame = pl.DataFrame(
        {
            "title": ["asthma study with flarglo"],
            "conditions": ["asthma"],
            "interventions": ["Zorbex"],
            "primary_outcomes": ["fev1"],
            "summary": ["x"],
            "any_arm": [False],
        }
    )
    return frame.with_columns(**changes)


def test_documents_removes_own_drug_names_only_for_that_trial():
    frame = pl.DataFrame(
        {
            "title": ["Zorbex versus placebo in asthma", "Zorbex versus placebo in asthma"],
            "interventions": ["Zorbex 10 mg", "Placebo"],
        }
    )
    kept = tm.documents(frame.head(1), ("title",), drop_own_drugs=False)
    stripped = tm.documents(frame, ("title",), drop_own_drugs=True)
    assert kept == ["zorbex versus placebo in asthma"]
    assert stripped == ["versus placebo in asthma", "zorbex versus placebo in asthma"]


def test_only_drug_names_are_stripped_not_ordinary_words_or_numbers():
    names = "Zorbex 10 mg oral tablet; Placebo matching zorbex; Quenlimab (Brandex) 12 weeks"
    assert tm.drug_tokens(names) == {"zorbex", "quenlimab", "brandex"}


def test_a_trials_own_ordinary_words_stay_in_its_text():
    frame = pl.DataFrame(
        {
            "title": ["Zorbex oral dose 12 weeks versus placebo"],
            "interventions": ["Zorbex 10 mg oral tablet; Placebo"],
        }
    )
    (document,) = tm.documents(frame, ("title",), drop_own_drugs=True)
    assert document == "oral dose 12 weeks versus placebo"


def test_fields_and_list_items_are_joined_with_a_boundary_so_no_phrase_spans_two():
    frame = pl.DataFrame({"title": ["alpha beta"], "conditions": ["gamma; delta"]})
    (document,) = tm.documents(frame, ("title", "conditions"), drop_own_drugs=False)
    assert document == "alpha beta\ngamma\ndelta"
    terms = tm._terms(document)
    assert "alpha beta" in terms and "beta gamma" not in terms and "gamma delta" not in terms


def test_weights_are_named_word_and_registration_with_the_encoders_prefix_removed():
    from synthetic import synthetic_trials

    frame = synthetic_trials(200, seed=1).with_columns(
        title=pl.lit("asthma study of drug"),
        conditions=pl.lit("asthma"),
        primary_outcomes=pl.lit("fev1 change"),
    )
    names = list(tm.fit_features_and_words(frame, BASELINE_FEATURES[REGISTRATION]).weights())
    assert "word:asthma" in names
    assert "registration:phase=phase2" in names


def test_honesty_is_worked_out_from_how_each_text_field_is_dated():
    assert tm.reads_honestly(tm.WORD_FIELDS)
    assert tm.reads_honestly(("interventions",))
    assert not tm.reads_honestly((*tm.WORD_FIELDS, "summary"))


def test_documents_handles_missing_text():
    frame = pl.DataFrame(
        {"title": [None], "interventions": [None]},
        schema={"title": pl.String, "interventions": pl.String},
    )
    assert tm.documents(frame, ("title",), drop_own_drugs=True) == [""]


def test_vocabulary_comes_from_training_rows_only():
    model = tm.fit_words(TRAIN, min_df=1)
    vocabulary = model.pipeline.named_steps["features"].transformer_list[0][1][-1].vocabulary_
    assert "asthma" in vocabulary
    assert "flarglo" not in vocabulary


def test_a_word_only_in_the_predicted_rows_has_no_effect():
    model = tm.fit_words(TRAIN, min_df=1)
    plain = model.predict(predict_frame())
    changed = model.predict(predict_frame(title=pl.lit("asthma study with another unseen word")))
    assert plain == pytest.approx(changed)


def test_drug_names_model_reads_nothing_but_interventions():
    model = tm.fit_words(TRAIN, fields=("interventions",), min_df=1)
    same_drug_other_text = predict_frame(title=pl.lit("cancer"), conditions=pl.lit("cancer"))
    assert model.predict(predict_frame()) == pytest.approx(model.predict(same_drug_other_text))


def test_without_drug_names_model_does_not_read_interventions():
    model = tm.fit_words(TRAIN, drop_own_drugs=True, min_df=1)
    other_drug = predict_frame(interventions=pl.lit("Quenlimab"))
    assert model.predict(predict_frame()) == pytest.approx(model.predict(other_drug))


def test_predictions_do_not_depend_on_the_label_column():
    model = tm.fit_words(TRAIN, min_df=1)
    flipped = predict_frame(any_arm=pl.lit(True))
    assert model.predict(predict_frame()) == pytest.approx(model.predict(flipped))
    assert model.predict(predict_frame().drop("any_arm")) == pytest.approx(
        model.predict(predict_frame())
    )


def test_top_terms_lead_with_the_terms_that_point_to_success():
    high, low = tm.fit_words(TRAIN, min_df=1).top_terms(2)
    assert high[0][1] > 0 > low[0][1]


def test_specs_are_registered_with_one_leaky_model():
    specs = [s for s in all_specs() if s.family == "text"]
    assert len(specs) == len(tm.SPECS) == 4
    assert [s.name for s in specs if not s.honest] == ["text: words with summary (leaky)"]
    assert all(s.tables == ("text",) and not s.pretrained for s in specs)
