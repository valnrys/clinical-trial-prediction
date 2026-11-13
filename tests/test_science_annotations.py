import polars as pl
import pytest

from trial_benchmark.science import annotations as an


def test_location_priority_is_membrane_then_secreted_then_nuclear_then_intracellular():
    assert (
        an.location_class(["Cell membrane ; Single-pass type I membrane protein", "Secreted"])
        == "membrane"
    )
    assert an.location_class(["Cytoplasm", "Secreted"]) == "secreted"
    assert an.location_class(["Cytoplasm", "Nucleus"]) == "nuclear"
    assert an.location_class(["Endoplasmic reticulum membrane", "Cytoplasm"]) == "intracellular"
    assert an.location_class([]) is None and an.location_class(None) is None


def test_an_organelle_membrane_is_not_the_cell_surface():
    assert an.location_class(["Nucleus membrane"]) == "nuclear"
    assert an.location_class(["Golgi apparatus membrane"]) == "intracellular"
    assert an.location_class(["Mitochondrion outer membrane"]) == "intracellular"


@pytest.mark.parametrize(
    "location",
    [
        "Membrane",
        "Membrane ; Multi-pass membrane protein",
        "Apical cell membrane",
        "Postsynaptic cell membrane",
        "Basolateral cell membrane",
        "Plasma membrane",
        "Cell surface",
        "Cell membrane ; Single-pass type I membrane protein",
    ],
)
def test_a_surface_membrane_with_a_qualifier_is_still_the_cell_surface(location):
    assert an.location_class([location]) == "membrane"


def test_level_one_class_ignores_other_levels_and_calls_several_classes_multiple():
    one = [
        {"id": 1, "label": "Kinase", "level": "l2"},
        {"id": 2, "label": "Enzyme", "level": "l1"},
    ]
    two = [*one, {"id": 3, "label": "Adhesion", "level": "l1"}]
    assert an.level_one_class(one) == "Enzyme"
    assert an.level_one_class(two) == "multiple"
    assert an.level_one_class([{"id": 1, "label": "Kinase", "level": "l2"}]) is None
    assert an.level_one_class(None) is None


def make_directory(tmp_path):
    cls = pl.Struct({"id": pl.Int64, "label": pl.String, "level": pl.String})
    loc = pl.Struct({"location": pl.String})
    pl.DataFrame(
        {
            "id": ["E1", "E2"],
            "targetClass": [[{"id": 1, "label": "Enzyme", "level": "l1"}], []],
            "subcellularLocations": [[{"location": "Nucleus"}], None],
        },
        schema={"id": pl.String, "targetClass": pl.List(cls), "subcellularLocations": pl.List(loc)},
    ).write_parquet(tmp_path / an.TARGET_SLIM)
    pl.DataFrame(
        {"targetId": ["E1", "E2"], "geneticConstraint": [0.9, -0.5], "mouseKOScore": [0.8, None]}
    ).write_parquet(tmp_path / an.PRIORITISATION_SLIM)


def test_scores_are_flipped_so_that_higher_means_less_tolerant(tmp_path):
    make_directory(tmp_path)
    genes = an.gene_table(tmp_path).sort("targetId")
    assert genes["constraint"].to_list() == [-0.9, 0.5]
    assert genes["mouse_knockout"].to_list() == [-0.8, None]
    assert genes["target_class"].to_list() == ["Enzyme", None]
    assert genes["target_location"].to_list() == ["nuclear", None]


def test_trial_features_take_the_most_constrained_target_and_multiple_for_mixed_classes():
    genes = pl.DataFrame(
        {
            "targetId": ["E1", "E2", "E3"],
            "target_class": ["Enzyme", "Ion channel", "Enzyme"],
            "target_location": ["membrane", "membrane", "nuclear"],
            "constraint": [0.1, 0.7, None],
            "mouse_knockout": [None, None, None],
        },
        schema_overrides={"mouse_knockout": pl.Float64},
    )
    targets = pl.DataFrame(
        {"nct_id": ["N1", "N1", "N2", "N2"], "targetId": ["E1", "E2", "E1", "E3"]}
    )
    out = an.trial_features(targets, genes).sort("nct_id")
    assert out["target_class"].to_list() == ["multiple", "Enzyme"]
    assert out["target_location"].to_list() == ["membrane", "multiple"]
    assert out["constraint"].to_list() == [pytest.approx(0.7), pytest.approx(0.1)]
    assert out["mouse_knockout"].to_list() == [None, None]


def test_the_prioritisation_columns_read_never_include_clinical_stage():
    names = " ".join((*an.PRIORITISATION_COLUMNS, *an.TARGET_COLUMNS)).lower()
    assert "stage" not in names and "safety" not in names and "clinical" not in names


def test_a_perinuclear_cytoplasm_annotation_is_not_nuclear():
    assert an.location_class(["Cytoplasm, perinuclear region"]) == "intracellular"
    assert an.location_class(["Cytoplasm, perinuclear region", "Nucleus"]) == "nuclear"
    assert an.location_class(["Nucleolus"]) == "nuclear"


def test_the_trace_uses_the_same_cell_surface_rule_as_the_build():
    from trial_benchmark.science import trace

    assert trace.SURFACE.pattern == an.SURFACE.pattern
    assert not an.SURFACE.search("membrane-bound organelle")
