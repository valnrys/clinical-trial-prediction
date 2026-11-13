from datetime import date

import polars as pl
import pytest

from elman_exercise import dataset as ds


def drug(name, kind="DRUG", arms=("Experimental",), other_names=()):
    item = {"type": kind, "name": name, "armGroupLabels": list(arms)}
    if other_names:
        item["otherNames"] = list(other_names)
    return item


def study(
    nct="NCT00000001",
    pcd="2019-06",
    start="2017-01-15",
    status="COMPLETED",
    interventions=None,
    arm_types=(("Experimental", "EXPERIMENTAL"), ("Placebo", "PLACEBO_COMPARATOR")),
    phases=("PHASE2",),
    results=True,
    p="0.01",
    mesh=("Lung Diseases",),
    ancestors=("Respiratory Tract Diseases",),
    why_stopped=None,
    **extra,
):
    """A small synthetic trial that is eligible unless a keyword says otherwise."""
    outcome = {
        "type": "PRIMARY",
        "groups": [{"id": "G0", "title": "Placebo"}, {"id": "G1", "title": "Drug X"}],
        "analyses": [{"groupIds": ["G0", "G1"], "pValue": p, "nonInferiorityType": "SUPERIORITY"}],
    }
    status_module = {
        "overallStatus": status,
        "startDateStruct": {"date": start},
        "resultsFirstPostDateStruct": {"date": "2020-03-01"},
    }
    if why_stopped:
        status_module["whyStopped"] = why_stopped
    if pcd:
        status_module["primaryCompletionDateStruct"] = {"date": pcd}
    record = {
        "hasResults": results,
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "statusModule": status_module,
            "sponsorCollaboratorsModule": {"leadSponsor": {"class": "INDUSTRY"}},
            "designModule": {
                "phases": list(phases),
                "enrollmentInfo": {"count": 120, "type": "ACTUAL"},
                "designInfo": {"allocation": "RANDOMIZED", "maskingInfo": {"masking": "DOUBLE"}},
            },
            "armsInterventionsModule": {
                "armGroups": [{"label": label, "type": kind} for label, kind in arm_types],
                "interventions": list(interventions or [drug("Drug X")]),
            },
        },
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [outcome]}},
        "derivedSection": {
            "conditionBrowseModule": {
                "meshes": [{"term": t} for t in mesh],
                "ancestors": [{"term": t} for t in ancestors],
            }
        },
    }
    record["protocolSection"].update(extra)
    return record


# ---------------------------------------------------------------- dates and splits


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2020-12-31", date(2020, 12, 31)),
        ("2021-01", date(2021, 1, 1)),
        ("2021", None),
        ("", None),
        (None, None),
        ("2021-13", None),
    ],
)
def test_parse_date(text, expected):
    assert ds.parse_date(text) == expected


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2020, 12, 31), "dev"),
        (date(2021, 1, 1), "test"),
        (date(2023, 12, 31), "test"),
        (date(2024, 1, 1), "late"),
        (date(2025, 12, 31), "late"),
    ],
)
def test_split_boundaries(day, expected):
    assert ds.split_of(day) == expected


def test_a_month_only_date_counts_as_the_first_of_the_month():
    assert ds.split_of(ds.parse_date("2021-01")) == "test"
    assert ds.split_of(ds.parse_date("2020-12")) == "dev"


# ---------------------------------------------------------------- eligibility


def failed_step(record):
    index, _ = ds.first_failed_step(record)
    return None if index is None else ds.FLOW_STEPS[index][0]


def test_a_plain_trial_is_eligible():
    assert failed_step(study()) is None


@pytest.mark.parametrize(
    ("record", "step"),
    [
        (study(interventions=(drug("Mindfulness", "BEHAVIORAL"),)), "drug, biological"),
        (study(status="RECRUITING"), "Status"),
        (study(pcd=None), "Primary completion date"),
        (study(pcd="2026-02"), "Primary completion date"),
        (study(results=False), "posted results"),
        (study(p="<0.2"), "determined"),
    ],
)
def test_each_trial_is_counted_against_the_first_rule_it_fails(record, step):
    assert step in failed_step(record)


def test_a_trial_with_no_control_comparison_fails_the_comparison_step():
    record = study()
    outcome = record["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"][0]
    outcome["analyses"][0]["groupIds"] = ["G1"]  # one group against its own baseline
    assert "treatment-versus-control" in failed_step(record)


def test_the_last_date_in_the_window_is_eligible():
    assert failed_step(study(pcd="2025-12-31")) is None


def test_a_trial_failing_two_rules_is_counted_only_at_the_first():
    tables = ds.make_tables([study(status="RECRUITING", results=False)])
    removed = {step: out for step, _, out in tables.flow}
    assert removed["Status completed or terminated"] == 1
    assert removed["Has posted results"] == 0


def test_flow_counts_add_up_and_rows_match_what_is_left():
    studies = [
        study(nct="NCT00000001"),
        study(nct="NCT00000002", status="WITHDRAWN"),
        study(nct="NCT00000003", results=False),
        study(nct="NCT00000004", p="NS"),
    ]
    tables = ds.make_tables(studies)
    first = tables.flow[0]
    assert first[1:] == (4, 0)
    remaining = 4
    for _, left, removed in tables.flow:
        remaining -= removed
        assert left == remaining
    assert tables.flow[-1][1] == tables.rows.height == 1


# ---------------------------------------------------------------- modality


def kind(name, intervention_type="DRUG", other_names=()):
    return ds.intervention_kind(drug(name, intervention_type, other_names=other_names))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("adalimumab", "antibody"),
        ("Etanercept", "fusion protein"),
        ("aflibercept", "fusion protein"),
        ("inclisiran", "oligonucleotide"),
        ("nusinersen", "oligonucleotide"),
        ("tisagenlecleucel", "cell"),
        ("CAR-T cells", "cell"),
        ("onasemnogene abeparvovec", "gene"),
        ("Influenza vaccine", "vaccine"),
        ("aspirin", "small molecule"),
    ],
)
def test_drug_name_stems(name, expected):
    assert kind(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Excel", "Intercept", "generic aspirin", "Parcel", "calcitonin gene-related peptide",
        "Gleevec-treated", "Sprycel-based", "Surgicel", "Oxycel", "oral hygiene", "VEC-162", "Cel",
    ],
)  # fmt: skip
def test_words_that_only_look_like_a_stem_are_not_biologics(name):
    assert kind(name) == "small molecule"


def test_a_hyphenated_biologic_is_still_found():
    assert kind("adalimumab-atto") == "antibody"


def test_a_brand_name_is_not_a_stem_but_the_generic_in_other_names_is():
    assert kind("Aricept") == "small molecule"
    assert kind("donepezil", other_names=["Aricept"]) == "small molecule"
    assert kind("XYZ-123", other_names=["Humira", "adalimumab"]) == "antibody"


def test_with_no_stem_the_registered_type_decides():
    assert kind("ABC-123", "BIOLOGICAL") == "other biological"
    assert kind("ABC-123", "GENETIC") == "gene"
    assert kind("ABC-123", "DRUG") == "small molecule"


def test_the_first_matching_stem_wins():
    assert kind("anti-CD19 CAR-T cell therapy") == "cell"
    assert kind("AAV vector vaccine") == "gene"


def modality(*interventions, arm_types=(("Experimental", "EXPERIMENTAL"),)):
    return ds.modality_of(study(interventions=interventions, arm_types=arm_types))


def test_a_small_molecule_trial():
    assert modality(drug("aspirin")) == ("small molecule", False)


def test_a_biologic_trial_and_an_advanced_trial():
    assert modality(drug("adalimumab", "BIOLOGICAL")) == ("biologic", False)
    assert modality(drug("nusinersen")) == ("advanced", False)


def test_a_vaccine_only_trial_has_its_own_modality():
    assert modality(drug("HPV vaccine", "BIOLOGICAL")) == ("vaccine", True)


def test_a_vaccine_given_with_a_drug_sets_the_flag_and_the_drug_sets_the_modality():
    assert modality(drug("HPV vaccine", "BIOLOGICAL"), drug("aspirin")) == ("small molecule", True)


def test_interventions_spanning_two_groups_are_mixed():
    assert modality(drug("adalimumab"), drug("methotrexate")) == ("mixed", False)
    assert modality(drug("nusinersen"), drug("etanercept")) == ("mixed", False)


def test_two_kinds_within_one_group_are_not_mixed():
    assert modality(drug("adalimumab"), drug("etanercept")) == ("biologic", False)


def test_the_comparator_drug_is_ignored_when_the_experimental_arm_has_its_own():
    arms = (("New", "EXPERIMENTAL"), ("Old", "ACTIVE_COMPARATOR"))
    new_biologic = drug("adalimumab", arms=["New"])
    old_pill = drug("methotrexate", arms=["Old"])
    assert modality(new_biologic, old_pill, arm_types=arms) == ("biologic", False)


def test_a_drug_in_both_experimental_and_comparator_arms_is_a_fallback():
    arms = (("New", "EXPERIMENTAL"), ("Old", "ACTIVE_COMPARATOR"))
    shared = drug("adalimumab", arms=["New", "Old"])
    assert modality(shared, arm_types=arms) == ("biologic", False)
    background = drug("methotrexate", arms=["New", "Old"])
    placebo = drug("Placebo", arms=["Old"])
    # Nothing is unique to the experimental arm, so the experimental arm's drugs are used.
    assert modality(background, placebo, arm_types=arms) == ("small molecule", False)


def test_with_no_experimental_arm_all_drug_like_interventions_are_used():
    interventions = (drug("adalimumab", arms=["A"]), drug("Diet", "DIETARY_SUPPLEMENT"))
    record = study(interventions=interventions, arm_types=(("A", "OTHER"),))
    assert ds.modality_of(record) == ("biologic", False)


@pytest.mark.parametrize(
    "control", ["Placebo", "Vehicle cream", "Sham injection", "Saline", "Dummy"]
)
def test_a_placebo_like_intervention_never_sets_the_modality(control):
    # No arm labels, so every drug-like intervention is a candidate: the placebo must be dropped.
    biologic = drug("adalimumab", "BIOLOGICAL", arms=())
    assert modality(biologic, drug(control, arms=())) == ("biologic", False)
    vaccine = drug("HPV vaccine", "BIOLOGICAL", arms=())
    assert modality(vaccine, drug(control, arms=())) == ("vaccine", True)


def test_a_placebo_only_trial_has_no_modality():
    assert modality(drug("Placebo", arms=())) == (None, None)


def test_a_genetic_test_is_not_a_gene_therapy():
    assert modality(drug("Genotyping", "GENETIC", arms=())) == (None, None)
    assert modality(drug("onasemnogene abeparvovec", "GENETIC", arms=())) == ("advanced", False)


def test_a_trial_with_no_drug_like_intervention_has_no_modality():
    record = study(interventions=(drug("Diet", "DIETARY_SUPPLEMENT"),))
    assert ds.modality_of(record) == (None, None)


# ---------------------------------------------------------------- therapeutic area


@pytest.mark.parametrize(
    ("ancestors", "expected"),
    [
        (["Neoplasms", "Respiratory Tract Diseases"], "oncology"),
        (["Virus Diseases", "Immune System Diseases"], "infectious"),
        (["Endocrine System Diseases"], "metabolic_endocrine"),
        (["Eye Diseases"], "other"),
    ],
)
def test_the_first_area_in_priority_order_wins(ancestors, expected):
    assert ds.therapeutic_area(study(mesh=(), ancestors=ancestors)) == expected


def test_a_top_level_term_found_only_in_meshes_counts():
    assert ds.therapeutic_area(study(mesh=("Mental Disorders",), ancestors=())) == "psychiatry"


def test_no_mesh_terms_at_all_is_null_not_other():
    assert ds.therapeutic_area(study(mesh=(), ancestors=())) is None


# ---------------------------------------------------------------- the row


def row_of(record):
    index, label = ds.first_failed_step(record)
    assert index is None
    return ds.make_row(record, label)


def test_the_row_has_every_column_once():
    row = row_of(study())
    assert set(row) == set(ds.SCHEMA)
    groups = [name for columns in ds.COLUMN_GROUPS.values() for name in columns]
    assert len(groups) == len(set(groups)) == len(ds.SCHEMA)


def test_the_design_group_holds_nothing_from_the_results_nothing_post_hoc_and_no_label():
    design = set(ds.columns_in_group("design"))
    for other in ("results_structure", "post_hoc", "labels", "identifier"):
        assert not design & set(ds.columns_in_group(other))
    assert {"n_arms", "control_kind"} == set(ds.columns_in_group("results_structure"))
    assert not {"n_arms", "control_kind", "enrollment", "any_arm"} & design


def test_an_unknown_column_group_is_refused():
    with pytest.raises(ValueError, match="Unknown column group"):
        ds.columns_in_group("design_ish")


def test_a_typical_row():
    row = row_of(
        study(
            nct="NCT00000009",
            pcd="2019-06",
            start="2017-01-15",
            why_stopped="Slow recruitment",
            contactsLocationsModule={"locations": [{}, {}, {}]},
            oversightModule={"oversightHasDmc": True},
            eligibilityModule={"healthyVolunteers": False},
        )
    )
    assert row["nct_id"] == "NCT00000009"
    assert (row["split"], row["primary_completion_date"]) == ("dev", date(2019, 6, 1))
    assert (row["phase"], row["sponsor_class"]) == ("phase2", "INDUSTRY")
    assert (row["allocation"], row["masking"]) == ("RANDOMIZED", "DOUBLE")
    assert row["n_registered_arms"] == 2 and row["control_kind"] == "placebo"
    assert row["start_year"] == 2017 and row["has_dmc"] is True
    assert row["healthy_volunteers"] is False
    assert row["enrollment"] == 120 and row["overall_status"] == "COMPLETED"
    assert row["why_stopped_present"] is True and row["n_locations"] == 3
    assert row["duration_months"] == pytest.approx(28.5, abs=0.1)
    assert row["results_lag_months"] == pytest.approx(9.0, abs=0.1)
    assert row["any_arm"] is True and row["has_dose_ladder"] is False


def test_missing_fields_stay_null():
    row = row_of(study())
    for column in ("n_locations", "has_dmc", "healthy_volunteers", "intervention_model"):
        assert row[column] is None


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (("PHASE2",), "phase2"),
        (("PHASE1", "PHASE2"), "phase2"),
        (("PHASE3",), "phase3"),
        (("PHASE2", "PHASE3"), "phase3"),
        (("PHASE1",), None),
    ],
)
def test_phase_is_collapsed(phases, expected):
    assert ds.phase_of(study(phases=phases)) == expected


@pytest.mark.parametrize(
    ("kind_in", "expected"),
    [
        ("INDUSTRY", "INDUSTRY"),
        ("NIH", "NIH"),
        ("OTHER", "OTHER"),
        ("FED", "OTHER_GOV_OR_NETWORK"),
        ("NETWORK", "OTHER_GOV_OR_NETWORK"),
        ("INDIV", "OTHER_GOV_OR_NETWORK"),
    ],
)
def test_sponsor_class_groups_the_small_classes(kind_in, expected):
    record = study(sponsorCollaboratorsModule={"leadSponsor": {"class": kind_in}})
    assert ds.sponsor_class_of(record) == expected


def test_a_trial_stopped_early_shows_its_why_stopped_flag():
    row = row_of(study(status="TERMINATED"))
    assert (row["overall_status"], row["why_stopped_present"]) == ("TERMINATED", False)


# ---------------------------------------------------------------- the tables


def three_splits():
    return [
        study(nct="NCT00000003", pcd="2024-02", p="0.01"),
        study(nct="NCT00000001", pcd="2019-05", p="0.01"),
        study(nct="NCT00000002", pcd="2022-09", p="0.60"),
    ]


def test_rows_are_sorted_and_split_by_date():
    tables = ds.make_tables(three_splits())
    assert tables.rows["nct_id"].to_list() == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert tables.rows["split"].to_list() == ["dev", "test", "late"]
    assert ds.split_counts(tables.rows) == {"dev": 1, "test": 1, "late": 1}


def test_an_empty_download_gives_an_empty_table_with_the_full_schema():
    tables = ds.make_tables([])
    assert tables.rows.height == 0 and tables.rows.columns == list(ds.SCHEMA)
    assert ds.split_counts(tables.rows) == {"dev": 0, "test": 0, "late": 0}


def test_build_writes_labels_only_where_the_split_allows(tmp_path):
    ds.build(iter(three_splits()), root=tmp_path)
    dev = pl.read_parquet(tmp_path / "data/interim/dev.parquet")
    features = pl.read_parquet(tmp_path / "data/interim/test_features.parquet")
    assert dev["split"].unique().to_list() == ["dev"]
    assert set(ds.LABEL_COLUMNS) <= set(dev.columns)
    assert set(features["split"]) == {"test", "late"}
    assert not set(ds.LABEL_COLUMNS) & set(features.columns)
    # The held-out file is written, and deliberately never read back here.
    assert (tmp_path / "data/heldout/test_labels.parquet").is_file()


def test_trials_csv_lists_every_eligible_trial_with_no_labels(tmp_path):
    ds.build(iter(three_splits()), root=tmp_path)
    csv = pl.read_csv(tmp_path / "trials.csv")
    assert csv.columns == ["nct_id", "split"]
    assert csv["nct_id"].to_list() == ["NCT00000001", "NCT00000002", "NCT00000003"]


def test_flow_markdown_shows_each_step_and_the_split_counts(tmp_path):
    tables = ds.build(iter(three_splits()), root=tmp_path)
    text = (tmp_path / "flow.md").read_text()
    assert text == ds.flow_markdown(tables.flow, ds.split_counts(tables.rows))
    assert "test + late (extended test set) | 2" in text
    assert "(any arm)" in text


def test_the_public_tables_carry_no_test_or_late_label():
    tables = ds.make_tables(three_splits())
    held_out = tables.rows.filter(pl.col("split") != "dev")
    assert held_out.height == 2
    for column in ds.LABEL_COLUMNS:
        assert held_out[column].null_count() == held_out.height
    dev = tables.rows.filter(pl.col("split") == "dev")
    assert dev["any_arm"].to_list() == [True]


def test_build_returns_no_held_out_labels_but_still_writes_them(tmp_path):
    tables = ds.build(iter(three_splits()), root=tmp_path)
    held_out = tables.rows.filter(pl.col("split") != "dev")
    assert all(held_out[c].null_count() == held_out.height for c in ds.LABEL_COLUMNS)
    assert (tmp_path / "data/heldout/test_labels.parquet").is_file()


# ---------------------------------------------------------------- the trial list


def test_an_existing_trial_list_is_kept_and_the_difference_reported(tmp_path, capsys):
    ds.build(iter(three_splits()), root=tmp_path)
    before = (tmp_path / "trials.csv").read_text()
    # One trial dropped, one new, one moved from test to dev.
    changed = [
        study(nct="NCT00000001", pcd="2019-05"),
        study(nct="NCT00000002", pcd="2019-09", p="0.60"),
        study(nct="NCT00000009", pcd="2019-01"),
    ]
    capsys.readouterr()
    ds.build(iter(changed), root=tmp_path)
    assert (tmp_path / "trials.csv").read_text() == before
    assert "1 added, 1 removed, 1 moved" in capsys.readouterr().out


def test_the_update_flag_rewrites_the_trial_list(tmp_path):
    ds.build(iter(three_splits()), root=tmp_path)
    ds.build(iter([study(nct="NCT00000009")]), root=tmp_path, update_trial_list=True)
    assert pl.read_csv(tmp_path / "trials.csv")["nct_id"].to_list() == ["NCT00000009"]


def test_comparing_trial_lists_counts_added_removed_and_moved():
    old = pl.DataFrame({"nct_id": ["A", "B", "C"], "split": ["dev", "dev", "test"]})
    new = pl.DataFrame({"nct_id": ["B", "C", "D"], "split": ["dev", "late", "dev"]})
    assert ds.compare_trial_list(old, new) == {"added": 1, "removed": 1, "moved": 1}
