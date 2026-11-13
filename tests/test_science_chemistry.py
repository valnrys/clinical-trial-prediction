import polars as pl
import pytest

from elman_exercise.science import chemistry as ch


def test_action_types_fall_into_five_groups():
    assert ch.action_group("INHIBITOR") == "inhibitor"
    assert ch.action_group("BLOCKER") == "antagonist"
    assert ch.action_group("PARTIAL AGONIST") == "agonist"
    assert ch.action_group("OPENER") == "activator"
    assert ch.action_group("VACCINE ANTIGEN") == "other"
    assert ch.action_group(None) is None


def test_a_drug_with_mechanisms_in_two_groups_is_multiple():
    mechanism = pl.DataFrame(
        {
            "actionType": ["INHIBITOR", "ANTISENSE INHIBITOR", "AGONIST", "ANTAGONIST"],
            "chemblIds": [["D1"], ["D1"], ["D2"], ["D2", "D3"]],
        }
    )
    out = dict(ch.drug_actions(mechanism).iter_rows())
    assert out == {"D1": "inhibitor", "D2": "multiple", "D3": "antagonist"}


def test_chemistry_comes_from_the_lead_drug_only():
    drugs = pl.DataFrame({"nct_id": ["N1", "N1"], "drugId": ["D1", "D2"], "is_lead": [True, False]})
    targets = pl.DataFrame({"nct_id": ["N1"], "drugId": ["D2"], "targetId": ["E1"]})
    table = pl.DataFrame(
        {
            "drugId": ["D1", "D2"],
            "chem_mol_weight": [180.0, 999.0],
            "chem_action_type": ["inhibitor", "agonist"],
        }
    )
    out = ch.trial_features(drugs, targets, table)
    assert out.rows(named=True) == [
        {"nct_id": "N1", "chem_mol_weight": 180.0, "chem_action_type": "inhibitor"}
    ]


def test_a_trial_with_a_lead_drug_but_no_target_gets_no_chemistry():
    """Every science column exists only for trials with a target (see `science`)."""
    drugs = pl.DataFrame({"nct_id": ["N1", "N2"], "drugId": ["D1", "D1"], "is_lead": [True, True]})
    targets = pl.DataFrame({"nct_id": ["N1"], "drugId": ["D1"], "targetId": ["E1"]})
    table = pl.DataFrame({"drugId": ["D1"], "chem_mol_weight": [180.0]})
    assert ch.trial_features(drugs, targets, table)["nct_id"].to_list() == ["N1"]


def test_a_drug_with_no_value_at_all_is_left_out_of_the_drug_table(tmp_path):
    (tmp_path / ch.PROPERTIES).write_text(
        "drug_id,mol_weight,logp,polar_surface_area,qed,rule_of_five_violations\n"
        "D1,180.2,1.3,63.6,0.55,0\nBIOLOGIC,,,,,\n"
    )
    pl.DataFrame(
        {"actionType": ["INHIBITOR"], "chemblIds": [["D1"]], "targets": [["E1"]]}
    ).write_parquet(tmp_path / ch.MECHANISM_SLIM)
    assert ch.drug_table(tmp_path)["drugId"].to_list() == ["D1"]


def test_target_actions_pair_each_drug_with_each_of_its_targets(tmp_path):
    pl.DataFrame(
        {
            "actionType": ["INHIBITOR", "AGONIST"],
            "chemblIds": [["D1", "D2"], ["D1"]],
            "targets": [["E1", "E2"], ["E3"]],
        }
    ).write_parquet(tmp_path / ch.MECHANISM_SLIM)
    assert ch.target_actions(tmp_path).rows() == [
        ("D1", "E1", "INHIBITOR"),
        ("D1", "E2", "INHIBITOR"),
        ("D1", "E3", "AGONIST"),
        ("D2", "E1", "INHIBITOR"),
        ("D2", "E2", "INHIBITOR"),
    ]


def test_properties_csv_is_joined_to_actions_on_the_drug_id(tmp_path):
    (tmp_path / ch.PROPERTIES).write_text(
        "drug_id,mol_weight,logp,polar_surface_area,qed,rule_of_five_violations\n"
        "D1,180.2,1.3,63.6,0.55,0\nD2,500.0,,,,2\n"
    )
    pl.DataFrame(
        {"actionType": ["INHIBITOR"], "chemblIds": [["D1", "D3"]], "targets": [["E1"]]}
    ).write_parquet(tmp_path / ch.MECHANISM_SLIM)
    table = ch.drug_table(tmp_path).sort("drugId")
    assert table["drugId"].to_list() == ["D1", "D2", "D3"]
    assert table["chem_mol_weight"].to_list() == [180.2, 500.0, None]
    assert table["chem_action_type"].to_list() == ["inhibitor", None, "inhibitor"]
    assert table["chem_logp"][1] is None


def test_no_outcome_like_field_is_requested_from_chembl():
    fields = " ".join(ch.CHEMBL_FIELDS).lower()
    assert "phase" not in fields and "approv" not in fields and "oral" not in fields


# A response in the shape the ChEMBL molecule endpoint returns: numbers as text, null properties
# for a biologic. The aspirin values are the ones saved in data/raw/chemistry/chembl_properties.csv.
RECORDED = {
    "molecules": [
        {
            "molecule_chembl_id": "CHEMBL25",
            "molecule_properties": {
                "mw_freebase": "180.16",
                "alogp": "1.31",
                "psa": "63.60",
                "qed_weighted": "0.55",
                "num_ro5_violations": 0,
            },
        },
        {"molecule_chembl_id": "CHEMBL2108041", "molecule_properties": None},
    ]
}


def fake_service(monkeypatch, release):
    def get(url):
        if url.endswith("status.json"):
            return {} if release is None else {"chembl_db_version": release}
        assert "molecule_chembl_id__in=CHEMBL25%2CCHEMBL2108041" in url
        return RECORDED

    monkeypatch.setattr(ch, "_get_json", get)


def test_fetch_properties_saves_the_five_fields_and_leaves_a_biologic_empty(tmp_path, monkeypatch):
    fake_service(monkeypatch, ch.CHEMBL_RELEASE)
    path = tmp_path / ch.PROPERTIES
    ch.fetch_properties(["CHEMBL25", "CHEMBL2108041"], path)
    assert path.read_text().splitlines() == [
        "drug_id,mol_weight,logp,polar_surface_area,qed,rule_of_five_violations",
        "CHEMBL2108041,,,,,",
        "CHEMBL25,180.16,1.31,63.6,0.55,0.0",
    ]


def test_a_different_chembl_release_is_refused(tmp_path, monkeypatch):
    fake_service(monkeypatch, "ChEMBL_38")
    with pytest.raises(RuntimeError, match="ChEMBL_38"):
        ch.fetch_properties(["CHEMBL25", "CHEMBL2108041"], tmp_path / ch.PROPERTIES)


def test_a_status_page_that_names_no_release_is_tolerated_with_a_message(
    tmp_path, monkeypatch, capsys
):
    fake_service(monkeypatch, None)
    ch.fetch_properties(["CHEMBL25", "CHEMBL2108041"], tmp_path / ch.PROPERTIES)
    assert "checksum decides" in capsys.readouterr().out
