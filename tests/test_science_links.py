import polars as pl
import pytest

from elman_exercise.science import links


def drugs_2006():
    return pl.DataFrame(
        {
            "id": ["C1", "C2", "C3"],
            "child_chembl_ids": [["C1S"], None, []],
            "mechanisms_of_action": [
                [
                    {
                        "action_type": "inhibitor",
                        "target_components": [{"ensembl": "ENSG1"}, {"ensembl": "ENSG2"}],
                    },
                    {"action_type": "blocker", "target_components": [{"ensembl": "ENSG1"}]},
                ],
                [{"action_type": "agonist", "target_components": [{"ensembl": "ENSG3"}]}],
                [{"action_type": "other", "target_components": None}],
            ],
        }
    )


def test_slim_targets_has_one_row_per_drug_target_and_action():
    slim = links.slim_targets_2006(drugs_2006())
    assert slim.rows() == [
        ("C1", "ENSG1", "blocker"),
        ("C1", "ENSG1", "inhibitor"),
        ("C1", "ENSG2", "inhibitor"),
        ("C2", "ENSG3", "agonist"),
    ]  # C3 has no target component, so no row


def test_action_words_for_one_drug_and_target_are_joined_and_sorted():
    slim = links.join_action_words(
        links.slim_targets_2006(drugs_2006()).with_columns(pl.col("drug_id"))
    )
    row = slim.filter((pl.col("drug_id") == "C1") & (pl.col("target_id") == "ENSG1"))
    assert row["action_type"].to_list() == ["blocker; inhibitor"]
    assert slim.height == 3


def test_a_drug_matches_the_2020_entry_through_parent_and_children():
    family = pl.DataFrame(
        {
            "drug_id": ["C1", "C1S", "NEWSALT", "C2", "NEW", "C3"],
            "parent_id": [None, "C1", "C2", None, None, None],
            "child_ids": [["C1S"], None, None, ["NEWSALT"], None, None],
        }
    )
    mapped = links.map_to_2006(family, links.slim_drugs_2006(drugs_2006()))
    pairs = set(mapped.iter_rows())
    assert ("C1", "C1") in pairs
    assert ("C1S", "C1") in pairs  # a salt known in 2020 as a child of C1
    assert ("NEWSALT", "C2") in pairs  # not known in 2020, but its parent is
    assert not any(d == "NEW" for d, _ in pairs)  # unknown in 2020: no 2020 targets


def test_investigational_drugs_exclude_comparators(tmp_path):
    drug = pl.Struct({"drugFromSource": pl.String, "drugId": pl.String})
    pl.DataFrame(
        {
            "id": ["NCT1"],
            "source": ["ClinicalTrials.gov"],
            "drugs": [
                [
                    {"drugFromSource": "newmab", "drugId": "D1"},
                    {"drugFromSource": "old pill", "drugId": "D2"},
                ]
            ],
            "diseases": [[]],
            "trialStartDate": ["2015-01-01"],
        },
        schema_overrides={"drugs": pl.List(drug)},
    ).write_parquet(tmp_path / "clinical_report.parquet")
    pl.DataFrame(
        {"id": ["D1", "D2"], "name": ["newmab", "old pill"], "drugType": ["x", "x"]}
    ).write_parquet(tmp_path / "drug_molecule.parquet")
    registry = pl.DataFrame({"nct_id": ["NCT1"], "exp_names": [["newmab"]]})
    drugs = links.build_trial_drugs(["NCT1"], registry, tmp_path)
    assert drugs.rows() == [
        ("NCT1", "D1")
    ]  # the comparator D2 is listed by Open Targets but not used


def test_join_gives_one_row_per_trial_drug_target_with_the_named_columns():
    trial_drug = pl.DataFrame({"nct_id": ["N1", "N1", "N2"], "drug_id": ["D1", "D1", "D9"]})
    drug_links = pl.DataFrame(
        {
            "drug_id": ["D1", "D1"],
            "target_id": ["ENSG1", "ENSG2"],
            "action_type": ["inhibitor", None],
        }
    )
    out = links.join_trial_targets(trial_drug, drug_links)
    assert out.columns == ["nct_id", "drug_id", "target_id", "action_type"]
    assert out.rows() == [("N1", "D1", "ENSG1", "inhibitor"), ("N1", "D1", "ENSG2", None)]


def test_public_function_uses_the_chosen_version(monkeypatch):
    monkeypatch.setattr(links, "download", lambda: None)
    monkeypatch.setattr(
        links, "trial_drugs", lambda: pl.DataFrame({"nct_id": ["N1"], "drug_id": ["D1"]})
    )
    seen = []

    def fake(version, merge_salts=False):
        seen.append((version, merge_salts))
        return pl.DataFrame(
            {"drug_id": ["D1"], "target_id": [f"T-{version}"], "action_type": [None]}
        )

    monkeypatch.setattr(links, "drug_targets", fake)
    assert links.investigational_targets("today")["target_id"].to_list() == ["T-today"]
    assert links.investigational_targets("2020")["target_id"].to_list() == ["T-2020"]
    assert seen == [
        ("today", True),
        ("2020", True),
    ]  # one rule everywhere: salts inherit the family
    with pytest.raises(ValueError):
        links.investigational_targets("2019")  # type: ignore[arg-type]


def test_the_20_06_download_is_pinned_with_version_and_checksums():
    assert len(links.DRUG_2006_FILES) == 7
    for file in links.DRUG_2006_FILES:
        assert "/platform/20.06/" in file.url
        assert len(file.sha256) == 64 and file.size > 0
    assert "20.06" in links.OT2006_RELEASE and "CC0" in links.LICENCE


def test_inherit_from_parent_gives_a_salt_the_targets_of_its_whole_family():
    table = pl.DataFrame(
        {
            "drug_id": ["P", "SALT1"],
            "target_id": ["ENSG1", "ENSG2"],
            "action_type": ["inhibitor", None],
        }
    )
    out = links.inherit_from_parent(table, {"SALT1": "P", "SALT2": "P"}, ["SALT2", "OTHER"])
    rows = sorted(out.select("drug_id", "target_id").rows())
    assert ("SALT2", "ENSG1") in rows and (
        "SALT2",
        "ENSG2",
    ) in rows  # from the parent and a sibling
    assert ("P", "ENSG2") in rows
    assert not any(d == "OTHER" for d, _ in rows)  # no family, no targets


def test_date_columns_read_a_month_as_the_whole_month():
    registry = pl.DataFrame(
        {"nct_id": ["N1", "N2"], "start": ["2015-03", None], "first_submit": ["2015-02-10", None]}
    )
    out = links.date_columns(registry)
    assert out["start_first"].to_list()[0].isoformat() == "2015-03-01"
    assert out["start_last"].to_list()[0].isoformat() == "2015-03-31"
    assert out["start_first"].to_list()[1] is None


def test_date_columns_give_the_first_and_last_day_of_a_month_only_submission_date():
    registry = pl.DataFrame(
        {
            "nct_id": ["N1", "N2", "N3"],
            "start": ["2015-03", "2015-03-10", None],
            "first_submit": ["2015-02", "2015-02-10", None],
        }
    )
    out = links.date_columns(registry).sort("nct_id")
    assert [d.isoformat() for d in out["submit_first"][:2]] == ["2015-02-01", "2015-02-10"]
    assert [d.isoformat() for d in out["submit_last"][:2]] == ["2015-02-28", "2015-02-10"]
    assert out["submit_last"][2] is None
    assert out["start_last"][1].isoformat() == "2015-03-10"  # a full date is a single day


def test_a_salt_with_its_own_catalogue_row_is_also_mapped_to_its_parent():
    drugs = pl.DataFrame({"drug_id": ["P", "S"], "child_ids": [["S"], None]})
    family = pl.DataFrame(
        {"drug_id": ["P", "S"], "parent_id": [None, None], "child_ids": [None, None]},
        schema={"drug_id": pl.String, "parent_id": pl.String, "child_ids": pl.List(pl.String)},
    )
    pairs = set(links.map_to_2006(family, drugs).iter_rows())
    assert ("S", "S") in pairs  # its own row is kept
    assert ("S", "P") in pairs  # and the parent that lists it as a child is added
    assert ("P", "P") in pairs and ("P", "S") not in pairs


# Row-for-row fingerprints of the two original versions. "2020" is unchanged since "2020_names" was
# added. "today" changed once, in round 12, when salts began to inherit their family's targets
# (18,912 rows then; the old digest was 133d7be6...): +99 rows, 21 more trials with a target.
FINGERPRINTS = {
    "today": (19011, "7ef887a75ddde98873aefbab09b19c164515f4975d7b757750313f34bcef60a1"),
    "2020": (16687, "9fa0c99c9c84fcfb837268025774895c8cd258164f422341adcdb3c205b1fdf5"),
}


@pytest.mark.skipif(not links.TRIAL_DRUGS.exists(), reason="needs the built data tables")
@pytest.mark.parametrize("version", ["today", "2020"])
def test_the_two_original_versions_are_unchanged_row_for_row(version):
    import hashlib

    table = links.investigational_targets(version).sort("nct_id", "drug_id", "target_id")
    rows, digest = FINGERPRINTS[version]
    assert table.height == rows
    assert hashlib.sha256(table.write_csv().encode()).hexdigest() == digest


def names_2006():
    drugs = pl.DataFrame(
        {
            "id": ["C1", "C2", "C3"],
            "pref_name": ["Newmab", "Old Pill", "Abc"],
            "synonyms": [["NM1", "12345678"], None, None],
            "trade_names": [["Brandex"], ["Oldpillex"], None],
        }
    )
    return links.slim_names_2006(drugs)


def test_20_06_names_are_normalised_and_short_or_numeric_ones_dropped():
    names = set(names_2006().iter_rows())
    assert ("C1", "newmab") in names and ("C1", "brandex") in names
    assert ("C2", "old pill") in names
    assert not any(n in {"nm1", "12345678", "abc"} for _, n in names)


def test_registry_names_match_only_the_20_06_table_by_whole_word_phrase():
    registry = pl.DataFrame(
        {
            "nct_id": ["N1", "N2", "N3"],
            "exp_names": [["newmab 10 mg"], ["placebo"], ["oldpillextra"]],
            "all_names": [["newmab 10 mg", "old pill"], ["placebo", "brandex"], ["oldpillextra"]],
        }
    )
    matched = links.match_names_2006(registry, names_2006())
    rows = set(matched.iter_rows())
    assert ("N1", "C1", True) in rows  # experimental-arm name holds a 20.06 name
    assert ("N1", "C2", False) in rows  # a comparator: matched, but not investigational
    assert ("N2", "C1", False) in rows  # trade name found only outside the experimental arm
    assert not any(n == "N3" for n, _, _ in rows)  # not a whole word, and no today's names help


def test_a_drug_catalogued_after_june_2020_cannot_be_found_by_name():
    registry = pl.DataFrame(
        {"nct_id": ["N1"], "exp_names": [["laterdrug"]], "all_names": [["laterdrug"]]}
    )
    assert links.match_names_2006(registry, names_2006()).is_empty()


def test_the_dated_version_gives_investigational_drugs_only_with_20_06_targets(monkeypatch):
    monkeypatch.setattr(links, "download", lambda: None)
    monkeypatch.setattr(links, "trial_ids", lambda: ["N1"])
    monkeypatch.setattr(
        links,
        "trial_drugs_2020_names",
        lambda: pl.DataFrame(
            {
                "nct_id": ["N1", "N1", "N2"],
                "drug_id": ["C1", "C2", "C1"],
                "is_exp": [True, False, True],
            }
        ),
    )
    monkeypatch.setattr(
        links,
        "drug_targets",
        lambda version, merge_salts=False: pl.DataFrame(
            {"drug_id": ["C1", "C2"], "target_id": ["ENSG1", "ENSG2"], "action_type": [None, None]}
        ),
    )
    out = links.investigational_targets("2020_names")
    assert out.rows() == [("N1", "C1", "ENSG1", None)]  # comparator C2 and non-benchmark N2 gone
    assert out.columns == ["nct_id", "drug_id", "target_id", "action_type"]


def test_generic_names_are_never_used_to_match_a_drug():
    drugs = pl.DataFrame(
        {
            "id": ["C1", "C2", "C3", "C4"],
            "pref_name": ["Insulin", "Saline", "Newmab", None],
            "synonyms": [["Water", "Placebo"], ["Vehicle", None], None, ["Oldpill"]],
            "trade_names": [None, ["Potassium"], None, None],
        }
    )
    names = links.slim_names_2006(drugs)  # a null preferred name or synonym does not break it
    assert ("C4", "oldpill") in set(names.iter_rows())
    registry = pl.DataFrame(
        {
            "nct_id": ["N1", "N2"],
            "exp_names": [["insulin glargine", "saline", "water", "placebo"], ["newmab"]],
            "all_names": [
                ["insulin glargine", "saline", "water", "placebo", "calcium", "oxygen"],
                ["newmab"],
            ],
        }
    )
    generic = ("insulin", "saline", "water", "oxygen", "glucose", "alcohol", "calcium")
    generic += ("potassium", "placebo", "vehicle")
    assert set(generic) <= links.GENERIC_NAMES
    matched = links.match_names_2006(registry, names)
    assert matched.rows() == [("N2", "C3", True)]  # the generics hit nothing, the real drug does
