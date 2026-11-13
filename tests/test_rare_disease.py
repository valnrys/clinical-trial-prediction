import polars as pl

from elman_exercise import rare_disease as rd

XML = """<?xml version="1.0"?>
<JDBOR><DisorderList count="2">
<Disorder id="1"><OrphaCode>100</OrphaCode><PrevalenceList count="2">
 <Prevalence id="1"><PrevalenceType><Name lang="en">Point prevalence</Name></PrevalenceType>
  <PrevalenceClass><Name lang="en">1-9 / 100 000</Name></PrevalenceClass>
  <PrevalenceGeographic><Name lang="en">Europe</Name></PrevalenceGeographic></Prevalence>
 <Prevalence id="2"><PrevalenceType><Name lang="en">Cases/families</Name></PrevalenceType>
  <PrevalenceClass/>
  <PrevalenceGeographic><Name lang="en">Worldwide</Name></PrevalenceGeographic></Prevalence>
</PrevalenceList></Disorder>
<Disorder id="2"><OrphaCode>200</OrphaCode></Disorder>
</DisorderList></JDBOR>
"""


def entries(*rows):
    return pl.DataFrame(
        [
            {"orpha_code": "1", "prevalence_type": t, "geography": g, "prevalence_class": c}
            for t, g, c in rows
        ],
        schema=rd.PREVALENCE_SCHEMA,
    )


def chosen(*rows):
    result = rd.one_class_per_disease(entries(*rows))
    return result["class_rank"].to_list()


def test_the_orphanet_file_is_read_into_code_type_geography_and_class(tmp_path):
    path = tmp_path / "prev.xml"
    path.write_text(XML)
    table = rd.parse_prevalence(path)
    assert table.rows() == [
        ("100", "Point prevalence", "Europe", "1-9 / 100 000"),
        ("100", "Cases/families", "Worldwide", None),
    ]


def test_a_worldwide_entry_beats_a_european_one():
    assert chosen(
        ("Point prevalence", "Europe", "1-9 / 100 000"),
        ("Point prevalence", "Worldwide", "6-9 / 10 000"),
    ) == [4]


def test_europe_beats_the_united_states_and_both_beat_other_places():
    rows = [
        ("Point prevalence", "Japan", "<1 / 1 000 000"),
        ("Point prevalence", "United States", "1-9 / 1 000 000"),
        ("Point prevalence", "Europe", "1-9 / 100 000"),
    ]
    assert chosen(*rows) == [2]
    assert chosen(*rows[:2]) == [1]


def test_point_prevalence_beats_birth_prevalence_which_beats_incidence():
    rows = [
        ("Annual incidence", "Worldwide", "<1 / 1 000 000"),
        ("Prevalence at birth", "Worldwide", "1-9 / 1 000 000"),
        ("Point prevalence", "Worldwide", "1-9 / 100 000"),
    ]
    assert chosen(*rows) == [2]
    assert chosen(*rows[:2]) == [1]


def test_the_kind_of_measure_outranks_geography():
    """A worldwide incidence must not beat a European prevalence: the cut is a prevalence cut."""
    rows = [
        ("Annual incidence", "Worldwide", "1-5 / 10 000"),
        ("Point prevalence", "Europe", ">1 / 1000"),
    ]
    assert chosen(*rows) == [5]


def test_geography_breaks_ties_between_entries_of_the_same_measure():
    rows = [
        ("Annual incidence", "Europe", "1-9 / 100 000"),
        ("Annual incidence", "Worldwide", "6-9 / 10 000"),
    ]
    assert chosen(*rows) == [4]


def test_the_earlier_geography_first_rule_is_still_available_for_comparison():
    rows = [
        ("Annual incidence", "Worldwide", "1-5 / 10 000"),
        ("Point prevalence", "Europe", ">1 / 1000"),
    ]
    old = rd.one_class_per_disease(entries(*rows), measure_first=False)
    assert old["class_rank"].to_list() == [3]


def test_birth_and_lifetime_prevalence_tie_and_the_more_common_class_wins():
    rows = [
        ("Prevalence at birth", "Worldwide", "1-9 / 1 000 000"),
        ("Lifetime Prevalence", "Worldwide", "1-5 / 10 000"),
    ]
    assert chosen(*rows) == [3]


def test_entries_without_a_usable_class_are_ignored():
    rows = [
        ("Point prevalence", "Worldwide", "Unknown"),
        ("Point prevalence", "Worldwide", None),
        ("Annual incidence", "Europe", "1-9 / 100 000"),
    ]
    assert chosen(*rows) == [2]


def test_a_code_with_only_unusable_entries_gets_no_class():
    assert chosen(("Point prevalence", "Worldwide", "Not yet documented")) == []


def test_disease_ids_link_to_orphanet_by_cross_reference_and_by_their_own_id():
    disease = pl.DataFrame(
        {
            "id": ["MONDO_1", "Orphanet_300", "EFO_9"],
            "dbXRefs": [["Orphanet:100", "OMIM:5"], [], ["MESH:D1"]],
        }
    )
    pairs = rd.disease_orphanet_codes(disease).sort("disease_id").rows()
    assert pairs == [("MONDO_1", "100"), ("Orphanet_300", "300")]


def classify(trial_disease_pairs, codes, classes):
    result = rd.classify_trials(
        pl.DataFrame(trial_disease_pairs, schema=["nct_id", "disease_id"], orient="row"),
        pl.DataFrame(codes, schema=["disease_id", "orpha_code"], orient="row"),
        pl.DataFrame(
            classes, schema={"orpha_code": pl.String, "class_rank": pl.Int64}, orient="row"
        ),
    )
    return {r["nct_id"]: r for r in result.iter_rows(named=True)}


CODES = [("rare1", "1"), ("rare2", "2"), ("common", "3"), ("noclass", "4")]
CLASSES = [("1", 2), ("2", 3), ("3", 4)]


def test_a_trial_with_a_rare_disease_is_rare():
    out = classify([("T1", "rare1")], CODES, CLASSES)
    assert out["T1"]["group"] == rd.RARE
    assert out["T1"]["band"] == 2


def test_the_rare_cut_is_one_to_five_per_ten_thousand():
    out = classify([("T1", "rare2"), ("T2", "common")], CODES, CLASSES)
    assert out["T1"]["group"] == rd.RARE
    assert out["T2"]["group"] == rd.NOT_RARE


def test_one_rare_disease_among_common_ones_makes_the_trial_rare_but_not_all_rare():
    out = classify([("T1", "rare1"), ("T1", "common"), ("T1", "other")], CODES, CLASSES)
    assert out["T1"]["group"] == rd.RARE
    assert out["T1"]["all_rare"] is False
    assert out["T1"]["n_diseases"] == 3


def test_a_trial_whose_every_disease_has_an_entry_but_no_class_is_unclear():
    out = classify([("T1", "noclass")], CODES, CLASSES)
    assert out["T1"]["group"] == rd.UNCLEAR


def test_an_unclear_disease_next_to_a_common_one_leaves_the_trial_not_rare():
    out = classify([("T1", "noclass"), ("T1", "common")], CODES, CLASSES)
    assert out["T1"]["group"] == rd.NOT_RARE


def test_a_disease_with_no_orphanet_entry_counts_as_not_rare():
    out = classify([("T1", "unknown_disease")], CODES, CLASSES)
    assert out["T1"]["group"] == rd.NOT_RARE


def test_two_orphanet_codes_for_one_disease_use_the_more_common_class():
    codes = [("d", "1"), ("d", "3")]
    assert classify([("T1", "d")], codes, CLASSES)["T1"]["group"] == rd.NOT_RARE


def test_the_band_is_the_class_of_the_rarest_disease_in_the_trial():
    out = classify([("T1", "rare1"), ("T1", "rare2")], CODES, CLASSES)
    assert out["T1"]["band"] == 2
    assert out["T1"]["band_commonest"] == 3


def test_a_common_disease_does_not_set_the_band():
    out = classify([("T1", "rare2"), ("T1", "common")], CODES, CLASSES)
    assert out["T1"]["band"] == 3
    assert out["T1"]["band_commonest"] == 3


def write_reports(path, rows):
    diseases = pl.Struct({"diseaseId": pl.String})
    pl.DataFrame(
        [{"id": nct, "diseases": [{"diseaseId": d} for d in ds]} for nct, ds in rows],
        schema={"id": pl.String, "diseases": pl.List(diseases)},
    ).write_parquet(path)


def test_trials_the_open_targets_table_does_not_list_are_unmapped(tmp_path):
    write_reports(tmp_path / "clinical_report.parquet", [("nct1", ["rare1"]), ("NCT2", [])])
    pl.DataFrame({"id": ["rare1"], "dbXRefs": [["Orphanet:1"]]}).write_parquet(
        tmp_path / "disease.parquet"
    )
    orphanet = tmp_path / "orphanet"
    orphanet.mkdir()
    entries(("Point prevalence", "Worldwide", "1-9 / 100 000")).write_parquet(
        orphanet / "prevalence.parquet"
    )
    flags = rd.rare_flags(["NCT1", "NCT2", "NCT3"], orphanet, tmp_path)
    assert dict(flags.select("nct_id", "group").iter_rows()) == {
        "NCT1": rd.RARE,
        "NCT2": rd.UNMAPPED,
        "NCT3": rd.UNMAPPED,
    }
