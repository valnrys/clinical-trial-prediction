import polars as pl

from trial_benchmark.science import targets as tg

DRUG = pl.Struct({"drugFromSource": pl.String, "drugId": pl.String})


def reports(*rows):
    """rows of (nct id, [(name in the registry, drug id), ...])"""
    return pl.DataFrame(
        [
            {"id": nct, "drugs": [{"drugFromSource": n, "drugId": d} for n, d in drugs]}
            for nct, drugs in rows
        ],
        schema={"id": pl.String, "drugs": pl.List(DRUG)},
    )


MOLECULES = pl.DataFrame(
    {"id": ["D1", "D2", "D3"], "name": ["alphamab", "betapill", "gammacillin"]}
)
DRUG_TARGETS = pl.DataFrame(
    {
        "drugId": ["D1", "D1", "D2", "D3"],
        "targetId": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
    }
)


def test_hgnc_keeps_approved_genes_and_entrez_ids_become_numbers(tmp_path):
    path = tmp_path / "hgnc.txt"
    path.write_text(
        "hgnc_id\tsymbol\tentrez_id\tensembl_gene_id\tstatus\n"
        "HGNC:1\tAAA\t10\tENSG1\tApproved\n"
        "HGNC:2\tOLD\t11\tENSG2\tEntry Withdrawn\n"
        "HGNC:3\tCCC\t\tENSG3\tApproved\n"
    )
    table = tg.read_hgnc(path)
    assert table["symbol"].to_list() == ["AAA", "CCC"]
    assert table["entrez_id"].to_list() == [10, None]


def test_an_ensembl_id_shared_by_two_entrez_ids_is_not_joined_either():
    hgnc = pl.DataFrame(
        {
            "hgnc_id": ["H1", "H2", "H3"],
            "symbol": ["A", "B", "C"],
            "entrez_id": [1, 2, 3],
            "ensembl_gene_id": ["E1", "E2", "E2"],
        }
    )
    assert tg.entrez_to_ensembl(hgnc).rows() == [(1, "E1")]


def test_the_hgnc_file_is_a_dated_archive_not_the_live_file():
    assert "archive" in tg.HGNC_FILE.url and "2026-10-06" in tg.HGNC_FILE.name


def test_an_entrez_id_with_two_ensembl_ids_is_not_joined():
    hgnc = pl.DataFrame(
        {
            "hgnc_id": ["H1", "H2", "H3", "H4"],
            "symbol": ["A", "B", "C", "D"],
            "entrez_id": [1, 2, 2, 3],
            "ensembl_gene_id": ["E1", "E2", "E2b", None],
        }
    )
    mapping = tg.entrez_to_ensembl(hgnc)
    assert mapping.rows() == [(1, "E1")]  # 2 is ambiguous, 3 has no Ensembl id


def test_join_by_entrez_id_never_by_symbol():
    mapping = pl.DataFrame({"entrez_id": [1, 2], "ensembl_gene_id": ["E1", "E2"]})
    frame = pl.DataFrame({"entrez": [2, 9], "symbol": ["WRONG", "WRONG"]})
    joined = tg.with_ensembl(frame, "entrez", mapping)
    assert joined["ensembl_gene_id"].to_list() == ["E2", None]


def test_comparator_drugs_and_their_targets_are_left_out():
    # NCT1 tests alphamab against betapill. Only alphamab is in the experimental arm.
    ids = ["NCT1"]
    drugs = tg.investigational_drugs(
        ids,
        reports(("NCT1", [("alphamab", "D1"), ("betapill", "D2")])),
        MOLECULES,
        {"NCT1": ["alphamab"]},
    )
    assert drugs.rows() == [("NCT1", "D1", True)]
    targets = tg.investigational_targets(drugs, DRUG_TARGETS)
    assert targets["targetId"].to_list() == ["ENSG1", "ENSG2"]  # not betapill's ENSG3


def test_the_lead_drug_is_the_first_investigational_drug_in_list_order():
    drugs = tg.investigational_drugs(
        ["NCT1"],
        reports(("NCT1", [("betapill", "D2"), ("alphamab", "D1")])),
        MOLECULES,
        {"NCT1": ["alphamab", "betapill"]},
    )
    assert dict(zip(drugs["drugId"], drugs["is_lead"], strict=True)) == {"D1": False, "D2": True}


def test_a_trial_with_no_matching_drug_has_no_investigational_drug_and_no_lead():
    # The registry's treatment (a code name) matches nothing Open Targets lists.
    drugs = tg.investigational_drugs(
        ["NCT1"],
        reports(("NCT1", [("gammacillin", "D3")])),
        MOLECULES,
        {"NCT1": ["xyz 123"]},
    )
    assert drugs.height == 0


def test_a_lead_drug_without_a_known_target_stays_in_the_drug_table_but_not_the_target_table():
    drugs = tg.investigational_drugs(
        ["NCT1"],
        reports(("NCT1", [("newdrug", "D9"), ("alphamab", "D1")])),
        pl.concat([MOLECULES, pl.DataFrame({"id": ["D9"], "name": ["newdrug"]})]),
        {"NCT1": ["alphamab", "newdrug"]},
    )
    assert drugs.filter(pl.col("is_lead"))["drugId"].to_list() == ["D9"]
    targets = tg.investigational_targets(drugs, DRUG_TARGETS)
    assert set(targets["drugId"]) == {"D1"}


def test_target_counts_count_genes_not_rows():
    targets = pl.DataFrame(
        {"nct_id": ["N1", "N1", "N1", "N2"], "drugId": ["A", "A", "B", "A"],
         "targetId": ["E1", "E2", "E1", "E1"]}
    )  # fmt: skip
    counts = tg.target_counts(targets).sort("nct_id")
    assert counts["n_target_genes"].to_list() == [2, 1]
    assert counts["has_target"].to_list() == [True, True]


def test_several_targets_agree_or_become_multiple():
    frame = pl.DataFrame(
        {
            "nct_id": ["a", "a", "b", "b", "c", "c", "d"],
            "kind": ["Enzyme", "Enzyme", "Enzyme", "Ion channel", None, None, "Enzyme"],
        }
    )
    out = frame.group_by("nct_id").agg(tg.agree_or_multiple("kind")).sort("nct_id")
    assert out["kind"].to_list() == ["Enzyme", "multiple", None, "Enzyme"]


def test_has_target_and_the_target_count_are_labelled_as_hindsight():
    assert tg.FEATURES == {"has_target": "after the start", "n_target_genes": "after the start"}


def test_a_null_target_id_is_not_a_target():
    drugs = pl.DataFrame({"nct_id": ["N1"], "drugId": ["D1"]})
    table = pl.DataFrame({"drugId": ["D1", "D1"], "targetId": ["E1", None]})
    targets = tg.investigational_targets(drugs, table)
    assert targets["targetId"].to_list() == ["E1"]
    assert tg.target_counts(targets)["n_target_genes"].to_list() == [1]


def test_a_drug_listed_twice_under_two_names_is_one_drug_and_one_lead():
    drugs = tg.investigational_drugs(
        ["NCT1"],
        reports(("NCT1", [("alphamab", "D1"), ("alphamab injection", "D1")])),
        MOLECULES,
        {"NCT1": ["alphamab"]},
    )
    assert drugs.rows() == [("NCT1", "D1", True)]
