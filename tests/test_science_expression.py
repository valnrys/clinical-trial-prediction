import zipfile

import numpy as np
import polars as pl
import pytest

from trial_benchmark.science import expression as ex


def test_tau_is_zero_for_a_gene_equal_everywhere_and_one_for_a_single_tissue():
    log = np.log2(np.array([[10.0, 10.0, 10.0, 10.0], [100.0, 0.0, 0.0, 0.0]]) + 1)
    values = ex.tau(log)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)


def test_tau_of_a_gene_with_no_expression_is_not_a_number():
    assert np.isnan(ex.tau(np.zeros((1, 3)))[0])


def test_gtex_names_lose_their_version_and_pseudo_autosomal_copies_are_dropped(tmp_path):
    import gzip

    path = tmp_path / "gtex.gct.gz"
    with gzip.open(path, "wt") as out:
        out.write("#1.2\n3\t2\nName\tDescription\tLiver\tHeart_Left_Ventricle\n")
        out.write("ENSG1.5\tA\t1.0\t2.0\nENSG2.3_PAR_Y\tB\t5.0\t5.0\nENSG2.4\tB\t3.0\t4.0\n")
    table = ex.read_gtex(path)
    assert table["Name"].to_list() == ["ENSG1", "ENSG2"]
    assert "Description" not in table.columns


def test_hpa_is_pivoted_to_one_row_per_gene_with_missing_tissues_as_zero(tmp_path):
    text = "Gene\tGene name\tTissue\tnTPM\nE1\tA\tliver\t5.0\nE1\tA\theart muscle\t1.5\nE2\tB\tliver\t7.0\n"
    path = tmp_path / "hpa.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rna_tissue_consensus.tsv", text)
    wide = ex.read_hpa(path)
    assert wide.columns == ["Gene", "liver", "heart muscle"] or set(wide.columns) == {
        "Gene", "liver", "heart muscle",
    }  # fmt: skip
    assert wide.filter(pl.col("Gene") == "E2")["heart muscle"][0] == 0.0


# ---------------------------------------------------------------- disease -> tissue


def test_a_cancer_is_matched_to_its_organ_of_origin():
    assert ex.disease_tissues(["Lung Neoplasms"], ["Neoplasms"], [], "oncology") == ["lung"]


def test_a_specific_match_is_not_widened_by_a_broader_parent():
    # "Genital Diseases, Male" lists prostate and testis, but a prostate cancer is a prostate trial.
    parents = ["Genital Diseases, Male", "Urogenital Diseases", "Prostatic Diseases"]
    assert ex.disease_tissues(["Prostatic Neoplasms"], parents, [], None) == ["prostate"]
    # The pain term is a symptom: the eye, named by the parent, decides.
    assert ex.disease_tissues(
        ["Eye Infections", "Pain"], ["Eye Diseases", "Infections"], [], None
    ) == ["eye"]


def test_breast_diseases_listed_under_skin_diseases_do_not_count_as_skin():
    # In MeSH, breast diseases sit under "Skin Diseases"; the skin must not be matched.
    assert ex.disease_tissues(["Breast Diseases"], ["Skin Diseases"], [], None) == ["breast"]
    both = ["Breast Neoplasms", "Skin Diseases", "Breast Diseases"]
    assert ex.disease_tissues(["Triple Negative Breast Neoplasms"], both, [], None) == ["breast"]
    assert ex.disease_tissues(["Psoriasis"], ["Skin Diseases"], [], None) == ["skin"]


def test_pulmonary_hypertension_does_not_count_as_systemic_hypertension():
    tissues = ex.disease_tissues(["Hypertension, Pulmonary"], ["Hypertension"], [], None)
    assert "kidney" not in tissues and "lung" in tissues


def test_a_no_tissue_term_is_read_from_the_trials_own_terms_only():
    assert ex.disease_tissues(["Osteoarthritis"], ["Nervous System Diseases"], [], None) == []
    assert ex.disease_tissues(["Neoplasms"], ["Lung Neoplasms"], [], None) == ["lung"]
    # "Infections" is only a parent here, so it must not stop the broader nervous-system match.
    parents = ["Central Nervous System Infections", "Infections", "Nervous System Diseases"]
    assert ex.disease_tissues(["Meningitis"], parents, [], None) == ["brain"]


def test_a_child_of_a_tissue_less_branch_is_not_handed_to_the_broader_rules():
    # "Sarcoma" sits under a tissue-less branch; its ancestors would otherwise reach skin.
    parents = ["Neoplasms, Connective and Soft Tissue", "Skin and Connective Tissue Diseases"]
    assert ex.disease_tissues(["Sarcoma"], parents, ["soft tissue sarcoma"], "oncology") == []
    parents = ["Bacterial Infections", "Infections", "Urogenital Diseases"]
    assert ex.disease_tissues(["Gonorrhea"], parents, [], None) == []


def test_a_dropped_broad_term_is_not_brought_back_by_its_parent_or_the_fallbacks():
    parents = ["Skin Diseases", "Skin and Connective Tissue Diseases"]
    assert (
        ex.disease_tissues(["Skin Diseases, Metabolic"], parents, ["porphyria"], "dermatology")
        == []
    )
    parents = ["Vascular Diseases", "Cardiovascular Diseases"]
    assert ex.disease_tissues(["Paraproteinemias"], parents, [], None) == []


def test_broader_system_terms_then_condition_text_then_area_are_tried_in_that_order():
    assert ex.disease_tissues(["Respiratory Tract Diseases"], [], ["psoriasis"], "neurology") == [
        "lung"
    ]
    assert ex.disease_tissues([], [], ["plaque psoriasis"], "neurology") == ["skin"]
    assert ex.disease_tissues([], [], ["unusual syndrome"], "neurology") == ["brain"]
    assert ex.disease_tissues([], [], ["unusual syndrome"], "oncology") == []


# ---------------------------------------------------------------- features


GENES = pl.DataFrame(
    {
        "ensembl_gene_id": ["E1", "E2"],
        "gtex_tau": [0.9, 0.3],
        "hpa_n_tissues": [5, 40],
        "gtex_vital_max": [1.0, 8.0],
    }
)


def targets(*pairs):
    return pl.DataFrame(pairs, schema={"nct_id": pl.String, "targetId": pl.String}, orient="row")


def test_several_targets_give_the_broadest_the_widest_and_the_highest_vital_tissue():
    out = ex.gene_features(targets(("N1", "E1"), ("N1", "E2")), GENES)
    row = out.row(0, named=True)
    assert row["expr_tau"] == 0.3  # the broadest target
    assert row["expr_n_tissues"] == 40
    assert row["expr_vital_max"] == 8.0


MATCHED = pl.DataFrame(
    {
        "ensembl_gene_id": ["E1", "E1", "E2", "E2"],
        "tissue": ["liver", "heart", "liver", "heart"],
        "log2_ntpm": [9.0, 2.0, 3.0, 1.0],
        "best_rank": [1, 20, 8, 30],
    }
)


def trial_tissues(*rows):
    return pl.DataFrame(
        rows, schema={"nct_id": pl.String, "tissues": pl.List(pl.String)}, orient="row"
    )


def test_disease_tissue_expression_is_the_best_over_targets_and_matched_tissues():
    out = ex.matched_features(
        targets(("N1", "E1"), ("N1", "E2")), trial_tissues(("N1", ["liver", "heart"])), MATCHED
    )
    row = out.row(0, named=True)
    assert row["expr_disease_tissue"] == 9.0
    assert row["expr_disease_tissue_top3"] is True  # E1's liver rank is 1


def test_top_three_is_false_when_no_target_has_the_tissue_in_its_top_three():
    out = ex.matched_features(targets(("N1", "E2")), trial_tissues(("N1", ["liver"])), MATCHED)
    assert out["expr_disease_tissue_top3"][0] is False


def test_a_trial_with_no_matched_tissue_gets_no_disease_value():
    out = ex.matched_features(targets(("N1", "E1")), trial_tissues(("N1", [])), MATCHED)
    assert out.height == 0


def test_matched_tissue_rank_uses_the_best_member_of_a_group(tmp_path):
    # Every HPA tissue the table names is listed, mostly at zero. "immune_blood" is the highest
    # of its members (bone marrow here); the liver is second.
    levels = {"bone marrow": 50.0, "lymph node": 5.0, "liver": 20.0, "spleen": 1.0}
    members = sorted({m for _, hpa in ex.TISSUES.values() for m in hpa})
    text = "Gene\tGene name\tTissue\tnTPM\n" + "".join(
        f"E1\tX\t{m}\t{levels.get(m, 0.0)}\n" for m in members
    )
    path = tmp_path / ex.HPA.name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rna_tissue_consensus.tsv", text)
    table = ex.matched_tissue_table(tmp_path)
    immune = table.filter(pl.col("tissue") == "immune_blood").row(0, named=True)
    liver = table.filter(pl.col("tissue") == "liver").row(0, named=True)
    assert immune["best_rank"] == 1 and liver["best_rank"] == 2
    assert immune["log2_ntpm"] == pytest.approx(np.log2(51.0))


def hpa_zip(tmp_path, levels):
    """An HPA file listing every tissue the table names for one gene, at `levels` or zero."""
    members = sorted({m for _, hpa in ex.TISSUES.values() for m in hpa})
    text = "Gene\tGene name\tTissue\tnTPM\n" + "".join(
        f"E1\tX\t{m}\t{levels.get(m, 0.0)}\n" for m in members
    )
    with zipfile.ZipFile(tmp_path / ex.HPA.name, "w") as archive:
        archive.writestr("rna_tissue_consensus.tsv", text)


def test_a_gene_that_is_off_everywhere_is_not_top_three_in_alphabetically_early_tissues(tmp_path):
    hpa_zip(tmp_path, {})
    table = ex.matched_tissue_table(tmp_path)
    assert table["best_rank"].min() > ex.TOP_RANKS
    assert table["log2_ntpm"].max() == 0.0


def test_tissues_below_the_expression_threshold_rank_after_expressed_ones(tmp_path):
    hpa_zip(tmp_path, {"liver": 3.0, "heart muscle": 0.4})
    table = ex.matched_tissue_table(tmp_path)
    rank = dict(zip(table["tissue"], table["best_rank"], strict=True))
    assert rank["liver"] == 1
    assert rank["heart"] > ex.TOP_RANKS  # 0.4 nTPM is below the threshold of 1


def test_tied_tissues_share_the_best_rank(tmp_path):
    hpa_zip(tmp_path, {"liver": 5.0, "kidney": 5.0, "lung": 2.0})
    table = ex.matched_tissue_table(tmp_path)
    rank = dict(zip(table["tissue"], table["best_rank"], strict=True))
    assert rank["liver"] == rank["kidney"] == 1 and rank["lung"] == 3


def test_dating_labels():
    assert ex.FEATURES["expr_tau"] == "timeless"
    assert ex.FEATURES["expr_disease_tissue"] == "protocol"
