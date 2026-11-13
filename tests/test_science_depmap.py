import polars as pl
import pytest

from elman_exercise.science import depmap as dm


def write_matrices(tmp_path):
    """Four cell lines (two skin, two breast) and three genes. One effect is missing."""
    header = ",G1 (11),G2 (22),G3 (33)\n"
    effect = header + "M1,-1.0,-0.2,0.1\nM2,-1.0,-0.4,\nM3,-1.0,0.2,0.0\nM4,-1.0,0.0,0.1\n"
    dependency = header + "M1,0.9,0.1,0.0\nM2,0.9,0.6,\nM3,0.9,0.1,0.0\nM4,0.9,0.1,0.0\n"
    model = "ModelID,OncotreeLineage\nM1,Skin\nM2,Skin\nM3,Breast\nM4,Breast\n"
    paths = []
    for name, text in (("effect.csv", effect), ("dep.csv", dependency), ("model.csv", model)):
        (tmp_path / name).write_text(text)
        paths.append(tmp_path / name)
    return paths


def test_the_matrices_are_reduced_to_gene_and_lineage_summaries(tmp_path):
    genes, lineages = dm.summarise(*write_matrices(tmp_path))
    g = {row["entrez_id"]: row for row in genes.iter_rows(named=True)}
    assert g[11]["mean_effect"] == pytest.approx(-1.0)
    assert g[22]["mean_effect"] == pytest.approx((-0.2 - 0.4 + 0.2 + 0.0) / 4)
    assert g[22]["fraction_dependent"] == pytest.approx(0.25)  # only M2 is above 0.5
    assert g[33]["n_lines"] == 3  # the empty cell is not a line
    assert g[33]["mean_effect"] == pytest.approx((0.1 + 0.0 + 0.1) / 3)
    skin = lineages.filter((pl.col("lineage") == "Skin") & (pl.col("entrez_id") == 22))
    assert skin["n_lines"][0] == 2 and skin["sum_effect"][0] == pytest.approx(-0.6)


def test_fraction_dependent_counts_only_lines_that_have_a_dependency_probability(tmp_path):
    effect, dependency, model = write_matrices(tmp_path)
    # G3 has a probability for one line only (M1, 0.9); its effect score exists for three lines.
    dependency.write_text(
        ",G1 (11),G2 (22),G3 (33)\nM1,0.9,0.1,0.9\nM2,0.9,0.6,\nM3,0.9,0.1,\nM4,0.9,0.1,\n"
    )
    genes, _ = dm.summarise(effect, dependency, model)
    g = {row["entrez_id"]: row for row in genes.iter_rows(named=True)}
    assert g[33]["fraction_dependent"] == pytest.approx(1.0)  # 1 of 1, not 1 of the 3 effects
    assert g[33]["n_lines"] == 3


def test_a_cell_line_with_no_lineage_does_not_crash_the_summary(tmp_path):
    effect, dependency, model = write_matrices(tmp_path)
    model.write_text("ModelID,OncotreeLineage\nM1,Skin\nM2,\nM3,Breast\nM4,Breast\n")
    _, lineages = dm.summarise(effect, dependency, model)
    assert "Unknown" in lineages["lineage"].to_list()


def test_matrices_with_different_rows_are_refused(tmp_path):
    effect, dependency, model = write_matrices(tmp_path)
    dependency.write_text(dependency.read_text().replace("M2,", "MX,"))
    with pytest.raises(ValueError, match="Row order"):
        dm.summarise(effect, dependency, model)


def test_entrez_id_is_read_from_the_column_name():
    assert dm.entrez_of("TP53 (7157)") == 7157


# ---------------------------------------------------------------- condition -> lineage


def test_mesh_terms_come_first_and_text_only_if_no_mesh_term_matched():
    assert dm.lineages_for(["Breast Neoplasms"], ["lung cancer"]) == ["Breast"]
    assert dm.lineages_for([], ["Non-small cell lung cancer"]) == ["Lung"]


def test_words_that_are_not_tumours_do_not_match():
    assert dm.lineages_for(["Breast Diseases", "Mastalgia"], ["breast pain"]) == []
    assert dm.lineages_for(["Fibroids"], ["uterine fibroid tumour"]) == []
    assert dm.lineages_for(["Keratosis, Actinic"], []) == []


def test_a_supportive_care_trial_in_cancer_patients_gets_no_lineage():
    assert dm.lineages_for(["Colorectal Neoplasms", "Mucositis"], []) == []
    assert dm.lineages_for(["Breast Neoplasms"], ["Breast cancer", "Nausea and vomiting"]) == []
    assert dm.lineages_for(["Lung Neoplasms", "Pain"], []) == []
    assert dm.lineages_for(["Lung Neoplasms"], []) == ["Lung"]


def test_cancers_whose_names_contain_a_benign_looking_word_are_still_reached():
    assert dm.lineages_for(["Thrombocythemia, Essential"], []) == ["Myeloid"]
    assert dm.lineages_for(["Myelodysplastic Syndromes", "Myelodysplasia"], []) == ["Myeloid"]
    assert dm.lineages_for(["Ovarian Cystadenocarcinoma"], []) == ["Ovary/Fallopian Tube"]
    assert dm.lineages_for(["Cystectomy"], []) == []
    assert dm.lineages_for(["Venous Thrombosis"], []) == []


def test_meningioma_and_ependymoma_reach_the_brain_lineage():
    assert dm.lineages_for(["Meningioma"], []) == ["CNS/Brain"]
    assert dm.lineages_for(["Ependymoma"], []) == ["CNS/Brain"]


def test_mesh_wording_that_puts_the_cell_type_after_leukemia_is_lymphoid():
    assert dm.lineages_for(["Leukemia, Lymphocytic, Chronic, B-Cell"], []) == ["Lymphoid"]
    assert dm.lineages_for(["Leukemia, Lymphoid"], []) == ["Lymphoid"]


def test_a_generic_leukemia_term_does_not_widen_a_specific_blood_cancer():
    assert dm.lineages_for(["Leukemia, Myeloid, Acute", "Leukemia"], []) == ["Myeloid"]
    assert dm.lineages_for(["Leukemia"], []) == ["Lymphoid", "Myeloid"]
    # The specific cancer is named in the condition text, the generic term only in MeSH.
    assert dm.lineages_for(["Leukemia"], ["Acute Myeloid Leukemia"]) == ["Myeloid"]


def test_a_vague_cancer_term_maps_to_no_lineage():
    assert dm.lineages_for(["Neoplasms"], ["advanced solid tumours"]) == []


def test_the_english_word_all_is_not_acute_lymphoblastic_leukaemia():
    assert dm.lineages_for([], ["All Solid Tumors"]) == []
    assert dm.lineages_for([], ["ALL SOLID TUMORS"]) == []
    assert dm.lineages_for([], ["Prostate cancer, all stages"]) == ["Prostate"]
    assert dm.lineages_for([], ["ALL"]) == ["Lymphoid"]
    assert dm.lineages_for([], ["Relapsed ALL in children"]) == ["Lymphoid"]


def test_acronyms_match_only_in_capitals_and_the_english_word_gist_is_not_a_tumour():
    assert dm.lineages_for([], ["the gist of the study"]) == []
    assert dm.lineages_for([], ["GIST"]) == ["Soft Tissue"]
    assert dm.lineages_for([], ["AML"]) == ["Myeloid"]
    assert dm.lineages_for([], ["aml"]) == []
    assert dm.lineages_for([], ["HCC"]) == ["Liver"]
    assert dm.lineages_for([], ["RCC"]) == ["Kidney"]
    assert dm.lineages_for([], ["NSCLC"]) == ["Lung"]
    assert dm.lineages_for([], ["cll", "mm", "mds", "hcc"]) == []  # lower-case ordinary text


def test_a_supportive_veto_does_not_blank_a_myelodysplastic_syndrome():
    assert dm.lineages_for(["Anemia, Refractory"], ["Myelodysplastic Syndrome"]) == ["Myeloid"]
    assert dm.lineages_for(["Anemia, Refractory, with Excess of Blasts"], ["MDS"]) == ["Myeloid"]
    assert dm.lineages_for(["Myelodysplastic Syndromes", "Anemia"], []) == ["Myeloid"]
    assert dm.lineages_for(["Primary Myelofibrosis", "Thrombocytopenia"], []) == ["Myeloid"]
    # Anaemia in a solid-tumour trial is still supportive care.
    assert dm.lineages_for(["Breast Neoplasms", "Anemia"], []) == []
    assert dm.lineages_for(["Lymphoma", "Neutropenia"], []) == []


def test_bare_sarcoma_and_melanoma_do_not_add_a_lineage_to_a_named_one():
    assert dm.lineages_for(["Sarcoma, Ewing"], []) == ["Bone"]
    assert dm.lineages_for(["Uterine Sarcoma"], []) == ["Uterus"]
    assert dm.lineages_for(["Osteogenic Sarcoma"], []) == ["Bone"]
    assert dm.lineages_for(["Myeloid Sarcoma"], []) == ["Myeloid"]
    assert dm.lineages_for(["Uveal Melanoma"], []) == ["Eye"]
    assert dm.lineages_for(["Sarcoma"], []) == ["Soft Tissue"]
    assert dm.lineages_for(["Melanoma"], []) == ["Skin"]


def test_a_target_with_no_mean_effect_gets_no_fraction_dependent_either():
    genes = pl.DataFrame(
        {
            "ensembl_gene_id": ["E1"],
            "common_essential": [False],
            "mean_effect": [None],
            "fraction_dependent": [0.4],
        },
        schema_overrides={"mean_effect": pl.Float64},
    )
    targets = pl.DataFrame({"nct_id": ["N1"], "targetId": ["E1"]})
    out = dm.gene_features(targets, genes)
    assert out["depmap_fraction_dependent"][0] is None


# ---------------------------------------------------------------- features


GENES = pl.DataFrame(
    {
        "ensembl_gene_id": ["E1", "E2", "E3"],
        "n_lines": [10, 10, 10],
        "mean_effect": [-1.0, -0.3, 0.1],
        "fraction_dependent": [0.9, 0.2, 0.0],
        "common_essential": [True, False, False],
    }
)


def targets(*pairs):
    return pl.DataFrame(pairs, schema={"nct_id": pl.String, "targetId": pl.String}, orient="row")


def test_gene_features_take_the_most_dependent_target_and_any_common_essential():
    out = dm.gene_features(targets(("N1", "E2"), ("N1", "E3"), ("N2", "E1"), ("N2", "E2")), GENES)
    rows = {r["nct_id"]: r for r in out.iter_rows(named=True)}
    assert rows["N1"]["depmap_mean_effect"] == pytest.approx(-0.3)
    assert rows["N1"]["depmap_common_essential"] is False
    assert rows["N2"]["depmap_mean_effect"] == pytest.approx(-1.0)
    assert rows["N2"]["depmap_fraction_dependent"] == pytest.approx(0.9)
    assert rows["N2"]["depmap_common_essential"] is True


def test_a_target_missing_from_depmap_gives_no_row():
    out = dm.gene_features(targets(("N1", "E9")), GENES)
    assert out.height == 0


LINEAGES = pl.DataFrame(
    {
        "lineage": ["Skin", "Breast", "Skin", "Breast"],
        "ensembl_gene_id": ["E2", "E2", "E3", "E3"],
        "n_lines": [6, 4, 6, 4],
        "sum_effect": [-6.0, 0.0, 0.6, 0.4],  # E2: skin mean -1.0, breast 0.0
    }
)
GENES_FOR_LINEAGE = GENES.with_columns(
    pl.Series("mean_effect", [-1.0, -0.6, 0.1]),  # E2: (-6.0 + 0.0) / 10
)


def lineage_table(trial_lineages, pairs):
    return dm.lineage_features(
        targets(*pairs),
        pl.DataFrame(trial_lineages, schema={"nct_id": pl.String, "lineages": pl.List(pl.String)}, orient="row"),
        GENES_FOR_LINEAGE,
        LINEAGES,
    )  # fmt: skip


def test_matched_lineage_mean_and_difference_from_the_other_lines():
    out = lineage_table([("N1", ["Skin"])], [("N1", "E2")])
    assert out["depmap_lineage_effect"][0] == pytest.approx(-1.0)
    # other lines: (total sum -6.0 minus matched -6.0) / (10 - 6) = 0.0
    assert out["depmap_lineage_minus_other"][0] == pytest.approx(-1.0)


def test_the_target_with_the_lowest_matched_mean_is_used():
    out = lineage_table([("N1", ["Skin"])], [("N1", "E2"), ("N1", "E3")])
    assert out["depmap_lineage_effect"][0] == pytest.approx(-1.0)


def test_too_few_matched_lines_or_too_many_lineages_give_no_value():
    assert lineage_table([("N1", ["Breast"])], [("N1", "E2")]).height == 0  # 4 lines, needs 5
    assert lineage_table([("N1", ["A", "B", "C", "D"])], [("N1", "E2")]).height == 0
    assert lineage_table([("N1", [])], [("N1", "E2")]).height == 0


def test_two_lineages_are_pooled_by_summing_lines_not_averaging_means():
    lineages = pl.DataFrame(
        {
            "lineage": ["Skin", "Breast", "Lung"],
            "ensembl_gene_id": ["E2", "E2", "E2"],
            "n_lines": [6, 4, 10],
            "sum_effect": [-6.0, 0.0, -2.0],
        }
    )
    genes = GENES.with_columns(
        pl.Series("n_lines", [20, 20, 20]), pl.Series("mean_effect", [-1.0, -0.4, 0.1])
    )  # E2: (-6.0 + 0.0 - 2.0) / 20
    trial = pl.DataFrame(
        {"nct_id": ["N1"], "lineages": [["Skin", "Breast"]]},
        schema={"nct_id": pl.String, "lineages": pl.List(pl.String)},
    )
    out = dm.lineage_features(targets(("N1", "E2")), trial, genes, lineages)
    assert out["depmap_lineage_effect"][0] == pytest.approx(-0.6)  # (-6.0 + 0.0) / 10 lines
    assert out["depmap_lineage_minus_other"][0] == pytest.approx(-0.6 - -0.2)  # others: -2.0 / 10


def test_matching_every_line_leaves_nothing_to_compare_with():
    out = lineage_table([("N1", ["Skin", "Breast"])], [("N1", "E2")])
    assert out.height == 0


HOST = ["E3"]
ACTIONS = pl.DataFrame(
    {
        "drugId": ["D1", "D1", "D1", "D2"],
        "targetId": ["E1", "E2", "E3", "E2"],
        "actionType": ["INHIBITOR", "AGONIST", "ANTAGONIST", "INHIBITOR"],
    }
)


def test_tumour_intrinsic_targets_are_turned_down_by_the_drug_and_not_host_directed():
    pairs = pl.DataFrame(
        {
            "nct_id": ["N1", "N1", "N1", "N2"],
            "drugId": ["D1", "D1", "D1", "D2"],
            "targetId": ["E1", "E2", "E3", "E2"],
        }
    )
    out = dm.tumour_intrinsic(pairs, ACTIONS, HOST)
    # E1: inhibited, kept. E2: an agonist target of D1, dropped; inhibited by D2, kept for N2.
    # E3: antagonised but host-directed, dropped.
    assert out.rows() == [("N1", "E1"), ("N2", "E2")]


def test_the_host_directed_list_resolves_through_hgnc_ids():
    hgnc = pl.DataFrame(
        {"symbol": ["PDCD1", "TP53", "KDR"], "ensembl_gene_id": ["E_PD1", "E_TP53", "E_KDR"]}
    )
    assert dm.host_directed_ids(hgnc) == ["E_KDR", "E_PD1"]


def test_a_trial_whose_targets_are_all_host_directed_gets_gene_features_but_no_lineage():
    conditions = pl.DataFrame(
        {"nct_id": ["N1"], "conditions": [["melanoma"]], "mesh": [["Melanoma"]], "ancestors": [[]]}
    )
    trials = pl.DataFrame({"nct_id": ["N1"], "therapeutic_area": ["oncology"]})
    out = dm.trial_features(
        targets(("N1", "E2")), targets(), trials, conditions, GENES_FOR_LINEAGE, LINEAGES
    )
    assert out["depmap_mean_effect"][0] == pytest.approx(-0.6)
    assert out["depmap_lineage_effect"][0] is None


def test_only_cancer_trials_get_matched_lineage_values():
    conditions = pl.DataFrame(
        {
            "nct_id": ["N1", "N2"],
            "conditions": [["melanoma"], ["melanoma"]],
            "mesh": [["Melanoma"], ["Melanoma"]],
            "ancestors": [[], []],
        }
    )
    trials = pl.DataFrame({"nct_id": ["N1", "N2"], "therapeutic_area": ["oncology", "dermatology"]})
    both = targets(("N1", "E2"), ("N2", "E2"))
    out = dm.trial_features(both, both, trials, conditions, GENES_FOR_LINEAGE, LINEAGES).sort(
        "nct_id"
    )
    assert out["depmap_lineage_effect"][0] == pytest.approx(-1.0)
    assert out["depmap_lineage_effect"][1] is None  # a dermatology trial: gene features only
    assert out["depmap_mean_effect"].null_count() == 0


def test_dating_labels():
    assert dm.FEATURES["depmap_mean_effect"] == "timeless"
    assert dm.FEATURES["depmap_lineage_effect"] == "protocol"
