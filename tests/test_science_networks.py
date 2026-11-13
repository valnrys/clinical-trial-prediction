import polars as pl

from elman_exercise.science import networks as nw


def index(*rows):
    """rows of (screen id, author, analysis, setup, phenotype, screen type, method, full list)"""
    return pl.DataFrame(
        rows,
        schema=["#SCREEN_ID", "AUTHOR", "ANALYSIS", "EXPERIMENTAL_SETUP", "PHENOTYPE",
                "SCREEN_TYPE", "LIBRARY_METHODOLOGY", "FULL_SIZE_AVAILABLE"],
        orient="row",
    )  # fmt: skip


GROWTH = ("Timecourse", "cell proliferation", "Negative Selection", "Knockout", "Yes")


def groups_of(*rows):
    return dict(nw.screen_groups(index(*rows)).iter_rows())


def test_a_growth_time_course_outside_the_depmap_family_is_a_viability_screen():
    assert groups_of((1, "Smith (2020)", "MAGeCK", *GROWTH)) == {1: "viability"}


def test_depmap_family_screens_are_left_out_by_author_or_by_ceres_analysis():
    found = groups_of(
        (1, "Meyers RM (2017)", "MAGeCK", *GROWTH),
        (2, "Smith (2020)", "CERES", *GROWTH),
        (3, "Smith (2020)", "MAGeCK", *GROWTH),
    )
    assert found == {3: "viability"}


def test_a_growth_screen_run_for_gain_of_growth_counts_as_neither():
    positive = ("Timecourse", "cell proliferation", "Positive Selection", "Knockout", "Yes")
    assert groups_of((1, "Smith (2020)", "MAGeCK", *positive)) == {}


def test_a_viability_screen_that_is_not_a_time_course_counts_as_neither_group():
    endpoint = ("Drug Exposure", "cell proliferation", "Negative Selection", "Knockout", "Yes")
    assert groups_of((1, "Smith (2020)", "MAGeCK", *endpoint)) == {}


def test_other_knockout_screens_count_but_activation_screens_and_partial_lists_do_not():
    drug = ("Drug Exposure", "response to chemicals", "Positive Selection", "Knockout", "Yes")
    activation = (
        "Drug Exposure",
        "response to chemicals",
        "Positive Selection",
        "Activation",
        "Yes",
    )
    partial = ("Drug Exposure", "response to chemicals", "Positive Selection", "Knockout", "No")
    found = groups_of((1, "A", "x", *drug), (2, "A", "x", *activation), (3, "A", "x", *partial))
    assert found == {1: "other"}


def test_hit_counts_per_group_count_each_gene_once_per_screen():
    screens = [
        pl.DataFrame(
            {"screen_id": [1, 1, 1], "entrez_id": [10, 10, 11], "hit": [False, True, False]}
        ),
        pl.DataFrame({"screen_id": [2, 2], "entrez_id": [10, 11], "hit": [False, False]}),
        pl.DataFrame({"screen_id": [3], "entrez_id": [10], "hit": [True]}),
    ]
    groups = pl.DataFrame({"screen_id": [1, 2, 3], "group": ["viability", "viability", "other"]})
    out = nw.count_hits(screens, groups).sort("entrez_id")
    row = out.filter(pl.col("entrez_id") == 10).row(0, named=True)
    assert (
        row["n_screens_viability"] == 2 and row["n_hits_viability"] == 1
    )  # duplicate row in screen 1
    assert row["n_screens_other"] == 1 and row["n_hits_other"] == 1
    other_gene = out.filter(pl.col("entrez_id") == 11).row(0, named=True)
    assert other_gene["n_screens_other"] is None  # never in an "other" screen


def test_degree_counts_distinct_partners_and_self_pairs_count_for_neither_degree_nor_papers():
    rows = pl.DataFrame(
        {
            "a": ["1", "1", "1", "2", "3", "4"],
            "b": ["2", "2", "3", "1", "3", "4"],  # 1-2 three times, 1-3, and 3-3 and 4-4 (self)
            "paper": ["P1", "P2", "P2", "P1", "P9", "P8"],
        }
    )
    out = nw.degree_and_papers(rows).sort("entrez_id")
    assert out["entrez_id"].to_list() == [1, 2, 3]  # gene 4 only ever paired with itself
    assert out["degree"].to_list() == [2, 1, 1]
    assert out["papers"].to_list() == [2, 2, 1]  # gene 3: P2 only, not P9 (a self-pair paper)


def write_summaries(directory, orcs, biogrid):
    pl.DataFrame(orcs).write_parquet(directory / nw.ORCS_SUMMARY)
    pl.DataFrame(biogrid).write_parquet(directory / nw.BIOGRID_SUMMARY)


def test_a_gene_needs_enough_screens_before_it_gets_a_hit_fraction(tmp_path):
    write_summaries(
        tmp_path,
        {
            "entrez_id": [1, 2, 3],
            "n_screens_viability": [10, 9, None],
            "n_hits_viability": [5, 9, None],
            "n_screens_other": [5, 4, 8],
            "n_hits_other": [1, 4, 2],
        },
        {"entrez_id": [1, 2], "degree": [8, 4], "papers": [2, 4]},
    )
    mapping = pl.DataFrame({"entrez_id": [1, 2, 3], "ensembl_gene_id": ["E1", "E2", "E3"]})
    genes = nw.gene_table(tmp_path, mapping).sort("ensembl_gene_id")
    assert genes["net_viability_hit_fraction"].to_list() == [
        0.5,
        None,
        None,
    ]  # 9 screens is too few
    assert genes["net_other_hit_fraction"].to_list() == [0.2, None, 0.25]
    # partners per paper on a log2 scale: 8 partners from 2 papers, 4 from 4
    assert genes["net_partners_per_paper"].to_list() == [2.0, 0.0, None]


def test_a_trial_takes_the_highest_value_over_its_targets():
    genes = pl.DataFrame(
        {
            "ensembl_gene_id": ["E1", "E2"],
            "net_viability_hit_fraction": [0.1, 0.8],
            "net_other_hit_fraction": [None, 0.2],
            "net_partners_per_paper": [0.5, -1.0],
        }
    )
    targets = pl.DataFrame({"nct_id": ["N1", "N1"], "targetId": ["E1", "E2"]})
    row = nw.trial_features(targets, genes).row(0, named=True)
    assert row["net_viability_hit_fraction"] == 0.8
    assert row["net_other_hit_fraction"] == 0.2
    assert row["net_partners_per_paper"] == 0.5


def test_a_partner_with_no_entrez_id_is_dropped_not_counted_or_crashed_on():
    rows = pl.DataFrame({"a": ["1", "1", "-"], "b": ["2", "-", "1"], "paper": ["P1", "P2", "P3"]})
    out = nw.degree_and_papers(rows).sort("entrez_id")
    assert out["entrez_id"].to_list() == [1, 2]
    assert out["degree"].to_list() == [1, 1]  # the "-" partner is not a partner
    assert out["papers"].to_list() == [1, 1]
