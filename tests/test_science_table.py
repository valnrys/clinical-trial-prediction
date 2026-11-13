import hashlib

import polars as pl
import pytest

from elman_exercise import dating, science
from elman_exercise.science import checks, fetch
from elman_exercise.science import table as tb


def trials():
    return pl.DataFrame({"nct_id": ["N1", "N2", "N3"], "split": ["dev", "dev", "test"]})


def test_every_trial_gets_a_row_and_no_target_means_has_target_false_and_zero_targets():
    targets = pl.DataFrame({"nct_id": ["N1", "N1"], "drugId": ["D", "D"], "targetId": ["E1", "E2"]})
    piece = pl.DataFrame({"nct_id": ["N1"], "annot_x": [1.0]})
    out = tb.assemble(trials(), targets, [piece])
    assert out["nct_id"].to_list() == ["N1", "N2", "N3"]
    assert out["has_target"].to_list() == [True, False, False]
    assert out["n_target_genes"].to_list() == [2, 0, 0]
    assert out["annot_x"].to_list() == [1.0, None, None]


def test_a_source_that_repeats_a_trial_is_refused():
    piece = pl.DataFrame({"nct_id": ["N1", "N1"], "v": [1, 2]})
    with pytest.raises(Exception, match="join"):
        tb.assemble(trials(), pl.DataFrame({"nct_id": [], "drugId": [], "targetId": []}, schema=
                    {"nct_id": pl.String, "drugId": pl.String, "targetId": pl.String}), [piece])  # fmt: skip


def test_funnel_counts_each_step_by_split():
    drugs = pl.DataFrame({"nct_id": ["N1", "N2"], "drugId": ["D", "D"], "is_lead": [True, True]})
    targets = pl.DataFrame({"nct_id": ["N1"], "drugId": ["D"], "targetId": ["E1"]})
    columns = {
        c: [None, None, None] for module in science.SOURCES.values() for c in module.FEATURES
    }
    columns["depmap_mean_effect"] = [-1.0, None, None]
    columns["chem_mol_weight"] = [None, 180.0, 200.0]
    table = tb.assemble(
        trials(), targets, [pl.DataFrame({"nct_id": ["N1", "N2", "N3"], **columns})]
    )
    out = tb.funnel(table, drugs, targets).sort("split")
    dev = out.filter(pl.col("split") == "dev").row(0, named=True)
    test = out.filter(pl.col("split") == "test").row(0, named=True)
    assert (dev["trials"], dev["investigational drug"], dev["target"]) == (2, 2, 1)
    assert dev["depmap value"] == 1 and dev["chemistry value"] == 1
    assert (test["trials"], test["investigational drug"], test["target"]) == (1, 0, 0)
    assert test["chemistry value"] == 1


def test_every_feature_has_a_known_dating_and_the_target_flags_are_not_honest():
    dating.check(science.FEATURES)
    assert science.FEATURES["has_target"] not in dating.HONEST
    assert science.FEATURES["n_target_genes"] not in dating.HONEST
    honest = [c for c, label in science.FEATURES.items() if label in dating.HONEST]
    assert len(honest) == sum(len(m.FEATURES) for m in science.SOURCES.values())


def test_no_two_sources_share_a_column_name():
    names = [c for module in science.SOURCES.values() for c in module.FEATURES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------- downloads


def remote(tmp_path, payload=b"some public data", **changes):
    source = tmp_path / "upstream.bin"
    source.write_bytes(payload)
    fields = {
        "name": "file.bin",
        "url": source.as_uri(),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return fetch.RemoteFile(**{**fields, **changes})


def test_a_file_is_fetched_and_checked(tmp_path):
    file = remote(tmp_path)
    path = fetch.fetch(file, tmp_path / "raw")
    assert path.read_bytes() == b"some public data"
    assert fetch.fetch(file, tmp_path / "raw") == path  # a second call checks and reuses


def test_a_wrong_checksum_deletes_the_file_and_stops(tmp_path):
    file = remote(tmp_path, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="expected"):
        fetch.fetch(file, tmp_path / "raw")
    assert not (tmp_path / "raw" / "file.bin").exists()


def test_a_wrong_size_stops(tmp_path):
    file = remote(tmp_path, size=3)
    with pytest.raises(RuntimeError):
        fetch.fetch(file, tmp_path / "raw")
    assert not (tmp_path / "raw" / "file.bin").exists()


def test_a_half_finished_download_is_resumed_not_restarted(tmp_path):
    payload = b"0123456789" * 100
    file = remote(tmp_path, payload=payload)
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "file.bin.part").write_bytes(payload[:400])
    started_from = []
    original = fetch._download_once

    def watching(url, partial, size):
        started_from.append(partial.stat().st_size)
        original(url, partial, size)

    fetch._download_once = watching
    try:
        fetch.fetch(file, directory)
    finally:
        fetch._download_once = original
    assert started_from == [400]
    assert (directory / "file.bin").read_bytes() == payload


class DroppedConnection:
    """A response that ends early without raising, as a closed connection can."""

    def __init__(self, payload, status=200):
        self.payload, self.status, self.sent = payload, status, False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        if self.sent:
            return b""
        self.sent = True
        return self.payload


def test_a_stream_that_ends_early_is_resumed_not_renamed_and_deleted(tmp_path, monkeypatch):
    payload = b"0123456789" * 100
    file = remote(tmp_path, payload=payload)
    calls = []

    def urlopen(request, timeout):
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            return DroppedConnection(payload[:300])  # closes after 300 bytes, no error
        return DroppedConnection(payload[300:], status=206)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    path = fetch.fetch(file, tmp_path / "raw")
    assert calls == [None, "bytes=300-"]
    assert path.read_bytes() == payload


def test_the_manifest_lists_every_file_with_its_url_and_checksum(tmp_path):
    file = remote(tmp_path)
    fetch.fetch_all(tmp_path / "raw", "release 1", "CC0", [file])
    manifest = (tmp_path / "raw" / "manifest.json").read_text()
    assert file.sha256 in manifest and "release 1" in manifest and "CC0" in manifest


def test_pinned_sources_have_sizes_and_checksums_of_the_right_shape():
    files = [
        *science.depmap.FILES, *science.expression.FILES, *science.networks.FILES,
        science.targets.HGNC_FILE, *science.annotations.TARGET_FILES,
        science.annotations.PRIORITISATION_FILE, science.chemistry.MECHANISM,
    ]  # fmt: skip
    for file in files:
        assert len(file.sha256) == 64 and file.size > 0 and file.url.startswith("https://")


# ---------------------------------------------------------------- the known-biology checks


def test_ids_of_finds_symbols_through_hgnc_and_skips_unknown_ones():
    hgnc = pl.DataFrame(
        {"hgnc_id": ["H1", "H2"], "symbol": ["AAA", "BBB"], "entrez_id": [1, 2],
         "ensembl_gene_id": ["E1", None]}
    )  # fmt: skip
    assert checks.ids_of(("AAA", "BBB", "CCC"), hgnc) == {"AAA": "E1"}


HGNC = pl.DataFrame(
    {
        "hgnc_id": [f"H{i}" for i in range(10)],
        "symbol": [
            "RPL5",
            "RPL11",
            "RPS14",
            "RPS19",
            "POLR2B",
            "POLR2E",
            "POLR2L",
            "INS",
            "ALB",
            "TNNT2",
        ],
        "entrez_id": list(range(10)),
        "ensembl_gene_id": [f"E{i}" for i in range(10)],
    }
)


def test_the_essential_gene_check_passes_on_right_data_and_fails_on_wrong_data():
    flags = [True] * 7 + [False] * 3
    genes = pl.DataFrame(
        {"ensembl_gene_id": [f"E{i}" for i in range(10)], "common_essential": flags}
    )
    assert checks.common_essential(genes, HGNC).passed
    wrong = genes.with_columns(pl.Series("common_essential", [True] * 10))  # INS flagged essential
    assert not checks.common_essential(wrong, HGNC).passed
    assert not checks.common_essential(genes.head(3), HGNC).passed  # genes missing from the source
    # The negative controls must be present too: a missing INS, ALB or TNNT2 is not a pass.
    assert not checks.common_essential(genes.head(7), HGNC).passed


def test_the_tissue_check_fails_if_the_expected_tissue_is_not_the_top_one():
    rows = [(g, t, 1) for g, t in (("E7", "pancreas"), ("E8", "liver"), ("E9", "heart"))]
    rows += [("E7", "liver", 5)]
    matched = pl.DataFrame(rows, schema=["ensembl_gene_id", "tissue", "best_rank"], orient="row")
    matched = matched.with_columns(pl.lit(0.0).alias("log2_ntpm"))
    result = checks.tissue_of_origin(matched, HGNC)
    assert not result.passed  # HMGCR and PCSK9 are not in this tiny HGNC table
    assert "INS/pancreas" in result.detail


def test_the_molecule_check_needs_aspirin_to_weigh_180():
    table = pl.DataFrame(
        {
            "drugId": ["CHEMBL25", "CHEMBL1487"],
            "chem_mol_weight": [180.16, 558.6],
            "chem_rule_of_five_violations": [0.0, 1.0],
            "chem_action_type": ["inhibitor", "inhibitor"],
        }
    )
    assert checks.known_molecules(table).passed
    assert not checks.known_molecules(
        table.with_columns(pl.lit(18.0).alias("chem_mol_weight"))
    ).passed


def test_statin_trials_must_reach_the_statin_target_through_the_join():
    hgnc = pl.DataFrame(
        {"hgnc_id": ["H1"], "symbol": ["HMGCR"], "entrez_id": [1], "ensembl_gene_id": ["EH"]}
    )
    ids = [f"N{i}" for i in range(25)]
    drugs = pl.DataFrame({"nct_id": ids, "drugId": ["CHEMBL1487"] * 25, "is_lead": [True] * 25})
    targets = pl.DataFrame({"nct_id": ids, "drugId": ["CHEMBL1487"] * 25, "targetId": ["EH"] * 25})
    assert checks.statin_trials_reach_hmgcr(drugs, targets, hgnc).passed
    wrong = targets.with_columns(pl.lit("EX").alias("targetId"))
    assert not checks.statin_trials_reach_hmgcr(drugs, wrong, hgnc).passed


def test_a_failed_check_stops_the_build_before_anything_is_written(tmp_path, monkeypatch):
    table = pl.DataFrame({"nct_id": ["N1"], "split": ["dev"], "has_target": [True]})
    monkeypatch.setattr(
        tb, "build_all", lambda download=True: (table, pl.DataFrame(), pl.DataFrame())
    )
    monkeypatch.setattr(science, "FEATURES", {"has_target": "after the start"})
    monkeypatch.setattr(tb.checks, "run_all", lambda d, t: [checks.Result("a check", False, "x")])
    monkeypatch.setattr(tb, "OUT_PATH", tmp_path / "science.parquet")
    monkeypatch.setattr(tb, "FUNNEL_PATH", tmp_path / "funnel.csv")
    monkeypatch.setattr("sys.argv", ["table", "--no-download"])
    with pytest.raises(SystemExit, match="Nothing was written"):
        tb.main()
    assert list(tmp_path.iterdir()) == []
