"""Download the Open Targets tables we need, keeping only what we are allowed to use.

    uv run python -m elman_exercise.opentargets

Open Targets (https://platform.opentargets.org) joins drugs, their protein
targets, diseases and human genetic evidence. Release 26.09 is pinned below, so a
rebuild gets the same tables. No account is needed.

What is kept, in data/raw/opentargets/ (git-ignored with the rest of data/raw/):

- `clinical_report.parquet`: ClinicalTrials.gov reports only, with five columns:
  NCT id, source, drugs, diseases, trial start date.
- `clinical_target.parquet`: drug to target (two columns).
- `drug_molecule.parquet`: drug type and name (three columns).
- `genetic_association.parquet`: genetic-association scores for the targets that
  appear in our trials, with their year-by-year history.
- `disease.parquet`: only if `rare_disease` asked for it; disease id and external
  references (two columns), to link diseases to Orphanet.
- `manifest.json`: the release, the files and their sizes.

Outcome-bearing columns are downloaded but never kept. The clinical report table
also carries trial status and stop-reason columns, and the drug and target tables
carry "maximum clinical stage" (which includes approval); the association file
includes the `clinical` (known_drug) datatype. Those are what our label is about,
and the held-out set must not be touched by any route. The full files do reach
this disk for a short time, as `_full_*` and `_raw_*` files. Each is read through
a fixed list of columns (`WANTED_COLUMNS`, `ASSOCIATION_COLUMNS`) or filtered to
genetic rows, and then deleted. Only the slim copies stay in data/raw/opentargets/.

The 2.5 GB association table is fetched part by part. Each part is cut down to the
genetic rows for our targets (the `aggregationValue` column is kept, so "genetic
rows only" can be checked from disk) and then deleted, so the disk holds at most
one raw part at a time. Kept files are written under a temporary name and renamed,
and a restart deletes any raw part or half-written file left by a crash, so a
partial file is never mistaken for a finished one.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from elman_exercise.dataset import ROOT

RELEASE = "26.09"
BASE_URL = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{RELEASE}/output/"
OT_DIR = ROOT / "data" / "raw" / "opentargets"
INTERIM = ROOT / "data" / "interim"

# The only columns ever read from each table. Nothing about status, stop reason,
# clinical stage or approval is on this list, and a test pins the exact lists.
WANTED_COLUMNS = {
    "clinical_report": ("id", "source", "drugs", "diseases", "trialStartDate"),
    "clinical_target": ("drugId", "targetId"),
    "drug_molecule": ("id", "name", "drugType"),
    "target": ("id", "approvedSymbol"),
}
SMALL_TABLES = tuple(WANTED_COLUMNS)
# The disease table is used only to link Open Targets disease ids to Orphanet codes
# (rare_disease.py). It is fetched on demand, not with the tables above, and has no
# outcome-bearing columns.
DISEASE_COLUMNS = ("id", "dbXRefs")
ASSOCIATION_TABLE = "association_by_datatype_indirect"
ASSOCIATION_COLUMNS = (
    "diseaseId",
    "targetId",
    "aggregationType",
    "aggregationValue",
    "associationScore",
    "timeseries",
)
GENETIC = "genetic_association"
CTGOV = "ClinicalTrials.gov"
CHUNK = 1 << 20


def list_parquet_files(table: str) -> list[str]:
    """File names of a table's parquet parts, read from the server's directory listing."""
    with urllib.request.urlopen(f"{BASE_URL}{table}/", timeout=60) as response:
        listing = response.read().decode()
    return sorted(set(re.findall(r'href="([^"?/]+\.parquet)"', listing)))


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(url: str, destination: Path) -> None:
    """Fetch `url` to `destination`, resuming a half-finished `.part` file if there is one."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    if destination.exists() and destination.stat().st_size == total:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if have > total:
        partial.unlink()
        have = 0
    if have < total:
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if have and response.status != 206:
                raise RuntimeError(f"{url}: the server ignored the resume request.")
            with partial.open("ab" if have else "wb") as out:
                while block := response.read(CHUNK):
                    out.write(block)
    if partial.stat().st_size != total:
        raise RuntimeError(f"{url}: got {partial.stat().st_size} bytes, expected {total}.")
    partial.replace(destination)


def slim_table(table: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Table-specific tidying of the wanted columns: for the clinical report, NCT trials only."""
    if table == "clinical_report":
        return frame.filter(pl.col("source") == CTGOV).with_columns(pl.col("id").str.to_uppercase())
    return frame


def fetch_small_table(
    table: str, out_dir: Path = OT_DIR, columns: tuple[str, ...] | None = None
) -> Path:
    """Download one small table, keep only its wanted columns, delete the full download."""
    target = out_dir / f"{table}.parquet"
    if target.exists():
        return target
    wanted = list(columns or WANTED_COLUMNS[table])
    pieces = []
    for name in list_parquet_files(table):
        full = out_dir / f"_full_{table}.parquet"
        download(f"{BASE_URL}{table}/{name}", full)
        pieces.append(slim_table(table, pl.read_parquet(full, columns=wanted)))
        full.unlink()
    tmp = target.with_suffix(".parquet.tmp")
    pl.concat(pieces).write_parquet(tmp)
    tmp.replace(target)
    return target


def fetch_disease_table(out_dir: Path = OT_DIR) -> Path:
    """The Open Targets disease table, two columns (id and external references)."""
    return fetch_small_table("disease", out_dir, columns=DISEASE_COLUMNS)


def trial_ids() -> list[str]:
    """NCT ids of every trial in the benchmark (all splits; ids only, no labels)."""
    ids: set[str] = set()
    for name in ("dev.parquet", "test_features.parquet"):
        ids.update(pl.read_parquet(INTERIM / name, columns=["nct_id"])["nct_id"].to_list())
    return sorted(ids)


def trial_targets(ids: list[str], out_dir: Path = OT_DIR) -> list[str]:
    """Every target (Ensembl gene id) of every drug Open Targets lists for these trials."""
    reports = pl.read_parquet(out_dir / "clinical_report.parquet", columns=["id", "drugs"])
    drugs = (
        reports.filter(pl.col("id").is_in(ids))
        .explode("drugs")
        .select(pl.col("drugs").struct.field("drugId").alias("drugId"))
        .drop_nulls()
        .unique()
    )
    links = pl.read_parquet(out_dir / "clinical_target.parquet")
    return sorted(links.join(drugs, on="drugId")["targetId"].drop_nulls().unique().to_list())


def genetic_rows_for(part: pl.LazyFrame, targets: list[str]) -> pl.DataFrame:
    """From one association part: genetic-association rows for the given targets."""
    return (
        part.filter(
            (pl.col("aggregationType") == "datatypeId")
            & (pl.col("aggregationValue") == GENETIC)
            & pl.col("targetId").is_in(targets)
        )
        .select("diseaseId", "targetId", "aggregationValue", "associationScore", "timeseries")
        .collect()
    )


def fetch_genetic_associations(targets: list[str], out_dir: Path = OT_DIR) -> Path:
    """Go through the association parts one at a time, keeping the genetic rows we need."""
    final = out_dir / "genetic_association.parquet"
    if final.exists():
        return final
    parts_dir = out_dir / "_association_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    clean_leftovers(out_dir, parts_dir)
    files = list_parquet_files(ASSOCIATION_TABLE)
    for i, name in enumerate(files, 1):
        kept = parts_dir / f"kept_{name}"
        if kept.exists():
            continue
        raw = out_dir / f"_raw_{name}"
        download(f"{BASE_URL}{ASSOCIATION_TABLE}/{name}", raw)
        rows = genetic_rows_for(pl.scan_parquet(raw), targets)
        tmp = kept.with_suffix(".parquet.tmp")
        rows.write_parquet(tmp)
        tmp.replace(kept)
        raw.unlink()
        print(f"  association part {i}/{len(files)}: kept {rows.height:,} genetic rows")
    combined = final.with_suffix(".parquet.tmp")
    pl.concat([pl.read_parquet(p) for p in sorted(parts_dir.glob("kept_*.parquet"))]).write_parquet(
        combined
    )
    combined.replace(final)
    for p in parts_dir.glob("kept_*.parquet"):
        p.unlink()
    parts_dir.rmdir()
    return final


def clean_leftovers(out_dir: Path, parts_dir: Path) -> None:
    """Delete what a crashed run can leave: raw parts (they hold outcome-bearing rows),
    and half-written `.tmp` files. Finished `kept_*.parquet` files stay."""
    for leftover in [
        *out_dir.glob("_raw_*"),
        *out_dir.glob("_full_*"),
        *out_dir.glob("*.tmp"),
        *parts_dir.glob("*.tmp"),
    ]:
        leftover.unlink()


def write_manifest(out_dir: Path = OT_DIR) -> dict:
    files = {p.name: p.stat().st_size for p in sorted(out_dir.glob("*.parquet"))}
    columns_read = {k: list(v) for k, v in WANTED_COLUMNS.items()} | {
        ASSOCIATION_TABLE: list(ASSOCIATION_COLUMNS)
    }
    if "disease.parquet" in files:
        columns_read["disease"] = list(DISEASE_COLUMNS)
    manifest = {
        "release": RELEASE,
        "source": BASE_URL,
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
        "columns_read": columns_read,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    OT_DIR.mkdir(parents=True, exist_ok=True)
    for table in SMALL_TABLES:
        print(f"{table}: {fetch_small_table(table).name}")
    targets = trial_targets(trial_ids())
    print(f"{len(targets):,} targets appear in our trials; scanning the association table")
    fetch_genetic_associations(targets)
    manifest = write_manifest()
    print(json.dumps(manifest["files"], indent=2))


if __name__ == "__main__":
    main()
