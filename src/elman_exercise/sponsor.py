"""The sponsor table: who ran each trial.

    uv run python -m elman_exercise.sponsor   # -> data/interim/sponsor.parquet

One row per trial in trials.csv (all splits). Columns:

- sponsor_name: the lead sponsor's name as the registry shows it TODAY, not the name
  at registration.
- sponsor_key: a cleaned form of the name: lower case, punctuation removed,
  trailing company words such as "Inc" or "Ltd" dropped, so "Pfizer Inc." and
  "Pfizer" match.

Both come from the protocol section of the raw record only. No results or
outcome is read. Dating: protocol (current version, see `dating`).

An accepted caveat: today's name can carry a little hindsight. A small company bought
after a good result now appears under its buyer. That is true of every protocol-dated
feature, which reads the current record. The pre-registered baseline reads
`sponsor_class` from the same current record, so it is in exactly the same position,
and these columns stay labelled "protocol", not "after the start".
"""

from __future__ import annotations

import re

import polars as pl

from elman_exercise.dataset import ROOT
from elman_exercise.dating import PROTOCOL
from elman_exercise.download import iter_studies

TABLE = "sponsor"
OUT_PATH = ROOT / "data" / "interim" / "sponsor.parquet"
TRIALS_PATH = ROOT / "trials.csv"
FEATURES = {"sponsor_name": PROTOCOL, "sponsor_key": PROTOCOL}
COMPANY_WORDS = {
    "inc", "incorporated", "ltd", "limited", "llc", "lp", "plc", "corp", "corporation",
    "co", "company", "gmbh", "ag", "sa", "bv", "nv", "kg", "kk", "pty", "srl", "spa",
}  # fmt: skip


def sponsor_key(name: str | None) -> str | None:
    """Lower-case letters and digits only, without trailing company words. None stays None."""
    if not name:
        return None
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    while len(words) > 1 and words[-1] in COMPANY_WORDS:
        words.pop()
    return " ".join(words) or None


def build_table(trials: pl.DataFrame, studies) -> pl.DataFrame:
    """nct_id, split, sponsor_name and sponsor_key for every trial in `trials`."""
    wanted = set(trials["nct_id"])
    names = {}
    for study in studies:
        protocol = study["protocolSection"]
        nct_id = protocol["identificationModule"]["nctId"]
        if nct_id in wanted:
            sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
            names[nct_id] = sponsor.get("name")
    return (
        trials.select("nct_id", "split")
        .with_columns(pl.col("nct_id").replace_strict(names, default=None).alias("sponsor_name"))
        .with_columns(
            pl.col("sponsor_name")
            .map_elements(sponsor_key, return_dtype=pl.String)
            .alias("sponsor_key")
        )
        .sort("nct_id")
    )


def main() -> None:
    table = build_table(pl.read_csv(TRIALS_PATH), iter_studies())
    table.write_parquet(OUT_PATH)
    print(f"{OUT_PATH}: {table.height} trials, {table['sponsor_key'].n_unique()} sponsors")
    print(table["split"].value_counts().sort("split"))


if __name__ == "__main__":
    main()
