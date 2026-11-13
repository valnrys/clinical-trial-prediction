"""Sentence embeddings of each trial's registration text, for the "embeddings" model family.

    uv run --extra embeddings python -m trial_benchmark.embeddings    # -> data/interim/embeddings_*.parquet

This is the one place the project uses a pretrained model. It is a model UNDER TEST: the
vectors feed only the models in `models/embedded.py` (marked `pretrained=True`). They never
feed a label or the pre-registered baseline.

What is built: one vector per trial, from one document made of the labelled protocol fields
(title, conditions, interventions, primary outcomes). The summary is never read, and neither
is the eligibility text, which is long enough to crowd the rest out of the encoder's window:

    Title: ...
    Conditions: ...
    Interventions: ...
    Primary outcomes: ...

A missing field is left out. Two tables, one per encoder, each with `nct_id`, `split` and
`<prefix>_000 ...` float columns (unit length). Test and late trials get vectors too: they
are features only, as in every other table.

The two encoders (names and revisions are pinned; a re-run gets the same files):

- general: sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41.
  384 numbers. Apache-2.0. Fine-tuned in 2021 on about a billion sentence pairs from the open
  web (Reddit, Wikipedia, S2ORC abstracts, and others). The card gives no exact data cut-off.
- biomedical: NeuML/pubmedbert-base-embeddings, revision b79526d6ef3645e0df4530322e266f24c829f5ef.
  768 numbers. Apache-2.0. PubMedBERT fine-tuned on PubMed title and abstract pairs. The card
  gives no data cut-off.

How the columns are dated: **protocol**, as the text they come from (see `text`).

Contamination warning, which the report repeats. A pretrained language model may have seen
these trials, and even their results, while it was trained: outcomes of 2021 to 2023 trials
are on the public web, and PubMed holds many of the papers. Small sentence encoders memorise
far less than large generative models, and these two were not trained to predict outcomes,
but the benchmark's time split does not protect against this. Treat a gain from these
vectors as "may include recognition", not as pure forecasting.

Deterministic: CPU, fixed batch size, no sampling. Text longer than the encoder's limit is
cut at its end. A table that already exists is not rebuilt; pass `--rebuild` to redo it.
The optional extra is needed only to build tables, not to use them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from trial_benchmark.dating import PROTOCOL
from trial_benchmark.opentargets import INTERIM
from trial_benchmark.text import OUT_PATH as TEXT_PATH

BATCH_SIZE = 32
FIELD_LABELS = {
    "title": "Title",
    "conditions": "Conditions",
    "interventions": "Interventions",
    "primary_outcomes": "Primary outcomes",
}
Encoder = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class EmbeddingModel:
    key: str  # "general" or "biomedical"
    hf_name: str
    revision: str  # exact commit of the model files
    dimension: int
    prefix: str  # column names are f"{prefix}_{i:03d}"
    dating: str = PROTOCOL  # of the text it reads, see `text`

    @property
    def table(self) -> str:
        return f"embeddings_{self.key}"

    @property
    def path(self) -> Path:
        return INTERIM / f"{self.table}.parquet"

    @property
    def columns(self) -> list[str]:
        return [f"{self.prefix}_{i:03d}" for i in range(self.dimension)]


GENERAL = EmbeddingModel(
    "general",
    "sentence-transformers/all-MiniLM-L6-v2",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    384,
    "emb_gen",
)
BIOMEDICAL = EmbeddingModel(
    "biomedical",
    "NeuML/pubmedbert-base-embeddings",
    "b79526d6ef3645e0df4530322e266f24c829f5ef",
    768,
    "emb_bio",
)
MODELS = (GENERAL, BIOMEDICAL)

# Every vector column is read from the protocol record (the text fields are all "protocol").
FEATURES: dict[str, str] = {column: model.dating for model in MODELS for column in model.columns}


def document(fields: dict[str, str | None]) -> str:
    """One trial's text: its labelled protocol fields, one per line, missing ones left out."""
    lines = [f"{FIELD_LABELS[f]}: {fields[f]}" for f in FIELD_LABELS if fields.get(f)]
    return "\n".join(lines)


def documents(text: pl.DataFrame) -> list[str]:
    return [document(row) for row in text.select(*FIELD_LABELS).iter_rows(named=True)]


def embedding_table(text: pl.DataFrame, model: EmbeddingModel, encode: Encoder) -> pl.DataFrame:
    """`nct_id`, `split` and the vector columns, one row per row of `text`, in its order."""
    vectors = np.asarray(encode(documents(text)), dtype=np.float32)
    if vectors.shape != (text.height, model.dimension):
        raise ValueError(f"Expected {(text.height, model.dimension)}, got {vectors.shape}.")
    columns = pl.DataFrame(vectors, schema=model.columns)
    return pl.concat([text.select("nct_id", "split"), columns], how="horizontal")


def load_encoder(model: EmbeddingModel) -> Encoder:
    """The pinned model on the CPU. Needs the optional `embeddings` extra."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model.hf_name, revision=model.revision, device="cpu")

    def encode(texts: Sequence[str]) -> np.ndarray:
        return encoder.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return encode


def build(model: EmbeddingModel, rebuild: bool = False) -> pl.DataFrame | None:
    """Write the model's table unless it exists. Returns the table, or None if it was kept."""
    if model.path.exists() and not rebuild:
        return None
    table = embedding_table(pl.read_parquet(TEXT_PATH), model, load_encoder(model))
    table.write_parquet(model.path)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild", action="store_true", help="embed again even if a table exists")
    args = parser.parse_args()
    for model in MODELS:
        table = build(model, args.rebuild)
        if table is None:
            print(f"kept {model.path} (exists; use --rebuild)", file=sys.stderr)
        else:
            print(f"wrote {model.path} ({table.height} trials, {model.dimension} numbers each)")


if __name__ == "__main__":
    main()
