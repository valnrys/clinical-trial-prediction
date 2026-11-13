"""How does our success label compare with the labels of two published benchmarks?

    uv run python -m elman_exercise.label_comparison --cto-dir PATH --top-file PATH
    uv run python -m elman_exercise.label_comparison --fetch --cto-dir PATH --top-file PATH

Writes results/dev_label_comparison.md. The benchmark files are NOT redistributed in
this repository. Get them from:

- CTO (Clinical Trial Outcome; Gao et al., https://arxiv.org/abs/2406.10292). Hugging Face
  dataset `chufangao/CTO`, files `phase2_CTO_rf.csv` and `phase3_CTO_rf.csv`
  (https://huggingface.co/datasets/chufangao/CTO). MIT licence. Put both in one folder
  and pass it as `--cto-dir`.
- TOP (Fu et al., trial outcome prediction). GitHub repository
  `futianfan/clinical-trial-outcome-prediction`, file `data/raw_data.csv`
  (https://github.com/futianfan/clinical-trial-outcome-prediction). No licence is stated.
  Pass the file as `--top-file`.

`--fetch` downloads both (about 62 MB) with a plain HTTPS request, into the paths you gave.

The held-out rule. Only trials in our development set (primary completion before
2021-01-01) are compared. The benchmark files are read row by row and a row's label
columns are touched only after its trial id is found in the development set, so the
labels of every other trial are never parsed. The reader refuses any id list that did
not come from a checked development table (`DevelopmentIds` can only be built with
`DevelopmentIds.from_frame`), and the development table is refused if it holds a trial
from 2021 or later. Columns about status or stop reason (TOP has them) are never read.
The hand-read disagreements are checked to be development trials too. The registry scan
that supplies two counts (see `registry_facts`) looks at a trial's analyses only if its
id is a development id, and at a trial's status only if it finished before 2021.

CTO's files give a probability, not a label, and no cut-off. The cut-off is our choice.
We report two (0.5, which we tried first, and 0.7, chosen after seeing how the
probabilities cluster) and a third view that keeps only the trials where CTO is
confident (below 0.1 or above 0.9), so the effect of that choice is visible.

Agreement is Cohen's kappa (agreement beyond what the two labels' success rates would
give by chance; 0 is chance, 1 is identical) with a 95% bootstrap interval over trials
(2,000 resamples, fixed seed). Trials are not independent (limit L10), so the intervals
are somewhat too narrow. CTO repeats some trials. Copies that fall on the same side of
every cut-off are one trial (probabilities averaged); copies that fall on different sides
of a cut-off are dropped and counted. The two CTO files share some trials: each file is
compared on its own, so those trials appear in both.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from elman_exercise import download
from elman_exercise.comparisons import classify_analyses, first_eligible_outcome
from elman_exercise.dataset import ROOT
from elman_exercise.dev_summary import check_is_development, load_dev
from elman_exercise.failures_summary import build_population, check_pre_2021
from elman_exercise.genetics_summary import md_table, pct
from elman_exercise.labels import one_control
from elman_exercise.metrics import auroc

RESULTS_PATH = ROOT / "results" / "dev_label_comparison.md"
DISAGREEMENTS_PATH = ROOT / "results" / "label_disagreements_read.csv"
BENCHMARK_DIR = ROOT / "data" / "raw" / "benchmarks"
CTO_FILES = {"phase 2 file": "phase2_CTO_rf.csv", "phase 3 file": "phase3_CTO_rf.csv"}
CTO_URL = "https://huggingface.co/datasets/chufangao/CTO/resolve/main/"
TOP_FILE = "raw_data.csv"
TOP_URL = (
    "https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/"
    + TOP_FILE
)
CTO_ID, CTO_PROBABILITY, CTO_P_VALUE = "nct_id", "pred_proba", "pvalues"
TOP_ID, TOP_LABEL = "nctid", "label"
CTO_CUTOFF = 0.5  # the first cut-off we tried
CTO_SECOND_CUTOFF = 0.7  # chosen after seeing the clusters in the probabilities
CTO_CUTOFFS = (CTO_CUTOFF, CTO_SECOND_CUTOFF)
CTO_CONFIDENT = (0.1, 0.9)  # confident: below the first or above the second
CTO_CLUSTERS = (
    ("below 0.1", 0.0, 0.1),
    ("0.1 to 0.55", 0.1, 0.55),
    ("0.55 to 0.75", 0.55, 0.75),
    ("0.75 to 0.9", 0.75, 0.9),
    ("0.9 and above", 0.9, 1.0001),
)
OURS = "any_arm"
N_RESAMPLES = 2000
SEED = 20261110
ONE_SIDED = re.compile(r"\b(?:one|1)[ -]?(?:sided|tailed)\b", re.IGNORECASE)

CATEGORY_STOPPED = "terminated but significant"
CATEGORY_WRONG_WAY = "significant in the wrong direction (our label wrong)"
CATEGORY_ONE_SIDED = "one-sided p-value accepted by our label"
CATEGORY_THRESHOLD = "p below 0.05 but said to be above the trial's own threshold"
CATEGORY_UNKNOWN = "their label says failure, reason not determinable from the registry record"
SOURCES = {"registry record", "model read of registry record", "model recall, unverified"}


# ---------------------------------------------------------------- reading, development only


class CheckedIds(frozenset):
    """Trial ids that come from a checked pre-2021 table. Nothing else is accepted.

    There is no public constructor: `DevelopmentIds(["NCT1"])` raises. Build one with the
    `from_frame` / `from_population` class methods, which refuse a table holding a trial
    from 2021 onwards.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("Use DevelopmentIds.from_frame(dev): ids must come from a checked table.")


class DevelopmentIds(CheckedIds):
    """Ids of the labelled development trials."""

    @classmethod
    def from_frame(cls, dev: pl.DataFrame) -> DevelopmentIds:
        check_is_development(dev)
        return frozenset.__new__(cls, dev["nct_id"].to_list())


class FinishedIds(CheckedIds):
    """Ids of every finished pre-2021 drug trial (the wider population in dev_failures.md)."""

    @classmethod
    def from_population(cls, population: pl.DataFrame) -> FinishedIds:
        check_pre_2021(population)
        return frozenset.__new__(cls, population["nct_id"].to_list())


def read_development_rows(
    path: Path,
    id_column: str,
    converters: dict[str, Callable[[str], object]],
    ids: CheckedIds,
) -> list[dict]:
    """Rows of a benchmark CSV for development trials only, with the named columns converted.

    The id is looked at first. A row whose id is not in `ids` is skipped without its
    label cells being touched, let alone converted. Refuses an `ids` that is not a
    `CheckedIds`, so a caller cannot ask for anything else.
    """
    if not isinstance(ids, CheckedIds):
        raise TypeError("Refusing to read labels: ids must come from the checked development set.")
    csv.field_size_limit(sys.maxsize)
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for name in (id_column, *converters):
            if name not in header:
                raise ValueError(f"{path} has no column {name!r}.")
        id_at = header.index(id_column)
        label_at = {name: header.index(name) for name in converters}
        for row in reader:
            nct_id = row[id_at].strip().upper()
            if nct_id not in ids:
                continue
            rows.append(
                {"nct_id": nct_id} | {n: converters[n](row[i]) for n, i in label_at.items()}
            )
    return rows


class CtoRead(NamedTuple):
    frame: pl.DataFrame  # one row per trial: nct_id, probability, p_value_signal
    repeated: int  # trials listed more than once
    dropped: int  # of those, trials whose copies fall on different sides of a cut-off


def collapse_cto(rows: list[dict]) -> CtoRead:
    """One row per trial: mean probability; p-value signal kept only if the copies agree.

    A trial with several copies that fall on different sides of any cut-off in
    CTO_CUTOFFS is dropped, since no one label describes it; the count is returned.
    """
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_trial[row["nct_id"]].append(row)
    out = []
    dropped = 0
    for nct_id, copies in by_trial.items():
        values = [c[CTO_PROBABILITY] for c in copies]
        if any(len({v >= cut for v in values}) > 1 for cut in CTO_CUTOFFS):
            dropped += 1
            continue
        signals = {c[CTO_P_VALUE] for c in copies}
        out.append(
            {
                "nct_id": nct_id,
                "probability": float(np.mean(values)),
                "p_value_signal": signals.pop() if len(signals) == 1 else -1,
            }
        )
    repeated = sum(1 for copies in by_trial.values() if len(copies) > 1)
    schema = {"nct_id": pl.String, "probability": pl.Float64, "p_value_signal": pl.Int64}
    return CtoRead(pl.DataFrame(out, schema=schema), repeated, dropped)


def read_cto(path: Path, ids: CheckedIds) -> CtoRead:
    rows = read_development_rows(
        path, CTO_ID, {CTO_PROBABILITY: float, CTO_P_VALUE: lambda text: int(float(text))}, ids
    )
    return collapse_cto(rows)


def read_top(path: Path, ids: CheckedIds) -> tuple[pl.DataFrame, int]:
    """One row per trial with `top_success`. Trials whose copies disagree are dropped and counted."""
    rows = read_development_rows(path, TOP_ID, {TOP_LABEL: lambda text: int(float(text)) == 1}, ids)
    labels: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        labels[row["nct_id"]].add(row[TOP_LABEL])
    clear = {k: v.pop() for k, v in labels.items() if len(v) == 1}
    frame = pl.DataFrame(
        {"nct_id": list(clear), "top_success": list(clear.values())},
        schema={"nct_id": pl.String, "top_success": pl.Boolean},
    )
    return frame, len(labels) - len(clear)


# ---------------------------------------------------------------- facts from the registry


class WiderCto(NamedTuple):
    """CTO's probability across every finished pre-2021 drug trial in a file, not just ours."""

    n: int
    success_half: float  # share with probability at or above the first cut-off
    stopped: float  # share that are terminated or withdrawn


@dataclass(frozen=True)
class RegistryFacts:
    """Things the benchmark files cannot give, counted from the raw registry records."""

    finished: int  # finished pre-2021 drug trials (completed, terminated or withdrawn)
    terminated: int
    withdrawn: int
    examined: int  # development trials with a deciding comparison that were looked at
    one_sided: int  # of those, comparisons whose p-value or statistical comment says one-sided
    wider: dict[str, WiderCto] | None = None
    inclusion: dict[str, float] | None = None  # share of each status that is in our set


def deciding_analyses(study: dict) -> list[dict]:
    """The analyses that decide the label: each arm's first-listed comparison with one control."""
    comparisons, _ = classify_analyses(study)
    chosen = first_eligible_outcome(comparisons)
    if not chosen:
        return []
    seen: set[str] = set()
    analyses = []
    for comparison in one_control(chosen):
        if comparison.treatment_id not in seen:
            seen.add(comparison.treatment_id)
            analyses.append(comparison.analysis)
    return analyses


def mentions_one_sided(analysis: dict) -> bool:
    text = f"{analysis.get('pValueComment') or ''} {analysis.get('statisticalComment') or ''}"
    return bool(ONE_SIDED.search(text))


def one_sided_counts(studies: Iterable[dict], ids: DevelopmentIds) -> tuple[int, int]:
    """(development trials with a deciding comparison, of those how many mention one-sided).

    A trial's analyses are read only if its id is a development id.
    """
    if not isinstance(ids, DevelopmentIds):
        raise TypeError(
            "Refusing to read analyses: ids must come from the checked development set."
        )
    examined = one_sided = 0
    for study in studies:
        if download.nct_id(study) in ids:
            analyses = deciding_analyses(study)
            if analyses:
                examined += 1
                one_sided += any(mentions_one_sided(a) for a in analyses)
    return examined, one_sided


def wider_cto(path: Path, population: pl.DataFrame) -> WiderCto:
    """CTO's success share and stopped share over every finished pre-2021 trial in one file."""
    read = read_cto(path, FinishedIds.from_population(population))
    joined = read.frame.join(population.select("nct_id", "status"), on="nct_id")
    return WiderCto(
        joined.height,
        float((joined["probability"] >= CTO_CUTOFF).mean()),
        float((joined["status"] != "COMPLETED").mean()),
    )


def registry_facts(
    population: pl.DataFrame,
    studies: Iterable[dict],
    ids: DevelopmentIds,
    wider: dict[str, WiderCto] | None = None,
) -> RegistryFacts:
    """Status counts over the finished pre-2021 population, and one-sided counts in `ids`."""
    examined, one_sided = one_sided_counts(studies, ids)
    inclusion = {}
    for status in ("COMPLETED", "TERMINATED", "WITHDRAWN"):
        part = population.filter(pl.col("status") == status)
        inclusion[status] = (
            part.filter(pl.col("nct_id").is_in(list(ids))).height / part.height
            if part.height
            else float("nan")
        )
    return RegistryFacts(
        population.height,
        int((population["status"] == "TERMINATED").sum()),
        int((population["status"] == "WITHDRAWN").sum()),
        examined,
        one_sided,
        wider,
        inclusion,
    )


# ---------------------------------------------------------------- agreement statistics


def confusion(ours: np.ndarray, theirs: np.ndarray) -> tuple[int, int, int, int]:
    """(both success, ours only, theirs only, both failure)."""
    return (
        int((ours & theirs).sum()),
        int((ours & ~theirs).sum()),
        int((~ours & theirs).sum()),
        int((~ours & ~theirs).sum()),
    )


def kappa(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Cohen's kappa for two boolean labels. nan if chance agreement is already perfect."""
    n = len(ours)
    agree = (ours == theirs).mean()
    p_ours, p_theirs = ours.mean(), theirs.mean()
    chance = p_ours * p_theirs + (1 - p_ours) * (1 - p_theirs)
    return float((agree - chance) / (1 - chance)) if chance < 1 and n else float("nan")


def bootstrap_interval(
    ours: np.ndarray, theirs: np.ndarray, statistic: Callable, seed: int = SEED
) -> tuple[float, float]:
    """95% percentile interval of `statistic(ours, theirs)` over resampled trials."""
    rng = np.random.default_rng(seed)
    n = len(ours)
    values = []
    for _ in range(N_RESAMPLES):
        pick = rng.integers(0, n, n)
        values.append(statistic(ours[pick], theirs[pick]))
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def kappa_text(ours: np.ndarray, theirs: np.ndarray) -> str:
    if len(ours) < 2:
        return "n/a"
    low, high = bootstrap_interval(ours, theirs, kappa)
    return f"{kappa(ours, theirs):.2f} ({low:.2f} to {high:.2f})"


def auroc_text(ours: np.ndarray, scores: np.ndarray) -> str:
    value = auroc(ours, scores)
    low, high = bootstrap_interval(ours, scores, auroc)
    return f"{value:.3f} ({low:.3f} to {high:.3f})"


# ---------------------------------------------------------------- tables


def arrays(frame: pl.DataFrame, theirs: str) -> tuple[np.ndarray, np.ndarray]:
    return frame[OURS].to_numpy().astype(bool), frame[theirs].to_numpy().astype(bool)


def compare_row(label: str, frame: pl.DataFrame, theirs: str) -> list[str]:
    ours_values, their_values = arrays(frame, theirs)
    n = len(ours_values)
    if n == 0:
        return [label, "0", "n/a", "n/a", "n/a", "n/a"]
    return [
        label,
        f"{n:,}",
        pct(ours_values.mean()),
        pct(their_values.mean()),
        pct((ours_values == their_values).mean()),
        kappa_text(ours_values, their_values),
    ]


COMPARE_HEADER = [
    "",
    "overlap",
    "our success",
    "their success",
    "raw agreement",
    "kappa (95% interval)",
]
MIN_SUBGROUP = 20


def subgroup_rows(name: str, frame: pl.DataFrame, theirs: str) -> list[list[str]]:
    """Rows by our phase and by sponsor class (industry or not), skipping groups under 20."""
    parts = [
        ("phase 2", frame.filter(pl.col("phase") == "phase2")),
        ("phase 3", frame.filter(pl.col("phase") == "phase3")),
        ("industry sponsor", frame.filter(pl.col("sponsor_class") == "INDUSTRY")),
        ("other sponsor", frame.filter(pl.col("sponsor_class") != "INDUSTRY")),
    ]
    return [
        compare_row(f"{name}: {label}", part, theirs)
        for label, part in parts
        if part.height >= MIN_SUBGROUP
    ]


def confusion_table(frame: pl.DataFrame, theirs: str, their_name: str) -> str:
    both, ours_only, theirs_only, neither = confusion(*arrays(frame, theirs))
    rows = [
        ["our label: success", f"{both:,}", f"{ours_only:,}"],
        ["our label: failure", f"{theirs_only:,}", f"{neither:,}"],
    ]
    return md_table(["", f"{their_name}: success", f"{their_name}: failure"], rows)


def cluster_table(frame: pl.DataFrame) -> str:
    """CTO's probability in bands: trials, how many were terminated, CTO's p-value signal, our success."""
    probability = frame["probability"].to_numpy()
    rows = []
    for name, low, high in CTO_CLUSTERS:
        inside = frame.filter((pl.col("probability") >= low) & (pl.col("probability") < high))
        n = inside.height
        if n == 0:
            rows.append([name, "0", "n/a", "n/a", "n/a"])
            continue
        rows.append(
            [
                name,
                f"{n:,}",
                pct((inside["overall_status"] == "TERMINATED").mean()),
                pct((inside["p_value_signal"] == 1).mean()),
                pct(inside[OURS].cast(pl.Float64).mean()),
            ]
        )
    assert sum(int(r[1].replace(",", "")) for r in rows) == len(probability)
    return md_table(
        [
            "CTO probability",
            "trials",
            "terminated (our record)",
            "CTO p-value signal says significant",
            "our label: success",
        ],
        rows,
    )


def threshold_frames(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The same trials with CTO's probability turned into a yes or no three ways."""
    low, high = CTO_CONFIDENT
    confident = frame.filter((pl.col("probability") < low) | (pl.col("probability") >= high))
    return {
        f"cut at {CTO_CUTOFF}": frame.with_columns(theirs=pl.col("probability") >= CTO_CUTOFF),
        f"cut at {CTO_SECOND_CUTOFF}": frame.with_columns(
            theirs=pl.col("probability") >= CTO_SECOND_CUTOFF
        ),
        f"confident trials only (below {low} or above {high}), cut at {CTO_CUTOFF}": (
            confident.with_columns(theirs=pl.col("probability") >= CTO_CUTOFF)
        ),
    }


# ---------------------------------------------------------------- the hand-read disagreements


def load_disagreements(path: Path = DISAGREEMENTS_PATH) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_disagreement_ids(rows: list[dict], dev: pl.DataFrame) -> None:
    """Every hand-read trial must be a development trial."""
    unknown = sorted({r["nct_id"] for r in rows} - set(dev["nct_id"].to_list()))
    if unknown:
        raise ValueError(f"Hand-read trials that are not development trials: {', '.join(unknown)}")


def _reading(counts: dict[str, int], facts: RegistryFacts | None) -> str:
    """One paragraph on what the categories mean, with the counts from the file."""
    stopped = counts.get(CATEGORY_STOPPED, 0)
    unknown = counts.get(CATEGORY_UNKNOWN, 0)
    wrong = counts.get(CATEGORY_WRONG_WAY, 0)
    one_sided = counts.get(CATEGORY_ONE_SIDED, 0)
    threshold = counts.get(CATEGORY_THRESHOLD, 0)
    text = (
        f"{stopped} of these trials were stopped early yet posted a significant primary result, which "
        "CTO or TOP counts as a failure and we count as a success: a difference in the question, not "
        f"an error in either label. For {unknown} completed trials with a significant result, TOP says "
        "failure, and the registry record does not show why; TOP's files do not say either. "
        f"{wrong + one_sided} are errors in our label, checked against the registry record: {wrong} "
        "had a significant result in the wrong direction (the registry has no field for which "
        f"direction is better, BELIEFS.md L4), and {one_sided} counted a one-sided p-value. "
        f"{threshold} more were below 0.05 but, as the reader recalls, above the trial's own "
        "threshold (L11); those two rest on recall and are unverified."
    )
    if facts is not None and facts.examined:
        text += (
            f"\n\nThe one-sided case shows a weakness: our label accepts any p-value below 0.05 and "
            "does not ask whether it is one-sided or two-sided. Among the "
            f"{facts.examined:,} development trials with a deciding comparison, {facts.one_sided:,} "
            f"({pct(facts.one_sided / facts.examined)}) have a deciding comparison whose p-value "
            "comment or statistical comment says one-sided (or one-tailed). That is a count of "
            "mentions, not of labels that would change."
        )
    return text


def disagreement_section(rows: list[dict], facts: RegistryFacts | None) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    summary = md_table(
        ["What the read found", "trials"],
        [[name, str(n)] for name, n in sorted(counts.items(), key=lambda item: -item[1])]
        + [["all", str(len(rows))]],
    )
    table = md_table(
        ["Trial", "direction", "category", "note", "source of judgement"],
        [[r["nct_id"], r["direction"], r["category"], r["note"], r["source"]] for r in rows],
    )
    return "\n\n".join(
        [
            (
                "These are trials where our label says success and another benchmark says failure (for "
                f"CTO, a probability below {CTO_CUTOFF}, which is the terminated trials). "
                f"{len(rows)} different trials were read (30 were drawn, 15 from each benchmark, and one "
                "was drawn from both). A language model read each registry record; it was not checked by "
                "a person. The 5 wrong-direction trials and the one-sided trial were then checked "
                "against the raw registry record and are marked 'registry record'; judgements marked "
                "'model recall, unverified' rest on the reader's memory of the trial, not on the "
                "record. The trials were drawn from the disagreements, so this is a description of "
                "why labels differ, not an error rate for either label."
            ),
            summary,
            _reading(counts, facts),
            table,
        ]
    )


# ---------------------------------------------------------------- definitions of success

CTO_DEFINITION = (
    "**CTO** (read in the paper, arXiv 2406.10292). Trials that were terminated, withdrawn, suspended "
    "or no longer available are failures, trials marked approved for marketing are successes, and a "
    "completed trial with any significant primary p-value (below 0.05) is a success. The rest are "
    "labelled by a weak-supervision model that combines several signals, including a language model's "
    "reading of publications, news headlines, links to later trials and approvals, and trial features. "
    "The `_rf` files used here hold a random forest's probability (`pred_proba`; read in the paper: the "
    "forest is fitted to TOP's human-annotated labels from those signals; that `pred_proba` is that "
    "forest's output is our reading of the file name). One of its inputs is a p-value signal "
    "(`pvalues`: 1 if any primary p-value is below 0.05, 0 if not, -1 if unknown; read in the paper). "
    "The files give a probability, not a label, and no cut-off: any yes-or-no here is our choice."
)
TOP_DEFINITION = (
    "**TOP** (Fu et al.). The TOP repository's README says only that `label` is \"0 (failure) or 1 "
    '(success)" and that "Outcome labels are provided by IQVIA". The CTO paper, citing TOP among '
    'others, describes the outcome as "whether the trial achieves its primary endpoints and can '
    'progress to the next stage of drug development" (arXiv 2406.10292). We could not verify more '
    "about how TOP's labels were made."
)


# ---------------------------------------------------------------- the report


def _selection_text(
    joined: dict[str, pl.DataFrame], facts: RegistryFacts | None, shared: int
) -> str:
    """How our overlap differs from CTO's trials as a whole, in the one respect we can measure."""
    text = (
        "**Who is in the overlap.** Our set holds only trials that posted a treatment-versus-control "
        "p-value, so stopped trials, which CTO counts as failures, are under-represented."
    )
    if facts is not None and facts.inclusion:
        rate = facts.inclusion
        text += (
            f" Of all finished pre-2021 drug trials, {pct(rate['COMPLETED'])} of completed, "
            f"{pct(rate['TERMINATED'])} of terminated and {pct(rate['WITHDRAWN'])} of withdrawn trials "
            "are in our set."
        )
    if facts is not None and facts.wider:
        rows = []
        for name, frame in joined.items():
            wide = facts.wider.get(name.removeprefix("CTO "))
            if wide is None:
                continue
            ours_stopped = (frame["overall_status"] != "COMPLETED").mean()
            ours_half = (frame["probability"] >= CTO_CUTOFF).mean()
            rows.append(
                [
                    name,
                    f"{wide.n:,}",
                    pct(wide.stopped),
                    pct(wide.success_half),
                    f"{frame.height:,}",
                    pct(ours_stopped),
                    pct(ours_half),
                ]
            )
        text += "\n\n" + md_table(
            [
                "",
                "all finished pre-2021 trials in the file",
                "terminated or withdrawn",
                f"CTO probability at or above {CTO_CUTOFF}",
                "our overlap",
                "terminated or withdrawn",
                f"CTO probability at or above {CTO_CUTOFF}",
            ],
            rows,
        )
        text += (
            "\n\nSo the share CTO calls success at a cut-off of 0.5 is higher in our overlap than "
            "across all these trials, partly because stopped trials are missing, but it is high across all "
            "of them too: at 0.5 CTO's label mostly separates completed trials from stopped ones. A figure "
            "like '90% successes' is a property of the cut-off and of which trials are counted, and "
            "does not measure how well CTO's model does."
        )
    text += (
        f" {shared:,} trials appear in both CTO files; each file is compared on its own, so they are "
        "counted in both rows of the next section and the two rows are not independent."
    )
    return text


def report(
    dev: pl.DataFrame,
    cto: dict[str, CtoRead],
    top: tuple[pl.DataFrame, int],
    disagreements: list[dict],
    facts: RegistryFacts | None = None,
) -> str:
    """The whole results file. Refuses a development table holding a trial from 2021 or later."""
    check_is_development(dev)
    check_disagreement_ids(disagreements, dev)
    ours = dev.select("nct_id", "phase", "sponsor_class", "overall_status", OURS).filter(
        pl.col(OURS).is_not_null()
    )
    cto_frames = {f"CTO {name}": ours.join(read.frame, on="nct_id") for name, read in cto.items()}
    top_frame, top_conflicts = top
    top_joined = ours.join(top_frame, on="nct_id").with_columns(theirs=pl.col("top_success"))
    shared = (
        len(set.intersection(*(set(f["nct_id"].to_list()) for f in cto_frames.values())))
        if len(cto_frames) > 1
        else 0
    )

    # Agreement: each CTO file at each way of turning its probability into a yes or no, then TOP.
    headline, confusions, breakdown = [], [], []
    for name, frame in cto_frames.items():
        for how, view in threshold_frames(frame).items():
            headline.append(compare_row(f"{name}, {how}", view, "theirs"))
        at_second = threshold_frames(frame)[f"cut at {CTO_SECOND_CUTOFF}"]
        confusions.append(
            f"**{name}, cut at {CTO_SECOND_CUTOFF}** (rows: our label, columns: theirs)\n\n"
            + confusion_table(at_second, "theirs", name)
        )
        breakdown += subgroup_rows(f"{name} (cut at {CTO_SECOND_CUTOFF})", at_second, "theirs")
    headline.append(compare_row("TOP", top_joined, "theirs"))
    confusions.append(
        "**TOP** (rows: our label, columns: theirs)\n\n"
        + confusion_table(top_joined, "theirs", "TOP")
    )
    breakdown += subgroup_rows("TOP", top_joined, "theirs")

    ranking, p_rows = [], []
    for name, frame in cto_frames.items():
        ours_values = frame[OURS].to_numpy().astype(bool)
        ranking.append(
            [name, f"{frame.height:,}", auroc_text(ours_values, frame["probability"].to_numpy())]
        )
        signal = frame.filter(pl.col("p_value_signal") != -1).with_columns(
            p_value_theirs=pl.col("p_value_signal") == 1
        )
        p_rows.append(compare_row(f"{name}: their p-value signal", signal, "p_value_theirs"))

    kappas = {}
    for name, frame in cto_frames.items():
        views = threshold_frames(frame)
        kappas[name] = [kappa(*arrays(view, "theirs")) for view in views.values()]
    detail = "; ".join(
        f"{k[0]:.2f} at {CTO_CUTOFF} and {k[1]:.2f} at {CTO_SECOND_CUTOFF} ({n})"
        for n, k in kappas.items()
    )
    values = [x for k in kappas.values() for x in k[:2]]
    swing = (
        "from slight to strong"
        if values and min(values) < 0.2 and max(values) > 0.7
        else "a good deal"
    )
    agreement_sentence = (
        "On the trials both cover, how CTO's probability is turned into a yes or no changes its "
        f"agreement with our label {swing}: kappa is {detail}. A benchmark's "
        '"accuracy" depends on its label definition and on the threshold, which is the reason to '
        "state both."
    )

    repeated = ", ".join(f"{name} {read.repeated}" for name, read in cto.items())
    dropped = ", ".join(f"{name} {read.dropped}" for name, read in cto.items())
    sections = [
        "# Our success label against two published benchmarks: development set\n",
        (
            "Do other benchmarks mean the same thing by success? Compared on the trials that are in both "
            "our development set (primary completion before 2021) and the other benchmark. Nothing from "
            "2021 onwards was read: the benchmark files are filtered to our development trial ids before "
            "any label column is read, and the code refuses anything else. The benchmark files are not "
            "in this repository (see the module docstring for where to get them).\n"
        ),
        "## 1. How each benchmark defines success\n\n"
        + CTO_DEFINITION
        + "\n\n"
        + TOP_DEFINITION
        + "\n\n**Ours.** At least one treatment arm beat its control at p < 0.05 on the first primary "
        "outcome that has a treatment-versus-control comparison (BELIEFS-CURRENT.md). It does not check "
        "which direction is better, or whether the p-value is one-sided.",
        "## 2. What CTO's probability looks like on these trials\n\n"
        + "\n\n".join(f"**{name}**\n\n{cluster_table(frame)}" for name, frame in cto_frames.items())
        + "\n\nAmong these trials CTO's probability falls in three groups and almost nowhere else: below "
        "0.1, which is the terminated trials (CTO counts those as failures); 0.55 to 0.75, completed trials "
        "whose p-value signal says not significant, where CTO's model is unsure; and 0.9 and above, "
        "completed trials whose p-value signal says significant. At a cut-off of 0.5, CTO's label is in "
        "effect whether the trial completed.\n\n" + _selection_text(cto_frames, facts, shared),
        "## 3. Agreement\n\n"
        + md_table(COMPARE_HEADER, headline)
        + "\n\n"
        + agreement_sentence
        + " Neither cut-off is CTO's: the files give none. 0.5 was our first try and 0.7 was chosen after "
        "seeing the groups in section 2, so the second view is shown to make the dependence visible, not "
        "as the better reading of CTO. The third view keeps only the trials CTO is confident about.\n\n"
        "The real disagreement is on completed trials without a significant p-value: CTO's model "
        "weakly leans towards success (probabilities of 0.55 to 0.75) and we count them as failures. "
        "Kappa near 0 means no more agreement than chance; near 1 means the same labels. "
        f"Trials listed more than once within a file: {repeated}. Of those, trials whose copies fall on "
        f"different sides of a cut-off, dropped: {dropped}. "
        f"TOP trials dropped because their copies disagree: {top_conflicts}.\n\n"
        + "\n\n".join(confusions)
        + "\n\nThe two-by-two tables: each trial counted once, by what each label says.",
        "## 4. By phase and by sponsor\n\n"
        + md_table(COMPARE_HEADER, breakdown)
        + "\n\nThe same comparison inside our phase and sponsor groups (groups under "
        f"{MIN_SUBGROUP} trials are left out). CTO is cut at {CTO_SECOND_CUTOFF} here. Phase is ours: "
        "combined phase 2/3 trials count as phase 3.",
        "## 5. How well does CTO's probability rank our label?\n\n"
        + md_table(["", "overlap", "AUROC against our label (95% interval)"], ranking)
        + "\n\nAUROC is the chance that a random trial with our success label gets a higher CTO "
        "probability than a random trial without it (0.5 is a coin toss). CTO's model takes a p-value as "
        "an input, and our label is a p-value rule, so a high value is partly the same signal seen "
        "twice, not independent confirmation.\n\n"
        + md_table(COMPARE_HEADER, p_rows)
        + "\n\nCTO's own p-value signal against our label, on trials where CTO has one. This is the "
        "closest like-for-like test of whether the two p-value rules read trials the same way.",
        "## 6. What the disagreements show about our own label\n\n"
        + disagreement_section(disagreements, facts),
    ]
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------- running it


def fetch(cto_dir: Path, top_file: Path) -> None:
    """Download the benchmark files with plain HTTPS requests, if they are not there yet."""
    downloads = [(CTO_URL + name, Path(cto_dir) / name) for name in CTO_FILES.values()]
    downloads.append((TOP_URL, Path(top_file)))
    for url, destination in downloads:
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "trial-outcome-benchmark/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(destination)
        print(f"fetched {destination.name} ({destination.stat().st_size:,} bytes)")


def main(argv: Iterable[str] | None = None, out_path: Path = RESULTS_PATH) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cto-dir", type=Path, default=BENCHMARK_DIR / "cto")
    parser.add_argument("--top-file", type=Path, default=BENCHMARK_DIR / "top" / TOP_FILE)
    parser.add_argument("--fetch", action="store_true", help="download the files first")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.fetch:
        fetch(args.cto_dir, args.top_file)
    dev = load_dev()
    ids = DevelopmentIds.from_frame(dev)
    cto = {name: read_cto(args.cto_dir / file, ids) for name, file in CTO_FILES.items()}
    population = build_population(download.iter_studies())
    wider = {name: wider_cto(args.cto_dir / file, population) for name, file in CTO_FILES.items()}
    facts = registry_facts(population, download.iter_studies(), ids, wider)
    text = report(dev, cto, read_top(args.top_file, ids), load_disagreements(), facts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
