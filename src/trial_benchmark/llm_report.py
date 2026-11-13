"""How a language model does on the sampled development trials: what is design, what is recall?

    uv run python -m trial_benchmark.llm_report     # -> results/dev_llm.md

Development trials only. This module reads outcomes through `dev_summary.load_dev`, which refuses
anything but development data, and `build_report` refuses a frame with any other trial. Test and
late trials may have stored predictions (they are features), but they are never scored here: that
belongs to the final logged run.

What it reports, all on the same sampled 2017 to 2020 development trials:

- AUROC, Brier score and calibration gap for each model and prompt variant (identified,
  masked, blinded), with bootstrap intervals, beside the registration baseline and "ladder:
  registration + words", both trained on development trials finishing before 2017.
- The blinded prompt on the trials the model could not name, and on those it could, against both
  baselines on the same trials. "Named" has two definitions, side by side: the `drug_guess` matches
  an alias of the tested drug (`llm_blinding.guess_matches`), or that, or the `reason` text holds
  an alias, or claims a result and names a trial the prompt does not hold. The split is chosen by
  the model's own output, so it is exploratory: recall is not removed there.
- That split crossed with the model's own "I recognise this trial", a row for trials neither
  model named, how the not-named trials differ from the named ones, and how often a reply claims
  to know an outcome.
- Paired differences between prompts. They are not clean measures of recall: blinding removes
  recall, but also the legitimate knowledge a forecaster has when a trial starts (the drug's
  class, its earlier results, who the comparator is). The wording of each contrast says so.
- The scores again without the trials whose arm or outcome text holds a post-hoc phrase.
- The share of trials the model says it recognises, by variant and by completion year; how often
  its drug guess is right; how often blinding left a word of an intervention name in the prompt
  (the leak check); how much a repeat of the same batch moves the answers.

Every interval resamples trials, all models share the same resamples, and a difference is taken
within each resample. Trials that share a drug or sponsor are not independent, so intervals are
somewhat too narrow. An interval is not reported if more than a twentieth of the resamples have
no AUROC (one outcome class only).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import numpy as np
import polars as pl

from trial_benchmark import dev_summary, metrics
from trial_benchmark.dataset import LABEL_COLUMNS, ROOT
from trial_benchmark.llm_blinding import KNOWN_GAPS, guess_matches, text_names_drug
from trial_benchmark.llm_predictor import (
    MODELS,
    PERIODS,
    PREDICTIONS_PATH,
    VARIANTS,
    leak_counts,
    load_records,
    read_predictions,
    read_sample,
    render_trial,
)
from trial_benchmark.models.linear import REGISTRATION_NAME
from trial_benchmark.models.registry import all_specs

REPORT_PATH = ROOT / "results" / "dev_llm.md"
LADDER_NAME = "ladder: registration + words"
BASELINE_TRAIN_BEFORE = 2017  # as in the rehearsal: train on trials finishing before this year
LAST_DEV_YEAR = 2020
MIN_FOR_AUROC = 10  # a subset smaller than this, or with one outcome only, gets no AUROC
MAX_DEGENERATE = 0.05  # more resamples than this without an AUROC: no interval is given
COULD_NOT_MEASURE = "interval could not be measured"


def combo(model: str, variant: str) -> str:
    return f"{model} ({variant})"


COMBOS = [combo(m, v) for m in MODELS for v in VARIANTS]
# (first, second): each is reported as first minus second, within each model.
CONTRASTS = (("identified", "blinded"), ("masked", "blinded"), ("identified", "masked"))


# ---------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def show(self, signed: bool = False, places: int = 3) -> str:
        def one(v: float) -> str:
            return f"{v:+.{places}f}".replace("-", "−") if signed else f"{v:.{places}f}"

        if np.isnan(self.value):
            return "n/a"
        if np.isnan(self.low) or np.isnan(self.high):
            return f"{one(self.value)} ({COULD_NOT_MEASURE})"
        return f"{one(self.value)} ({one(self.low)} to {one(self.high)})"


SCORES = {
    "auroc": metrics.auroc,
    "brier": metrics.brier,
    "gap": metrics.calibration_gap,
}


def interval(values: np.ndarray, point: float) -> Estimate:
    """The point and its percentile interval; no interval if too many resamples had no score."""
    missing = np.isnan(values)
    if np.isnan(point) or missing.mean() > MAX_DEGENERATE:
        return Estimate(point, float("nan"), float("nan"))
    low, high = np.percentile(values[~missing], metrics.INTERVAL_PERCENTILES)
    return Estimate(point, float(low), float(high))


def paired_bootstrap(
    outcomes: np.ndarray,
    predictions: dict[str, np.ndarray],
    pairs: list[tuple[str, str]] = (),
    clusters: np.ndarray | None = None,
) -> tuple[dict[str, dict[str, Estimate]], dict[tuple[str, str], dict[str, Estimate]]]:
    """Each score (AUROC, Brier, gap) of each model, and `first - second` for each pair.

    One set of resamples serves every model, so a difference is within each resample. With
    `clusters` (one label per trial, such as the lead drug) each resample draws whole clusters.
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    for name, p in predictions.items():
        if len(p) != len(outcomes):
            raise ValueError(f"{name}: {len(p)} predictions for {len(outcomes)} outcomes")
    rng = np.random.default_rng(metrics.BOOTSTRAP_SEED)
    members = None if clusters is None else metrics.cluster_members(clusters, len(outcomes))
    draws = {n: {s: np.full(metrics.N_RESAMPLES, np.nan) for s in SCORES} for n in predictions}
    for i in range(metrics.N_RESAMPLES):
        if members is None:
            chosen = rng.integers(0, len(outcomes), size=len(outcomes))
        else:
            picks = rng.integers(0, len(members), size=len(members))
            chosen = np.concatenate([members[k] for k in picks])
        for name, p in predictions.items():
            for score, function in SCORES.items():
                draws[name][score][i] = function(outcomes[chosen], p[chosen])
    points = {n: {s: f(outcomes, p) for s, f in SCORES.items()} for n, p in predictions.items()}
    single = {n: {s: interval(draws[n][s], points[n][s]) for s in SCORES} for n in predictions}
    paired = {
        (a, b): {
            s: interval(draws[a][s] - draws[b][s], points[a][s] - points[b][s]) for s in SCORES
        }
        for a, b in pairs
    }
    return single, paired


# ---------------------------------------------------------------- the data


def check_development_only(dev: pl.DataFrame) -> None:
    """Refuse any frame that is not purely development trials (nothing from 2021 on)."""
    dev_summary.check_is_development(dev)


def sampled_development(dev: pl.DataFrame, sample: pl.DataFrame) -> pl.DataFrame:
    """The sampled development trials that are in `dev`, one row each, all finishing 2017 to 2020.

    Raises if a sampled trial finished before 2017: the baselines are trained on those years.
    """
    check_development_only(dev)
    ids = sample.filter(pl.col("split") == "dev")["nct_id"].to_list()
    trials = dev.filter(pl.col("nct_id").is_in(ids))
    years = trials["primary_completion_date"].dt.year()
    if not years.is_between(BASELINE_TRAIN_BEFORE, LAST_DEV_YEAR).all():
        raise ValueError(
            f"A sampled trial finished outside {BASELINE_TRAIN_BEFORE}-{LAST_DEV_YEAR}."
        )
    return trials


def wide_predictions(
    predictions: pl.DataFrame, ids: list[str], run: int = 0, complete: bool = True
) -> pl.DataFrame:
    """One row per trial in `ids`, with a column per model and variant (and `<name> recognised`,
    `<name> guess`, `<name> reason`).

    Only successful predictions count, one row per trial, model, variant and run (a repeat that
    would double a row raises). With `complete`, trials missing any probability are dropped;
    otherwise they stay, with nulls. A null guess is not missing: it is a model saying "unknown".
    """
    wide = pl.DataFrame({"nct_id": ids})
    needed = []
    for model in MODELS:
        for variant in VARIANTS:
            name = combo(model, variant)
            part = predictions.filter(
                (pl.col("model") == model)
                & (pl.col("variant") == variant)
                & (pl.col("run") == run)
                & pl.col("probability").is_not_null()
            ).select(
                "nct_id",
                pl.col("probability").alias(name),
                pl.col("recognised").alias(f"{name} recognised"),
                pl.col("drug_guess").alias(f"{name} guess"),
                pl.col("reason").alias(f"{name} reason"),
            )
            wide = wide.join(part, on="nct_id", how="left", validate="1:1")
            needed += [name, f"{name} recognised"]
    return wide.drop_nulls(subset=needed) if complete else wide


def fit_baselines(dev_with_text: pl.DataFrame, ids: list[str]) -> dict[str, dict[str, float]]:
    """The registration baseline and "ladder: registration + words", trained on trials
    that finished before 2017 and predicting the trials in `ids` (all finishing 2017 to 2020)."""
    check_development_only(dev_with_text)
    specs = {s.name: s for s in all_specs() if s.name in (REGISTRATION_NAME, LADDER_NAME)}
    year = dev_with_text["primary_completion_date"].dt.year()
    train = dev_with_text.filter(year < BASELINE_TRAIN_BEFORE)
    target = dev_with_text.filter(pl.col("nct_id").is_in(ids))
    if (
        not target["primary_completion_date"]
        .dt.year()
        .is_between(BASELINE_TRAIN_BEFORE, LAST_DEV_YEAR)
        .all()
    ):
        raise ValueError("A trial to predict finished before the baselines' training cut.")
    if set(train["nct_id"]) & set(target["nct_id"]):
        raise ValueError("A baseline would be scored on a trial it was trained on.")
    blind = target.drop(*LABEL_COLUMNS, strict=False)
    return {
        name: dict(zip(target["nct_id"], spec.fit(train).predict(blind), strict=True))
        for name, spec in specs.items()
    }


# ---------------------------------------------------------------- sections


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def headline_section(single, names: list[str]) -> list[str]:
    rows = [
        [
            n,
            single[n]["auroc"].show(),
            single[n]["brier"].show(),
            single[n]["gap"].show(places=1),
        ]
        for n in names
    ]
    return table(["model", "AUROC (95% interval)", "Brier", "calibration gap, points"], rows)


def difference_section(paired, pairs: list[tuple[str, str]]) -> list[str]:
    rows = [
        [
            f"{a} minus {b}",
            paired[(a, b)]["auroc"].show(signed=True),
            paired[(a, b)]["brier"].show(signed=True),
        ]
        for a, b in pairs
    ]
    return table(["pair", "AUROC difference", "Brier difference (negative is better)"], rows)


def can_score(outcomes: np.ndarray) -> bool:
    """True if there are enough trials, and both outcomes, to give an AUROC."""
    return len(outcomes) >= MIN_FOR_AUROC and 0 < outcomes.sum() < len(outcomes)


def recognition_section(outcomes: np.ndarray, wide: pl.DataFrame) -> list[str]:
    rows = []
    for name in COMBOS:
        said = wide[f"{name} recognised"].to_numpy()
        p = wide[name].to_numpy()
        row = [name]
        for flag in (True, False):
            keep = said == flag
            n = int(keep.sum())
            if can_score(outcomes[keep]):
                single, _ = paired_bootstrap(outcomes[keep], {name: p[keep]})
                score = single[name]["auroc"].show()
            else:
                score = "too few to score"
            rate = f"{outcomes[keep].mean():.0%}" if n else "n/a"
            row += [str(n), rate, score]
        rows.append(row)
    return table(
        [
            "model",
            "said yes: trials",
            "success rate",
            "AUROC",
            "said no: trials",
            "success rate",
            "AUROC",
        ],
        rows,
    )


def leak_section(leaks: dict[str, int] | None, naming: dict[str, float] | None) -> list[str]:
    """The three pieces of evidence that name removal does not blind a model to a trial."""
    if leaks is None:
        leak = "the leak check was not run for this report"
    elif leaks["trials"] == 0:
        leak = "the leak check had no trials"
    else:
        leak = (
            f"After blinding, {leaks['any']} of {leaks['trials']} sampled development trials "
            f"({leaks['any'] / leaks['trials']:.0%}) still show a word or company code from one of "
            "their own intervention names or other names. That undercounts: it cannot see an "
            'abbreviation ("MF" for mometasone), a brand or device name that was never listed, or '
            "a programme recognised from its design"
        )
    named = (
        "not run for this report"
        if naming is None
        else ", ".join(f"{name} {rate:.0%}" for name, rate in naming.items())
    )
    return [
        "Name removal cannot blind a model to a trial it has read about. Three pieces of evidence:",
        "",
        f"- **Leak check.** {leak}.",
        (
            "- **Hand read.** Reading 30 blinded prompts (`results/llm_blinded_read.csv`, made with an "
            "earlier version of the blinding rules), the builder found the drug obvious in 13 and "
            "partly in 5. A reviewer reading a fresh 30 found it obvious in about 15 and partly in 8."
        ),
        (
            "- **The model's own guess.** On blinded prompts the model's `drug_guess` names the "
            f"tested drug in: {named}."
        ),
    ]


def recognised_rate_section(wide: pl.DataFrame, named: dict[str, np.ndarray] | None) -> list[str]:
    rows = []
    for name in COMBOS:
        row = [name, f"{wide[f'{name} recognised'].to_numpy().mean():.0%}"]
        row.append("n/a" if named is None else f"{named[name].mean():.0%}")
        rows.append(row)
    return table(
        [
            "model",
            "share it said it recognised",
            "share where its drug guess names the tested drug",
        ],
        rows,
    )


def share_by_year_section(wide: pl.DataFrame, years: np.ndarray) -> list[str]:
    rows = []
    for year in sorted(set(years.tolist())):
        keep = years == year
        row = [str(year), str(int(keep.sum()))]
        row += [f"{wide[f'{n} recognised'].to_numpy()[keep].mean():.0%}" for n in COMBOS]
        rows.append(row)
    return table(["completion year", "trials", *COMBOS], rows)


def post_hoc_section(sample: pl.DataFrame) -> list[str]:
    """Trials per period whose arm or outcome text holds a post-hoc phrase (all 900 sampled)."""
    rows = []
    for period in PERIODS:
        group = sample.filter(pl.col("period") == period)
        rows.append([period, str(group.height), str(int(group["post_hoc"].sum()))])
    return table(["period", "sampled trials", "with a post-hoc phrase"], rows)


def run_to_run_section(predictions: pl.DataFrame, ids: list[str]) -> list[str]:
    """How far a repeat of the same question moves the answer. Uses no outcome."""
    first, repeat = (wide_predictions(predictions, ids, run, complete=False) for run in (0, 1))
    rows = []
    for name in COMBOS:
        both = (
            first.select("nct_id", a=name, ra=f"{name} recognised")
            .join(repeat.select("nct_id", b=name, rb=f"{name} recognised"), on="nct_id")
            .drop_nulls()
        )
        if both.height < 2:
            continue
        a, b = both["a"].to_numpy(), both["b"].to_numpy()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        flipped = float((both["ra"] != both["rb"]).mean())
        rows.append(
            [name, str(both.height), f"{corr:.2f}", f"{np.abs(a - b).mean():.3f}", f"{flipped:.0%}"]
        )
    if not rows:
        return ["No repeated batches have been run."]
    return table(
        [
            "model",
            "trials asked twice",
            "correlation",
            "mean absolute change",
            "recognised flipped",
        ],
        rows,
    )


def reading(effect: Estimate) -> str:
    """One clause saying whether a paired AUROC difference is clearly above zero."""
    if np.isnan(effect.value) or np.isnan(effect.low) or np.isnan(effect.high):
        return "could not be measured"
    if effect.low > 0:
        return "clearly above zero"
    if effect.high < 0:
        return "clearly below zero"
    return "not clearly different from zero"


# ---------------------------------------------------------------- did the model name the drug?

GUESS = "guess names the drug"
STRICTER = "guess or reason names it"
DEFINITIONS = (GUESS, STRICTER)
NAMED = "drug named"
NOT_NAMED = "drug not named"

# A reply that claims to know how a trial came out, by wording alone. Rule-based and readable.
OUTCOME_CLAIM = re.compile(
    r"\b(recall|was (strongly |clearly )?(positive|negative)|strongly positive|met (its|the|both|all)"
    r"|missed|hit its|did not (meet|show|reach)|failed|reported|published)\b",
    re.IGNORECASE,
)
# A word that looks like a trial name: a run of four capitals (SPARTAN, STRIVE) or a capital run
# inside a word (TrustTSC). Not an ordinary outcome abbreviation, because those are in the prompt.
TRIAL_NAME = re.compile(r"[A-Z]{4,}|(?<=[a-z])[A-Z]{3,}")


def claims_an_outcome(reason: str | None) -> bool:
    return bool(OUTCOME_CLAIM.search(reason or ""))


def names_a_trial(reason: str | None, prompt: str) -> bool:
    """True if the reason holds a trial-name-like word that the prompt does not: the model's own."""
    shown = prompt.lower()
    return any(word.lower() not in shown for word in TRIAL_NAME.findall(reason or ""))


def named_stricter(guess: str | None, reason: str | None, record: dict) -> bool:
    """True if the blinded reply identifies the drug or trial, by the guess or by its reason.

    Any of: the guess matches an alias of the tested intervention; the reason holds an alias; the
    reason claims a result and names a trial (a word of capitals that the blinded prompt does
    not hold). The last two catch what the guess field misses: a synonym the alias list lacks, a
    trial named in the reason only. It cannot catch a synonym written in lower case with no
    result claimed ("melatonin" for Circadin), so it still undercounts.
    """
    if guess_matches(guess, record) or text_names_drug(reason, record):
        return True
    return claims_an_outcome(reason) and names_a_trial(reason, render_trial(record, "blinded", 1))


def named_arrays(
    wide: pl.DataFrame, kept: list[str], records: dict[str, dict]
) -> dict[str, dict[str, np.ndarray]]:
    """For each definition of "named" and each reply set, True where the model named the drug.

    The guess definition covers every prompt variant; the stricter one only the blinded prompt.
    """
    guess = {
        n: np.array(
            [guess_matches(g, records[i]) for i, g in zip(kept, wide[f"{n} guess"], strict=True)]
        )
        for n in COMBOS
    }
    stricter = {}
    for model in MODELS:
        n = combo(model, "blinded")
        replies = zip(kept, wide[f"{n} guess"], wide[f"{n} reason"], strict=True)
        stricter[n] = np.array([named_stricter(g, r, records[i]) for i, g, r in replies])
    return {GUESS: guess, STRICTER: stricter}


# ---------------------------------------------------------------- the trials the model could not name

MIN_FOR_CELL = 20  # a four-cell group smaller than this gets no AUROC


@dataclass(frozen=True)
class Subset:
    """The blinded prompt of one model on a subset of trials, beside the two baselines."""

    n: int
    rate: float
    blinded: Estimate | None = None
    registration: Estimate | None = None
    difference: Estimate | None = None
    ladder: Estimate | None = None
    ladder_difference: Estimate | None = None


def subset_result(
    outcomes: np.ndarray,
    arrays: dict[str, np.ndarray],
    keep: np.ndarray,
    name: str,
    minimum: int = MIN_FOR_AUROC,
) -> Subset:
    """AUROC of `name` and of each baseline on the trials in `keep`, and each paired gap."""
    n = int(keep.sum())
    rate = float(outcomes[keep].mean()) if n else float("nan")
    if n < minimum or not can_score(outcomes[keep]):
        return Subset(n, rate)
    refs = [r for r in (REGISTRATION_NAME, LADDER_NAME) if r in arrays]
    pairs = [(name, r) for r in refs]
    single, paired = paired_bootstrap(
        outcomes[keep], {k: arrays[k][keep] for k in [name, *refs]}, pairs
    )

    def against(ref: str) -> tuple[Estimate | None, Estimate | None]:
        if ref not in refs:
            return None, None
        return single[ref]["auroc"], paired[(name, ref)]["auroc"]

    registration, difference = against(REGISTRATION_NAME)
    ladder, ladder_difference = against(LADDER_NAME)
    return Subset(
        n, rate, single[name]["auroc"], registration, difference, ladder, ladder_difference
    )


def not_named_results(
    outcomes: np.ndarray,
    arrays: dict[str, np.ndarray],
    named: dict[str, dict[str, np.ndarray]] | None,
) -> dict[tuple[str, str, str], Subset]:
    """For each definition and model, its blinded prompt on trials it did not name, and did."""
    if named is None:
        return {}
    results = {}
    for definition, by_name in named.items():
        for model in MODELS:
            name = combo(model, "blinded")
            for label, keep in ((NOT_NAMED, ~by_name[name]), (NAMED, by_name[name])):
                results[(definition, model, label)] = subset_result(outcomes, arrays, keep, name)
    return results


def four_cell_results(
    outcomes: np.ndarray,
    arrays: dict[str, np.ndarray],
    named: dict[str, np.ndarray],
    recognised: dict[str, np.ndarray],
) -> dict[tuple[str, bool, bool], Subset]:
    """Each model's blinded prompt by (named, said it recognised the trial), stricter definition."""
    results = {}
    for model in MODELS:
        name = combo(model, "blinded")
        for was_named in (False, True):
            for knew in (False, True):
                keep = (named[name] == was_named) & (recognised[name] == knew)
                results[(model, was_named, knew)] = subset_result(
                    outcomes, arrays, keep, name, MIN_FOR_CELL
                )
    return results


def neither_named_results(
    outcomes: np.ndarray, arrays: dict[str, np.ndarray], named: dict[str, np.ndarray]
) -> dict[str, Subset]:
    """Each model's blinded prompt on the trials that no model named (stricter definition)."""
    keep = np.ones(len(outcomes), dtype=bool)
    for model in MODELS:
        keep &= ~named[combo(model, "blinded")]
    return {m: subset_result(outcomes, arrays, keep, combo(m, "blinded")) for m in MODELS}


SUBSET_HEADER = [
    "trials",
    "success rate",
    "AUROC",
    "registration AUROC, same trials",
    "difference",
    "ladder AUROC, same trials",
    "difference",
]


def subset_cells(r: Subset) -> list[str]:
    def show(estimate: Estimate | None, signed: bool = False) -> str:
        return "too few to score" if estimate is None else estimate.show(signed=signed)

    rate = "n/a" if np.isnan(r.rate) else f"{r.rate:.0%}"
    return [
        str(r.n),
        rate,
        show(r.blinded),
        show(r.registration),
        show(r.difference, signed=True),
        show(r.ladder),
        show(r.ladder_difference, signed=True),
    ]


def subset_table(lead: list[str], rows: list[tuple[list[str], Subset]]) -> list[str]:
    return table([*lead, *SUBSET_HEADER], [[*label, *subset_cells(r)] for label, r in rows])


def not_named_section(results: dict[tuple[str, str, str], Subset]) -> list[str]:
    if not results:
        return ["Not run for this report (the raw records were not given)."]
    rows = [([d, m, label], r) for (d, m, label), r in results.items()]
    return subset_table(["definition of named", "model, blinded prompt", "subset"], rows)


def four_cell_section(cells: dict[tuple[str, bool, bool], Subset]) -> list[str]:
    if not cells:
        return ["Not run for this report (the raw records were not given)."]
    rows = [
        ([m.title(), NAMED if was_named else NOT_NAMED, "said yes" if knew else "said no"], r)
        for (m, was_named, knew), r in cells.items()
    ]
    return subset_table(["model", "drug", "said it recognised the trial"], rows)


def neither_section(results: dict[str, Subset]) -> list[str]:
    return subset_table(["model, blinded prompt"], [([m.title()], r) for m, r in results.items()])


def profile_cells(group: pl.DataFrame) -> list[str]:
    """Phase 3 share, median enrolment, industry share and mean completion year of a group."""
    if group.is_empty():
        return ["n/a"] * 4
    return [
        f"{(group['phase'] == 'phase3').mean():.0%}",
        f"{group['enrollment'].median():.0f}",
        f"{(group['sponsor_class'] == 'INDUSTRY').mean():.0%}",
        f"{group['primary_completion_date'].dt.year().mean():.1f}",
    ]


def profile_section(
    trials: pl.DataFrame,
    named: dict[str, np.ndarray],
    subsets: dict[tuple[str, str, str], Subset],
) -> list[str]:
    """How the trials a model did not name differ from the ones it did (stricter definition)."""
    rows = []
    for model in MODELS:
        for label in (NOT_NAMED, NAMED):
            is_named = named[combo(model, "blinded")]
            group = trials.filter(pl.Series(is_named if label == NAMED else ~is_named))
            result = subsets[(STRICTER, model, label)]
            registration = (
                "too few to score" if result.registration is None else result.registration.show()
            )
            phase3, enrolment, industry, year = profile_cells(group)
            rate = "n/a" if np.isnan(result.rate) else f"{result.rate:.0%}"
            rows.append(
                [
                    model.title(),
                    label,
                    str(group.height),
                    phase3,
                    rate,
                    enrolment,
                    industry,
                    year,
                    registration,
                ]
            )
    return table(
        [
            "model",
            "subset",
            "trials",
            "phase 3",
            "success rate",
            "median enrolment",
            "industry sponsor",
            "mean completion year",
            "registration AUROC on these trials",
        ],
        rows,
    )


def claims_section(
    wide: pl.DataFrame, named: dict[str, dict[str, np.ndarray]]
) -> tuple[list[str], dict[str, int]]:
    """Blinded replies whose reason claims to know an outcome: all, and within the not-named."""
    rows, counts = [], {}
    for model in MODELS:
        name = combo(model, "blinded")
        claims = np.array([claims_an_outcome(r) for r in wide[f"{name} reason"]])
        counts[model] = int(claims.sum())
        row = [model.title(), f"{claims.sum()} of {len(claims)}"]
        for definition in DEFINITIONS:
            not_named = ~named[definition][name]
            row.append(f"{(claims & not_named).sum()} of {not_named.sum()}")
        rows.append(row)
    header = ["model", "blinded replies claiming an outcome"] + [
        f"among the not-named ({d})" for d in DEFINITIONS
    ]
    return table(header, rows), counts


# ---------------------------------------------------------------- the reading

POWER = [
    ("one AUROC, 300 trials", "about plus or minus 0.05"),
    ("identified minus masked", "can detect about 0.03"),
    ("identified minus blinded, or masked minus blinded", "can detect about 0.06 to 0.08"),
    ("blinded minus registration", "can detect about 0.08 to 0.10"),
    ("a subset of 100 to 200 trials against a baseline", "about 0.10 or more"),
    ("share recognised, by completion year", "only differences of 15 to 20 points"),
]


def power_section() -> list[str]:
    return table(
        ["comparison on the 300 development trials", "what the sample resolves"],
        [list(r) for r in POWER],
    )


# What a reviewer found by reading the blinded replies by hand, with a language model's help (a
# model-assisted read, not a computed figure). Quoted as the yardstick for the stricter rule.
REVIEWER = {
    "sonnet": {
        "misjudged": 28, "of": 159, "trials": 131,
        "levels": "0.807 against 0.690", "edge": "+0.117 (+0.034 to +0.204)",
        "ladder": "about +0.057 (−0.019 to +0.135)",
    },
    "haiku": {
        "misjudged": 12, "of": 209, "trials": 197,
        "levels": None, "edge": "+0.049 (−0.032 to +0.128)",
        "ladder": "about −0.014",
    },
}  # fmt: skip


def hand_read_section(subsets: dict[tuple[str, str, str], Subset]) -> list[str]:
    """What the drug-guess match gets wrong, by a hand read, and how near the stricter rule gets."""
    lines = [
        (
            "**The automatic match misses about one in six.** A reviewer read every blinded reply "
            "it calls not named, with a language model's help (a model-assisted hand read, not a "
            "computed figure). Of the 159 replies of Sonnet's that the match left (it now leaves 165, "
            "after the tightening described above), 28 did name the drug or trial: 16 by a synonym, "
            "brand or company code that is not in the alias list (nirsevimab for MEDI8897, "
            "erenumab for AMG334, melatonin for Circadin) and 12 in the reason text instead of "
            "the guess field. Of Haiku's 209, 12 did. So the not-named subset still contains trials "
            "the model identified. With those moved to the named group the reviewer found, "
            f"against registration: Sonnet {REVIEWER['sonnet']['levels']}, "
            f"{REVIEWER['sonnet']['edge']}, on {REVIEWER['sonnet']['trials']} trials; Haiku "
            f"{REVIEWER['haiku']['edge']} on {REVIEWER['haiku']['trials']} trials. Against the "
            f"ladder: Sonnet {REVIEWER['sonnet']['ladder']}, Haiku {REVIEWER['haiku']['ladder']}."
        ),
        "",
        "The stricter rule recomputes this, without the hand read:",
        "",
    ]
    for model in MODELS:
        guess = subsets.get((GUESS, model, NOT_NAMED))
        strict = subsets.get((STRICTER, model, NOT_NAMED))
        if guess is None or strict is None:
            continue
        said = REVIEWER[model]
        line = (
            f"- {model.title()}: it moves {guess.n - strict.n} trials to named (the reviewer: "
            f"{said['misjudged']}), leaving {strict.n} not named (the reviewer: {said['trials']})."
        )
        if strict.difference is not None:
            line += (
                f" Its edge over registration is {strict.difference.show(signed=True)} "
                f"(the reviewer: {said['edge']})."
            )
        lines.append(line)
    lines += [
        "",
        (
            "So the rule gets the edge about right but finds only about half of the trials the "
            "reviewer found. It catches an alias or a trial acronym in the reason; it cannot "
            "catch a synonym written in plain words."
        ),
    ]
    return lines


CONFIDENT_CALLS = (
    "What the confident, correct calls on not-named trials rest on was read by a reviewer, with a "
    "language model's help, in 21 of Sonnet's replies (a model-assisted read of 21, not a "
    "computed figure). About half rest on the drug class plus an easy endpoint against placebo "
    "(blood sugar, cholesterol, weight, haemoglobin, lung function) or on the design. About half "
    "rest on a remembered result of the same or a similar programme."
)

REVIEWER_FOUR_CELL = (
    "The reviewer's reading of Sonnet, from hand-corrected labels: the advantage does not "
    "concentrate in recognised trials. It is present everywhere and certain nowhere. In trials "
    "not named and not recognised Sonnet is +0.122 (+0.019 to +0.227) against registration on 100 "
    "trials, but about +0.040 (−0.053 to +0.133) against the ladder. Here the cells are small, "
    "and the gaps that clear zero against both baselines sit in cells of a few dozen trials."
)


def four_cell_reading(cells: dict[tuple[str, bool, bool], Subset]) -> list[str]:
    """In how many scored cells each model's gap over each baseline is above zero, and clearly so."""
    lines = []
    for model in MODELS:
        scored = [r for (m, _, _), r in cells.items() if m == model and r.blinded is not None]
        parts = []
        for label, pick in (
            ("registration", lambda r: r.difference),
            ("the ladder", lambda r: r.ladder_difference),
        ):
            gaps = [pick(r) for r in scored if pick(r) is not None]
            above = sum(g.value > 0 for g in gaps)
            clear = sum(reading(g) == "clearly above zero" for g in gaps)
            parts.append(
                f"against {label} above zero in {above} of {len(gaps)}, clearly in {clear}"
            )
        lines.append(f"- {model.title()}: the gap is " + "; ".join(parts) + ".")
    both = [
        f"{m.title()}, {NAMED if was_named else NOT_NAMED}, said {'yes' if knew else 'no'} ({r.n} trials)"
        for (m, was_named, knew), r in cells.items()
        if r.difference is not None
        and r.ladder_difference is not None
        and reading(r.difference) == reading(r.ladder_difference) == "clearly above zero"
    ]
    lines.append(
        "- Clearly above zero against both baselines in: " + ("; ".join(both) or "no cell") + "."
    )
    return lines


TOP_BIN = 0.8


def calibration_section(
    outcomes: np.ndarray, arrays: dict[str, np.ndarray], names: list[str], earlier_rate: float
) -> list[str]:
    """Calibration, said plainly: where the baselines' optimism comes from, and the top of the scale."""
    rows = []
    for name in names:
        top = arrays[name] >= TOP_BIN
        n = int(top.sum())
        observed = f"{outcomes[top].mean():.0%}" if n else "n/a"
        mean = f"{arrays[name][top].mean():.0%}" if n else "n/a"
        rows.append([name, str(n), mean, observed])
    lines = []
    if not np.isnan(earlier_rate):
        lines += [
            (
                "The baselines' over-confidence is mostly the fall in the success rate between the "
                f"years they learn from and these: trials finishing before {BASELINE_TRAIN_BEFORE} "
                f"succeeded {earlier_rate:.0%} of the time, these trials "
                f"{outcomes.mean():.0%}."
            ),
            "",
        ]
    lines += [
        (
            "Given the names, the language models are under-confident at the top of the scale "
            "(table below); blinded, they are about right. Calibration says nothing about recall "
            "versus reasoning."
        ),
        "",
    ]
    header = [
        "model",
        f"trials given {TOP_BIN:.0%} or more",
        "mean probability given",
        "share that succeeded",
    ]
    return [*lines, *table(header, rows)]


def share(value: float) -> str:
    return f"{value:.0%}"


def gap_text(r: Subset) -> str:
    """A model's gap over each baseline on its not-named trials, as the number and its interval."""
    if r.difference is None:
        return f"{r.n} trials, too few to score"
    text = f"{r.difference.show(signed=True)} over registration"
    if r.ladder_difference is not None:
        text += f" and {r.ladder_difference.show(signed=True)} over the ladder"
    return text


def edge_line(model: str, subsets: dict[tuple[str, str, str], Subset]) -> str:
    """One model's not-named gap under each way of counting "named"."""
    guess, stricter = (subsets[(d, model, NOT_NAMED)] for d in DEFINITIONS)
    said = REVIEWER[model]
    return (
        f"   - {model.title()}. Guess alone, {guess.n} trials: {gap_text(guess)}. Stricter rule, "
        f"{stricter.n} trials: {gap_text(stricter)}. Reviewer's hand-corrected labels, "
        f"{said['trials']} trials: {said['edge']} over registration and {said['ladder']} over the "
        "ladder."
    )


def what_this_shows(
    n: int,
    single,
    shares: dict[str, dict[str, float]] | None,
    claims: dict[str, int] | None,
    subsets: dict[tuple[str, str, str], Subset],
    cells: dict[tuple[str, bool, bool], Subset],
) -> list[str]:
    """The four things this study can say, in plain English with the numbers."""
    auroc = {name: single[name]["auroc"].value for name in single}
    lines = [
        (
            f"1. **The scores.** On {n} development trials finished 2017 to 2020, Sonnet reading "
            f"the registration record scored AUROC {auroc[combo('sonnet', 'identified')]:.2f} and "
            f"Haiku {auroc[combo('haiku', 'identified')]:.2f}, against "
            f"{auroc[REGISTRATION_NAME]:.2f} for the registration baseline"
            + (
                f" and {auroc[LADDER_NAME]:.2f} for the best model built on registration details "
                "and wording."
                if LADDER_NAME in auroc
                else "."
            )
        ),
    ]
    if shares is None or claims is None:
        return [*lines, "2. and 3. Not run for this report (the raw records were not given)."]
    scores = [
        f"{m.title()} {cells[(m, True, True)].blinded.value:.2f} on {cells[(m, True, True)].n} trials"
        for m in sorted(MODELS, reverse=True)
        if cells[(m, True, True)].blinded is not None
    ]
    lines.append(
        "2. **Most of that cannot be separated from recall.** With drug names removed, Sonnet "
        f"still named the drug in {share(shares[GUESS]['sonnet'])} of trials by its guess "
        f"({share(shares[STRICTER]['sonnet'])} by the stricter rule) and Haiku in "
        f"{share(shares[GUESS]['haiku'])} ({share(shares[STRICTER]['haiku'])}). Sonnet's "
        f"stated reasons often cite a named trial and its result: {claims['sonnet']} of "
        f"{n} blinded replies claim knowledge of an outcome. On trials a model both named and "
        f"said it recognised, it scored best of any group: {' and '.join(scores)}."
    )
    lines += [
        (
            "3. **Without the name, a smaller edge.** On trials it did not name, with names "
            "removed, Sonnet still ranked outcomes better than the registration model but not "
            "clearly better than the registration-plus-words model (the ladder); Haiku showed no "
            'clear edge. The split is exploratory, and "not named" is counted three ways. Gaps in '
            "AUROC (Sonnet's interval against the ladder only just clears zero under the stricter "
            "rule):"
        ),
        "",
    ]
    lines += [edge_line(model, subsets) for model in sorted(MODELS, reverse=True)]
    lines += [
        "",
        (
            "   This mixes design and drug-class knowledge with whatever recall survives "
            "blinding. It is not evidence of reasoning from design."
        ),
        "",
        (
            "4. **What would settle it.** A claim that a language model forecasts trial outcomes "
            "needs trials that finished after its training cut-off. None of the three periods "
            "here qualifies (both models report a cut-off of June 2026)."
        ),
    ]
    return lines


LIMITS = [
    (
        "- **Blinding lowers recall; it does not remove it.** A drug-class word in an outcome, an "
        "unusual endpoint, an abbreviation or a named scale can still point to the drug, and a "
        "model can recognise a programme from its conditions, design and endpoint with no drug "
        "name at all. Evidence is in the leak check section."
    ),
    (
        "- **Blinding also removes legitimate knowledge.** A forecaster at the start of a trial "
        "knows the drug's class, its earlier results and who the comparator is. So a gap between "
        "prompts is recall plus that knowledge, and no gap here is a measure of recall alone."
    ),
    (
        "- **The not-named split is exploratory.** The model's own output chooses it. It is "
        "harder for the baseline, and not free of recall: the automatic drug-guess match "
        "misjudges about one in six of the replies it calls not named (see the hand read below)."
    ),
    (
        "- **Known blinding gaps, unfixed.** The stored predictions were made from prompts built "
        "by the blinding rules as they stood, so the rules are frozen (a test re-renders a sample "
        "of stored prompts and compares their hashes). Gaps left in: "
        + "; ".join(KNOWN_GAPS)
        + ". Counts are of the 300 sampled development trials."
    ),
    (
        "- **No clean period.** All three periods (2017-2020, 2021-2023, 2024-2025) are inside "
        "these models' training data (each reports a knowledge cut-off of June 2026). A fall in "
        "score from the early to the late period reflects how much had been published before the "
        "cut-off, mixed with changes in trial mix. The late period is the least contaminated, not "
        "clean. Nothing here is an out-of-sample estimate of forecasting skill."
    ),
    (
        "- **The model is told the date and its own cut-off.** The command line adds an "
        "environment block with today's date and the model's knowledge cut-off, and a block with the "
        "account's email. They are the same on every call, so they do not affect comparisons, but "
        "they are not nothing (see `llm_predictor`)."
    ),
    (
        "- **Post-hoc text.** Sponsors rewrite outcome time frames when they post results (cut-off "
        "dates, achieved follow-up, termination). The blinded prompt cuts the clauses a rule "
        "finds; masked and identified prompts keep them. The scores without flagged trials are "
        "below. A rule cannot find every rewrite."
    ),
    (
        "- **Self-reported recognition is weak evidence.** A model saying it recognises a trial is "
        "not proof of memory, and one saying it does not is not proof of its absence. A correct drug "
        "guess is stronger, but a design can make a drug easy to guess without any memory of the trial."
    ),
    (
        "- **Not repeatable to the digit.** The command line has no temperature setting, so a repeat "
        "of the same batch gives somewhat different answers (see run-to-run below)."
    ),
    (
        "- **Batches of ten, frozen.** Each call holds ten trials, so a model sees neighbours. The "
        "batches and the order in them were fixed once, are mixed across the three periods, and are "
        "the same for every model and variant."
    ),
    (
        "- **Arm descriptions.** The protocol record's arm text sometimes states a planned number of "
        "participants. That is a plan, not the final enrolment, which is never shown."
    ),
    "- **Intervals** resample trials as if independent; trials sharing a drug or sponsor are not.",
]


MIN_FOR_RESULT = 100


def small_sample_warning(n: int) -> list[str]:
    """A line that says the numbers are a plumbing check, if too few trials were scored."""
    if n >= MIN_FOR_RESULT:
        return []
    return [
        (
            f"**Only {n} trials have a prediction from every model and variant. This is a check "
            "that the pipeline runs, not a result: no interval means anything at this size.**"
        ),
        "",
    ]


def build_report(
    dev: pl.DataFrame,
    sample: pl.DataFrame,
    predictions: pl.DataFrame,
    baselines: dict[str, dict[str, float]],
    leaks: dict[str, int] | None = None,
    records: dict[str, dict] | None = None,
) -> str:
    """The report text. `dev` must be development trials only; anything else raises.

    `leaks` is the leak check (`llm_predictor.leak_counts`) over the sampled development trials.
    `records` are their `extract_record` records; without them the drug-guess sections are skipped.
    """
    trials = sampled_development(dev, sample)
    ids = sorted(trials["nct_id"].to_list())
    wide = wide_predictions(predictions, ids).sort("nct_id")
    kept = wide["nct_id"].to_list()
    if not kept:
        raise ValueError(
            "No sampled development trial has a prediction from every model and variant."
        )
    scored = trials.filter(pl.col("nct_id").is_in(kept)).sort("nct_id")
    if scored["nct_id"].to_list() != kept:
        raise ValueError("The scored trials and their predictions are not the same trials.")
    outcomes = scored["any_arm"].to_numpy().astype(bool)
    years = scored["primary_completion_date"].dt.year().to_numpy()
    arrays = {n: wide[n].to_numpy() for n in COMBOS}
    for name, by_id in baselines.items():
        arrays[name] = np.array([by_id[i] for i in kept])
    refs = [REGISTRATION_NAME, LADDER_NAME]
    pairs = [(combo(m, a), combo(m, b)) for m in MODELS for a, b in CONTRASTS]
    pairs += [(c, r) for c in COMBOS for r in refs if r in arrays]
    named = named_arrays(wide, kept, records) if records is not None else None
    single, paired = paired_bootstrap(outcomes, arrays, pairs)
    subsets = not_named_results(outcomes, arrays, named)
    shown = COMBOS + [r for r in refs if r in arrays]

    flag_of = dict(zip(sample["nct_id"], sample["post_hoc"], strict=True))
    clean = np.array([not flag_of[i] for i in kept])
    contrast_pairs = [(combo(m, a), combo(m, b)) for m in MODELS for a, b in CONTRASTS[:2]]
    if can_score(outcomes[clean]):
        clean_single, clean_paired = paired_bootstrap(
            outcomes[clean], {n: arrays[n][clean] for n in shown}, contrast_pairs
        )
        sensitivity = [
            *headline_section(clean_single, shown),
            "",
            *difference_section(clean_paired, contrast_pairs),
        ]
    else:
        sensitivity = ["Too few trials without a post-hoc phrase to score."]

    blinded = [combo(m, "blinded") for m in MODELS]
    recognised = {n: wide[f"{n} recognised"].to_numpy().astype(bool) for n in blinded}
    cells = shares = claims = None
    four_cell, neither, profile = [], [], []
    if named is not None:
        cells = four_cell_results(outcomes, arrays, named[STRICTER], recognised)
        shares = {
            d: {m: float(named[d][combo(m, "blinded")].mean()) for m in MODELS} for d in DEFINITIONS
        }
        claim_rows, claims = claims_section(wide, named)
        profile = profile_section(scored, named[STRICTER], subsets)
        neither = neither_section(neither_named_results(outcomes, arrays, named[STRICTER]))
        four_cell = four_cell_section(cells)
    naming = None
    if named is not None:
        naming = {n: float(named[GUESS][n].mean()) for n in blinded}
    earlier = dev.filter(pl.col("primary_completion_date").dt.year() < BASELINE_TRAIN_BEFORE)
    earlier_rate = float(earlier["any_arm"].mean()) if earlier.height else float("nan")
    lines = [
        "# A language model as predictor: how much is design, how much is recall?",
        "",
        "Development trials only (finishing 2017 to 2020). No test or late trial is scored here.",
        "",
        *small_sample_warning(len(kept)),
        "## What this shows",
        "",
        *what_this_shows(len(kept), single, shares, claims, subsets, cells or {}),
        "",
        "## Limits",
        "",
        *LIMITS,
        "",
        "## What this sample can resolve",
        "",
        *power_section(),
        "",
        f"## Scores on the same {len(kept)} sampled trials",
        "",
        (
            f"{len(ids) - len(kept)} of {len(ids)} sampled trials are left out for lack of a "
            "successful prediction from every model and variant. Success rate in these trials: "
            f"{outcomes.mean():.1%}. The two baselines are trained on development trials finishing "
            f"before {BASELINE_TRAIN_BEFORE}. Calibration gap: mean predicted minus observed, in "
            "points; positive is too optimistic."
        ),
        "",
        *headline_section(single, shown),
        "",
        "### Calibration",
        "",
        *calibration_section(outcomes, arrays, shown, earlier_rate),
        "",
        "## Paired differences",
        "",
        (
            "Within the same resamples. The first rows are the contrasts between prompts. Identified "
            "minus blinded is recall plus the legitimate drug knowledge a forecaster has at the "
            "start (the drug's class, its earlier results, who the comparator is), not a floor on "
            "recall. Masked minus blinded is drug knowledge, known before the start and recalled "
            "afterwards, together. Identified minus masked is what the id, title, sponsor name, "
            "acronym and study ids add. The rest are against the two baselines: the blinded prompt, "
            "in which recall is reduced, not removed."
        ),
        "",
        *difference_section(paired, pairs),
        "",
        "## The blinded prompt on trials the model could not name (exploratory)",
        "",
        (
            "Each model's blinded reply is sorted by whether it named the tested drug, under two "
            f'definitions. "{GUESS}": its `drug_guess` matches a name, other name or company code '
            "of the tested intervention (case aside; a guess that is only a class word such as "
            '"monoclonal antibody" does not count, nor does a drug that a control arm takes '
            f'too). "{STRICTER}": that, or the `reason` text holds such a name, or it claims a '
            "result and names a trial the prompt does not hold. "
            f'"{NOT_NAMED}" includes a null guess. Each row is paired with both baselines on the '
            "same trials. This is an exploratory split, chosen by the model's own output; the "
            "intervals are wide because the subsets are small."
        ),
        "",
        *not_named_section(subsets),
        "",
        *(hand_read_section(subsets) if subsets else []),
        "",
        "### By whether the model also said it recognised the trial (stricter definition)",
        "",
        *four_cell,
        "",
        *(four_cell_reading(cells) if cells else []),
        "",
        REVIEWER_FOUR_CELL,
        "",
        "### Trials neither model named (stricter definition)",
        "",
        *neither,
        "",
        "### How the not-named trials differ from the named ones (stricter definition)",
        "",
        *profile,
        "",
        (
            "The subset is chosen by the model's own output, is harder for the baseline, and is "
            "not free of recall."
        ),
        "",
        "### Replies that claim to know an outcome",
        "",
        (
            "A rule-based phrase list on the blinded reply's reason: "
            "recall, met its, did not meet, was positive, was negative, failed, reported, "
            "published (and a few close forms)."
        ),
        "",
        *(claim_rows if named is not None else []),
        "",
        CONFIDENT_CALLS,
        "",
        "## Without the trials with a post-hoc phrase",
        "",
        (
            f"{int((~clean).sum())} of {len(kept)} scored trials have a post-hoc phrase in arm or "
            "outcome text (cut-off dates, achieved follow-up, termination). The same scores "
            "without them:"
        ),
        "",
        *sensitivity,
        "",
        "Trials flagged, by period (all sampled trials):",
        "",
        *post_hoc_section(sample),
        "",
        "## By whether the model said it recognised the trial",
        "",
        (
            'The model\'s own answer to "do you know this specific trial or its result?", prompt '
            "variant as in the row. Many models say yes to most trials, which makes the yes group "
            "most of the sample."
        ),
        "",
        *recognition_section(outcomes, wide.sort("nct_id")),
        "",
        "## How often the model said it recognised the trial, and named the drug",
        "",
        *recognised_rate_section(wide, named[GUESS] if named is not None else None),
        "",
        "## Share recognised, by completion year",
        "",
        *share_by_year_section(wide, years),
        "",
        "## Leak check: what blinding leaves behind",
        "",
        *leak_section(leaks, naming),
        "",
        "## Run-to-run variation",
        "",
        *run_to_run_section(predictions, ids),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from trial_benchmark.validate import load_development_with

    dev_with_text = load_development_with({"text"})
    sample = read_sample()
    trials = sampled_development(dev_with_text, sample)
    baselines = fit_baselines(dev_with_text, trials["nct_id"].to_list())
    records = load_records(trials["nct_id"])
    leaks = leak_counts(records.values())
    report = build_report(
        dev_with_text, sample, read_predictions(PREDICTIONS_PATH), baselines, leaks, records
    )
    REPORT_PATH.write_text(report)
    print(f"wrote {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
