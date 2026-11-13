"""Write results/dev_whathow.md: how much of a trial's chance is WHAT it tries, and how much HOW it is run?

    uv run python -m elman_exercise.whathow           # build the tables first (once)
    uv run python -m elman_exercise.whathow_report    # -> results/dev_whathow.md

Development trials only, through the guarded loader in `validate`. Scoring is the repo's own: three
time-ordered rounds and grouped 10-fold by lead drug (`validate.run_time_ordered`,
`validate.run_grouped`). The numbers shown are the same mean-within-round (or fold) AUROC, with
the shared lead-drug bootstrap (`validate.mean_auroc_bootstrap(clusters=...)`: each resample draws
whole lead drugs once for all rounds) and paired differences taken on the same resamples.

The other statistics (the correlation of HOW and WHAT, the share of WHAT's gap that survives, the
success by thirds) use the same draws of lead drugs, as weights on the trials (`cluster_weights`),
with thirds and strata fixed from the out-of-sample predictions and cut inside each round or fold.

The analysis functions below take plain arrays (outcomes, predictions, which round or fold, phase,
sponsor class, therapeutic area, lead drug), not a development-only frame. To score the test set
later, the owner predicts it through the repo's guarded test-set entry point (`heldout.open_heldout`
opens the file; `validate.predict_all` is only the function that fits and predicts, it guards
nothing) with the models of `models.whathow`, trained on the development trials, and passes the
arrays to the same functions. This module never touches the test set.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from elman_exercise import metrics, validate, whathow
from elman_exercise.models import linear
from elman_exercise.models import precedent as precedent_models
from elman_exercise.models import whathow as models
from elman_exercise.models.registry import ModelSpec
from elman_exercise.science import links

REPORT_PATH = validate.ROOT / "results" / "dev_whathow.md"
REGISTRATION = linear.REGISTRATION_NAME
CLUSTER_RESAMPLES = metrics.N_RESAMPLES
LABEL = {"HOW": "HOW", "WHAT": "WHAT", "WHAT + phase": "WHAT + phase", "HOW + WHAT": "HOW + WHAT"}
PHASES = ("phase2", "phase3")
LINK_TITLE = {
    "today": "Today's links (Open Targets 26.09)",
    "2020": "2020 links (Open Targets 20.06 targets, today's drug match)",
    "2020_names": "2020 names and links (drug matched by 20.06 names, 20.06 targets)",
}


def strict_precedent_name(version: str) -> str:
    """The registration logistic plus the strict precedent flag of `version` (WHAT's own flag)."""
    return precedent_models.spec_for(f"precedent_exp_{version}", True, version).name


# ---------------------------------------------------------------- intervals

Triple = tuple[float, float, float]


def clustered_scores(
    outcomes: np.ndarray,
    predictions: dict[str, np.ndarray],
    parts: np.ndarray,
    clusters: np.ndarray,
    pairs: list[tuple[str, str]],
    n_resamples: int = CLUSTER_RESAMPLES,
) -> tuple[dict[str, Triple], dict[tuple[str, str], Triple]]:
    """Mean within-part AUROC per model and (better - baseline) per pair, drug-clustered intervals.

    A thin adapter over `validate.mean_auroc_bootstrap(clusters=...)`, the repo's one clustered
    bootstrap. That function keys a difference by its better model only, so pairs are run in
    groups with one baseline each; every call uses the same seed and so the same draws of drugs.
    Returns ({model: (value, low, high)}, {(better, baseline): (value, low, high)}).
    """
    by_baseline: dict[str, list[tuple[str, str]]] = {}
    for pair in pairs:
        by_baseline.setdefault(pair[1], []).append(pair)
    groups = list(by_baseline.values()) or [[]]
    scores: dict[str, Triple] = {}
    differences: dict[tuple[str, str], Triple] = {}
    for group in groups:
        means, diffs = validate.mean_auroc_bootstrap(
            outcomes, predictions, parts, group, n_resamples=n_resamples, clusters=clusters
        )
        for name, mean in means.items():
            scores.setdefault(name, (mean.value, mean.interval.low, mean.interval.high))
        for better, baseline in group:
            d = diffs[better]
            differences[(better, baseline)] = (d.value, d.interval.low, d.interval.high)
    return scores, differences


def cluster_weights(
    clusters: np.ndarray, n_resamples: int = CLUSTER_RESAMPLES, seed: int = metrics.BOOTSTRAP_SEED
) -> np.ndarray:
    """(n_resamples, n_trials): how many times each trial is drawn when whole clusters are drawn.

    Each resample draws as many clusters (lead drugs) as there are, with replacement, ONCE for all
    rounds or folds; every trial of a drawn cluster gets the cluster's count. A trial with no known
    cluster is its own (`metrics.cluster_codes`). The same scheme as
    `validate.mean_auroc_bootstrap(clusters=...)`, as weights, so it works for any statistic.
    """
    clusters = np.asarray(clusters, dtype=object)
    codes = metrics.cluster_codes(clusters, len(clusters))
    k = int(codes.max()) + 1 if len(codes) else 0
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, len(codes)), dtype=np.int32)
    for i in range(n_resamples):
        times = np.bincount(rng.integers(0, k, size=k), minlength=k)
        out[i] = times[codes]
    return out


def percentile_pair(values: np.ndarray) -> tuple[float, float]:
    found = validate.percentile_interval(np.asarray(values, dtype=float))
    return found.low, found.high


# ---------------------------------------------------------------- the questions


def log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def weighted_correlation(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Pearson correlation of a and b with trial weights `w` (NaN if either has no spread)."""
    total = w.sum()
    if total <= 0:
        return float("nan")
    ma, mb = (w * a).sum() / total, (w * b).sum() / total
    cov = (w * (a - ma) * (b - mb)).sum()
    spread = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return float(cov / spread) if spread > 0 else float("nan")


def correlation(
    how: np.ndarray, what: np.ndarray, parts: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Correlation of the two models' out-of-sample log-odds: pooled, and mean within part."""
    a, b = log_odds(how), log_odds(what)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    within = [
        weighted_correlation(a[parts == p], b[parts == p], w[parts == p]) for p in np.unique(parts)
    ]
    return weighted_correlation(a, b, w), float(np.nanmean(within))


def correlation_interval(
    how: np.ndarray,
    what: np.ndarray,
    parts: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = CLUSTER_RESAMPLES,
    within: bool = False,
    weights: np.ndarray | None = None,
) -> Triple:
    """(value, low, high) of the pooled correlation (or, with `within`, the mean within part),
    with an interval that resamples whole lead drugs."""
    if weights is None:
        weights = cluster_weights(clusters, n_resamples)
    which = 1 if within else 0
    draws = np.array([correlation(how, what, parts, w)[which] for w in weights])
    point = correlation(how, what, parts)[which]
    return (point, *percentile_pair(draws))


def thirds_within(values: np.ndarray, *groups: np.ndarray) -> np.ndarray:
    """0, 1, 2 for the bottom, middle, top third of `values` inside every combination of `groups`."""
    out = np.zeros(len(values), dtype=int)
    keys = np.stack([np.asarray(g).astype(str) for g in groups], axis=1)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    labels = labels.ravel()
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        out[rows] = whathow.thirds(values[rows])
    return out


@dataclass(frozen=True)
class Survival:
    top_minus_bottom: float  # raw gap in success per 100, WHAT's top third less its bottom third
    within_strata: float  # the same gap inside strata, per 100
    share: float  # within_strata / top_minus_bottom
    n_strata: int  # strata that held trials in both WHAT's top and bottom third
    n_strata_total: int  # every stratum with any trial


def stratum_codes(
    how3: np.ndarray, parts: np.ndarray, strata: tuple[np.ndarray, ...]
) -> np.ndarray:
    """One integer per trial for its stratum: round or fold x each `strata` array x HOW third."""
    labels = np.stack(
        [parts.astype(str), *(np.asarray(s).astype(str) for s in strata), how3.astype(str)], axis=1
    )
    return np.unique(labels, axis=0, return_inverse=True)[1].ravel()


def survival_numbers(
    y: np.ndarray, what3: np.ndarray, codes: np.ndarray, w: np.ndarray
) -> tuple[float, float, int]:
    """(raw gap, within-strata gap, strata with both tails), per trial, with trial weights `w`.

    The raw gap is WHAT's top third's success rate less its bottom third's, pooled over rounds. The
    within-strata gap is the same gap in every stratum with both tails, averaged with weight
    n_top * n_bottom / (n_top + n_bottom) (Mantel-Haenszel).
    """
    top, bottom = what3 == 2, what3 == 0
    raw = (w * y)[top].sum() / w[top].sum() - (w * y)[bottom].sum() / w[bottom].sum()
    size = int(codes.max()) + 1
    n_t, n_b = np.bincount(codes, w * top, size), np.bincount(codes, w * bottom, size)
    y_t, y_b = np.bincount(codes, w * y * top, size), np.bincount(codes, w * y * bottom, size)
    both = (n_t > 0) & (n_b > 0)
    if not both.any():
        return float(raw), float("nan"), 0
    gap = y_t[both] / n_t[both] - y_b[both] / n_b[both]
    weight = n_t[both] * n_b[both] / (n_t[both] + n_b[both])
    return float(raw), float(np.average(gap, weights=weight)), int(both.sum())


def survival_setup(
    how: np.ndarray, what: np.ndarray, parts: np.ndarray, strata: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """(WHAT third, stratum code) per trial. Thirds are cut within each round or fold."""
    what3 = thirds_within(what, parts)
    how3 = thirds_within(how, parts)
    return what3, stratum_codes(how3, parts, strata)


def survival(
    outcomes: np.ndarray,
    how: np.ndarray,
    what: np.ndarray,
    parts: np.ndarray,
    *strata: np.ndarray,
    weights: np.ndarray | None = None,
) -> Survival:
    """How much of WHAT's top-versus-bottom-third success gap is left among alike-HOW trials.

    Thirds of WHAT and of HOW are cut within each round (or fold), never pooled across rounds
    scored by differently calibrated models. Strata are round or fold x each array in `strata`
    (for example phase, sponsor class) x HOW third. `share` is the within-strata gap over the raw
    gap, so 1.0 would mean HOW explains none of it. Ties are broken by average rank (`whathow.thirds`).
    `weights` (one per trial, default 1) lets a bootstrap resample reuse the same thirds.
    """
    y = np.asarray(outcomes, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    what3, codes = survival_setup(how, what, parts, strata)
    raw, adjusted, n_both = survival_numbers(y, what3, codes, w)
    share = adjusted / raw if raw else float("nan")
    return Survival(100 * raw, 100 * adjusted, share, n_both, int(codes.max()) + 1)


def survival_draws(
    outcomes: np.ndarray,
    how: np.ndarray,
    what: np.ndarray,
    parts: np.ndarray,
    clusters: np.ndarray,
    *strata: np.ndarray,
    n_resamples: int = CLUSTER_RESAMPLES,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(shares, within-strata gaps per 100) over lead-drug resamples; thirds and strata fixed."""
    y = np.asarray(outcomes, dtype=float)
    what3, codes = survival_setup(how, what, parts, strata)
    if weights is None:
        weights = cluster_weights(clusters, n_resamples)
    shares, gaps = np.empty(len(weights)), np.empty(len(weights))
    for i, w in enumerate(weights):
        raw, adjusted, _ = survival_numbers(y, what3, codes, w.astype(float))
        shares[i], gaps[i] = (adjusted / raw if raw else np.nan), 100 * adjusted
    return shares, gaps


def survival_interval(
    outcomes: np.ndarray,
    how: np.ndarray,
    what: np.ndarray,
    parts: np.ndarray,
    clusters: np.ndarray,
    *strata: np.ndarray,
    n_resamples: int = CLUSTER_RESAMPLES,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """95% interval of `Survival.share`, resampling whole lead drugs."""
    shares, _ = survival_draws(
        outcomes, how, what, parts, clusters, *strata, n_resamples=n_resamples, weights=weights
    )
    return percentile_pair(shares)


def success_by_thirds(
    outcomes: np.ndarray, what: np.ndarray, parts: np.ndarray, phase: np.ndarray
) -> dict[str, tuple[list[float], list[int]]]:
    """Per phase: success per 100 in WHAT's bottom, middle and top third, and the trial counts.

    Thirds are cut within each round (or fold) and phase, so a trial is compared with others of the
    same phase scored by the same fitted model.
    """
    y = np.asarray(outcomes, dtype=float)
    third = thirds_within(what, parts, phase)
    result = {}
    for name in PHASES:
        rows = phase == name
        counts = [int((rows & (third == t)).sum()) for t in range(3)]
        rates = [100 * float(y[rows & (third == t)].mean()) if c else float("nan")
                 for t, c in zip(range(3), counts, strict=True)]  # fmt: skip
        result[name] = (rates, counts)
    return result


def thirds_intervals(
    outcomes: np.ndarray,
    what: np.ndarray,
    parts: np.ndarray,
    phase: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = CLUSTER_RESAMPLES,
    weights: np.ndarray | None = None,
) -> dict[str, tuple[list[float], list[tuple[float, float]], tuple[float, float]]]:
    """Per phase: the success rate per 100 in each third, a lead-drug interval for each, and an
    interval for top minus bottom. Thirds are those of `success_by_thirds`, held fixed."""
    y = np.asarray(outcomes, dtype=float)
    third = thirds_within(what, parts, phase)
    if weights is None:
        weights = cluster_weights(clusters, n_resamples)
    w_all = weights.astype(float)
    result = {}
    for name in PHASES:
        cells = [(phase == name) & (third == t) for t in range(3)]
        rates = [100 * float(y[c].mean()) if c.any() else float("nan") for c in cells]
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = [100 * (w_all[:, c] @ y[c]) / w_all[:, c].sum(axis=1) for c in cells]
        intervals = [percentile_pair(d) for d in draws]
        gap = percentile_pair(draws[2] - draws[0])
        result[name] = (rates, intervals, gap)
    return result


# ---------------------------------------------------------------- running the models


@dataclass(frozen=True)
class Scored:
    """Out-of-sample predictions on one scoring scheme, with what the analyses need per trial."""

    outcomes: np.ndarray
    predictions: dict[str, np.ndarray]
    parts: np.ndarray  # round or fold of each scored trial
    phase: np.ndarray
    sponsor_class: np.ndarray
    therapeutic_area: np.ndarray
    clusters: np.ndarray  # lead drug, or the trial itself when unknown
    unit: str = "round"  # what `parts` are called in the text: "round" or "fold"

    def named(self, version: str, block: str, exclude: bool = False) -> np.ndarray:
        return self.predictions[models.spec_name(block, version, "lead" if exclude else None)]


def run_schemes(frame: pl.DataFrame, specs: list[ModelSpec]) -> dict[str, Scored]:
    """Fit and score `specs` in the time-ordered rounds and in the grouped folds."""
    groups = validate.drug_groups(frame)
    rows = np.concatenate([validate.time_round(frame, a, b)[1] for a, b in validate.ROUNDS])
    timed = validate.run_time_ordered(specs, frame)
    folded = validate.run_grouped(specs, frame)
    everyone = np.arange(frame.height)
    area = frame["therapeutic_area"].cast(pl.String).fill_null("(missing)").to_numpy()

    def scored(run, row_of: np.ndarray, parts: np.ndarray, unit: str) -> Scored:
        return Scored(
            run.outcomes,
            run.predictions,
            parts,
            frame["phase"].to_numpy()[row_of],
            frame["sponsor_class"].to_numpy()[row_of],
            area[row_of],
            groups[row_of],
            unit,
        )

    return {
        "rounds": scored(timed, rows, timed.round_of, "round"),
        "folds": scored(folded, everyone, folded.fold_of, "fold"),
    }


def version_specs(version: str) -> list[ModelSpec]:
    """The four models, the registration baseline, the registration + strict precedent flag (the
    flag WHAT itself uses), and the two own-drug-excluded WHAT models."""
    return [
        *models.make_specs(version),
        *models.make_specs(version, exclude="lead"),
        next(s for s in linear.SPECS if s.name == REGISTRATION),
        precedent_models.spec_for(f"precedent_exp_{version}", True, version),
    ]


def load_frame(version: str) -> pl.DataFrame:
    return validate.load_development_with({whathow.table_name(version), "precedent"})


# ---------------------------------------------------------------- report text


def cell(triple: Triple, signed: bool = False) -> str:
    value, low, high = triple
    return validate.interval_text(value, metrics.Interval(low, high), signed=signed)


def four_model_rows(scored: Scored, version: str) -> tuple[list[list[str]], list[list[str]]]:
    """(AUROC rows, paired-difference rows) for one scoring scheme."""
    name = {b: models.spec_name(b, version) for b in models.BLOCKS}
    flag = strict_precedent_name(version)
    names = [*name.values(), REGISTRATION, flag]
    both = name["HOW + WHAT"]
    pairs = [
        (both, name["HOW"]),
        (both, REGISTRATION),
        (flag, REGISTRATION),
        (both, flag),
        (name["WHAT"], name["HOW"]),
        (name["WHAT + phase"], name["HOW"]),
    ]
    means, differences = clustered_scores(
        scored.outcomes,
        {n: scored.predictions[n] for n in names},
        scored.parts,
        scored.clusters,
        pairs,
    )
    aurocs = [[LABEL.get(b, b), cell(means[name[b]])] for b in models.BLOCKS]
    aurocs.append(["registration baseline (with therapeutic area)", cell(means[REGISTRATION])])
    aurocs.append(["registration baseline + strict precedent flag alone", cell(means[flag])])
    texts = {
        (both, name["HOW"]): "HOW + WHAT minus HOW",
        (both, REGISTRATION): "HOW + WHAT minus the registration baseline",
        (
            flag,
            REGISTRATION,
        ): "registration baseline + strict precedent flag alone, minus the registration baseline",
        (both, flag): "HOW + WHAT minus (registration baseline + strict precedent flag alone)",
        (name["WHAT"], name["HOW"]): "WHAT minus HOW",
        (name["WHAT + phase"], name["HOW"]): "WHAT + phase minus HOW",
    }
    rows = [[texts[p], cell(differences[p], signed=True)] for p in pairs]
    return aurocs, rows


def excluded_rows(scored: Scored, version: str) -> list[list[str]]:
    """WHAT and HOW + WHAT with the scored trial's own lead drug left out of every rate."""
    how = models.spec_name("HOW", version)
    chosen = {
        "WHAT, own lead drug excluded": models.spec_name("WHAT", version, "lead"),
        "HOW + WHAT, own lead drug excluded": models.spec_name("HOW + WHAT", version, "lead"),
    }
    names = [how, *chosen.values()]
    means, differences = clustered_scores(
        scored.outcomes,
        {n: scored.predictions[n] for n in names},
        scored.parts,
        scored.clusters,
        [(n, how) for n in chosen.values()],
    )
    rows = []
    for label, n in chosen.items():
        rows.append([label, cell(means[n]), cell(differences[(n, how)], signed=True)])
    return rows


def survival_lines(scored: Scored, version: str, exclude: bool = False) -> list[str]:
    how = scored.named(version, "HOW")
    what = scored.named(version, "WHAT", exclude)
    weights = cluster_weights(scored.clusters)
    pooled = correlation_interval(how, what, scored.parts, scored.clusters, weights=weights)
    within = correlation_interval(
        how, what, scored.parts, scored.clusters, within=True, weights=weights
    )
    lines = [
        (
            "- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): "
            f"{cell(pooled)} pooled, {cell(within)} on average within a round or fold."
        ),
    ]
    for label, strata in (
        ("phase, sponsor class and HOW third", (scored.phase, scored.sponsor_class)),
        (
            "phase, sponsor class, therapeutic area and HOW third",
            (scored.phase, scored.sponsor_class, scored.therapeutic_area),
        ),
    ):
        s = survival(scored.outcomes, how, what, scored.parts, *strata)
        shares, gaps = survival_draws(
            scored.outcomes, how, what, scored.parts, scored.clusters, *strata, weights=weights
        )
        low, high = percentile_pair(shares)
        gap_low, gap_high = percentile_pair(gaps)
        lines.append(
            f"- WHAT's top third succeeds {s.top_minus_bottom:.1f} per 100 more than its bottom "
            f"third (thirds cut within each {scored.unit}). Among trials of the same {label} "
            f"within the same {scored.unit}, the gap is {s.within_strata:.1f} per 100 "
            f"({gap_low:.1f} to {gap_high:.1f}): **{100 * s.share:.0f}% survives** "
            f"({100 * low:.0f}% to {100 * high:.0f}%, lead-drug resamples). "
            f"{s.n_strata} of {s.n_strata_total} strata had trials in both tails."
        )
    return lines


def thirds_table(scored: Scored, version: str, exclude: bool = False) -> list[str]:
    what = scored.named(version, "WHAT", exclude)
    counts = success_by_thirds(scored.outcomes, what, scored.parts, scored.phase)
    cells = thirds_intervals(scored.outcomes, what, scored.parts, scored.phase, scored.clusters)
    rows = []
    for phase in PHASES:
        rates, intervals, gap = cells[phase]
        n = counts[phase][1]
        rows.append(
            [
                phase.replace("phase", "Phase "),
                *(
                    f"{r:.1f} ({lo:.1f} to {hi:.1f}), n={c}"
                    for r, (lo, hi), c in zip(rates, intervals, n, strict=True)
                ),
                f"{rates[2] - rates[0]:+.1f} ({gap[0]:+.1f} to {gap[1]:+.1f})",
            ]
        )
    return validate.table(
        ["", "bottom third", "middle third", "top third", "top minus bottom"], rows
    )


def version_section(version: str, schemes: dict[str, Scored]) -> list[str]:
    lines = [f"## {LINK_TITLE[version]}", ""]
    for key, title in (
        ("rounds", "Time-ordered rounds"),
        ("folds", "Grouped 10-fold by lead drug"),
    ):
        scored = schemes[key]
        aurocs, differences = four_model_rows(scored, version)
        lines += [
            f"### {title}",
            "",
            (
                f"{len(scored.outcomes):,} scored trials. Mean within-{scored.unit} AUROC, 95% "
                "interval resampling whole lead drugs."
            ),
            "",
            *validate.table(["model", "AUROC"], aurocs),
            "",
            "Paired differences (same resamples):",
            "",
            *validate.table(["comparison", "AUROC difference"], differences),
            "",
            *survival_lines(scored, version),
            "",
            "Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):",
            "",
            *thirds_table(scored, version),
            "",
            (
                "With the scored trial's own lead drug left out of every group rate (training rows "
                "too). Difference is from HOW:"
            ),
            "",
            *validate.table(["model", "AUROC", "minus HOW"], excluded_rows(scored, version)),
            "",
            *survival_lines(scored, version, exclude=True),
            "",
            "Success per 100 by third of WHAT, own lead drug excluded:",
            "",
            *thirds_table(scored, version, exclude=True),
            "",
        ]
    return lines


def report(sections: dict[str, dict[str, Scored]], missing: dict[str, str]) -> str:
    head = [
        "# WHAT versus HOW (development trials)",
        "",
        (
            "Can what a trial tries to do predict its outcome beyond how it is run? Two blocks of "
            "predictors, compared model against model."
        ),
        "",
        (
            "- **HOW**: the registration columns less therapeutic area (phase, sponsor class, "
            "allocation, masking, intervention model, arm count, modality, vaccine, healthy "
            "volunteers, data monitoring committee)."
        ),
        (
            "- **WHAT**: therapeutic area; success rates for the trial's target, its disease and the "
            "pair (earlier trials only, see below); how many trials back each rate; the strict "
            "precedent flag (experimental-arm drugs only, no comparator fallback); genetic support "
            "when the trial started; generic target biology; flags for "
            '"has a mapped target" and "has a mapped disease".'
        ),
        (
            "- **WHAT + phase** and **HOW + WHAT** add phase, and all of HOW. All WHAT models are "
            "logistic regressions with a fixed penalty (C = 0.1)."
        ),
        "",
        (
            "**HOW here is the registration baseline WITHOUT therapeutic area.** Against the full "
            "registration baseline (which has it) the gain of HOW + WHAT is smaller, and most of "
            "that gain is the precedent flag: see the rows for the registration baseline plus the "
            "strict precedent flag alone, and HOW + WHAT minus that, in each table."
        ),
        "",
        (
            "**The rule for group rates.** A trial's rate counts only labelled training trials whose "
            "results were first posted before that trial's start, and never the trial itself. The "
            "training rows follow the same rule. Each rate is (successes + 5) / (trials + 10): "
            "pulled toward 0.5 by a fixed ten trials' worth, with nothing tuned on a label. The "
            "outcomes counted are dated; the keys that group trials are not (see below)."
        ),
        "",
        (
            "**Which links.** The drug-to-target links come from a release published after the trials "
            'started, so the WHAT models are not "honest" (the group rates included: their grouping '
            "keys come from the link version and today's disease mapping). The 2020 versions use the "
            "20.06 release (the last development year's); the third also matches the registry's "
            "drug names to 20.06's names (a short list of generic names such as insulin or saline "
            "is never matched) and leaves the chemistry columns empty, because they are keyed by "
            "today's drug ids. The disease mapping and the registry record are today's in every "
            "version, and so is `lead_drug`, which the lead-drug resampling and the own-drug "
            "exclusion use, even in the third version."
        ),
        "",
        (
            "**Test-set caution.** The 20.06 links postdate most development trials' starts but "
            "predate many test trials' starts, so gains from the dated versions may not carry over "
            "to the test set."
        ),
        "",
        (
            "Scoring is the repo's own (three rounds; grouped folds by lead drug). Intervals resample "
            "whole lead drugs, drawn once for all rounds; differences are paired on the same "
            "resamples. The correlation, the share and the thirds use the same draws of drugs, with "
            "the thirds and strata fixed from the predictions and cut inside each round or fold. "
            "Where a lead drug is unknown the trial is its own cluster; pairs of trials where "
            "either lead drug is unknown are kept in the group rates. Success is per 100 trials."
        ),
        "",
    ]
    body = []
    for version, schemes in sections.items():
        body += version_section(version, schemes)
    for version, reason in missing.items():
        body += [f"## {LINK_TITLE[version]}", "", f"Not run: {reason}", ""]
    return "\n".join(head + body).rstrip() + "\n"


# ---------------------------------------------------------------- command line


def available(version: str) -> str | None:
    """Why `version` cannot run, or None."""
    for name in (whathow.table_name(version), whathow.keys_name(version), "precedent"):
        if not validate.table_path(name).exists():
            return f"`{name}.parquet` is missing (run `python -m elman_exercise.whathow`)."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--versions", nargs="+", default=list(links.VERSIONS))
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    sections, missing = {}, {}
    for version in args.versions:
        reason = available(version)
        if reason:
            missing[version] = reason
            print(f"{version}: {reason}", file=sys.stderr)
            continue
        print(f"== {version}", file=sys.stderr)
        sections[version] = run_schemes(load_frame(version), version_specs(version))
    args.out.write_text(report(sections, missing))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
