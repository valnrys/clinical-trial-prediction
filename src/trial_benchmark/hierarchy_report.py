"""Write results/dev_hierarchy.md: what the hierarchical model adds, and what it learned.

    uv run --extra hierarchical python -m trial_benchmark.hierarchy_report

Development trials only (completed before 2021). Reads fits cached by `models.hierarchical`; run
its `fit` command for each link version, variant and split first (a missing fit is an error here,
not a slow refit). Scoring is the repo's own (`validate.run_time_ordered`, `validate.summarise`):
three time-ordered rounds, the AUROC averaged within rounds, every difference taken from
"logistic: registration" on the same resamples. The intervals resample whole lead drugs inside
each round (`validate.summarise(clusters=...)`), because the trials of one drug are alike; the
trial-level interval is shown beside for comparison.

Two link versions are shown side by side. "today" uses the drug-to-target links of today's Open
Targets release (26.09). "2020" changes ONLY the drug-to-target step, using Open Targets release
20.06 (mid 2020); the trial-to-drug match, the lead drug and the MeSH condition are today's in both.
A trial's targets are its investigational drugs' targets in both; comparators are not counted.
The genetics columns of the "2020" variant are rebuilt from those 2020 targets with the evidence
dated before each trial's start year; the "today" variant reads the repo's genetics table (the
targets of every drug Open Targets lists for the trial, today's links), as before.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from trial_benchmark import metrics, validate
from trial_benchmark.models import hierarchical as hm
from trial_benchmark.models.linear import REGISTRATION_NAME
from trial_benchmark.models.registry import ModelSpec, all_specs

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "results" / "dev_hierarchy.md"
VERSIONS = ("today", "2020")
MIN_TRIALS_FOR_EXAMPLES = 10  # a level with fewer training trials is not shown by name
N_EXAMPLES = 5  # per direction, for each of diseases and targets
EFFECT_DRAWS = 1000  # posterior draws used for effects on the full frame (every 4th of 4,000)


def inverse_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def per_100_between_one_sd(sd, overall_rate: float):
    """Successes per 100 trials between a group one SD below and one SD above average.

    Both ends are at the overall success rate's log-odds shifted by minus and plus `sd`.
    """
    centre = np.log(overall_rate / (1 - overall_rate))
    return 100 * (inverse_logit(centre + sd) - inverse_logit(centre - sd))


def target_scale(prepared: pl.DataFrame) -> tuple[float, float]:
    """(trial-level scale of a target effect, share of trials with no target).

    A trial's target effect is the average of the effects of its k targets, so across trials its
    spread is the group's SD times the square root of the mean of 1/k over trials that have a
    target. A trial with no target gets no target effect.
    """
    counts = np.array([min(len(t or []), hm.MAX_TARGETS) for t in prepared["target_ids"].to_list()])
    with_targets = counts[counts > 0]
    scale = float(np.sqrt(np.mean(1.0 / with_targets))) if len(with_targets) else 1.0
    return scale, float(np.mean(counts == 0))


def summary(values) -> tuple[float, float, float]:
    values = np.asarray(values).ravel()
    low, high = np.percentile(values, metrics.INTERVAL_PERCENTILES)
    return float(values.mean()), float(low), float(high)


def interval_text(values, places: int = 2, signed: bool = False) -> str:
    mean, low, high = summary(values)
    sign = "+" if signed else ""
    return f"{mean:{sign}.{places}f} ({low:{sign}.{places}f} to {high:{sign}.{places}f})"


# ---------------------------------------------------------------- scores


@dataclass(frozen=True)
class Provision:
    """Where one link version's targets and precedent flag come from."""

    provider: object  # frame -> frame (see `hierarchical.Provider`); None means the repo's own
    source: str  # "repo", or a label for another source; part of the cache key
    description: str = ""  # said in the report; defaults to `source`


def specs_for(provisions: dict[str, Provision], allow_fit: bool) -> list[ModelSpec]:
    registration = [s for s in all_specs() if s.name == REGISTRATION_NAME]
    specs = list(registration)
    for version, provision in provisions.items():
        specs += hm.make_specs(version, provision.provider, provision.source, allow_fit=allow_fit)  # type: ignore[arg-type]
    return specs


def score_models(frame: pl.DataFrame, specs: list[ModelSpec]):
    """Time-ordered predictions through the repo's validation, then gains over the registration.

    Returns the predictions, the scores with intervals that resample whole lead drugs inside each
    round, and the same scores with intervals that resample single trials (narrower; shown beside).
    """
    result = validate.run_time_ordered(specs, frame)
    by_drug = validate.summarise(
        result.outcomes, result.predictions, result.round_of, clusters=result.drug
    )
    by_trial = validate.summarise(result.outcomes, result.predictions, result.round_of)
    return result, by_drug, by_trial


# ---------------------------------------------------------------- what the posterior learned


def thin(model: hm.FittedModel, draws: int = EFFECT_DRAWS) -> hm.FittedModel:
    step = max(1, len(model.posterior["a0"]) // draws)
    posterior = {k: v[::step] for k, v in model.posterior.items()}
    return hm.FittedModel(
        model.design, posterior, model.diagnostics, model.n_train, model.overall_rate
    )


def covariate_effects(model: hm.FittedModel, frame: pl.DataFrame) -> list[dict]:
    """Per 100 trials, the average change in success from a yes (or one SD more) in each column.

    For yes/no columns every trial is set to "no" and then to "yes"; for numbers every trial is
    moved one SD up. The average is over the trials of `frame` and over the posterior.
    """
    model = thin(model)
    design = model.design
    data = hm.encode(frame, design)
    eta = hm.linear_predictor(model, data, np.random.default_rng(0), fresh=False)
    out = []
    for j, name in enumerate(design.coef_names):
        g = model.posterior["g"][:, j]
        x = data["C"][:, j]
        yes_no = design.coef_scale[j] == 1.0 and frame[name].fill_null(0).n_unique() <= 2
        base = eta
        if yes_no:  # every trial set to "no"
            no = (0.0 - design.coef_mean[j]) / design.coef_scale[j]
            base = eta - g[:, None] * (x - no)[None]
        effect = 100 * (inverse_logit(base + g[:, None]) - inverse_logit(base)).mean(axis=1)
        out.append(
            {
                "name": name,
                "unit": "yes vs no" if yes_no else "+1 SD",
                "per100": effect,
                "odds_ratio": np.exp(g),
            }
        )
    return out


def level_effects(
    model: hm.FittedModel, group: str, overall_rate: float, names: dict[str, str]
) -> pl.DataFrame:
    """Each level of `group` seen often enough: its effect on a typical trial, per 100 trials."""
    design, post = model.design, model.posterior
    centre = np.log(overall_rate / (1 - overall_rate))
    rows = []
    for level, index in design.levels[group].items():
        n = design.counts[group][level]
        if n < MIN_TRIALS_FOR_EXAMPLES:
            continue
        effect = post[f"s_{group}"] * post[f"z_{group}"][:, index]
        per100 = 100 * (inverse_logit(centre + effect) - inverse_logit(centre))
        mean, low, high = summary(per100)
        rows.append(
            {
                "level": level,
                "name": names.get(level, level),
                "trials": n,
                "per100": mean,
                "low": low,
                "high": high,
            }
        )
    return pl.DataFrame(rows).sort("per100", descending=True)


def condition_names() -> dict[str, str]:
    path = validate.table_path(hm.CONDITION_TABLE)
    table = pl.read_parquet(path).drop_nulls("condition").unique("condition")
    return dict(zip(table["condition"], table["condition_name"], strict=True))


def target_symbols() -> dict[str, str]:
    path = ROOT / "data" / "raw" / "opentargets" / "target.parquet"
    table = pl.read_parquet(path, columns=["id", "approvedSymbol"])
    return dict(zip(table["id"], table["approvedSymbol"], strict=True))


# ---------------------------------------------------------------- the report


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def variant_name(variant: hm.Spec, version: str) -> str:
    suffix = "" if version == "today" else f" [{version} links]"
    return f"hierarchical: {variant.title}{suffix}"


def gain_section(frame, provisions, specs) -> tuple[list[str], list]:
    result, evaluation, by_trial = score_models(frame, specs)
    registration = evaluation.aurocs[REGISTRATION_NAME]
    rows = []
    for version in provisions:
        for variant in hm.VARIANTS:
            name = variant_name(variant, version)
            gain = evaluation.differences[name]
            trial_level = by_trial.differences[name].interval
            rows.append(
                [
                    variant.title,
                    version,
                    f"{evaluation.aurocs[name].value:.3f}",
                    f"{gain.value:+.3f}",
                    f"{gain.interval.low:+.3f} to {gain.interval.high:+.3f}",
                    f"{trial_level.low:+.3f} to {trial_level.high:+.3f}",
                ]
            )
    lines = [
        f"## AUROC gain over the registration logistic ({len(result.outcomes)} validation trials)",
        "",
        (
            f"The registration logistic scores {registration.value:.3f} "
            f"(95% interval, drugs resampled, {registration.interval.low:.3f} to "
            f"{registration.interval.high:.3f}). "
            "The table is the paired gain over it: mean AUROC within each of the three time-ordered "
            "rounds, each variant fitted inside each round on trials completed before it. A model's "
            "prediction is the posterior mean of its probability of success. The drug-clustered "
            "interval resamples whole lead drugs (a drug with trials in two rounds is in or out of "
            "both) and is the one to trust; the last column resamples single trials as if they were "
            "independent, for comparison."
        ),
        "",
        *table(
            [
                "variant",
                "links",
                "AUROC",
                "gain",
                "95% interval, drugs resampled",
                "95% interval, trials resampled",
            ],
            rows,
        ),
    ]
    return lines, rows


def full_fit(
    frame, provisions, version: str, variant: hm.Spec
) -> tuple[hm.FittedModel, pl.DataFrame]:
    provision = provisions[version]
    use = provision.provider or hm.repo_provider(version)
    prepared = use(frame)
    model = hm.fit_cached(prepared, variant, version, provision.source, allow_fit=False)
    return model, prepared


def sd_section(frame, provisions) -> list[str]:
    overall = float(frame["any_arm"].mean())
    rows = []
    notes = []
    for group in ("tgt", "cond", "sponsor", "drug", "area"):
        row = [hm.GROUP_LABEL[group]]
        for version in provisions:
            model, prepared = full_fit(frame, provisions, version, hm.VARIANT["full"])
            sd = model.posterior[f"s_{group}"]
            trial_sd = sd
            if group == "tgt":
                scale, none = target_scale(prepared)
                trial_sd = sd * scale
                notes.append(
                    f"{version} links: {100 * none:.0f}% of trials have no target and get no target "
                    f"effect; a trial with several targets gets the average of their effects, so its "
                    f"effect varies by {scale:.2f} times the SD in the table."
                )
            row += [interval_text(sd), interval_text(per_100_between_one_sd(trial_sd, overall), 1)]
        rows.append(row)
    header = ["group"]
    for version in provisions:
        header += [f"sd, log-odds ({version})", f"per 100 ({version})"]
    return [
        "## How much each group matters",
        "",
        (
            "The full hierarchy fitted on all development trials. Each group has one spread (SD, in "
            'log-odds). "Per 100" turns it into successes per 100 trials between a group one SD '
            f"below average and one SD above it, for a trial at the overall success rate "
            f"({100 * overall:.1f} per 100). Read the drug and target rows together, not apart: "
            "a drug and its targets overlap, so the model can give credit to either. For targets the "
            "per-100 figure is the trial-level one (the SD is first scaled for averaging over a "
            "trial's targets)."
        ),
        "",
        *[f"- {note}" for note in notes],
        "",
        *table(header, rows),
    ]


def covariate_section(frame, provisions) -> list[str]:
    rows = []
    for version in provisions:
        model, prepared = full_fit(frame, provisions, version, hm.VARIANT["full_pg"])
        for effect in covariate_effects(model, prepared):
            rows.append(
                [
                    effect["name"],
                    version,
                    effect["unit"],
                    interval_text(effect["per100"], 1, signed=True),
                    interval_text(effect["odds_ratio"]),
                ]
            )
    return [
        "## Precedent and genetics",
        "",
        (
            "From the full hierarchy plus precedent and genetics, fitted on all development trials. "
            '"Per 100" is, for each posterior draw, the average over the development trials of the '
            "change in predicted probability of success when the column goes from no to yes (or, for "
            "the genetic score, up one SD), group effects included, in sample, times 100. These "
            "coefficients have a tight prior (SD 0.2 in log-odds), so a small number is partly the "
            "prior's doing."
        ),
        "",
        *table(["column", "links", "unit", "per 100 (95% interval)", "odds ratio"], rows),
        *plain_precedent_lines(frame, provisions),
    ]


def plain_precedent_lines(frame, provisions) -> list[str]:
    """The same precedent flag in the plain registration logistic, computed the same way.

    Fit on all development trials; the number is the average over those trials of the change in
    predicted probability when the flag goes from no to yes, times 100. A point estimate only.
    """
    from trial_benchmark.models.precedent import fit_with_flag

    rows = []
    for version in provisions:
        column = f"precedent_exp_{version}"
        joined = validate.join_tables(frame, {"precedent"}).with_columns(
            pl.col(column).fill_null(False)
        )
        model = fit_with_flag(column, joined)
        yes = model.predict(joined.with_columns(pl.lit(True).alias(column)))
        no = model.predict(joined.with_columns(pl.lit(False).alias(column)))
        rows.append([version, f"{100 * float(np.mean(yes - no)):+.1f}"])
    return [
        (
            "For comparison, the plain registration logistic with the same precedent flag added, "
            "fitted on all development trials and read the same way (average change in predicted "
            "probability, no to yes, in sample, times 100; no interval):"
        ),
        "",
        *table(["links", "per 100"], rows),
    ]


def example_section(frame, provisions) -> list[str]:
    version = next(iter(provisions))
    model, _ = full_fit(frame, provisions, version, hm.VARIANT["full"])
    overall = float(frame["any_arm"].mean())
    lines = [
        "## Named examples",
        "",
        (
            f"Largest effects among levels with at least {MIN_TRIALS_FOR_EXAMPLES} training trials, "
            f"full hierarchy, {version} links, all development trials. Each is the change in "
            "successes per 100 trials for an otherwise average trial, as a posterior mean with its "
            "95% interval. Sparse levels are shrunk, but these are still the largest of the "
            "levels shown ({counts}), picked by size, so they overstate what a fresh trial of "
            "that kind would show; most intervals include zero ({zero})."
        ),
        "",
    ]
    shown_counts: list[str] = []
    zero_counts: list[str] = []
    for group, names, title in (
        ("cond", condition_names(), "Conditions (first MeSH term)"),
        ("tgt", target_symbols(), "Targets"),
    ):
        effects = level_effects(model, group, overall, names)
        shown_counts.append(f"{effects.height} {title.split(' (')[0].lower()}")
        spans_zero = int(((effects["low"] <= 0) & (effects["high"] >= 0)).sum())
        zero_counts.append(f"{spans_zero} of {effects.height} {title.split(' (')[0].lower()}")
        for label, part in (
            ("higher", effects.head(N_EXAMPLES)),
            ("lower", effects.tail(N_EXAMPLES).sort("per100")),
        ):
            rows = [
                [
                    r["name"],
                    str(r["trials"]),
                    f"{r['per100']:+.1f} ({r['low']:+.1f} to {r['high']:+.1f})",
                ]
                for r in part.iter_rows(named=True)
            ]
            lines += [f"### {title}: success {label} than average", ""]
            lines += table(["name", "training trials", "per 100 (95% interval)"], rows)
    lines[2] = lines[2].format(counts=", ".join(shown_counts), zero=", ".join(zero_counts))
    return lines


def diagnostics_section(frame, provisions) -> list[str]:
    rows = []
    for version in provisions:
        for variant in hm.VARIANTS:
            for split in ("round0", "round1", "round2", "full"):
                train = hm.training_rows(frame, split)
                provision = provisions[version]
                use = provision.provider or hm.repo_provider(version)
                model = hm.fit_cached(
                    use(train), variant, version, provision.source, allow_fit=False
                )
                d = model.diagnostics
                rows.append(
                    [
                        version,
                        variant.title,
                        split,
                        str(d["divergences"]),
                        f"{d['max_rhat']:.3f}",
                        f"{d['min_ess']:.0f}",
                    ]
                )
    worst = max(float(r[4]) for r in rows)
    above = sum(float(r[4]) > 1.01 for r in rows)
    over_limit = sum(float(r[4]) >= 1.05 for r in rows)
    summary = (
        f"Over {len(rows)} fits the worst R-hat is {worst:.3f}; {above} fits are above 1.01 and "
        f"{over_limit} are at or above 1.05. Divergences in total: {sum(int(r[3]) for r in rows)}."
    )
    return [
        "## Sampler checks",
        "",
        (
            "Four chains per fit. Divergences should be 0; the worst R-hat (over every parameter) "
            "is acceptable below 1.05 and clean below 1.01; the smallest effective sample size (over "
            f"every parameter) should be in the hundreds. {summary}"
        ),
        "",
        *table(
            ["links", "variant", "split", "divergences", "worst R-hat", "smallest effective n"],
            rows,
        ),
    ]


def build_report(provisions: dict[str, Provision] | None = None, allow_fit: bool = False) -> str:
    provisions = provisions or {v: Provision(None, "repo") for v in VERSIONS}
    frame = validate.load_development_with({"sponsor", "genetics", hm.CONDITION_TABLE})
    specs = specs_for(provisions, allow_fit)
    gains, _ = gain_section(frame, provisions, specs)
    lines = [
        "# The hierarchical model on the development trials",
        "",
        "Development trials only (completed before 2021). No test or late trial is used.",
        "",
        (
            "A Bayesian logistic model of trial success: the registration columns plus partially "
            "pooled effects for therapeutic area, lead sponsor, condition, target and lead drug "
            "(`models/hierarchical.py`). A group that has few trials is pulled towards the average, "
            "so it learns from what is common and cannot just memorise. Every score is out of "
            "sample: each fit is made inside its round and sees only trials completed before it."
        ),
        "",
        "Where the targets and precedent flags came from: "
        + "; ".join(f"{v} links, {p.description or p.source}" for v, p in provisions.items())
        + ' ("repo" is `science.links` and `precedent`). "2020" dates only the drug-to-target step '
        "(Open Targets release 20.06, mid 2020); the trial-to-drug match, lead drug and MeSH condition "
        "are today's. The 2020 genetics columns are rebuilt from the 2020 investigational targets, "
        "dated to each trial's start year.",
        "",
        *gains,
        *sd_section(frame, provisions),
        *covariate_section(frame, provisions),
        *example_section(frame, provisions),
        *diagnostics_section(frame, provisions),
    ]
    return "\n".join(lines)


def write_report(
    path: Path = REPORT_PATH,
    provisions: dict[str, Provision] | None = None,
    allow_fit: bool = False,
) -> Path:
    """Write `results/dev_hierarchy.md`. Needs every fit cached unless `allow_fit` is true."""
    path.write_text(build_report(provisions, allow_fit))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit-missing", action="store_true", help="fit what is not cached (slow)")
    args = parser.parse_args()
    print(write_report(allow_fit=args.fit_missing))


if __name__ == "__main__":
    main()
