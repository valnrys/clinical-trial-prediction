"""Write results/dev_precedent.md: does the precedent flag help, and how much do the links matter?

    uv run python -m trial_benchmark.precedent          # build data/interim/precedent.parquet first
    uv run python -m trial_benchmark.precedent_report   # -> results/dev_precedent.md

Development trials only. No test or late outcome is read: the flag table covers every split (it
holds no outcome), but this report joins it to the development frame only, through the guarded
loader in `validate`.

Scoring is the repo's own: three time-ordered rounds and grouped 10-fold by lead drug
(`validate.run_time_ordered`, `validate.run_grouped`, `validate.summarise`), paired differences
from "logistic: registration". Intervals come from `validate.summarise`: once resampling trials,
and once resampling whole lead drugs (`clusters=`), because one drug's trials are not independent.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from trial_benchmark import metrics, precedent, validate
from trial_benchmark.models import linear
from trial_benchmark.models import precedent as precedent_models
from trial_benchmark.models.registry import ModelSpec
from trial_benchmark.science import links
from trial_benchmark.whathow_report import cluster_weights

REPORT_PATH = validate.ROOT / "results" / "dev_precedent.md"
REGISTRATION = linear.REGISTRATION_NAME
CLUSTER_RESAMPLES = metrics.N_RESAMPLES
MAIN = {
    "today": "logistic: registration + precedent (today's links)",
    "2020": "logistic: registration + precedent (2020 links)",
}
STRICT = {
    "today": "logistic: registration + precedent (today's links, experimental arms only)",
    "2020": "logistic: registration + precedent (2020 links, experimental arms only)",
}


# ---------------------------------------------------------------- numbers


# ---------------------------------------------------------------- sections


def specs_to_score() -> list[ModelSpec]:
    registration = next(s for s in linear.SPECS if s.name == REGISTRATION)
    return [registration, *precedent_models.SPECS, *precedent_models.sensitivity_specs()]


def paired_table(frame, specs) -> list[str]:
    time_result = validate.run_time_ordered(specs, frame)
    drugs = validate.drug_groups(frame)
    plain = validate.summarise(time_result.outcomes, time_result.predictions, time_result.round_of)
    clustered = validate.summarise(
        time_result.outcomes,
        time_result.predictions,
        time_result.round_of,
        clusters=time_result.drug,
    )
    grouped = validate.run_grouped(specs, frame)
    grouped_eval = validate.summarise(
        grouped.outcomes, grouped.predictions, grouped.fold_of, clusters=drugs
    )
    rows = []
    for spec in specs:
        name = spec.name
        row = [name, validate.auroc_cell(clustered, name)]
        if name == REGISTRATION:
            row += ["(reference)", "(reference)", "(reference)"]
        else:
            row += [
                validate.difference_cell(name, plain),
                validate.difference_cell(name, clustered),
                validate.difference_cell(name, grouped_eval),
            ]
        rows.append(row)
    header = [
        "model",
        "mean AUROC, three rounds (95% interval, lead drugs resampled)",
        "gain over registration, trials resampled",
        "gain over registration, lead drugs resampled",
        "gain over registration, grouped 10-fold, lead drugs resampled",
    ]
    return validate.table(header, rows)


def flagged_table(frame: pl.DataFrame) -> list[str]:
    """How many development trials each flag marks, headline against the Phase 2/3 sensitivity."""
    rows = []
    for column in (*precedent.ALL_FLAGS, *precedent.SENSITIVITY_FLAGS):
        k = int(frame[column].sum())
        rows.append([column, f"{k:,}", f"{100 * k / frame.height:.1f}"])
    return validate.table(["flag", "development trials flagged", "per 100 trials"], rows)


RATE_ROWS = (
    ("today's links", "precedent_today"),
    ("2020 links (targets dated)", "precedent_2020"),
    ("2020 names and links (drug match and targets dated)", "precedent_2020_names"),
    ("today's links, experimental arms only", "precedent_exp_today"),
    ("2020 links, experimental arms only", "precedent_exp_2020"),
    ("2020 names and links, experimental arms only", "precedent_exp_2020_names"),
)


def clustered_rates(
    y: np.ndarray,
    flag: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = CLUSTER_RESAMPLES,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Success per 100 with the flag, without it, and the difference, each (value, low, high).

    The intervals resample whole lead drugs (`whathow_report.cluster_weights`) among the trials
    given, one draw for both groups, so the difference interval is paired. A group empty in a
    resample leaves that resample out.
    """
    y = np.asarray(y, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    weights = cluster_weights(clusters, n_resamples).astype(float)

    def draws(mask: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * (weights[:, mask] @ y[mask]) / weights[:, mask].sum(axis=1)

    with_flag, without = draws(flag), draws(~flag)
    point = (
        100 * y[flag].mean() if flag.any() else float("nan"),
        100 * y[~flag].mean() if (~flag).any() else float("nan"),
    )

    def triple(value: float, values: np.ndarray) -> tuple[float, float, float]:
        found = validate.percentile_interval(values)
        return float(value), found.low, found.high

    return (
        triple(point[0], with_flag),
        triple(point[1], without),
        triple(point[0] - point[1], with_flag - without),
    )


def rates_table(frame: pl.DataFrame, own_version: bool = False) -> list[str]:
    """Phase 2 trials that are the drug's first in the disease: success with and without precedent.

    By default the population is fixed: first in disease by TODAY's drug match, the same trials in
    every row. With `own_version` each row uses the first-in-disease column of its own link
    version (today's drug match for the first two, the 20.06 name match for the third).
    """
    rows = []
    for label, column in RATE_ROWS:
        first = (
            precedent.FIRST_NAMES if own_version and "2020_names" in column else "first_in_disease"
        )
        sub = frame.filter((pl.col("phase") == "phase2") & pl.col(first).fill_null(False))
        with_, without = sub.filter(pl.col(column)), sub.filter(~pl.col(column))
        r1, r0, diff = clustered_rates(
            sub["any_arm"].to_numpy(), sub[column].to_numpy(), validate.drug_groups(sub)
        )

        def show(r, n):
            return f"{r[0]:.1f} ({r[1]:.1f} to {r[2]:.1f}) of {n:,}"

        rows.append(
            [
                label,
                show(r1, with_.height),
                show(r0, without.height),
                f"{diff[0]:+.1f} ({diff[1]:+.1f} to {diff[2]:+.1f})",
            ]
        )
    header = [
        "links",
        "with precedent: successes per 100 trials (95% interval, lead drugs resampled)",
        "without precedent",
        "difference per 100 (95% interval)",
    ]
    return validate.table(header, rows)


def link_funnel(dev_ids: list[str]) -> tuple[list[str], dict[str, int]]:
    """Dev trials with a target under each link version, and who is lost going to 2020."""
    ids = set(dev_ids)
    drugs = links.trial_drugs().filter(pl.col("nct_id").is_in(list(ids)))
    today = set(links.investigational_targets("today")["nct_id"]) & ids
    old = set(links.investigational_targets("2020")["nct_id"]) & ids
    named = links.trial_drugs_2020_names().filter(pl.col("nct_id").is_in(list(ids)))
    named_inv = set(named.filter(pl.col("is_exp"))["nct_id"])
    dated = set(links.investigational_targets("2020_names")["nct_id"]) & ids
    counts = {
        "trials": len(ids),
        "investigational drug": drugs["nct_id"].n_unique(),
        "target today": len(today),
        "target 2020": len(old),
        "lost": len(today - old),
        "gained": len(old - today),
        "both": len(today & old),
        "named drug": len(named_inv),
        "named any drug": named["nct_id"].n_unique(),
        "target names": len(dated),
    }
    rows = [
        ["development trials", f"{counts['trials']:,}", ""],
        [
            "with an investigational drug (a drug of an experimental arm that Open Targets lists)",
            f"{counts['investigational drug']:,}",
            "",
        ],
        ["with a target gene, today's links (26.09)", f"{counts['target today']:,}", ""],
        [
            "with a target gene, 2020 links (20.06)",
            f"{counts['target 2020']:,}",
            f"{counts['lost']:,} lost, {counts['gained']:,} gained",
        ],
        [
            (
                "with an investigational drug found by 20.06 names (an experimental-arm name "
                "matches a 20.06 drug name; comparators are not counted)"
            ),
            f"{counts['named drug']:,}",
            f"{counts['named any drug']:,} trials have any 20.06-named drug, comparators included",
        ],
        [
            "with a target gene, 2020 names and links (20.06 names, 20.06 targets)",
            f"{counts['target names']:,}",
            f"{len(today - dated):,} of today's lost, {len(dated - today):,} gained",
        ],
    ]
    return validate.table(["step", "trials", "change from today's links"], rows), counts


def precedent_funnel_tables(inputs: precedent.Inputs) -> list[str]:
    out = []
    for strict in (False, True):
        columns = {}
        for version in links.VERSIONS:
            version_inputs = precedent.inputs_for_version(inputs, version)
            drug_links = links.drug_targets(version, merge_salts=links.MERGE_SALTS)
            steps = precedent.funnel(version_inputs, drug_links, strict)
            columns[version] = steps
        rows = [
            [label, *(f"{columns[v][i][1]:,}" for v in links.VERSIONS)]
            for i, (label, _) in enumerate(columns["today"])
        ]
        title = "experimental arms only, no fallback" if strict else "main flag (with the fallback)"
        out += [f"**{title}**", ""]
        out += validate.table(
            [
                "step (development trials)",
                "today's links",
                "targets dated (2020 links)",
                "names and targets dated (2020 names and links)",
            ],
            rows,
        )
        out.append("")
    return out


# Known answers from the published record. KEYNOTE-010 (pembrolizumab, non-small-cell lung cancer,
# started August 2013) came after the nivolumab Phase 3 trials CheckMate 017 (October 2012) and
# 057 (November 2012), both on PD-1 (PDCD1). The two nivolumab trials themselves have no earlier
# PD-1 Phase 3 in lung cancer: 017 is the first, and 057's only earlier PD-1 trial is its own drug.
KNOWN_YES = {"NCT01905657": {"NCT01642004", "NCT01673867"}}
KNOWN_NO = ("NCT01642004", "NCT01673867")
PD1 = "ENSG00000188389"


def named_cases(inputs: precedent.Inputs) -> list[str]:
    """The known answers above, run through the same code for each link version."""
    lines = []
    experimental = dict(inputs.registry.select("nct_id", "exp_names").rows())
    for version in links.VERSIONS:
        drug_links = links.drug_targets(version, merge_salts=links.MERGE_SALTS)
        version_inputs = precedent.inputs_for_version(inputs, version)
        scored = precedent.Inputs(
            inputs.registry,
            version_inputs.drugs,
            inputs.diseases,
            (*KNOWN_YES, *KNOWN_NO),
        )
        pairs = precedent.precedent_pairs(scored, drug_links)
        for nct, expected in KNOWN_YES.items():
            found = set(pairs.filter(pl.col("nct_id") == nct)["u_nct"])
            verdict = "found" if expected <= found else f"NOT FOUND (found {sorted(found)})"
            lines.append(
                f"- {version} links: KEYNOTE-010 ({nct}, pembrolizumab, lung cancer, 2013): the "
                f"nivolumab Phase 3 trials {', '.join(sorted(expected))} are {verdict} as precedent."
            )
        for nct in KNOWN_NO:
            mine = pairs.filter(pl.col("nct_id") == nct)
            on_pd1 = mine.filter(pl.col("target_id") == PD1)
            other = mine.filter(pl.col("target_id") != PD1)["target_id"].n_unique()
            labelled = any("docetaxel" in n for n in experimental.get(nct, []))
            lines.append(
                f"- {version} links: {nct} (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials "
                f"in lung cancer): {'no PD-1 precedent, as it should be' if on_pd1.is_empty() else 'WRONGLY flagged on PD-1'}. "
                f"It is flagged on {other} other target(s) by the main flag"
                + (
                    ", because its registry record lists docetaxel as an experimental-arm "
                    "intervention, so the comparator is read as the trial's own drug."
                    if labelled and other
                    else "."
                )
            )
    return lines


def known_answers(inputs: precedent.Inputs, reports: pl.DataFrame) -> list[str]:
    lines = named_cases(inputs)
    for version in links.VERSIONS:
        drug_links = links.drug_targets(version, merge_salts=links.MERGE_SALTS)
        version_inputs = precedent.inputs_for_version(inputs, version)
        pairs = precedent.precedent_pairs(version_inputs, drug_links, strict=False)
        problems = precedent.check_pairs(
            version_inputs,
            drug_links,
            pairs,
            sample=precedent.SAMPLE,
            reports=reports,
            drugs_in_reports=version != "2020_names",
        )
        verdict = "no problems" if not problems else "PROBLEMS: " + "; ".join(problems[:5])
        drugs_text = (
            "both drugs in Open Targets' own list for their trial"
            if version != "2020_names"
            else "(no Open Targets list to check the 20.06 name match against)"
        )
        lines.append(
            f"- {version} links: {min(precedent.SAMPLE, pairs.height):,} of {pairs.height:,} pairs "
            f"rechecked one by one in plain Python. Recomputed from raw data: the Phase 3, "
            f"withdrawn, start and first-submission rules (from the registry's own date strings), "
            f"the shared disease and {drugs_text}. Not independent: the target links and which "
            f"drug is the trial's experimental one are read from the same tables the flag used: "
            f"{verdict}."
        )
    return lines


# ---------------------------------------------------------------- the report


WHAT_FLAG = (
    "The flag says: before this trial started, another molecule acting on the same target had "
    "already started a Phase 3 trial in the same disease. Only the fact that the earlier trial "
    "had started is used, never how it ended. The earlier trial must be a pure Phase 3 (a combined "
    "Phase 2/3 record does not count; it can be a Phase 2 relabelled later) and not withdrawn. "
    "Month-only dates are read conservatively: the earlier trial's latest possible start and "
    "first-submission days must both be before the scored trial's earliest possible start day. "
    'The exact rule is in `precedent.py`. Rows marked "with Phase 2/3" in section 1 are the '
    "sensitivity version that also counts combined Phase 2/3 records."
)
WHAT_LINKS = (
    "Three versions, each dating more of the chain. **Today's links:** Open Targets 26.09 for "
    "both the trial-to-drug match and the drug-to-target links. **Targets dated:** the same drug "
    "match, targets from Open Targets 20.06 (2 July 2020); a drug gets a target in the catalogue "
    "mostly because it went far, so today's links favour drugs that later succeeded. **Names and "
    "targets dated (the headline):** registry intervention names are matched only to the names "
    "in the 20.06 drug table (preferred names, synonyms, trade names; four or more characters, "
    "whole-word phrases; a short list of generic names such as insulin, saline or water is never "
    "matched), and targets are 20.06's, so a drug catalogued after June 2020 cannot be "
    "found. Targets come from the experimental-arm (investigational) drugs of the trial. The main "
    "flag falls back to every matched drug when none matches an experimental arm; the strict "
    "flag does not, and also needs the earlier trial's drug to be an experimental-arm drug. In "
    "every version a salt or brand form inherits the targets of its molecule family."
)
STILL_TODAYS = (
    "**Still today's in all three versions:** the registry record itself (intervention names, "
    "arms, phase, status and start dates, as they are now; a sponsor can have edited them since), "
    "Open Targets' disease mapping (26.09), and `lead_drug` (the drug trials are clustered by in "
    "the intervals; today's match even in the names version). **Still today's in the first "
    "two:** the trial-to-drug name match. In the third, the 20.06 names cover the table's parent "
    "molecules only: a salt or brand that 20.06 lists as a child of a parent is not matched on "
    "its own. No version rebuilds what could have been known at the start, so the models that "
    "read the flags are not marked honest. **Test-set caution:** the 20.06 links postdate most "
    "development trials' starts but predate many test trials' starts, so gains from the dated "
    "versions may not carry over to the test set."
)
AUROC_NOTE = (
    "AUROC is the chance a randomly chosen success is ranked above a randomly chosen failure "
    "(0.5 is a coin flip). Mean of three time-ordered rounds (scored 2015-16, 2017-18, 2019-20, "
    "each trained on earlier completions), and the repo's grouped 10-fold by lead drug. "
    "Differences are paired against `logistic: registration`. Intervals come from "
    "`validate.summarise`: the first gain column resamples trials; the others resample whole "
    "lead drugs, drawn once for all rounds, because one drug's trials are not independent."
)
RATES_NOTE = (
    "Successes per 100 trials (any arm), with and without a precedent, for Phase 2 trials with no "
    "earlier trial of the same molecule in the same disease. A raw comparison, not adjusted for "
    "anything else. Intervals resample whole lead drugs; the last column is an interval on the "
    'difference. **First table: a fixed population.** "First in disease" is decided once, by '
    "TODAY's drug match (Open Targets 26.09), so every row describes the same trials and the rows "
    "differ only in the flag. **Second table:** each row's own link version decides \"first in "
    "disease\" (today's drug match for the first four rows; for the 20.06-names rows, only "
    "registry Phase 2 and 3 records matched by 20.06 name can be the earlier trial), so the "
    "populations differ from row to row and from the first table."
)
FUNNEL_NOTE = (
    "Counts only. A trial with no target under the 2020 links is one whose drug the 2020 "
    "catalogue did not yet know, or gave no target. In the flag tables below, the 2020 names "
    "version counts every trial with a 20.06-named drug (comparators included) in its first rows. "
    "All versions use one salt rule (a salt inherits its molecule family's targets), so the "
    "target counts here and in the flag tables agree."
)
LEAKAGE_NOTE = (
    "Leakage: `tests/test_precedent.py` fails if a Phase 3 trial that starts after the scored "
    "trial, starts or is first registered on the scored trial's start day, or is first registered "
    "after it, sets the flag."
)


def write_report(frame: pl.DataFrame, inputs: precedent.Inputs, reports: pl.DataFrame) -> str:
    specs = specs_to_score()
    dev_ids = frame["nct_id"].to_list()
    dev_inputs = precedent.Inputs(
        inputs.registry, inputs.drugs, inputs.diseases, tuple(dev_ids), inputs.names_drugs
    )
    funnel_rows, _ = link_funnel(dev_ids)
    lines = [
        "# Does a precedent on the same target help predict trial success?",
        "",
        "Development trials only (completed before 2021). No test or late outcome is read.",
        "",
        "## What is being tested",
        "",
        WHAT_FLAG,
        "",
        WHAT_LINKS,
        "",
        STILL_TODAYS,
        "",
        "## 1. Paired gain over the registration baseline",
        "",
        AUROC_NOTE,
        "",
        *paired_table(frame, specs),
        "",
        "How many development trials each flag marks:",
        "",
        *flagged_table(frame),
        "",
        "## 2. Phase 2 trials that are the drug's first in the disease",
        "",
        RATES_NOTE,
        "",
        "**Fixed population (first in disease by today's drug match):**",
        "",
        *rates_table(frame),
        "",
        "**Each row's own link version decides first in disease:**",
        "",
        *rates_table(frame, own_version=True),
        "",
        "## 3. Funnel",
        "",
        *funnel_rows,
        "",
        FUNNEL_NOTE,
        "",
        *precedent_funnel_tables(dev_inputs),
        "## 4. Checks",
        "",
        *known_answers(dev_inputs, reports),
        "",
        LEAKAGE_NOTE,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    inputs, reports = precedent.load_inputs()
    frame = validate.load_development_with({precedent.TABLE})
    REPORT_PATH.write_text(write_report(frame, inputs, reports))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
