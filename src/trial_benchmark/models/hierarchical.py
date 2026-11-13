"""A hierarchical Bayesian model of trial success: partial pooling over who, what and where.

    logit P(success) = intercept + registration columns             (one coefficient each)
                     + area effect + area-specific Phase 3 slope    (therapeutic area, 13 levels)
                     + sponsor effect                               (lead sponsor)
                     + condition effect                             (first MeSH condition)
                     + target effect                                (average over the drug's targets)
                     + drug effect                                  (the lead drug itself)
                     + genetics and precedent coefficients          (variants that include them)

Every group effect is `sd * z`, with `z ~ Normal(0, 1)` per level and one `sd` per group learned
from the data (a half-normal prior). Many trials in a level: the effect follows the data. Two
trials: it is pulled most of the way back to the average. That pulling is the "partial pooling",
and it is why a model with thousands of sparse levels does not just memorise them.

What a scored trial may learn from. The levels, the encoding and the posterior all come from the
training trials of one split and nothing else. A level not seen in training gets no effect of its
own: at prediction time it is a fresh draw from the group's learned spread, so its prediction is
the population prediction, widened by that spread. Predicting reads no outcome column.

Targets. A trial's targets are those of its INVESTIGATIONAL drugs only (`science.links`,
`investigational_targets`). A comparator's target says nothing about the drug being tested; the
genetics table's `target_ids` mixes the two, so it is replaced here.

Fits take minutes. `fit_cached` stores each posterior under `data/interim/hierarchical_fits/`,
keyed by the variant, the link version and the exact training trials of the split, so the
validation and the report never refit what they already have.

    uv run --extra hierarchical python -m trial_benchmark.models.hierarchical fit \
        --version today --variant full --split round0
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark.models.registry import ModelSpec, Predictor

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "interim" / "hierarchical_fits"
CONDITION_TABLE = "condition"  # data/interim/condition.parquet: nct_id, condition, condition_name

Version = Literal["today", "2020"]
Provider = Callable[[pl.DataFrame], pl.DataFrame]  # adds target_ids (and, if needed, precedent)

GROUP_COLUMN = {
    "sponsor": "sponsor_key",
    "area": "therapeutic_area",
    "cond": "condition",
    "drug": "lead_drug",
}
GROUP_LABEL = {
    "area": "therapeutic area",
    "sponsor": "lead sponsor",
    "cond": "condition",
    "tgt": "target",
    "drug": "lead drug",
}
MAX_TARGETS = 80  # targets kept per trial (almost no trial has more than a handful)
GENETICS_COLUMNS = ("genetic_score_at_start", "supported_at_start")
COEFFICIENT_SCALE = 0.2  # prior sd of genetics and precedent coefficients, per sd or yes-vs-no
REGISTRATION = bl.registration_features_without("therapeutic_area")  # area is a random effect


# ---------------------------------------------------------------- variants


@dataclass(frozen=True)
class Spec:
    """One model variant: which group effects it has and which extra columns it reads."""

    key: str
    title: str  # the name shown in reports and the registry
    groups: tuple[str, ...]
    genetics: bool = False
    precedent: bool = False
    group_scale: float = 1.0  # prior scale of every group's sd (half-normal)

    @property
    def honest(self) -> bool:
        """Could every input have been known when the trial started? (see `dating`)

        Targets, the lead drug, genetics and the precedent flag all depend on a drug mapping or a
        release made after the trials started ("after the start"), so a variant with any of them
        is not honest. Area, sponsor and condition are read from the registry record.
        """
        return not ({"tgt", "drug"} & set(self.groups) or self.genetics or self.precedent)


ALL_GROUPS = ("area", "sponsor", "cond", "drug", "tgt")
VARIANTS = (
    Spec("registration", "registration only", ("area",)),
    Spec("target", "+ target", ("area", "tgt")),
    Spec("condition", "+ condition", ("area", "cond")),
    Spec("sponsor", "+ sponsor", ("area", "sponsor")),
    Spec("full", "full hierarchy", ALL_GROUPS),
    Spec("full_pg", "full + precedent + genetics", ALL_GROUPS, genetics=True, precedent=True),
)
VARIANT = {v.key: v for v in VARIANTS}


@dataclass(frozen=True)
class Sampler:
    warmup: int = 800
    samples: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    seed: int = 0


DEFAULT_SAMPLER = Sampler()


# ---------------------------------------------------------------- inputs the frame must carry


def coefficient_columns(spec: Spec) -> tuple[str, ...]:
    """Columns entering as ordinary coefficients with a tight prior (genetics, precedent)."""
    return (GENETICS_COLUMNS if spec.genetics else ()) + (("precedent",) if spec.precedent else ())


def coefficient_matrix(frame: pl.DataFrame, spec: Spec) -> np.ndarray:
    columns = [
        frame[c].fill_null(0.0 if c != "supported_at_start" else False).cast(pl.Float64).to_numpy()
        for c in coefficient_columns(spec)
    ]
    return np.column_stack(columns) if columns else np.zeros((frame.height, 0))


# ---------------------------------------------------------------- the design (learned from training)


@dataclass
class Design:
    """Everything learned from the training trials to encode any trial the same way."""

    spec: Spec
    encoder: object = None
    keep: np.ndarray | None = None
    fixed_names: list[str] = field(default_factory=list)
    levels: dict[str, dict[str, int]] = field(default_factory=dict)  # group -> level -> index
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # group -> level -> trials
    coef_names: tuple[str, ...] = ()
    coef_mean: np.ndarray | None = None
    coef_scale: np.ndarray | None = None  # sd for numbers, 1 for yes/no


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _level_values(frame: pl.DataFrame, group: str) -> list[list[str]]:
    """The levels each trial belongs to for `group` (one, or several for targets)."""
    if group == "tgt":
        return [list(dict.fromkeys(ts or [])) for ts in frame["target_ids"].to_list()]
    return [[v] if v is not None else [] for v in frame[GROUP_COLUMN[group]].to_list()]


def fit_design(train: pl.DataFrame, spec: Spec) -> Design:
    design = Design(spec)
    encoder = bl.build_pipeline(REGISTRATION)[:-1].fit(train, bl.training_outcomes(train))
    raw = _dense(encoder.transform(train))
    names = [bl._readable(n) for n in encoder.named_steps["encode"].get_feature_names_out()]
    keep = raw.std(0) > 0
    for column in REGISTRATION.categorical:  # drop each one's commonest level: the baseline
        found = [i for i, n in enumerate(names) if n.startswith(column + "=")]
        if found:
            keep[found[int(np.argmax(raw[:, found].sum(0)))]] = False
    design.encoder, design.keep = encoder, keep
    design.fixed_names = [n for n, k in zip(names, keep, strict=True) if k]
    for group in spec.groups:
        tally: dict[str, int] = {}
        for levels in _level_values(train, group):
            for level in levels:
                tally[level] = tally.get(level, 0) + 1
        design.counts[group] = dict(sorted(tally.items()))
        design.levels[group] = {level: i for i, level in enumerate(design.counts[group])}
    design.coef_names = coefficient_columns(spec)
    if design.coef_names:
        matrix = coefficient_matrix(train, spec)
        design.coef_mean = matrix.mean(0)
        sd = matrix.std(0)
        yes_no = np.array(
            [set(np.unique(matrix[:, j])) <= {0.0, 1.0} for j in range(matrix.shape[1])]
        )
        design.coef_scale = np.where(yes_no | (sd == 0), 1.0, sd)
    return design


def encode(frame: pl.DataFrame, design: Design) -> dict[str, np.ndarray]:
    """Arrays for the model. A level the design has not seen is marked new, never given an index."""
    n = frame.height
    out = {"X": _dense(design.encoder.transform(frame))[:, design.keep].astype(np.float32)}
    out["ph3"] = ((frame["phase"] == "phase3").cast(pl.Float64).to_numpy() - 0.5).astype(np.float32)
    for group in design.spec.groups:
        known = design.levels[group]
        none = len(known)  # the extra slot that carries no effect
        members = _level_values(frame, group)
        width = MAX_TARGETS if group == "tgt" else 1
        index = np.full((n, width), none, dtype=np.int32)
        weight = np.zeros((n, width), dtype=np.float32)
        new = np.zeros((n, width), dtype=bool)
        for i, levels in enumerate(members):
            levels = levels[:width]
            for k, level in enumerate(levels):
                if level in known:
                    index[i, k] = known[level]
                else:
                    new[i, k] = True
                weight[i, k] = 1.0 / len(levels)
        out[group + "_index"], out[group + "_weight"], out[group + "_new"] = index, weight, new
    matrix = coefficient_matrix(frame, design.spec)
    if matrix.shape[1]:
        matrix = (matrix - design.coef_mean) / design.coef_scale
    out["C"] = matrix.astype(np.float32)
    return out


# ---------------------------------------------------------------- the model


def numpyro_model(data: dict, sizes: dict, spec: Spec, y=None) -> None:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    X = data["X"]
    eta = numpyro.sample("a0", dist.Normal(0.0, 1.5))
    eta = eta + X @ numpyro.sample("b", dist.Normal(0.0, 1.0).expand([X.shape[1]]).to_event(1))
    for group in spec.groups:
        sd = numpyro.sample(f"s_{group}", dist.HalfNormal(spec.group_scale))
        z = numpyro.sample(f"z_{group}", dist.Normal(0.0, 1.0).expand([sizes[group]]).to_event(1))
        padded = jnp.concatenate([z, jnp.zeros(1)])
        eta = eta + sd * jnp.sum(padded[data[group + "_index"]] * data[group + "_weight"], axis=1)
        if group == "area":  # the Phase 3 vs Phase 2 gap differs by area
            sd2 = numpyro.sample("s_area_phase", dist.HalfNormal(spec.group_scale))
            z2 = numpyro.sample(
                "z_area_phase", dist.Normal(0.0, 1.0).expand([sizes[group]]).to_event(1)
            )
            padded2 = jnp.concatenate([z2, jnp.zeros(1)])
            eta = eta + sd2 * padded2[data["area_index"][:, 0]] * data["ph3"]
    if data["C"].shape[1]:
        coefficient = numpyro.sample(
            "g", dist.Normal(0.0, COEFFICIENT_SCALE).expand([data["C"].shape[1]]).to_event(1)
        )
        eta = eta + data["C"] @ coefficient
    numpyro.sample("y", dist.Bernoulli(logits=eta), obs=y)


def diagnostics(posterior: dict[str, np.ndarray], divergences: int) -> dict:
    """Worst R-hat and smallest effective sample size over every parameter (chains first)."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    rhat, ess = [], []
    for values in posterior.values():
        rhat.append(np.atleast_1d(split_gelman_rubin(values)).ravel())
        ess.append(np.atleast_1d(effective_sample_size(values)).ravel())
    rhat, ess = np.concatenate(rhat), np.concatenate(ess)
    return {
        "divergences": int(divergences),
        "max_rhat": float(np.nanmax(rhat)),
        "min_ess": float(np.nanmin(ess)),
    }


@dataclass
class FittedModel:
    """A fitted variant: the design from its training trials and the posterior draws."""

    design: Design
    posterior: dict[str, np.ndarray]  # name -> (draws, ...) with the chains stacked
    diagnostics: dict
    n_train: int
    overall_rate: float  # success rate of the training trials

    def predict(self, frame: pl.DataFrame, seed: int = 1) -> np.ndarray:
        """Probability of success per trial: the posterior mean of the inverse-logit."""
        eta = linear_predictor(self, encode(frame, self.design), np.random.default_rng(seed))
        return (1.0 / (1.0 + np.exp(-eta))).mean(axis=0)


def fit(train: pl.DataFrame, spec: Spec, sampler: Sampler = DEFAULT_SAMPLER) -> FittedModel:
    """Fit `spec` on `train` (needs the `any_arm` label). Reads nothing but `train`."""
    import jax
    from numpyro.infer import MCMC, NUTS

    design = fit_design(train, spec)
    data = encode(train, design)
    sizes = {g: len(design.levels[g]) for g in spec.groups}
    outcomes = bl.training_outcomes(train).astype(np.int32)
    method = "parallel" if jax.local_device_count() >= sampler.chains else "sequential"
    mcmc = MCMC(
        NUTS(numpyro_model, target_accept_prob=sampler.target_accept),
        num_warmup=sampler.warmup,
        num_samples=sampler.samples,
        num_chains=sampler.chains,
        chain_method=method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(sampler.seed), data, sizes, spec, y=outcomes)
    by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    divergences = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
    stacked = {k: v.reshape((-1, *v.shape[2:])).astype(np.float32) for k, v in by_chain.items()}
    return FittedModel(
        design, stacked, diagnostics(by_chain, divergences), train.height, float(outcomes.mean())
    )


def group_contribution(model: FittedModel, data: dict, group: str, rng, fresh: bool) -> np.ndarray:
    """(draws, trials): the sd-scaled effect of `group` on each trial's log-odds."""
    post = model.posterior
    z = np.concatenate([post[f"z_{group}"], np.zeros((len(post["a0"]), 1), np.float32)], axis=1)
    index, weight, new = data[group + "_index"], data[group + "_weight"], data[group + "_new"]
    draws, n = z.shape[0], index.shape[0]
    contribution = np.zeros((draws, n))
    for k in range(index.shape[1]):  # one column per member level (targets have several)
        if not weight[:, k].any():
            continue
        contribution += z[:, index[:, k]] * weight[:, k][None]
    if fresh and new.any():  # unseen levels: independent fresh draws, one per posterior draw
        spread = np.sqrt(((new * weight) ** 2).sum(axis=1))
        contribution += rng.standard_normal((draws, n)) * spread[None]
    return post[f"s_{group}"][:, None] * contribution


def linear_predictor(model: FittedModel, data: dict, rng, fresh: bool = True) -> np.ndarray:
    """(draws, trials) posterior draws of the log-odds."""
    post = model.posterior
    eta = post["a0"][:, None] + post["b"] @ data["X"].T
    for group in model.design.spec.groups:
        eta = eta + group_contribution(model, data, group, rng, fresh)
        if group == "area":
            z2 = np.concatenate([post["z_area_phase"], np.zeros((len(post["a0"]), 1))], axis=1)
            slope = z2[:, data["area_index"][:, 0]]
            if fresh and data["area_new"][:, 0].any():
                m = data["area_new"][:, 0]
                slope[:, m] = rng.standard_normal((len(slope), int(m.sum())))
            eta = eta + post["s_area_phase"][:, None] * slope * data["ph3"][None]
    if data["C"].shape[1]:
        eta = eta + post["g"] @ data["C"].T
    return eta


# ---------------------------------------------------------------- cache


def input_columns(spec: Spec) -> list[str]:
    """Every column a fit of `spec` reads: label, registration, group and coefficient columns."""
    columns = [bl.LABEL, "phase", *REGISTRATION.categorical, *REGISTRATION.boolean]
    columns += list(REGISTRATION.numeric)
    for group in spec.groups:
        columns.append("target_ids" if group == "tgt" else GROUP_COLUMN[group])
    columns += coefficient_columns(spec)
    return list(dict.fromkeys(columns))


def training_digest(train: pl.DataFrame, spec: Spec) -> str:
    """A hash of the training trials' ids AND of every value the fit reads, label included.

    Two training tables with the same trial ids but a different label, target list, precedent
    flag, genetics value, sponsor, condition, lead drug or registration column get different
    digests, so a cached fit is never reused for different inputs.
    """
    columns = ["nct_id", *input_columns(spec)]
    rows = train.select(columns).sort("nct_id").rows()
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:12]


def cache_key(spec: Spec, sampler: Sampler, train: pl.DataFrame, version: str, source: str) -> str:
    """Names the fit by variant and by exactly which trials it trained on.

    A variant that reads genetics also depends on where its genetics came from, so that goes in
    the key (`genetics_source`); other variants keep the key they always had.
    """
    ids = training_digest(train, spec)
    where = genetics_source(version) if spec.genetics else ""
    setting = repr((spec, sampler, version, source)) + where
    setting = hashlib.sha256(setting.encode()).hexdigest()[:8]
    return f"{version}__{spec.key}__n{train.height}_{ids}_{setting}"


def fit_cached(
    train: pl.DataFrame,
    spec: Spec,
    version: str,
    source: str = "repo",
    sampler: Sampler = DEFAULT_SAMPLER,
    cache_dir: Path | None = None,
    allow_fit: bool = True,
) -> FittedModel:
    """The cached posterior for this exact variant, version and training set, else fit and store."""
    directory = cache_dir or CACHE_DIR
    path = directory / (cache_key(spec, sampler, train, version, source) + ".pkl")
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if not allow_fit:
        raise FileNotFoundError(f"No cached fit {path.name}; run the fit command first.")
    model = fit(train, spec, sampler)
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as handle:
        pickle.dump(model, handle)
    os.replace(partial, path)
    return model


# ---------------------------------------------------------------- the registry-shaped models


@dataclass
class HierarchicalPredictor:
    model: FittedModel
    provider: Provider

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.predict(self.provider(frame))


def genetics_source(version: str) -> str:
    """Where the genetics columns of a link version come from (part of a fit's cache key).

    "today": the repo's `genetics` table (targets of every drug Open Targets lists, today's links).
    Any other version: rebuilt from that version's investigational targets (`genetics_at_start`).
    """
    return "" if version == "today" else f"genetics at start, {version} investigational targets"


def genetics_at_start(version: Version) -> pl.DataFrame:
    """`genetic_score_at_start` and `supported_at_start` for each trial under one link version.

    Same rule as `genetics.build_table`: the best genetic-association score, over the trial's
    targets and its mapped diseases, in the latest history year strictly before the trial's start
    year. The targets are those of the trial's INVESTIGATIONAL drugs under `version`'s links
    (`science.links.investigational_targets`); the diseases are Open Targets' mapping of the
    trial's conditions (today's: there is no 2020 mapping to use). A trial with no target has
    nulls, as in the repo's table. Features only: reads no outcome.
    """
    from trial_benchmark import genetics
    from trial_benchmark.science import links

    reports = pl.read_parquet(
        genetics.OT_DIR / "clinical_report.parquet", columns=["id", "diseases"]
    )
    associations = pl.read_parquet(genetics.OT_DIR / "genetic_association.parquet")
    trials = genetics.load_trials()
    diseases = genetics.trial_diseases(reports, trials["nct_id"].to_list())
    targets = (
        links.investigational_targets(version)
        .select("nct_id", pl.col("target_id").alias("targetId"))
        .unique()
    )
    pairs = targets.join(diseases, on="nct_id")
    scored = genetics.pair_scores(pairs, trials, associations)
    best = scored.group_by("nct_id").agg(pl.col("at_start").max().alias("genetic_score_at_start"))
    return best.with_columns((pl.col("genetic_score_at_start") > 0).alias("supported_at_start"))


@functools.lru_cache(maxsize=4)
def dated_genetics(version: str) -> pl.DataFrame:
    return genetics_at_start(version)  # type: ignore[arg-type]


def repo_provider(version: Version) -> Provider:
    """Replace a trial's targets with its investigational drugs' targets, and add the precedent flag.

    Targets: `science.links.investigational_targets(version)`. Precedent: the STRICT
    `precedent_exp_<version>` column of `data/interim/precedent.parquet` (experimental-arm drugs on
    both sides, no comparator fallback; the same flag the what-versus-how models use). Build the
    table with `python -m trial_benchmark.precedent`.
    Genetics: for "today" the frame's own columns (the repo's `genetics` table); for any other
    version they are replaced by `genetics_at_start(version)`, so the 2020 model never reads
    genetics built from today's drug-to-target links.
    """

    def provide(frame: pl.DataFrame) -> pl.DataFrame:
        from trial_benchmark import precedent
        from trial_benchmark.science import links

        targets = (
            links.investigational_targets(version)
            .group_by("nct_id")
            .agg(pl.col("target_id").drop_nulls().unique().sort().alias("target_ids"))
        )
        flags = pl.read_parquet(
            ROOT / "data" / "interim" / f"{precedent.TABLE}.parquet",
            columns=["nct_id", f"precedent_exp_{version}"],
        ).rename({f"precedent_exp_{version}": "precedent"})
        frame = frame.drop("target_ids", "precedent", strict=False)
        if version != "today":
            frame = frame.drop(GENETICS_COLUMNS, strict=False).join(
                dated_genetics(version), on="nct_id", how="left"
            )
        return (
            frame.join(targets, on="nct_id", how="left")
            .join(flags, on="nct_id", how="left")
            .with_columns(pl.col("target_ids").fill_null([]), pl.col("precedent").fill_null(False))
        )

    return provide


def make_specs(
    version: Version = "today",
    provider: Provider | None = None,
    source: str = "repo",
    sampler: Sampler = DEFAULT_SAMPLER,
    allow_fit: bool = True,
) -> list[ModelSpec]:
    """Registry specs for every variant under one link version.

    `provider` supplies the targets and precedent flag (default: the repo's own); `source`
    names where they came from so a cached fit is never reused across sources. With
    `allow_fit=False` a missing cached fit is an error, not a slow refit.
    """
    use = provider or repo_provider(version)
    suffix = "" if version == "today" else f" [{version} links]"
    specs = []
    for variant in VARIANTS:

        def fit_variant(train: pl.DataFrame, variant: Spec = variant) -> Predictor:
            prepared = use(train)
            return HierarchicalPredictor(
                fit_cached(prepared, variant, version, source, sampler, allow_fit=allow_fit), use
            )

        specs.append(
            ModelSpec(
                f"hierarchical: {variant.title}{suffix}",
                "hierarchical",
                "Bayesian logistic model: registration columns plus partially pooled effects for "
                + ", ".join(GROUP_LABEL[g] for g in variant.groups)
                + (", genetics and precedent" if variant.genetics else "")
                + ".",
                fit=fit_variant,
                tables=("sponsor", "genetics", CONDITION_TABLE),
                own_tables=("precedent",) if variant.precedent else (),
                honest=variant.honest,
            )
        )
    return specs


SPECS = make_specs("today")  # same shape as every other family's `SPECS`


# ---------------------------------------------------------------- the condition table


def build_condition_table(path: Path | None = None) -> pl.DataFrame:
    """One pass over the registry: each trial's first MeSH condition (id and name). No outcomes."""
    from trial_benchmark.download import iter_studies

    rows = []
    for study in iter_studies():
        browse = (study.get("derivedSection") or {}).get("conditionBrowseModule") or {}
        meshes = browse.get("meshes") or []
        first = meshes[0] if meshes else {}
        rows.append(
            {
                "nct_id": study["protocolSection"]["identificationModule"]["nctId"],
                "condition": first.get("id"),
                "condition_name": first.get("term"),
            }
        )
    table = pl.DataFrame(
        rows, schema={k: pl.String for k in ("nct_id", "condition", "condition_name")}
    )
    if path is not None:
        table.write_parquet(path)
    return table


# ---------------------------------------------------------------- command line


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser(
        "fit", help="fit (or load) one variant on one split of the development trials"
    )
    one.add_argument("--version", choices=["today", "2020"], required=True)
    one.add_argument("--variant", choices=list(VARIANT), required=True)
    one.add_argument("--split", required=True, help="round0..round2, fold0..fold9 or full")
    sub.add_parser("condition", help="write data/interim/condition.parquet")
    args = parser.parse_args()
    if args.command == "condition":
        table = build_condition_table(ROOT / "data" / "interim" / f"{CONDITION_TABLE}.parquet")
        print(table.shape, "non-null condition:", table["condition"].is_not_null().sum())
        return
    from trial_benchmark import validate

    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    frame = validate.load_development_with({"sponsor", "genetics", CONDITION_TABLE})
    train = training_rows(frame, args.split)
    prepared = repo_provider(args.version)(train)
    model = fit_cached(prepared, VARIANT[args.variant], args.version)
    print(args.version, args.variant, args.split, model.diagnostics)


def training_rows(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """The training trials of a named split, by the same rules as `validate`."""
    from trial_benchmark import validate

    if split == "full":
        return frame
    if split.startswith("round"):
        rows, _ = validate.time_round(frame, *validate.ROUNDS[int(split[5:])])
        return frame[rows]
    if split.startswith("fold"):
        folds = validate.grouped_folds(validate.drug_groups(frame))
        return frame[folds[int(split[4:])][0]]
    raise ValueError(f"Unknown split {split!r}.")


if __name__ == "__main__":
    # Run under the package name, so the classes in a pickled fit are not saved as `__main__.*`.
    from trial_benchmark.models.hierarchical import _main as main

    main()
