"""Small made-up trial tables with the real column layout, for tests."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from trial_benchmark import dataset as ds


def synthetic_trials(
    n: int,
    seed: int = 0,
    split: str = "dev",
    first_day: date = date(2008, 1, 1),
    days: int = 4700,
    prefix: str = "NCT0",
    with_labels: bool = True,
) -> pl.DataFrame:
    """`n` trials. Industry sponsors and phase 3 succeed more often; terminated ones rarely do."""
    rng = np.random.default_rng(seed)
    industry = rng.random(n) < 0.6
    phase3 = rng.random(n) < 0.5
    terminated = rng.random(n) < 0.1
    chance = 0.2 + 0.3 * industry + 0.2 * phase3 - 0.25 * terminated
    success = rng.random(n) < chance
    kinds = np.array(["small molecule", "biologic", "advanced", "mixed", "vaccine"])
    rows = {
        "nct_id": [f"{prefix}{i:07d}" for i in range(n)],
        "split": [split] * n,
        "primary_completion_date": [
            first_day + timedelta(days=int(d)) for d in rng.integers(0, days, n)
        ],
        "phase": np.where(phase3, "phase3", "phase2").tolist(),
        "sponsor_class": np.where(industry, "INDUSTRY", "OTHER").tolist(),
        "allocation": rng.choice(["RANDOMIZED", "NON_RANDOMIZED"], n, p=[0.9, 0.1]).tolist(),
        "masking": rng.choice(["NONE", "DOUBLE", "QUADRUPLE"], n).tolist(),
        "intervention_model": rng.choice(["PARALLEL", "CROSSOVER"], n).tolist(),
        "n_registered_arms": rng.integers(1, 5, n).tolist(),
        "modality": rng.choice(kinds, n, p=[0.6, 0.2, 0.05, 0.1, 0.05]).tolist(),
        "is_vaccine": rng.random(n) < 0.05,
        "therapeutic_area": rng.choice(["oncology", "neurology", "other"], n).tolist(),
        "start_year": rng.integers(2000, 2020, n).tolist(),
        "healthy_volunteers": rng.random(n) < 0.1,
        "has_dmc": rng.random(n) < 0.3,
        "control_kind": rng.choice(["placebo", "active"], n, p=[0.8, 0.2]).tolist(),
        "n_arms": rng.integers(1, 4, n).tolist(),
        "enrollment": rng.integers(20, 2000, n).tolist(),
        "overall_status": np.where(terminated, "TERMINATED", "COMPLETED").tolist(),
        "why_stopped_present": terminated,
        "duration_months": rng.uniform(3, 60, n),
        "n_locations": rng.integers(1, 100, n).tolist(),
        "results_lag_months": rng.uniform(6, 30, n),
        "any_arm": success if with_labels else [None] * n,
        "all_arms": success if with_labels else [None] * n,
        "highest_dose": [None] * n,
        "has_dose_ladder": [False if with_labels else None] * n,
    }
    return pl.DataFrame(rows, schema=ds.SCHEMA)
