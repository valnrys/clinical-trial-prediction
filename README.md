# A benchmark for clinical-trial outcome prediction

The question: several companies claim their AI can predict whether a clinical trial
will succeed. This is a benchmark that can test claims like that.

This repo builds that benchmark from public ClinicalTrials.gov data, scores plain
baselines on a held-out test set, and tests what else (biology, precedent,
language models) adds. Every step can be rebuilt from the commands below.

## Where to look first

| Read | Why |
|---|---|
| `report/two_page.pdf` | The two-page report. Source: `report/two_page.md`. |
| `PREREGISTRATION.md` | What I expected to find, written and emailed before any test labels were read. |
| `results/test_final.md`, `results/test_scorecard.md` | The first planned run on the test set (`FINAL_PLAN.md`) and how the seven pre-registered predictions fared. |
| `results/test_final2.md` | The follow-up run (`TEST_PLAN_2.md`): the fourth and last opening of the test labels. |
| `BELIEFS.md` | The working record: what I believed before data, after the development set and after the test set, with each decision and limit. Last updated 11 Nov; later work is in `TEST_PLAN_2.md` and the report. |
| `results/test_baselines.md` | The pre-registered baselines on the test set, with intervals and subsets. |
| `WHAT_ELSE_I_TRIED.md` | What I tried that did not make the report. |

The report says "training set" where these files say "development". They are the
same trials (finished before 2021). Test trials finished 2021 to 2023; a reserve
set finished 2024 to 2025.

## Status

Everything has been run. The test labels have been opened four times, each logged in
`evals/heldout_log.jsonl` and listed under "Access" in the result file it made:

| # | When (UTC) | What | Result file |
|---|---|---|---|
| 1 | 11 Nov 06:48 | The pre-registered baselines (`evaluate test --extended`) | `results/test_baselines.md` |
| 2 | 11 Nov 19:36 | The final run (`final test`, plan in `FINAL_PLAN.md`) | `results/test_final.md` |
| 3 | 11 Nov 19:36 | The scorecard for the seven pre-registered predictions (`scorecard test`) | `results/test_scorecard.md` |
| 4 | 12 Nov 17:46 | The second scoring (`final2 --real`, plan in `TEST_PLAN_2.md`) | `results/test_final2.md`, `.json`, `test_final2_item7_final.md`, `test_final2_item7_scorecard.md` |

The second plan was written after I had seen the first round of test results.
The first round showed that the words of a trial record carried
nearly all the gain over the baseline; the second plan asks what those words stand for
(disease, target, sponsor, precedent, biology). Its item 7 recomputes the first round's
intervals by resampling whole drugs instead of single trials, to see whether any verdict changes.
No model was changed in response to a test result. Development-only analyses are in `results/dev_*.md`.

**About this copy.** The public repository is one commit: the finished state, without the
working history. So the `prereg` git tag that the guard checks is not here, and the commit
hashes quoted in `evals/heldout_log.jsonl` and the result files point to commits that are
not published. The proof that the pre-registration came first is the email, sent Wed 11 Nov
2026 at 06:21 GMT, before the first logged read at 06:48.

## What it covers

**Build**

1. A labelled trial dataset with a development set and a held-out test set,
   from the public ClinicalTrials.gov v2 API (other public sources allowed).
2. Eligibility, labels and label checks, exclusions, and the split — with the
   choices that matter and their limitations documented.
3. At least one baseline: performance, uncertainty around the estimate, what
   the result does and does not show, and a breakdown for at least two
   meaningful subsets where performance should differ, with the reason.
4. Which claims the benchmark can test. Could it assess a claim of "90%
   accuracy"? When could it mislead? What can public data not answer?

**Send**

1. This repository: code and instructions to rebuild the benchmark and
   reproduce the baseline.
2. A short pre-registration, emailed **before inspecting test-set results**:
   a few lines on what I expect to find and why. What counts is the comparison,
   not whether the guess was right.
3. A report of at most two pages (`report/two_page.pdf`) that stands alone, including
   three one-line answers on pretrained models, outside information, and
   coding assistants.

**Constraints:** about one focused day of active time (report how much), public
data only, and judgement counts — including what is left out.

## Dates

| Event | When (GMT) |
|---|---|
| Pre-registration sent | Wed 11 Nov 2026, 06:21 |

## Where each item is answered

| Item | Where |
|---|---|
| Dataset and split | `trials.csv`, `flow.md`, `src/trial_benchmark/dataset.py` |
| Labels and label checks | `src/trial_benchmark/labels.py`, `results/dev_label_comparison.md`, `results/label_disagreements_read.csv` and `results/label_disagreements_read_reverse.csv` |
| Baseline with uncertainty | `results/test_baselines.md` |
| Subsets (placebo against active control; trial size) | The subset table in `results/test_baselines.md` |
| Claims, "90% accuracy", when it could mislead, what public data cannot answer | Report section "What this analysis can and can't do"; `results/dev_label_comparison.md` |

## The held-out guard

Test-set labels live in `data/heldout/` and are loaded only through
`trial_benchmark.open_heldout(path, note)`. It raises until the `prereg` git tag
exists and `PREREGISTRATION.md` is unchanged since that tag, and it logs every
access to `evals/heldout_log.jsonl`.

This is a tripwire, not a seal. The raw download holds every trial's outcome,
so nothing in the code can stop test-set results being computed another way,
and git tags and local logs can be rewritten. The independent proof of timing
is the pre-registration email. The guard is there to prevent accidents, and to
give the report an honest count of how often the test labels were opened.

```bash
uv run trial-benchmark status       # locked or not, and every held-out access
uv run trial-benchmark lock-prereg  # after the pre-registration is committed and emailed
```

## Rebuild

Each command prints what it did; "~" is a rough run time. Run them in this order.

```bash
uv sync
uv run pytest

# 1. The trials
uv run python -m trial_benchmark.download     # raw records -> data/raw/ctgov/ (~20 min, resumable)
uv run python -m trial_benchmark.dataset      # eligible trials, split, tables (prints the flow)
                                             # add --update-trial-list to rewrite an existing trials.csv
uv run python -m trial_benchmark.dev_summary  # success rates, development set only
uv run python -m trial_benchmark.evaluate dev # pre-registered baselines, development set -> results/dev_baselines.md

# 2. Feature tables, one row per trial, every split -> data/interim/
uv run python -m trial_benchmark.endpoint     # what the primary endpoint is -> endpoint.parquet
uv run python -m trial_benchmark.sponsor      # who ran the trial -> sponsor.parquet
uv run python -m trial_benchmark.text         # words of title, conditions, outcomes -> text.parquet
uv run python -m trial_benchmark.opentargets  # Open Targets 26.09 tables -> data/raw/opentargets/ (~15 min, resumable)
uv run python -m trial_benchmark.genetics     # genetic support for the drug's target -> genetics.parquet
uv run python -m trial_benchmark.history      # the sponsor's and drug's earlier trials -> history.parquet
uv run python -m trial_benchmark.science.table  # public science databases for the drug's target -> science.parquet
                                             # (downloads ~1.9 GB into data/raw/, checked against recorded checksums, resumable; ~2 min after that)

# 3. Optional: sentence embeddings. Pulls in PyTorch (about 1.2 GB). The other models do not need it;
#    the harness skips the embedding models if these tables are missing.
uv sync --extra embeddings
uv run --extra embeddings python -m trial_benchmark.embeddings   # -> embeddings_general.parquet, embeddings_biomedical.parquet

# 3b. Optional: a language model as predictor (Claude Haiku and Sonnet through the `claude` command line,
#     no tools, no API key; see the module docstring). The sample is fixed in results/llm_sample.csv.
#     Predictions are features (no label is read); scoring on test and late belongs to the final run.
uv run python -m trial_benchmark.llm_predictor check-isolation --model haiku   # proves the model has no tools
uv run python -m trial_benchmark.llm_predictor run                              # 90 frozen batches x 2 models x 3 prompts = 540 calls, about 40 minutes, resumable (--batches 1-2 runs only those)
uv run python -m trial_benchmark.llm_predictor leak-check                       # how often blinding leaves a drug-name word
uv run python -m trial_benchmark.llm_report                                     # development sample only -> results/dev_llm.md

# 4. Side analyses, development set only
uv run python -m trial_benchmark.genetics_summary  # -> results/dev_genetics.md
uv run python -m trial_benchmark.science.report    # public science data vs success -> results/dev_science.md (~5 min)
uv run python -m trial_benchmark.rare_disease      # Orphanet prevalence + Open Targets disease table
uv run python -m trial_benchmark.failures_summary  # why finished trials stopped -> results/dev_failures.md (~1 min)
uv run python -m trial_benchmark.label_comparison --fetch  # our label against CTO and TOP (~62 MB) -> results/dev_label_comparison.md

# 5. Compare every model on development trials only.
#    `validate` skips any model whose table is missing, so to reproduce the committed
#    results/dev_models.md run the table builders of steps 6 and 7 first, then this step.
#    The same goes for `split_study` and `final2`: build every table before them.
uv run python -m trial_benchmark.validate          # -> results/dev_models.md (~10 min with embeddings)
uv run python -m trial_benchmark.scorecard rehearse  # the final scorecard's code on dev data -> results/rehearsal_scorecard.md
uv run python -m trial_benchmark.final rehearse      # the final run's code on dev data -> results/rehearsal_final.md (~2 min)
uv run python -m trial_benchmark.final check         # every label-free step of the real run plus a dress rehearsal on made-up labels; opens nothing

# 6. Optional: the links, precedent, hierarchical and split-study analyses (development trials only).
#    Run from a clean checkout, in this order.
uv sync --extra hierarchical                           # JAX + NumPyro for the Bayesian models
uv run --extra hierarchical python -m trial_benchmark.models.hierarchical condition   # -> condition.parquet (first MeSH condition)
uv run python -m trial_benchmark.science.links          # downloads the Open Targets 20.06 drug table (checksummed), builds the link tables
uv run python -m trial_benchmark.precedent              # -> precedent.parquet (needs the links and the registry)
# The fits: 12 variant/links pairs x 4 splits (round0..round2, full), minutes each, cached under
# data/interim/hierarchical_fits/ and keyed by every input a fit reads. Fit them by hand with
#   uv run --extra hierarchical python -m trial_benchmark.models.hierarchical fit --version today --variant full --split round0
# or let the report fit what is missing (an hour or more):
uv run --extra hierarchical python -m trial_benchmark.hierarchy_report --fit-missing   # -> results/dev_hierarchy.md
uv run python -m trial_benchmark.precedent_report       # -> results/dev_precedent.md
uv run python -m trial_benchmark.split_study --models all   # the 41 models fixed by name in the module; an hour or more -> results/dev_split_study.md

# 7. Optional: the WHAT/HOW and biology analyses (development trials only)
uv run python -m trial_benchmark.whathow                # builds the WHAT/HOW tables (reads no outcome)
uv run python -m trial_benchmark.whathow_report         # -> results/dev_whathow.md
uv run python -m trial_benchmark.biology_report         # biology-only score -> results/dev_biology.md

# 8. Rehearse the second test run (slow: it fits the models). Reads no test labels, opens nothing;
#    writes under outputs/final2_dry/. `--hierarchy` limits which hierarchical variants it fits.
uv run --extra hierarchical python -m trial_benchmark.final2 --dry-run

# 9. The report: report/two_page.md -> two_page.html -> two_page.pdf (needs Google Chrome at its usual macOS path; fails unless it is exactly 2 pages)
uv run --with pypdf --with pyarrow report/build_two_page.py
```

The report's curves figure is drawn from `outputs/final2/predictions.parquet`, which only the
real test run writes (it holds outcomes, so it is never committed). Without that file the
build uses the committed `report/data/two_page_*.json` files, which hold the numbers the
report draws. In that case it prints a note saying so. The build itself writes those files, from
`results/` and, when it is there, the predictions file.

`validate` leaves the hierarchical models out unless you ask (`--only hierarchical`): each is a
sampler run of minutes. Its intervals resample whole lead drugs.

### The commands that read the test labels

```bash
uv run python -m trial_benchmark.evaluate test --extended --note "why"   # -> results/test_baselines.md (--extended adds the 2024 to 2025 trials)
uv run python -m trial_benchmark.final test --note "why"       # -> results/test_final.md (needs FINAL_PLAN.md committed)
uv run python -m trial_benchmark.scorecard test --note "why"   # -> results/test_scorecard.md
uv run --extra hierarchical python -m trial_benchmark.final2 --real --note "why"   # -> results/test_final2.md etc. (needs TEST_PLAN_2.md committed)
```

These open the sealed test labels. Each run is logged in `evals/heldout_log.jsonl`, needs the
`prereg` tag, and is meant to be run once, deliberately. Do not run them casually, and
never to try something out. **All four have been run, once each** (see Status). Running one
again would open the labels a fifth time, and `final2` refuses unless you pass `--rerun "reason"`.

- `final test` reads the test and late labels in one logged opening and computes only what
  `FINAL_PLAN.md` lists; it refuses unless the plan is committed and unchanged. Run it before
  `scorecard test`, and commit `results/test_final.md` and the log in between; otherwise the
  next opening's log line says the code was dirty.
- `final2 --real` is the second scoring, `TEST_PLAN_2.md`, eight items, one opening. It also
  writes `outputs/final2/predictions.parquet` (git-ignored: it holds outcomes). `final2 --dry-run`
  is the same code on development data and opens nothing.

What the side analyses are for: `genetics_summary` asks whether genetic support for the
drug's target goes with trial success. `opentargets` downloads some
outcome-bearing Open Targets files, keeps only named columns and genetic rows, and deletes
the rest. `failures_summary` and `label_comparison` look at trials that finished before 2021: why they stopped, and how our label compares with two published
benchmarks (`--fetch` downloads their files to `data/raw/benchmarks/`, or pass `--cto-dir`
and `--top-file`). `stopping.py` sorts free-text stop reasons with fixed keyword rules and
`rare_disease.py` flags rare diseases; both are rough, and their limits are in the module
docstrings and the result files. `results/label_disagreements_read.csv` and
`results/stop_reason_read.csv` are hand reads, by a language model, that measure those rules.

The API serves the current version of each record only, and refreshes about
once a day. A pull must finish inside one snapshot: resuming across a refresh
is refused, and a pull that crossed one is rejected when it is read. The
snapshot time and a count check against the API's own total are in
`data/raw/ctgov/manifest.json`. A rebuild on a later day will differ slightly,
so the exact trials and split used for the reported result are committed as a
list of trial IDs (`trials.csv`).

`dataset` writes `data/interim/dev.parquet` (development trials, with labels),
`data/interim/test_features.parquet` (2021 onwards, no labels),
`data/heldout/test_labels.parquet` (their labels; read only through the guard),
`trials.csv` (trial and split, no labels) and `flow.md` (how many trials each
eligibility rule removed). `trials.csv` is the exact benchmark membership, so an
existing one is never overwritten: `dataset` prints how many trials the new
derivation would add, remove or move, and rewrites the file only with
`--update-trial-list`. The tables `dataset` returns carry no test-set labels.
`dev_summary` reads only the development file and refuses anything else.

Columns come in groups (`dataset.columns_in_group(name)` lists each): `design`
(known at registration, the only group a baseline should use as inputs),
`results_structure` (`control_kind`, `n_arms`: read from the results section, for
slicing results, not model inputs), `post_hoc` (edited or known only after the
trial ended), and `labels`.

### Baselines

`evaluate dev` fits a ladder of six plain predictors (constant, phase lookup, a
logistic regression on registration-time columns, the same without modality, and
two deliberately leaky ones) and scores them on the development set two ways:
trained on trials finishing before 2017 and scored on 2017 to 2020, and
five-fold shuffled cross-validation. It also scores the cross-validated
predictions on those same 2017 to 2020 trials (a random split hides how
over-optimistic a model is; it does not flatter its ranking) and shows what the
registration model loses, in time-forward AUROC, when each design column is
dropped in turn. It reads `data/interim/dev.parquet` only and writes
`results/dev_baselines.md` (tracked in git; two runs give the same file apart
from the commit hash it records). Intervals are bootstrap intervals over trials,
which treat trials as independent, so they are somewhat too narrow. The columns
the registration rungs read are listed once, in `baselines.REGISTRATION_COLUMNS`
(the `design` group less `REGISTRATION_EXCLUDED`, empty for now).

`evaluate test` fits on all development trials and scores the 2021 to 2023 test set (with
`--extended`, also the 2024 to 2025 late set on its own, and 2021 to 2025). The labels come
only through `open_heldout`, so it stops until the `prereg` tag exists, logs the access, and
stops if the trial ids in the features, the labels and `trials.csv` do not match or repeat.
It writes `results/test_baselines.md`, which records the commit, whether the code had
uncommitted changes, and every time the test labels have been opened.

## How features are dated

A model is honest only if it could have run when the trial started. Every feature table lists
each of its columns with one of four datings, defined in `src/trial_benchmark/dating.py`:

- **timeless**: does not depend on any trial's result (a word count of the title).
- **dated**: restricted to information from before the trial started.
- **protocol**: read from the trial's own record, current version. Sponsors can edit it later;
  the pre-registered baseline reads the same record, so this is allowed.
- **after the start**: may include what was learned afterwards. Not allowed in honest models.

"Honest" means a model reads only timeless, dated and protocol columns. It is worked out from
the columns a model reads, never set by hand. Deliberately leaky models are kept and labelled
"not honest", to show how much hindsight is worth.

## Adding a model

`src/trial_benchmark/models/registry.py` is the list of models the benchmark compares. Each
family is a module in `models/` with a `SPECS` list of `ModelSpec`s (a name, how to fit, the
feature tables it needs, and whether it is honest). To add one, write the spec and add its
module name to `FAMILY_MODULES`. `validate` picks it up.

## Layout

| Path | What goes there |
|---|---|
| `src/trial_benchmark/` | Code that the results depend on. |
| `tests/` | Tests for it. |
| `src/trial_benchmark/dataset.py` | Eligibility, split and the trial table: design, results-structure and post-hoc (leaky) columns and labels, in named groups kept apart. |
| `src/trial_benchmark/dev_summary.py` | Success rates with intervals, development set only. |
| `src/trial_benchmark/baselines.py` | The ladder of plain predictors, each a function from a training table to a model with `predict_probability`; the registration columns are defined once here. |
| `src/trial_benchmark/metrics.py` | AUROC, Brier, accuracy, calibration gap, and bootstrap intervals and paired differences. |
| `src/trial_benchmark/evaluate.py` | Time-forward and cross-validated evaluation, the same-trials comparison and drop-one-column table (`dev`), and the held-out evaluation (`test`); writes `results/`. |
| `src/trial_benchmark/opentargets.py` | Downloads the Open Targets tables, keeping only named columns and genetic-association rows; crash-safe and resumable. |
| `src/trial_benchmark/genetics.py` | Per-trial genetic-support features (today, and dated before the trial started), investigational-drug flag, lead drug, targets. Features only. |
| `src/trial_benchmark/genetics_summary.py` | Development-set genetic-support analysis with intervals and added sensitivity analyses; writes `results/dev_genetics.md`. |
| `src/trial_benchmark/stopping.py` | Keyword rules that sort a trial's free-text stop reason (stopped early for benefit, recruitment, efficacy, safety, funding, other, none). |
| `src/trial_benchmark/adjusted.py` | Mantel-Haenszel ratios and differences with an interval that resamples sponsors. |
| `src/trial_benchmark/rare_disease.py` | Downloads Orphanet's prevalence file; flags a trial rare, not rare, unclear or unmapped from its Open Targets diseases. |
| `src/trial_benchmark/failures_summary.py` | Trials that stopped without a result, rare against not rare (crude, by cancer, and adjusted), and the labelled trials' success by rarity; writes `results/dev_failures.md`. |
| `src/trial_benchmark/dating.py` | The four datings and which are honest. |
| `src/trial_benchmark/endpoint.py` | What the primary endpoint measures and its time frame; features only. |
| `src/trial_benchmark/history.py` | What earlier trials of the same drug and sponsor had shown before this one started; features only. |
| `src/trial_benchmark/sponsor.py` | Sponsor name and cleaned key; features only. |
| `src/trial_benchmark/text.py` | The words of each trial's registration, one column per field; features only. |
| `src/trial_benchmark/embeddings.py` | Optional. Sentence vectors of the registration text from two pretrained encoders. |
| `src/trial_benchmark/science/` | Do public science databases help? One module per source (`depmap`, `expression`, `networks`, `annotations`, `chemistry`), `targets` (trial to investigational drug to target genes, gene identifiers), `fetch` (checked downloads), `table` (joins them into `science.parquet`), `checks` (known-biology checks), `trace` (25 random trials followed from the raw files to the table) and `report` (writes `results/dev_science.md`, with a positive control). Features only. |
| `src/trial_benchmark/llm_predictor.py` | A language model as predictor: the fixed sample, the protocol-only prompt (identified, masked and blinded), the isolated tool-free calls, strict parsing and the predictions table. Features only. |
| `src/trial_benchmark/llm_blinding.py` | Takes the names a model could recognise a trial by out of a record (drug names become "Drug A", the start date goes) and measures what is left. |
| `src/trial_benchmark/llm_posthoc.py` | Rules that find phrases sponsors write after the results are in (cut-off dates, achieved follow-up, termination) in outcome and arm text; flags a trial and cuts the clause out of the blinded prompt. |
| `src/trial_benchmark/llm_report.py` | Development-only report on the sampled 2017 to 2020 trials: reasoning versus recall; writes `results/dev_llm.md`. |
| `src/trial_benchmark/models/llm.py` | The six language-model specs (two models, three prompts); they look up stored predictions and are not ranked on all trials. |
| `src/trial_benchmark/models/` | Every model compared, by family. `registry.py` lists them; see "Adding a model". `biology.py` holds the models that read the science table. |
| `src/trial_benchmark/validate.py` | Compares every model on development trials only; writes `results/dev_models.md`. |
| `src/trial_benchmark/scorecard.py` | Scores the pre-registered predictions: `rehearse` on development data, `test` on the test set (logged, one-shot). |
| `src/trial_benchmark/final.py` | The final run: every registered model fitted on all development trials, then the analyses of `FINAL_PLAN.md` on the test and late labels, one logged opening. `rehearse` is the same code on development data; `check` stops before the labels. Before opening, the whole plan runs once on the real test and late features with made-up labels, and any failure refuses the run. |
| `src/trial_benchmark/final2.py` | The second test scoring (`TEST_PLAN_2.md`, eight items, one logged opening). `--dry-run` is the same code on development data. |
| `src/trial_benchmark/whathow.py`, `whathow_report.py` | WHAT/HOW tables for each trial (what is tested, how it is tested) and their development report, `results/dev_whathow.md`. |
| `src/trial_benchmark/biology_score.py`, `biology_report.py` | The biology-only score (risk flags and a logistic on biology columns) and its development report, `results/dev_biology.md`. |
| `src/trial_benchmark/precedent.py`, `precedent_report.py` | The "same target already in Phase 3" flag and its report, `results/dev_precedent.md`. |
| `src/trial_benchmark/hierarchy_report.py` | Development report on the hierarchical models, `results/dev_hierarchy.md`. |
| `src/trial_benchmark/split_study.py` | Does a random split flatter a model? Development only; `results/dev_split_study.md`. |
| `src/trial_benchmark/models/hierarchical.py`, `precedent.py`, `whathow.py` | The model families behind those analyses. |
| `src/trial_benchmark/science/links.py` | Trial to drug to target links, under today's or June 2020's Open Targets. |
| `src/trial_benchmark/label_comparison.py` | Our label against CTO and TOP on development trials, agreement and the hand-read disagreements; writes `results/dev_label_comparison.md`. |
| `FINAL_PLAN.md` | What the final run will compute, with the development expectation for each item. Committed before the run; the run refuses if it is missing, has a `<<` marker or has changed. |
| `results/` | `rehearsal_final.md` (the final run's code on development data), `dev_whathow.md`, `dev_biology.md`, `dev_hierarchy.md`, `dev_precedent.md`, `dev_split_study.md`, `test_scorecard.md`, `test_final.md`, `test_final2.md` (with `.json` and the two `test_final2_item7_*.md`), `dev_baselines.md`, `dev_models.md` (every model, development only), `rehearsal_scorecard.md`, `dev_genetics.md`, `dev_science.md`, `dev_failures.md`, `dev_label_comparison.md`, `dev_llm.md`, `llm_sample.csv` (the fixed sample of 900 trial ids, with the frozen batch and position of each and a post-hoc flag), `llm_blinded_read.csv` (a hand read of 30 blinded prompts), `test_baselines.md` (the pre-registered baselines on the test set), and hand reads: `label_disagreements_read.csv`, `label_disagreements_read_reverse.csv`, `stop_reason_read.csv`, `endpoint_type_read.csv`, `history_pairs_read.csv`. In git. |
| `data/raw/`, `data/interim/` | Downloaded and processed public data. Not in git. |
| `trials.csv` | Every eligible trial and its split, no labels. In git. Rewritten only with `--update-trial-list`. |
| `flow.md` | The eligibility flow and trials per split. Written by `dataset`. |
| `data/heldout/` | Test-set labels. Not in git. Guarded. |
| `data/README.md` | Source and download date of every dataset. |
| `evals/` | The held-out access log. In git. |
| `LOG.md` | Active time spent and notes on coding-assistant use. |
| `PREREGISTRATION.md` | The pre-registration. Locked at the `prereg` tag; never edited since. |
| `report/` | The two-page report: `two_page.md` (source), `two_page.pdf` and `.html` (output), `build_two_page.py` (builder), `data/two_page_*.json` (the numbers the figures draw). |
| `WHAT_ELSE_I_TRIED.md` | Things tried that are not in the report. |
| `TEST_PLAN_2.md` | The plan for the second test scoring, written before it ran. |
| `task/` | Task description and background reading. Not in git. |
