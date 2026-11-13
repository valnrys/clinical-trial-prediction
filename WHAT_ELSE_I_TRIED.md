# What else I tried

Things I tried that are not in the two-page report, or get only a line there. Each row is one line on
what happened and, except for four exploratory rows, the file that shows it. "Development" means the trials finished before 2021 (the report calls
them the training set). AUROC is the chance a random success is ranked above a
random failure; 0.5 is a coin flip.

Four rows are marked "Exploratory; not in this repo". Those were quick side experiments on
development trials only. Their code was never brought up to the standard of the rest, so it
is left out and nobody can rerun them from here. The report quotes one figure from them, the graph network's gain on development trials, and nothing else.

| What I tried | What happened | Where |
|---|---|---|
| Trees, random forest and a small neural network on the registration columns | All scored about 0.717 to 0.718 against 0.709 for the plain logistic baseline, so little gained from a fancier model. | `results/dev_models.md` |
| Tuning the baseline's penalty on an inner split | No change: 0.709, difference -0.000 (-0.005 to +0.005). | `results/dev_models.md` |
| Five public biology databases added to registration | No gain in development: -0.001 AUROC (-0.010 to +0.008). | `results/dev_science.md` |
| A graph neural network over trials, drugs, targets and diseases | Alone it ranked below registration (0.677 against 0.709). Added to registration it gained 0.021 (0.013 to 0.028), or 0.013 (0.006 to 0.020) with links dated to 2020. `TEST_PLAN_2.md` says it "added nothing"; for this model that was wrong. The hierarchical model gave a similar gain with effects that can be read, so that went to the test instead. Never scored on the test set. | Exploratory; not in this repo |
| A Bayesian network learned from the data | Worse than registration under both scoring rules (0.688 and 0.693). Not scored on the test set. | Exploratory; not in this repo |
| Protein language-model embeddings of each target | No help on targets the model had never seen (95 trials); some settings did worse. Not scored on the test set. | Exploratory; not in this repo |
| AlphaFold confidence for the target | Adds nothing beyond whether the target has an entry at all: +0.0009 (-0.0015 to +0.0032). | Exploratory; not in this repo |
| Claude Haiku and Sonnet as predictors, reading the registration record | Sonnet scored AUROC 0.92 and Haiku 0.88 on 300 development trials, against 0.72 for the baseline. With drug names removed Sonnet still named the drug in 45% of trials, so recall and reasoning cannot be told apart. | `results/dev_llm.md` |
| Does a random split flatter a model? | 40 of 41 models scored higher on a random holdout than on later trials. The baseline: 0.700 on later trials, 0.728 on a random holdout. No interval for the difference excluded zero. | `results/dev_split_study.md` |
| Why finished trials have no result | Of 55,329 finished drug trials, 5,849 (10.6%) are in the benchmark. Recruitment is the stated reason for 27.9% of stopped trials. | `results/dev_failures.md` |
| Rare diseases and stopping for recruitment | The crude ratio of 1.44 shrinks to 0.97 (0.88 to 1.08) once phase, sponsor, era and cancer are allowed for. | `results/dev_failures.md` |
| Memorising the lead drug (deliberately leaky) | Scored 0.638 against 0.709 for the baseline (difference -0.071, -0.098 to -0.042). | `results/dev_models.md` |
