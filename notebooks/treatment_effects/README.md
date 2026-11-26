# Treatment effects: what an intervention changes, and for whom

**Question.** When a treatment was not assigned at random, how well can its effect be estimated, on average and for each person, and how much does a strong tabular predictor such as TabPFN help when it is used as the model inside the standard effect estimators?

**Why it matters.** Care programs, clinical trials and observational health studies all ask which intervention changes an outcome and for whom: which members a coach should call, which patients a drug helps, which message to send to whom. Most of the data available to answer such questions were not produced by a randomized trial. Clinicians and coaches chose whom to treat, and they chose people who differ from the others in ways that affect the outcome on their own. A model that predicts outcomes very well can then still be wrong about what a treatment would change. The effect estimators of causal inference correct for this, and every one of them is built from ordinary prediction problems: the probability that a person was treated, and the expected outcome with and without treatment. A tabular in-context learner is strong at exactly those problems for the sample sizes these studies have, from hundreds to a few thousand people. These notebooks measure how far that strength carries into the effects themselves.

## Notebooks

| Notebook | What it measures |
|---|---|
| [`01_effects_with_a_known_answer.ipynb`](01_effects_with_a_known_answer.ipynb) | On a simulated care program where every member's true effect is known: how far the plain comparison of treated and untreated is off, how well linear models, tuned gradient boosting and TabPFN fit the propensity and outcome models, which estimators recover the average effect (regression adjustment, inverse propensity weighting, the doubly robust AIPW, with DoubleML as a cross-check), which recover each member's effect (S-, T-, X- and DR-learners), how well each targets the members who benefit most, and what happens as the treated and untreated stop overlapping. Repeated over ten draws and in a harder scenario on the [results page](docs/results/01-effects-with-a-known-answer.md). |
| [`02_observational_study_nhefs.ipynb`](02_observational_study_nhefs.ipynb) | A real observational study, the NHEFS data of Hernán and Robins' *Causal Inference: What If*: whether quitting smoking causes weight gain. Reproduces the book's estimates with its own models, then the doubly robust estimators on the book's linear specification, tuned gradient boosting and TabPFN, with the checks a real study relies on in place of ground truth: held-out prediction error, overlap, covariate balance after weighting, tests of whether estimated differences between people are real, and the strength an unmeasured confounder would need to change the conclusion. |

Planned: a randomized trial used as the ground truth for observational methods (including variance reduction with pre-treatment predictions, CUPED and CUPAC), uplift modelling and treatment policies, and in-context learners pretrained specifically for causal effects.

## Results

The [results ledger](docs/results/index.md) repeats each notebook's analysis over independent draws, records every number in JSON reports written by the notebook's report script, and says what the characterizations add up to.

## Code

The estimators are general and live in the lab's package, [`tabicllab.treatment_effects`](../../src/tabicllab/treatment_effects.py), with unit tests in [`tests/test_treatment_effects.py`](../../tests/test_treatment_effects.py). Every estimator takes the prediction model as an argument, so the same estimator runs on any base model. [`helper.py`](helper.py) holds what is particular to these notebooks: the simulator, the three base models and the plots.

## Running

Use the `tabicllab` environment and its Jupyter kernel (see the repository README); DoubleML is installed from `environment.yml`. The notebooks fit TabPFN dozens of times, which takes minutes on a GPU and far longer on a laptop CPU, so they are executed on a GPU pod: [`run_te_pod.sh`](run_te_pod.sh) with the SkyPilot task [`ops/skypilot/te.yaml`](../../ops/skypilot/te.yaml), as described in [`ops/README.md`](../../ops/README.md). Setting the environment variable `TE_SMOKE=1` runs a notebook on a small table with a light TabPFN, which checks that every cell works on a laptop in a few minutes; `TE_MEMBERS` and `TE_FOLDS` set the number of simulated members and of cross-fitting folds for either mode, and nothing else in the code assumes a size.

The committed notebook 01 was executed on an NVIDIA A40 with TabPFN-3.5 (TabPFN at upstream commit `3f2e9f9`), scikit-learn 1.9.1, XGBoost 2.1.3 and DoubleML 0.11.4; the full run takes under four minutes there. Notebook 02 was executed the same way, in about three and a half minutes, with the NHEFS file downloaded to the pod's data lake and checked against its SHA-256.
