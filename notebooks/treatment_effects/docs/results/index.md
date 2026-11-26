# The results ledger: treatment effects

*An append-only record of how effect estimators behave when built on TabPFN, linear models or tuned gradient boosting, and of what the characterizations add up to. Each characterization has its own page, which says what was asked, how it was run and what was measured, with every number traced to the script that produced it and the report that holds it.*

---

## The characterizations

| # | Question | Answer | Page |
|---|---|---|---|
| 01 | On a simulated care program with known effects, over ten draws and two scenarios, which base model gives the most accurate average and individual effects? | Individual effects: TabPFN, in 15 of 16 paired contrasts (simultaneous 95%), with a best PEHE of 0.084 against about 0.120 for either reference in the smooth scenario, and 0.128 against 0.147 for gradient boosting's best learner in the harder one. Average effect: AIPW is accurate with all three base models in the smooth scenario; in the harder one it is biased and covers the truth in 3 draws of 10 with linear models, whose outcome and propensity models are both wrong, and in 9 of 10 with TabPFN. With poor overlap in the harder scenario no estimator's interval can be trusted. | [effects with a known answer](01-effects-with-a-known-answer.md) |

---

## What they add up to

**A strong predictor helps most where effects vary between people.** For the average effect, the doubly robust estimator already corrects most of what a mediocre nuisance model gets wrong, and the base model matters little as long as one of the two nuisance models is close. For each person's effect there is no such correction, and the base model's accuracy carries through: TabPFN, untuned, gave the lowest mean error for individual effects with every meta-learner in both scenarios, resolved in 15 of the 16 paired comparisons.

**Double robustness needs one of the two models to be right.** When the outcome model and the propensity model are both misspecified, AIPW can be more biased than inverse propensity weighting alone, and its standard error does not show it. A flexible base model is the practical protection.

**Overlap comes first.** When many people had almost no chance of one of the arms, every estimator leans on extrapolation, and in the harder scenario none of them, TabPFN included, produced intervals that covered the truth in more than half the draws.

---

## The rules every characterization follows

- **Simulated truth before real data.** An estimator is characterized where every effect is known before it is applied where none is.
- **The same estimator on every base model.** The estimators take the base model as an argument, so a difference between rows is a difference between base models.
- **Every number comes from a saved report** written by a script whose docstring fixes the protocol before the full run, and the script first checks that it reproduces its notebook.
- **Draws are the unit of replication.** Comparisons are paired by draw and count only when a simultaneous 95% interval lies wholly on one side of zero.

## The settings every characterization shares

- **Code:** the estimators in [`tabicllab.treatment_effects`](../../../../src/tabicllab/treatment_effects.py); the simulator, base models and plots in [`helper.py`](../../helper.py).
- **Compute:** a RunPod A40 through [`ops/skypilot/te_01.yaml`](../../../../ops/skypilot/te_01.yaml); a laptop runs the smoke versions.
