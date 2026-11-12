# Perturbation response: an in-context learner for genetic interactions

**Question.** When two genes are perturbed together, how far does the effect depart from the sum of the two single-gene effects, and can a tabular in-context learner predict that departure from the single-gene measurements alone?

**Why it matters.** On the Norman et al. (2019) Perturb-seq combinations, the training-free additive prediction, the sum of the two measured single-gene effects, beats every learned model tested in the sibling project [ssl-lab](https://github.com/pleiadian53/ssl-lab) on predicting the mean effect of unseen pairs ([its results ledger](https://github.com/pleiadian53/ssl-lab/blob/main/examples/perturbation_response/docs/results/index.md)). Any model that wants to beat additivity has to predict the genetic interaction, the residual left after adding the singles. That residual is a small-data, tabular problem: about a hundred training combinations, each described by its two single-gene profiles.

**Plan.** The learner sees, as context, training combinations whose residuals were measured, and predicts the residual for held-out combinations. Its prediction is added back to the additive one and scored on the same twenty held-out pairs and the same endpoints as ssl-lab, so the numbers are directly comparable. The protocol is fixed in writing before any number is computed.

## Results

| Step | Question | Answer | Page |
|---|---|---|---|
| R1.1 | Is the residual reproducible, or mostly measurement noise? | Mostly signal: 18 of 20 pairs at least half signal (median 0.82). A perfect residual predictor would cut the size-aware error from 0.289 to about 0.08. | [the residual ceiling](results/r1-1-residual-ceiling.md) |

**Next.** The first scored round: TabPFN-3.5 predicting the residual, against additivity, a scaled and a shrunk additive reference, and tuned conventional learners, with its protocol fixed before any model is scored.
