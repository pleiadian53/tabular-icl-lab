# Perturbation response: an in-context learner for genetic interactions

**Question.** When two genes are perturbed together, how far does the effect depart from the sum of the two single-gene effects, and can a tabular in-context learner predict that departure from the single-gene measurements alone?

**Why it matters.** On the Norman et al. (2019) Perturb-seq combinations, the training-free additive prediction, the sum of the two measured single-gene effects, beats every learned model tested in the sibling project [ssl-lab](https://github.com/pleiadian53/ssl-lab) on predicting the mean effect of unseen pairs ([its results ledger](https://github.com/pleiadian53/ssl-lab/blob/main/examples/perturbation_response/docs/results/index.md)). Any model that wants to beat additivity has to predict the genetic interaction, the residual left after adding the singles. That residual is a small-data, tabular problem: about a hundred training combinations, each described by its two single-gene profiles.

**Plan.** The learner sees, as context, training combinations whose residuals were measured, and predicts the residual for held-out combinations. Its prediction is added back to the additive one and scored on the same twenty held-out pairs and the same endpoints as ssl-lab, so the numbers are directly comparable. The protocol is fixed in writing before any number is computed.

## Results

| Step | Question | Answer | Page |
|---|---|---|---|
| R1.1 | Is the residual reproducible, or mostly measurement noise? | Mostly signal: 18 of 20 pairs at least half signal (median 0.82). A perfect residual predictor would cut the size-aware error from 0.289 to about 0.08. | [the residual ceiling](results/r1-1-residual-ceiling.md) |
| R1.3 | Round 1: does TabPFN-3.5, untuned, predict the residual better than additivity? | A tie on the 20 test pairs under the pre-registered rule (size-aware error 0.260 against 0.289, better on 13 of 20, interval includes zero). The larger readouts agree and are clearer: 0.221 against 0.318 on the validation pairs, and 0.233 against 0.312 across the 91 training pairs, cross-validated, where it beats additivity, both no-interaction references and tuned gradient boosting. | [round 1](results/r1-3-round1.md) |

**Next.** Route 1 is under review after the tie. Two follow-ups are being weighed. One is a confirmation round that scores all 131 combinations by cross-validation, which is affordable here because an in-context learner needs no retraining. The other gives the learner descriptors of the genes involved, aimed at interactions like the one TabPFN missed, which appear only in the combination and cannot be foreseen from the singles alone.
