# The results ledger

*An append-only record of every experimental round on predicting the genetic-interaction residual. This page holds what the rounds add up to: the standing verdict, the rules every round follows, the settings they share and what comes next. Each round has its own page, which says what was asked, how it was run, what was measured and what the verdict was under the rule fixed before the run, with a link from every number to the script and the report that produced it.*

> **What this ledger covers.** Every round is evaluated on the same benchmark as the sibling project [ssl-lab](https://github.com/pleiadian53/ssl-lab)'s [results ledger](https://github.com/pleiadian53/ssl-lab/blob/main/examples/perturbation_response/docs/results/index.md): the twenty held-out two-gene combinations of the Norman (2019) Perturb-seq screen, each scored on its 20 most strongly changed genes, with the same data, the same split and the same scores. That is what makes a number here comparable with a number there, and it is this ledger's only scope claim.

---

## The rounds

| Round | Question | Answer | Page |
|---|---|---|---|
| 0 | Is the part of a pair's effect that additivity misses reproducible, or mostly measurement noise? | Mostly signal: in 18 of the 20 pairs at least half of it reappears in independent cells (median 0.82). A perfect predictor would cut the size-aware error from 0.289 to about 0.08. | [the residual ceiling](round-00-residual-ceiling.md) |
| 1 | Does TabPFN-3.5, untuned, predict that residual well enough to beat additivity? | A tie on the 20 test pairs under the pre-registered rule (error 0.260 against 0.289, better on 13 of 20, interval includes zero). The validation pairs (0.221 against 0.318) and the 91 training pairs, cross-validated (0.233 against 0.312), point the same way, and on the latter TabPFN beats every other arm, tuned gradient boosting included. | [an in-context learner for the residual](round-01-tabpfn-residual.md) |
| 2 | Over all 131 pairs, cross-validated, does TabPFN beat additivity and both no-interaction references? | Yes: error 0.227 against 0.309, better on 102 of 131, every simultaneous interval below zero; credited with learning interactions. It also beats tuned ridge and, narrowly, tuned LightGBM (0.011 [0.002, 0.020]). | [confirmation over all 131 pairs](round-02-confirmation-cv.md) |
| 3 | Does the gain survive when no other combination of either gene is in the context? | Yes: error 0.254 against 0.309, better on 94 of 131, every simultaneous interval below zero, still ahead of tuned LightGBM; about two thirds of round 2's gain is a rule that transfers to unseen genes, the rest was familiarity with genes seen in other pairs. | [genes never seen in combination](round-03-leave-genes-out.md) |

Rounds are numbered in the order they were run. Round 0 scored no model: it measured how much of the residual a model could predict at all, which is what round 1's endpoint and references were built on. Round 2 re-ran round 1's protocol, unchanged, with all 131 pairs as the primary readout, and round 3 re-ran it with each pair's genes kept out of its context. Each page also names the route item it answers (round 0 is R1.1, round 1 is R1.3, round 2 is R1.6, round 3 is R1.7), the identifier used in this lab's planning.

---

## The standing verdict

**A residual learner beats additivity on the size of the change.** Across all 131 two-gene combinations, each predicted by models that never saw it, TabPFN-3.5 without tuning lowers the size-aware error from additivity's 0.309 to 0.227 and does better on 102 pairs ([round 2](round-02-confirmation-cv.md)). It beats a globally scaled additivity and an additivity with noise-shrunk singles as well, so under the rule every round follows it is credited with learning interactions, not only with correcting the inputs. Its predicted residuals correlate about 0.66 with the measured ones within a pair. A tuned LightGBM on the same table comes close, and TabPFN is ahead of it by a small margin that holds in every fold assignment.

**On ssl-lab's twenty test pairs alone, the gain is not yet decisive.** There TabPFN scores 0.245 and LightGBM 0.235 against additivity's 0.289, the first learned models in either project below additivity on these pairs, but twenty pairs leave the interval touching zero ([round 1](round-01-tabpfn-residual.md) found the same tie with less context).

**The rule transfers to genes never seen in combination.** Round 2's folds let a held-out pair's genes appear in other pairs of the context, so part of its gain could be familiarity with particular genes. [Round 3](round-03-leave-genes-out.md) removed every other combination of either gene from each pair's context. TabPFN still beats additivity, both no-interaction references and both tuned learners, at 0.254 against 0.309, keeping about two thirds of round 2's gain; seeing one of the pair's genes in other pairs is almost as good as seeing both.

**Two limits travel with the result.** Every pair was scored in round 1 before rounds 2 and 3 used all of them, so these are fixed-protocol tests on the same data, not on unseen data, and a test on an independent combinatorial screen is still to come. Pairs share genes (73 genes make up the 131 pairs), so the intervals are somewhat narrower than they should be. Much of the learnable residual also remains: TabPFN closes about 35% of the distance from additivity to the noise floor when genes recur, about a quarter when they do not.

**Additivity remains the reference** every arm is compared with.

---

## The rules every round follows

- **The protocol is fixed before the run.** Each round's script carries its protocol in its docstring: the arms, the primary comparison, the decision rule and the secondary readouts. It is committed before the full run, and development runs touch no test pair.
- **The primary readout is the twenty test pairs**, so that results stay commensurable with ssl-lab. Other readouts, such as the validation pairs or cross-validation over training pairs, are reported beside it with the same arithmetic and no verdict, unless a round's protocol names one of them as its primary.
- **Comparisons are paired over pairs.** For every pair, each arm's error is averaged over its seeds, and arms are compared through per-pair differences. Intervals come from a joint paired bootstrap over pairs, 10,000 resamples shared by every contrast, with max-t simultaneous 95% intervals over the primary family, as in ssl-lab's round 8. A difference counts only when its simultaneous interval lies wholly on one side of zero.
- **Learning interactions needs more than beating additivity.** A learner can beat plain additivity by shrinking every prediction a little or by discounting noisy single measurements. So every round that scores learners includes two no-interaction references, a globally scaled additivity and an additivity with noise-shrunk singles, and credits a learner with learning interactions only if it beats both as well.
- **Parity is checked, not assumed.** Before scoring, each round checks that additivity reproduces ssl-lab's published mean size-aware error, 0.289096, to $10^{-12}$, and stops if it does not.

---

## The settings every round shares

- **Data.** The processed Norman (2019) cache built by ssl-lab, read-only: 5,000 highly variable genes in log1p of counts per 10,000, 11,726 control cells, 106 single perturbations, and two-gene combinations split into 91 training, 20 validation and 20 test pairs. `de_genes.json` md5 `9844c2c25d2903872e1c0a92c722f105`, `splits.json` md5 `82f1f43952731c64551629701c5dc041`.
- **Effects.** For a perturbation $p$ and a gene $g$, $\Delta_g(p)$ is the mean expression of $g$ in the cells carrying $p$ minus its mean in all control cells. A single gene's effect uses its cells on the training side of the split, as ssl-lab's additive reference does; a pair's measured effect uses all of its cells.
- **The residual.** For a pair $AB$, the additive prediction is $\Delta_g(A) + \Delta_g(B)$ and the residual is $\varepsilon_g(AB) = \Delta_g(AB) - \Delta_g(A) - \Delta_g(B)$.
- **Scored genes.** Each pair's 20 most strongly changed genes by ssl-lab's Wilcoxon criterion, with its two guards (detected in at least 10% of the pair's cells, Benjamini–Hochberg adjusted $p < 0.05$); some training pairs have fewer genes that pass.
- **The primary score** is the size-aware error on the scored genes, $e_p = \lVert \hat\Delta(AB) - \Delta(AB) \rVert / \lVert \Delta(AB) \rVert$, which counts both the shape and the size of the predicted change. For additivity it equals the residual's relative size. $\Delta$-correlation, the correlation of predicted and measured change over the same genes, is reported beside it as the link to ssl-lab's earlier rounds.
- **Where runs happen.** Every full run goes to a RunPod GPU pod; the tasks are in [`ops/`](https://github.com/pleiadian53/tabular-icl-lab/tree/main/ops).

---

## What comes next

Gene descriptors. Round 3 showed both that the model learns a rule from how the singles moved each gene and that knowing which genes are involved is worth something more: about a third of round 2's gain came from having seen the genes in other pairs. Descriptors that say how two genes relate, such as their similarity in control cells, shared pathways or learned gene embeddings, may recover that part for genes never seen in combination, and may also reach interactions like BCL2L11+TGFBR2's, which appear only in the combination. Round 3's arm is the reference for unseen genes and round 2's for recurring ones.
