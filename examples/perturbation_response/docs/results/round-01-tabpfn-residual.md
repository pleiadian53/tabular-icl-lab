# Round 1 — an in-context learner for the interaction residual

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. This is route item R1.3. Previous: [round 0](round-00-residual-ceiling.md).*

**Question.** Given how each of two genes changes expression when perturbed alone, can TabPFN-3.5, shown the 91 training combinations as examples and never tuned, predict how the two combine better than simply adding the two single effects? Adding them is the reference that every learned model in the sibling project ssl-lab has failed to beat on these data.

**Answer, under the rule fixed before the run: a tie on the twenty held-out test pairs.** TabPFN lowers the size-aware error from additivity's 0.289 to 0.260 on average and does better on 13 of the 20 pairs, but with only twenty pairs, one of which it gets badly wrong, the simultaneous interval for the difference includes zero. The two larger readouts point the same way and are clearer. On the twenty validation pairs TabPFN cuts the error from 0.318 to 0.221. Across the 91 training pairs, each predicted once by models fitted without it, it cuts the error from 0.312 to 0.233, and there it beats additivity, both no-interaction references and both tuned learners with every simultaneous interval below zero. Those two readouts were fixed in advance as secondary and carry no verdict, so the result stands as a tie that the rest of the evidence says is underpowered rather than empty.

## What was being predicted

For a perturbation $p$ and a gene $g$, write $\Delta_g(p)$ for the change in mean log-normalized expression (log1p of counts per 10,000) of $g$ between the cells carrying $p$ and the control cells. For two perturbed genes $A$ and $B$, the additive prediction is $\Delta_g(A) + \Delta_g(B)$, and the residual is what it misses:

$$
\varepsilon_g(AB) = \Delta_g(AB) - \Delta_g(A) - \Delta_g(B).
$$

The earlier step, [round 0](round-00-residual-ceiling.md), showed that this residual is mostly a reproducible effect rather than noise: in 18 of the 20 test pairs at least half of it reappears in an independent half of the cells. So there is something to learn. Each model below predicts $\hat\varepsilon_g$, and its prediction of the pair's effect is the additive prediction plus that.

**The score.** Each pair is scored on its 20 most strongly changed genes, the same genes ssl-lab scores, with the size-aware error

$$
e_p = \frac{\lVert \hat\Delta(AB) - \Delta(AB) \rVert}{\lVert \Delta(AB) \rVert},
$$

the size of the prediction's error relative to the size of the pair's measured change, so $e_p = 0.25$ means the prediction is off by a quarter of the change. Additivity's $e_p$ is the residual's own relative size, 0.289 on average. Round 0 showed that even a perfect predictor would score about 0.08, because the pair's measured profile, the target every prediction is compared with, is itself an average over a limited number of cells.

## The table the models see

Each row is one gene in one pair, about 1,800 rows from the 91 training pairs. The label is that gene's measured residual. The seven features say how the two singles moved the gene (the larger effect first, then the smaller, so that naming order does not matter), how precisely each of those effects was measured (its standard error), the gene's mean and spread in control cells, and whether the gene is one of the two perturbed genes itself. Nothing in a row names the gene or the pair, so a model can learn a general rule for how two effects combine, such as "when both singles push a gene strongly the same way, the pair pushes it less than the sum", but nothing specific to particular genes.

## What was compared

| Arm | What it predicts |
|---|---|
| **Additive** | the sum of the two single effects; the reference |
| **Scaled additive** | the sum times one number $\beta$ fitted on the training pairs ($\beta = 0.907$: combinations do about 9% less than the sum). It knows nothing about interactions beyond a uniform shrink. |
| **Shrunk additive** | each single effect first shrunk toward zero according to its own noise, $\Delta \max(0, 1 - \mathrm{SE}^2/\Delta^2)$, then summed. It corrects noisy single measurements and nothing else. |
| **Ridge** | a regularized linear model on the same rows, with all pairwise products of the features; degree and penalty tuned |
| **Gradient boosting** | LightGBM on the same rows; tree size, learning rate, number of trees and leaf size tuned |
| **TabPFN-3.5** | the pretrained in-context learner, given the training rows as context and asked for the test rows, with its default settings and no tuning; three seeds |

The two no-interaction references exist because a learner could beat plain additivity without learning anything about how genes interact, either by learning one global shrink or by discounting noisy single measurements. A model is credited with learning interactions only if it beats both of them as well as additivity. Ridge and boosting were tuned by five-fold cross-validation over the training pairs, keeping each pair's rows together, on the same $e_p$; nothing was tuned on validation or test pairs, and no development run touched a test pair.

**The rule, fixed in the script before the run.** For each pair, take the difference between TabPFN's $e_p$ and each other arm's. Resample the twenty pairs 10,000 times and form max-t simultaneous 95% intervals over the five differences, the procedure of ssl-lab's Round 8. TabPFN beats an arm if the interval lies wholly below zero, loses if wholly above, and ties otherwise.

## Results on the twenty test pairs

| Arm | Mean $e_p$ | Pairs better than additive | Mean $\Delta$-correlation | Interaction recovery |
|---|---|---|---|---|
| Additive | 0.289 | — | 0.961 | — |
| Scaled additive | 0.288 | 9 | 0.961 | 0.03 |
| Shrunk additive | 0.299 | 7 | 0.953 | −0.26 |
| Ridge | 0.293 | 10 | 0.955 | 0.31 |
| Gradient boosting | **0.246** | **16** | 0.969 | 0.57 |
| TabPFN-3.5 | 0.260 | 13 | **0.971** | **0.60** |

*Interaction recovery* is the correlation, across a pair's 20 genes, between the residual a model predicts and the residual measured, averaged over pairs: 0 means the predicted interaction has nothing to do with the real one, 1 means it has exactly the right pattern. $\Delta$-correlation is the standing benchmark from ssl-lab, the correlation between predicted and measured change over the same genes.

| TabPFN minus | Difference in mean $e_p$ | Simultaneous 95% interval | Reading |
|---|---|---|---|
| Additive | −0.029 | [−0.118, +0.059] | tie |
| Scaled additive | −0.028 | [−0.110, +0.054] | tie |
| Shrunk additive | −0.040 | [−0.124, +0.044] | tie |
| Ridge | −0.034 | [−0.118, +0.051] | tie |
| Gradient boosting | +0.014 | [−0.033, +0.061] | tie |

**Verdict: a tie with additivity.** By the route's rule TabPFN is reported as a strong tuning-free reference, and the route is reviewed rather than advanced.

The intervals are wide for two reasons. Twenty pairs is a small sample, and one pair dominates it. On BCL2L11+TGFBR2 TabPFN's error is 0.749 against additivity's 0.383. Its largest residuals sit on genes that neither single moves: CTSL barely changes under either single perturbation (−0.025 and +0.002) but falls by 0.40 under the pair. A table that describes a gene only by how the two singles moved it cannot foresee an effect that appears only in the combination, so this kind of interaction is outside what round 1 could learn. The diagnosis was made after the run, and the pair stays in every number above. Leaving it out, as a description and not a result, TabPFN's mean difference from additivity is −0.050 and it does better on 13 of the remaining 19 pairs; the median difference over all twenty is −0.043.

## The two larger readouts

Both were fixed in the protocol as secondary, with simultaneous intervals and no verdict.

| Readout | Pairs | Additive | Scaled | Shrunk | Ridge | Boosting | TabPFN |
|---|---|---|---|---|---|---|---|
| Test (primary) | 20 | 0.289 | 0.288 | 0.299 | 0.293 | 0.246 | 0.260 |
| Validation | 20 | 0.318 | 0.288 | 0.318 | 0.267 | 0.235 | **0.221** |
| Training pairs, cross-validated | 91 | 0.312 | 0.299 | 0.314 | 0.289 | 0.248 | **0.233** |

**Validation pairs.** These are twenty more combinations no model saw, scored with models fitted exactly as for the test pairs. TabPFN is better than additivity on 15 of the 20, and its simultaneous intervals against additivity (−0.098, interval [−0.193, −0.003]) and against shrunk additivity lie wholly below zero; against scaled additivity, ridge and boosting they include zero.

**Training pairs, cross-validated.** The 91 training pairs were split into five groups; each group was predicted by models fitted on the other four, with ridge and boosting re-tuned inside every fold and $\beta$ refitted, so every training pair is predicted once by a model that never saw it. With 91 pairs this readout has the most statistical power. TabPFN beats every other arm with every simultaneous interval below zero: additivity by 0.079 [−0.108, −0.050], scaled additivity by 0.067 [−0.095, −0.039], shrunk additivity by 0.082 [−0.111, −0.052], ridge by 0.056 [−0.081, −0.031] and boosting by 0.015 [−0.027, −0.003]. Had this been the primary readout, TabPFN would have been credited with learning interactions.

## What the comparisons say

- **The gain is not a correction of the singles' noise.** Shrinking each single effect by its own noise never helps: it is slightly worse than additivity on the test pairs (0.299 against 0.289) and on the training pairs. So the learners' gains do not come from discounting noisy single measurements.
- **Nor is it a uniform shrink.** One global scale helps only a little (0.312 to 0.299 on the training pairs), as round 0 predicted when it found that a global scale explains 8.5% of the residual. The learners gain several times more.
- **TabPFN and the residual it predicts.** Its predicted residuals correlate about 0.6 with the measured ones within a pair, on every readout, so it is predicting the pattern of the interaction and not only its size. The clearest case is CEBPE+RUNX1T1, the pair additivity gets most wrong ($e_p$ 0.541). There the two singles together would push LST1 up by 1.90, but the pair raises it by only 1.15; VIM, GYPA and the haemoglobin gene HBZ show the same buffering. TabPFN brings the error to 0.222.
- **TabPFN without tuning matches a tuned gradient-boosting model.** Boosting is slightly ahead on the test pairs and slightly behind on the other two readouts; only the 91-pair readout separates them, by 0.015 in TabPFN's favour. Ridge, even with pairwise products, gains little, which says the combination rule is not close to linear in these features.
- **Shape barely moves; size does.** Every arm's $\Delta$-correlation stays between 0.95 and 0.98, as round 0 anticipated (additivity already reaches 0.961). The differences are in $e_p$.
- **Still far from the floor.** The best arm's $e_p$ on the test pairs, 0.246, is well above the noise floor of 0.079, so most of the learnable residual is still unexplained.

## Settings and reproducibility

- **Script:** [`02_round1_residual_learners.py`](../../02_round1_residual_learners.py), with the protocol in its docstring, committed before the full run. Table: [`src/tabicllab/residual_table.py`](https://github.com/pleiadian53/tabular-icl-lab/blob/main/src/tabicllab/residual_table.py). Full report, with per-pair numbers for every arm and readout and the tuning grids' scores: [`round-01-tabpfn-residual.json`](reports/round-01-tabpfn-residual.json).
- **Data:** the processed Norman (2019) cache shared with [ssl-lab](https://github.com/pleiadian53/ssl-lab), read-only; `de_genes.json` md5 `9844c2c25d2903872e1c0a92c722f105`, `splits.json` md5 `82f1f43952731c64551629701c5dc041`. 1,792 context rows from 91 training pairs; 400 test rows; 400 validation rows. Before scoring, the run checked that additivity's mean $e_p$ reproduces ssl-lab's published 0.289096 to $10^{-12}$.
- **Models:** TabPFN 9.2.0 (upstream commit `3f2e9f9b`), checkpoint `tabpfn-v3.5-20260909`, `create_default_for_version` with its default number of estimators, seeds 0, 1 and 2, errors averaged over seeds per pair; averaging the three seeds' predictions first gives the same numbers to three decimals. Tuned choices on the full training set: ridge with pairwise products and penalty 316; LightGBM with 16 leaves, learning rate 0.1, 100 trees and at least 10 rows per leaf.
- **Run:** a RunPod A40 pod; the whole round, including the cross-validated readout, took 67 seconds and peaked at 6 GB of memory.
- **Limits.** Twenty test pairs give wide intervals, and pairs share single genes, so they are not fully independent. The cross-validated readout reuses the training pairs that the earlier steps also examined, though no model was tuned on any pair it predicted.
