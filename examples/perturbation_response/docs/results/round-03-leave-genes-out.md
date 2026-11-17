# Round 3 — genes never seen in combination: the learned rule transfers

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. This is route item R1.7. Previous: [round 2](round-02-confirmation-cv.md).*

**Question.** In [round 2](round-02-confirmation-cv.md) TabPFN-3.5 beat additivity and both no-interaction references across all 131 two-gene combinations, but a held-out pair's two genes usually appeared in other pairs of its context. Does the gain survive when each pair is predicted by a model that has never seen any other combination of either of its genes?

**Answer: yes, with about two thirds of the gain kept.** Predicted with every pair that shares a gene removed from the context, TabPFN lowers the size-aware error from additivity's 0.309 to 0.254 and does better on 94 of the 131 pairs. Its simultaneous intervals against additivity, the globally scaled additivity and the noise-shrunk additivity all lie wholly below zero, so under the rule fixed before the run it is credited with a combination rule that transfers to genes it has never seen combined. It also still beats a tuned ridge and a tuned LightGBM. Familiarity with the genes did contribute, though: round 2's gain over additivity was 0.082, and 0.055 of it, 67%, remains.

## Why this round was needed

Each row of the table describes one gene in one pair by how the two single perturbations moved it, how precisely those moves were measured, and how the gene behaves in control cells. Nothing in a row names a gene. But a gene's single effect is the same set of numbers in every pair that contains it. In round 2's folds, drawn at random over pairs, a held-out pair such as CEBPA+CEBPE usually had other CEBPE pairs in its context, and a model could recognize CEBPE from those numbers and reuse what it had learned about CEBPE's other combinations. CEBPE is in eleven of the 131 pairs, and round 2's largest gains were on CEBP pairs, so this was a live possibility. That would be familiarity with particular genes, not a rule that holds for any two genes, and the two no-interaction references cannot detect it.

## How it was run

The notation is that of the earlier rounds. For a perturbation $p$ and a gene $g$, $\Delta_g(p)$ is the change in mean log-normalized expression of $g$ between the cells carrying $p$ and the control cells. For a pair $AB$, the residual is $\varepsilon_g(AB) = \Delta_g(AB) - \Delta_g(A) - \Delta_g(B)$, the part of the pair's effect that the additive prediction misses, and each pair is scored on its 20 most strongly changed genes with the size-aware error

$$
e_p = \frac{\lVert \hat\Delta(AB) - \Delta(AB) \rVert}{\lVert \Delta(AB) \rVert},
$$

the size of the prediction's error relative to the size of the measured change.

**The primary readout, leave both genes out.** For each of the 131 pairs, the context was every other pair that contains neither of its two genes. For CEBPA+CEBPE, every pair containing CEBPA or CEBPE was removed. Every arm was fitted on that context alone: the global scale refitted, ridge and LightGBM re-tuned by an inner cross-validation, TabPFN given it as its context with three seeds. Then the arm predicted that one pair. Removing a pair's relatives left between 109 and 130 context pairs, a median of 120, against about 105 in round 2's folds, so no pair was predicted from fewer examples than in round 2.

**A readout beside it, leave one gene out.** For each of the 73 genes, every pair containing it was held out together and predicted from the rest, so the context could still contain other pairs of the pair's second gene. Each pair was predicted twice, once with each of its genes held out, and the two errors were averaged.

The arms, grids, table and rule are round 2's, and the code that fitted and scored them is round 1's script, loaded unchanged.

## Results over all 131 pairs, leaving both genes out

| Arm | Mean $e_p$ | Pairs better than additive | Typical error relative to additive | Interaction recovery |
|---|---|---|---|---|
| Additive | 0.309 | — | 1.00 | — |
| Scaled additive | 0.302 | 62 | 0.98 | 0.19 |
| Shrunk additive | 0.312 | 55 | 1.01 | −0.09 |
| Ridge | 0.296 | 69 | 0.95 | 0.33 |
| LightGBM | 0.268 | 91 | 0.87 | 0.54 |
| **TabPFN-3.5** | **0.254** | **94** | **0.82** | **0.59** |

The *typical error relative to additive* is the geometric mean over pairs of an arm's $e_p$ divided by additivity's. *Interaction recovery* is the correlation, across a pair's genes, between the residual a model predicts and the residual measured, averaged over pairs.

| TabPFN minus | Difference in mean $e_p$ | Simultaneous 95% interval | Reading |
|---|---|---|---|
| Additive | −0.055 | [−0.077, −0.034] | TabPFN better |
| Scaled additive | −0.048 | [−0.069, −0.027] | TabPFN better |
| Shrunk additive | −0.058 | [−0.080, −0.037] | TabPFN better |
| Ridge | −0.042 | [−0.060, −0.023] | TabPFN better |
| LightGBM | −0.014 | [−0.021, −0.006] | TabPFN better |

**Verdict: the combination rule transfers to genes never seen in combination.** TabPFN beats additivity and both no-interaction references with every interval wholly below zero, and it beats both tuned learners.

## How much of round 2's gain was familiarity

| Readout | What the context may contain | TabPFN | LightGBM | Additive |
|---|---|---|---|---|
| Round 2, random folds | other pairs of both genes | 0.227 | 0.238 | 0.309 |
| Round 3, leave one gene out | other pairs of the pair's second gene | 0.234 | 0.248 | 0.309 |
| Round 3, leave both genes out | no other pair of either gene | 0.254 | 0.268 | 0.309 |

Three things follow.

- **Most of the gain is a rule that transfers.** With no other combination of either gene in view, TabPFN keeps 0.055 of its 0.082 gain over additivity. Per pair, its error rose by 0.027 on average from round 2, interval [0.016, 0.037], so familiarity was real but accounted for about a third of the gain.
- **Seeing one of the two genes elsewhere is almost as good as seeing both.** Leaving one gene out keeps 91% of round 2's gain. The loss comes mainly from pairs whose two genes are both new to the model.
- **TabPFN transfers slightly better than LightGBM.** LightGBM keeps 59% of its round 2 gain, TabPFN 67%, and TabPFN's lead over LightGBM is between 0.011 and 0.014 in every readout.

The CEBP pairs, the case that prompted this round, keep most of their gain. Across the 21 pairs containing CEBPA, CEBPB or CEBPE, additivity's mean error is 0.399; TabPFN's was 0.238 in round 2 and is 0.274 here, so 78% of the gain survives without any other CEBP combination in view. Some individual pairs did lean on familiarity. CEBPE+ZC3HAV1 went from 0.411 to 0.631, still better than additivity's 0.856, and CDKN1A+CDKN1C from 0.189 to 0.380, close to additivity's 0.494, once the other pairs of these two genes were out of view. Others barely moved. CEBPA+CEBPE went from 0.226 to 0.312, against additivity's 0.733, so the buffering seen across the CEBP factors is something the model predicts from how the singles moved each gene, not something it remembers.

## The strata

These are descriptive, with percentile intervals.

| Stratum | Pairs | Additive | LightGBM | TabPFN | TabPFN minus additive |
|---|---|---|---|---|---|
| All pairs (primary) | 131 | 0.309 | 0.268 | 0.254 | −0.055 |
| ssl-lab's test pairs | 20 | 0.289 | 0.253 | 0.253 | −0.036 [−0.076, +0.010] |
| Validation pairs | 20 | 0.318 | 0.270 | 0.252 | −0.067 [−0.123, −0.015] |
| Training pairs | 91 | 0.312 | 0.270 | 0.255 | −0.057 [−0.076, −0.038] |
| At least 100 cells | 118 | 0.307 | 0.266 | 0.253 | −0.055 [−0.072, −0.037] |
| Pairs with a CEBP gene | 21 | 0.399 | 0.301 | 0.274 | −0.125 [−0.180, −0.073] |

On ssl-lab's twenty test pairs alone the gain is again not decisive, as in rounds 1 and 2, and there TabPFN and LightGBM tie. BCL2L11+TGFBR2 remains the pair the model gets most wrong, 0.639 against additivity's 0.383, for the reason round 1 found: its residual sits on genes neither single moves.

## What this changes

Round 2's credit stands, now on a stricter footing. An in-context learner, given examples of how pairs of other genes combine and nothing that names any gene, predicts a real part of how two unseen genes combine. About two thirds of round 2's gain is that rule; the rest came from having seen the genes in other pairs. For the next step, gene descriptors, this round sets two references: round 2's arm for what the model can do when genes recur, and this round's arm for what it can do when they do not. A descriptor that tells the model which genes are related should recover part of the difference between them.

## Settings and reproducibility

- **Script:** [`04_round3_leave_genes_out.py`](../../04_round3_leave_genes_out.py), protocol in its docstring, committed before the full run, loading round 1's [`02_round1_residual_learners.py`](../../02_round1_residual_learners.py) for its arms, tuning and scoring. Full report, with per-pair numbers and context sizes for both readouts and the retention against round 2: [`round-03-leave-genes-out.json`](reports/round-03-leave-genes-out.json).
- **Data:** the processed Norman (2019) cache shared with [ssl-lab](https://github.com/pleiadian53/ssl-lab), read-only, with the fingerprints in [the ledger](index.md). 131 pairs over 73 genes, 2,592 rows. The run checked that additivity's mean $e_p$ on the twenty test pairs reproduces ssl-lab's published 0.289096 to $10^{-12}$.
- **Models:** TabPFN 9.2.0 (upstream commit `3f2e9f9b`), checkpoint `tabpfn-v3.5-20260909`, version defaults, seeds 0 to 2; ridge and LightGBM re-tuned in all 204 contexts over round 1's grids.
- **Run:** a RunPod A40 pod, 42 minutes, most of it re-tuning the grids on the pod's single CPU; 6 GB peak memory.
- **Limits.** Inner tuning folds were drawn at random over the context pairs, not by gene. The bootstrap treats pairs as independent, but pairs share genes. Every pair had been scored before, in rounds 1 and 2, under the same table and arms.
