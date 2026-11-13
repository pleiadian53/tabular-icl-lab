# Round 2 — confirmation over all 131 pairs: the in-context learner beats additivity

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. This is route item R1.6. Previous: [round 1](round-01-tabpfn-residual.md).*

**Question.** Round 1 found a tie on its twenty test pairs but a clear gain on its larger secondary readouts. With every one of the 131 two-gene combinations predicted by models that never saw it, and all 131 as the primary readout, does TabPFN-3.5, untuned, predict the genetic-interaction residual well enough to beat additivity, and does the gain survive the two references that cannot learn interactions?

**Answer: yes, under the rule fixed before the run.** Across the 131 pairs TabPFN lowers the size-aware error from additivity's 0.309 to 0.227 and does better on 102 of them. The simultaneous intervals for TabPFN minus additivity, minus the globally scaled additivity and minus the noise-shrunk additivity all lie wholly below zero, which is the protocol's condition for crediting it with learning interactions. It also beats both tuned learners: a ridge regression by 0.057 and a tuned LightGBM by a small margin, 0.011, interval [−0.020, −0.002].

## What this round can and cannot confirm

Every pair here had already been scored once in [round 1](round-01-tabpfn-residual.md): the twenty test pairs as its primary readout, the twenty validation pairs and the 91 training pairs as secondary ones. So this round does not test the model on data no one has looked at. What it does test is whether round 1's pattern holds when the whole dataset is the readout, under a protocol in which nothing was changed after round 1. The table, the six arms, the tuning grids and the decision rule are round 1's, and the code that fitted and scored them is round 1's script, loaded unchanged. A test on independent data needs another combinatorial screen.

## How it was run

The notation is round 1's. For a perturbation $p$ and a gene $g$, $\Delta_g(p)$ is the change in mean log-normalized expression of $g$ between the cells carrying $p$ and the control cells. For a pair $AB$, the residual is

$$
\varepsilon_g(AB) = \Delta_g(AB) - \Delta_g(A) - \Delta_g(B),
$$

the part of the pair's effect that the additive prediction $\Delta_g(A) + \Delta_g(B)$ misses. Each model predicts it from a table with one row per gene of each pair, whose seven features say how the two singles moved that gene, how precisely those effects were measured and how the gene behaves in control cells; nothing in a row names the gene or the pair. Each pair is scored on its 20 most strongly changed genes with the size-aware error

$$
e_p = \frac{\lVert \hat\Delta(AB) - \Delta(AB) \rVert}{\lVert \Delta(AB) \rVert},
$$

the size of the prediction's error relative to the size of the measured change. Additivity's $e_p$ is the residual's own relative size.

The 131 pairs were split into five groups, each pair's rows kept together. Each group was predicted by every arm fitted on the other four: the global scale $\beta$ refitted, ridge and LightGBM re-tuned by an inner cross-validation over the context pairs, TabPFN given those pairs as its context with seeds 0, 1 and 2. The whole procedure was repeated with three different random assignments of pairs to groups, so every pair was predicted three times, and its error is the average over those predictions (and over TabPFN's seeds). The arms are round 1's:

| Arm | What it predicts |
|---|---|
| **Additive** | the sum of the two single effects; the reference |
| **Scaled additive** | that sum times one number fitted on the context pairs ($\beta$ between 0.88 and 0.91 across folds); no interactions, only a uniform shrink |
| **Shrunk additive** | each single effect first shrunk toward zero by its own noise, $\Delta \max(0, 1 - \mathrm{SE}^2/\Delta^2)$, then summed; corrects noisy measurements and nothing else |
| **Ridge** | a regularized linear model with all pairwise products of the features, tuned |
| **LightGBM** | gradient-boosted trees, tuned |
| **TabPFN-3.5** | the pretrained in-context learner, default settings, no tuning |

**The rule.** For each pair, take the difference between TabPFN's $e_p$ and each other arm's. Resample the 131 pairs 10,000 times and form max-t simultaneous 95% intervals over the five differences. TabPFN beats an arm if its interval lies wholly below zero. It is credited with learning interactions only if it beats additivity and both no-interaction references.

## Results over all 131 pairs

| Arm | Mean $e_p$ | Pairs better than additive | Typical error relative to additive | Mean $\Delta$-correlation | Interaction recovery |
|---|---|---|---|---|---|
| Additive | 0.309 | — | 1.00 | 0.967 | — |
| Scaled additive | 0.296 | 63 | 0.97 | 0.967 | 0.19 |
| Shrunk additive | 0.312 | 55 | 1.01 | 0.965 | −0.09 |
| Ridge | 0.285 | 80 | 0.92 | 0.968 | 0.38 |
| LightGBM | 0.238 | 100 | 0.77 | 0.977 | 0.62 |
| **TabPFN-3.5** | **0.227** | **102** | **0.72** | **0.980** | **0.66** |

The *typical error relative to additive* is the geometric mean over pairs of the arm's $e_p$ divided by additivity's, so 0.72 means TabPFN's error on a typical pair is 28% smaller than additivity's. *Interaction recovery* is the correlation, across a pair's genes, between the residual a model predicts and the residual measured, averaged over pairs: 0 means the predicted interaction has nothing to do with the real one, 1 means it has exactly its pattern. $\Delta$-correlation is ssl-lab's standing benchmark, the correlation of predicted and measured change.

| TabPFN minus | Difference in mean $e_p$ | Simultaneous 95% interval | Reading |
|---|---|---|---|
| Additive | −0.082 | [−0.108, −0.057] | TabPFN better |
| Scaled additive | −0.069 | [−0.092, −0.046] | TabPFN better |
| Shrunk additive | −0.085 | [−0.111, −0.060] | TabPFN better |
| Ridge | −0.057 | [−0.077, −0.038] | TabPFN better |
| LightGBM | −0.011 | [−0.020, −0.002] | TabPFN better |

**Verdict: TabPFN learns interactions.** It beats additivity and both no-interaction references, every interval wholly below zero, and it also beats both tuned learners. The result does not depend on how the pairs were grouped: in each of the three fold assignments, scored alone, TabPFN's advantage over additivity is −0.079, −0.083 and −0.083, and over LightGBM −0.008, −0.010 and −0.016.

## What the comparisons say

- **The gain is an interaction, not a correction of the inputs.** Shrinking each single effect by its noise makes things slightly worse (0.312 against 0.309, the interval of that difference wholly above zero), so the learners are not winning by discounting noisy single measurements. One global scale helps only a little, 0.013, with an interval that includes zero. TabPFN gains six times that, and its predicted residuals correlate 0.66 with the measured ones within a pair, so it predicts which genes depart from additivity and in which direction, not only by how much.
- **The combination rule is not linear.** Ridge, even with every pairwise product of the features, gains 0.025 over additivity, under a third of TabPFN's 0.082 and of LightGBM's 0.071.
- **An untuned in-context learner edges out a tuned one.** TabPFN's margin over LightGBM is small, about 5% of LightGBM's error, but it holds in every fold assignment and in every stratum except the twenty test pairs, where LightGBM is slightly ahead. LightGBM reaches its score after a grid search of 72 settings, re-run in every fold; TabPFN uses its defaults.
- **Much of the learnable residual is still unexplained.** Noise in each pair's own cells sets a floor of 0.073 on $e_p$ (round 0's measurement, recomputed here over all 131 pairs). TabPFN closes about 35% of the distance from additivity to that floor.

## Where it helps most, and where it fails

The largest gains are on combinations of CEBP transcription factors. On CEBPA+CEBPE additivity is off by 73% of the change and TabPFN by 23%; on CEBPA+CEBPB the errors are 0.57 and 0.21; on CEBPE+RUNX1T1, round 1's clearest case of buffering, 0.54 and 0.19. CEBPE appears in eleven of the 131 pairs, so when one of them is predicted, others that share CEBPE are in the context. That is legitimate under this split, as it is in ssl-lab's, where the test pairs' singles also occur in training pairs. It does mean that part of the gain may come from having seen a gene's other combinations, through the values of its single effect, and not only from a rule that holds for any two genes. Holding out every pair that contains a given gene would separate the two, and is a natural next check.

TabPFN is worse than additivity on 29 of the 131 pairs. The worst is BCL2L11+TGFBR2 again, 0.611 against 0.383, the pair whose residual sits on genes neither single moves, which round 1 traced to a limit of the table: an effect that appears only in the combination cannot be read off how the singles moved each gene. The next worst are MAP2K3+SLC38A2 (0.419 against 0.233), CNN1+ETS2 (0.355 against 0.210) and KIF18B+KIF2C (0.731 against 0.590).

## The strata

These are descriptive, with percentile intervals.

| Stratum | Pairs | Additive | LightGBM | TabPFN | TabPFN minus additive |
|---|---|---|---|---|---|
| All pairs (primary) | 131 | 0.309 | 0.238 | 0.227 | −0.082 |
| ssl-lab's test pairs | 20 | 0.289 | 0.235 | 0.245 | −0.044 [−0.094, +0.006] |
| Validation pairs | 20 | 0.318 | 0.231 | 0.216 | −0.102 [−0.170, −0.040] |
| Training pairs | 91 | 0.312 | 0.241 | 0.226 | −0.086 [−0.109, −0.063] |
| At least 100 cells | 118 | 0.307 | 0.236 | 0.225 | −0.083 [−0.104, −0.062] |
| Fewer than 100 cells | 13 | 0.327 | 0.258 | 0.252 | −0.075 |

On ssl-lab's twenty test pairs TabPFN now scores 0.245, against 0.260 in round 1, where it had 91 pairs of context instead of about 105. Twenty pairs on their own still leave the interval touching zero, which is why round 1 could not decide and this round could. These are the pairs every ssl-lab result is reported on. On them the two residual learners score 0.235 and 0.245 against additivity's 0.289, the first learned models in either project to score below additivity on the size of the change there, though not by a margin twenty pairs alone can confirm. The best learned model of the full expression profile in ssl-lab, the conditioned NB-VAE, scores 0.455 on the same pairs.

## Settings and reproducibility

- **Script:** [`03_round2_confirmation_cv.py`](../../03_round2_confirmation_cv.py), protocol in its docstring, committed before the full run. It loads round 1's script, [`02_round1_residual_learners.py`](../../02_round1_residual_learners.py), for its arms, tuning and scoring, so the two rounds run identical code. Full report, with per-pair numbers for every arm, the settings chosen in every fold and the per-repeat results: [`round-02-confirmation-cv.json`](reports/round-02-confirmation-cv.json).
- **Data:** the processed Norman (2019) cache shared with [ssl-lab](https://github.com/pleiadian53/ssl-lab), read-only, with the fingerprints listed in [the ledger](index.md). 131 pairs, 2,592 rows. Before scoring, the run checked that additivity's mean $e_p$ on the twenty test pairs reproduces ssl-lab's published 0.289096 to $10^{-12}$.
- **Cross-validation:** 5 folds by pair, fold assignments drawn with seeds 0, 1 and 2; inner tuning folds seed 0. In 13 of the 15 outer folds ridge chose pairwise products with penalty 316; LightGBM most often chose 16 leaves, learning rate 0.03 and 300 trees.
- **Models:** TabPFN 9.2.0 (upstream commit `3f2e9f9b`), checkpoint `tabpfn-v3.5-20260909`, version defaults, seeds 0 to 2; averaging the seeds' predictions first gives the same scores to three decimals.
- **Run:** a RunPod A40 pod, 159 seconds for the whole round, 6 GB peak memory.
- **Limits.** The bootstrap treats pairs as independent, but 73 genes make up the 131 pairs, so pairs that share a gene are not independent and the intervals are somewhat narrower than they should be. Every pair had been scored once before, in round 1. And a held-out pair's single genes occur in other pairs of its context, which this round does not separate from a general combination rule.
