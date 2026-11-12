# R1.1: the interaction residual is mostly signal

**Question.** On the twenty held-out Norman (2019) gene pairs, is the part of a pair's effect that the additive prediction misses reproducible, or is it mostly measurement noise? If it were noise, no model could beat additivity on these pairs, and the experiment would end here.

**Answer.** It is mostly signal. For 18 of the 20 pairs at least half of the measured residual is reproducible, with a median signal fraction of 0.82, so the pre-registered rule says continue. A perfect predictor of the residual would cut the size-aware error from additivity's 0.289 to about 0.08 on average. That is the room an in-context learner has to work in.

## What was measured

For a perturbation $p$ and gene $g$, write $\Delta_g(p)$ for the change in mean log-normalized expression (log1p of counts per 10,000) of gene $g$ between cells carrying $p$ and control cells. For a pair of perturbed genes $A$ and $B$, the additive prediction is $\Delta_g(A) + \Delta_g(B)$, and the residual is

$$
\varepsilon_g(AB) = \Delta_g(AB) - \Delta_g(A) - \Delta_g(B).
$$

Writing $m_p$ for the measured mean expression of group $p$ and $c$ for the control mean, $\varepsilon = m_{AB} - m_A - m_B + c$. Each of the four means is estimated from a finite number of cells, so each carries its own sampling error, and the measured residual is the true one plus a sum of four errors.

The obvious way to test reproducibility, splitting the pair's cells in half and checking whether the two halves' residuals agree, is biased upwards in two ways:

- **Shared noise.** If only the pair's cells are split, both halves subtract the same single-gene and control means, so they share those means' errors and agree partly because of it.
- **Gene selection.** Each pair is scored on its 20 most strongly changed genes, chosen by a Wilcoxon test on all of its cells. Genes picked because their measured change came out large are genes whose noise tends to push the same way in every subset of those cells (the winner's curse).

So every estimate here splits all four groups (the pair, both singles and the control cells) into halves that share no cells. The primary estimate also re-selects the 20 genes on one half and measures on the other.

## The primary measure and the rule

In each of 50 random splits and in both directions, the pair's top 20 genes were chosen on half 1 with the same criterion the sibling project uses. Then, on half 2, the residual $\varepsilon^{(2)}$ over those genes was compared with its noise variance, computed from the cells:

$$
\mathrm{SE}_g^2 = \frac{s^2_{AB,g}}{n_{AB}} + \frac{s^2_{A,g}}{n_A} + \frac{s^2_{B,g}}{n_B} + \frac{s^2_{c,g}}{n_c},
$$

where $s^2_{p,g}$ is the variance of gene $g$ across the cells of group $p$ in that half and $n_p$ their number. The signal energy is $S = \lVert \varepsilon^{(2)} \rVert^2 - \sum_g \mathrm{SE}_g^2$. At the full sample size the noise variance halves, so the full-sample signal fraction is

$$
\phi = \frac{S}{S + \tfrac{1}{2} \sum_g \mathrm{SE}_g^2},
$$

averaged over the splits and both directions to give each pair's $\phi_p$. The rule, fixed in the script before the full run: continue if at least 10 of the 20 pairs have $\phi_p \ge 0.5$, stop if at most 4 do, and otherwise continue with the reliable pairs reported separately.

## Results

| Measure, median over the 20 pairs | Value |
|---|---|
| **Cross-fitted signal fraction $\phi_p$ (primary)** | **0.82**, with 18 of 20 pairs at or above 0.5 |
| Signal fraction on the fixed scored genes, half sample | 0.87 |
| Signal fraction on the fixed scored genes, full data | 0.88 |
| Split-half correlation of $\varepsilon$, all four groups split, stepped up to full sample | 0.88 |
| The same with only the pair's cells split | 0.94 |
| Additivity's size-aware error $e_p = \lVert \varepsilon \rVert / \lVert \Delta(AB) \rVert$ | 0.272 (mean 0.289) |
| Floor of $e_p$ for any predictor, from the pair's own cell noise | 0.074 (mean 0.079) |

The size-aware error $e_p$ is the norm of the prediction error over the norm of the measured change, on the pair's scored genes. Its floor counts only the noise in the pair's own mean, because a model never sees a held-out pair's cells. It was estimated two ways that agree: from the cell variances (mean 0.079) and from the difference between two halves' means (mean 0.077).

**Both biases are real, and both are modest at the median.** Sharing the singles' and control's noise raises the median split-half reliability from 0.88 to 0.94. The effect is largest where the pair has few cells: for PLK4+STIL (81 cells) it is 0.42 against 0.70, and for PTPN12+ZBTB10 0.63 against 0.88. Gene selection raises the half-sample signal fraction from 0.82, with genes chosen on the other half, to 0.87 on the fixed genes.

**The two pairs below the line are both among the four with fewer than 100 cells.** They are PLK4+STIL (81 cells, $\phi_p$ negative: its re-selected residual is no larger than its noise) and SNAI1+ZBTB10 (97 cells, $\phi_p = 0.25$); the other two small pairs, DLX2+ZBTB10 (87) and FOXL2+HOXB9 (92), clear the line. Every pair with more than 100 cells is above 0.5.

**The residual is mostly pair-specific, not a global shrinkage.** On the 91 training pairs, used here so that no held-out pair is touched, one global scale $\beta$ fitted on the additive direction, $\Delta(AB) \approx \beta (\Delta(A) + \Delta(B))$, comes out at $\beta = 0.91$ (combinations do about 9% less than their singles' sum). Fitted leaving each pair out, it explains only 8.5% of the residual energy. A model that only learned "shrink the sum a little" would therefore gain little; most of the residual differs from pair to pair.

| Pair | Cells | $\phi_p$ | Split-half, all split | Split-half, pair only | $e_p$ (additive) | $e_p$ floor |
|---|---|---|---|---|---|---|
| PLK4+STIL | 81 | −2.36 | 0.42 | 0.70 | 0.341 | 0.202 |
| SNAI1+ZBTB10 | 97 | 0.25 | 0.58 | 0.74 | 0.202 | 0.108 |
| PTPN12+ZBTB10 | 259 | 0.56 | 0.63 | 0.88 | 0.202 | 0.072 |
| FOXL2+HOXB9 | 92 | 0.64 | 0.81 | 0.85 | 0.272 | 0.120 |
| IGDCC3+ZBTB25 | 141 | 0.71 | 0.80 | 0.85 | 0.172 | 0.072 |
| CBFA2T3+POU3F2 | 214 | 0.74 | 0.85 | 0.91 | 0.322 | 0.088 |
| BCL2L11+TGFBR2 | 459 | 0.77 | 0.85 | 0.94 | 0.383 | 0.102 |
| DLX2+ZBTB10 | 87 | 0.78 | 0.94 | 0.96 | 0.377 | 0.100 |
| FOXA1+FOXF1 | 267 | 0.79 | 0.87 | 0.92 | 0.269 | 0.089 |
| SAMD1+ZBTB1 | 181 | 0.79 | 0.84 | 0.90 | 0.203 | 0.075 |
| KLF1+MAP2K6 | 260 | 0.85 | 0.87 | 0.91 | 0.193 | 0.064 |
| CBL+TGFBR2 | 186 | 0.87 | 0.92 | 0.94 | 0.299 | 0.081 |
| FOSB+UBASH3B | 422 | 0.89 | 0.91 | 0.95 | 0.373 | 0.086 |
| CEBPE+FOSB | 150 | 0.90 | 0.91 | 0.94 | 0.262 | 0.070 |
| UBASH3A+UBASH3B | 540 | 0.90 | 0.93 | 0.97 | 0.329 | 0.065 |
| AHR+KLF1 | 476 | 0.91 | 0.88 | 0.95 | 0.216 | 0.044 |
| OSR2+UBASH3B | 775 | 0.93 | 0.93 | 0.97 | 0.285 | 0.048 |
| CBL+CNN1 | 345 | 0.96 | 0.96 | 0.98 | 0.273 | 0.040 |
| TBX2+TBX3 | 1,145 | 0.97 | 0.98 | 0.99 | 0.267 | 0.023 |
| CEBPE+RUNX1T1 | 1,198 | 0.99 | 0.99 | 1.00 | 0.541 | 0.034 |

## What it means for the next round

- **There is room to beat additivity, and it is in the size of the change.** On $\Delta$-correlation additivity already scores 0.961, so at most 0.039 is left. On $e_p$ the residual that a model could in principle predict is about three quarters of additivity's error (0.289 against a floor near 0.08). That makes the size-aware error the endpoint where an in-context learner can show something, and it stays the primary endpoint.
- **The no-interaction references remain necessary.** A global scale explains little, but a learner can also gain by shrinking noisy single-gene estimates. The scaled-additive and shrunk-additive references stay in the next round's protocol, so a win is credited to learning interactions only when it beats them too.
- **Small pairs are noisy.** Pairs with fewer than 100 cells have $e_p$ floors of 0.10 to 0.20, against a median of 0.074, and the next round reports per-pair results so they can be read separately.

## Settings and reproducibility

- **Script:** [`../01_residual_ceiling.py`](../01_residual_ceiling.py), protocol in its docstring, committed before the full run. Full report: [`r1_1_residual_ceiling.json`](r1_1_residual_ceiling.json).
- **Data:** the processed Norman cache shared with [ssl-lab](https://github.com/pleiadian53/ssl-lab), read-only; `de_genes.json` md5 `9844c2c25d2903872e1c0a92c722f105`, `splits.json` md5 `82f1f43952731c64551629701c5dc041`. Singles use their training cells, control uses all 11,726 control cells, and pairs use all their cells.
- **Gene selection** reimplements ssl-lab's Wilcoxon criterion with its two guards (detected in at least 10% of the pair's cells, Benjamini–Hochberg adjusted $p < 0.05$) and reproduces its stored top-20 lists exactly on all 40 pairs checked.
- **Run:** 50 random splits, seed 0; on a RunPod A40 pod, about 7 minutes. The report's `peak_rss_gb` reads 0.00 because the run predates a fix to how peak memory is reported on Linux; nothing else is affected.
- **One detail of the rule.** The protocol says $\phi$ is clipped to $[0, 1]$; the script averaged the unclipped values per pair. Clipping each repeat could only raise a pair's $\phi_p$, so the count of 18 is a lower bound and the decision is unchanged.
- **Limit.** The two halves of a group come from one experiment and share its guides, sequencing lanes and processing. Reproducibility across independent experiments is not testable with these data.
