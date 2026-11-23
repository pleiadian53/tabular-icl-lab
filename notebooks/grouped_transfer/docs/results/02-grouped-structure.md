# 02 · The group structure: twelve unseen geometries, references that use the grouping, and bands

*Part of [the results ledger](index.md), which holds what the characterizations add up to, the rules they follow and the settings they share. Notebook: [`02_grouped_structure.ipynb`](../../02_grouped_structure.ipynb). Report: [`02-grouped-structure.json`](reports/02-grouped-structure.json), from [`02_grouped_structure_report.py`](../../02_grouped_structure_report.py).*

**Question.** When each of the twelve building geometries is held out in turn, how do TabPFN, gradient boosting and two models built around the grouping compare; what does a random split of designs hide; and do 80% prediction bands cover 80% of the designs of a geometry the model has never seen?

**Answer.** The grouping dominates the table: 91% of the heating-load variance lies between geometries, so the 768 designs carry about 13 independent observations' worth of information about a geometry's level. Held out one geometry at a time, TabPFN-3.5 has an MAE of 2.92 averaged over three seeds, level with a linear mixed-effects model (2.68) and a two-stage model (2.80), ahead of TabPFN-2 (3.59) and far ahead of XGBoost (4.97) in point value, but no difference has a simultaneous interval that excludes zero over twelve geometries. A random split of designs scores the same models at 0.31 and 0.22, nine and twenty-three times lower, and ranks XGBoost first. Every model's overall band coverage is near nominal, 0.76 to 0.83, while two neighbouring two-storey geometries are covered barely or not at all. **Transfer to a new geometry is limited by the geometries whose level the shape does not predict, and averages over designs hide both that limit and the failure of the bands.**

## What was measured

For design $i$ with true heating load $y_i$, predicted mean $\hat y_i$ and band from $q_{0.1,i}$ to $q_{0.9,i}$, the 10% and 90% quantiles of the predictive distribution:

- **Leave one geometry out.** For each geometry $g$, a model fitted on the other eleven geometries, 704 designs, predicts the 64 designs of $g$. Every design is predicted once.
- **Per-geometry MAE**, the mean of $|\hat y_i - y_i|$ over the designs of one geometry, and its **mean signed error**, the mean of $\hat y_i - y_i$. Overall MAE and $R^2$ over all 768 designs.
- **Band coverage and width** per geometry: the share of designs with $q_{0.1,i} \le y_i \le q_{0.9,i}$, nominally 0.80, and the mean of $q_{0.9,i} - q_{0.1,i}$.
- **Variance structure.** The share of variance explained by geometry means, and by geometry means plus one shift per glazing setup; the intraclass correlation $\rho$; and the effective sample size $n / (1 + (m - 1)\rho)$ for $n = 768$ designs in groups of $m = 64$.
- **Inference over geometries.** Per-geometry MAE differences between arms, with max-t simultaneous 95% intervals from one bootstrap over the twelve geometries, 10,000 resamples. Twelve units make these intervals coarse.

## What was compared

- **TabPFN-3.5**, default constructor, the eight design columns, seeds 0, 1 and 2.
- **TabPFN-3.5, short shape description**: relative compactness and overall height in place of the five shape columns, with orientation and the two glazing columns.
- **TabPFN-2**, eight columns, three seeds.
- **XGBoost**, default settings, one thread, eight columns.
- **Mixed effects**: a linear model with a random intercept per geometry, $y_{gi} = x_{gi}^\top \beta + u_g + \varepsilon_{gi}$ with $u_g \sim \mathcal{N}(0, \tau^2)$ and $\varepsilon_{gi} \sim \mathcal{N}(0, \sigma^2)$, where $x_{gi}$ holds compactness, height, glazing area, a glazing-by-height term, glazing distribution and orientation. For an unseen geometry the prediction is $x_{gi}^\top \beta$ and the band uses the predictive variance $\tau^2 + \sigma^2$.
- **Two-stage**: a Gaussian process over the five shape columns, fitted to one mean per context geometry, plus the context's average within-geometry shift for the design's height and glazing setup.
- **TabPFN-3.5's band after group calibration**: conformalized quantile regression with geometries as the unit, each geometry's band shifted by the 80% quantile of the other geometries' conformity scores.
- **Random split**: 192 random designs held out, the other 576 as context, three split seeds; TabPFN-3.5 and XGBoost.

## Results

### The variance structure

| Quantity | Value |
|---|---|
| Variance explained by geometry means | 0.908 |
| … plus one shift per glazing setup | 0.990 |
| Intraclass correlation $\rho$ | 0.914 |
| Effective sample size, of 768 designs | 13.1 |
| Mixed model on all twelve geometries: $\tau$, $\sigma$ | 3.21, 0.65 |

### Leave one geometry out, all 768 designs

| Arm | MAE | Range over seeds | $R^2$ |
|---|---|---|---|
| Mixed effects | **2.68** | | 0.870 |
| Two-stage | 2.80 | | 0.816 |
| TabPFN-3.5 | 2.92 | 2.78 to 3.03 | 0.836 |
| TabPFN-3.5, short shape | 3.20 | 3.13 to 3.26 | 0.834 |
| TabPFN-2 | 3.59 | 3.47 to 3.80 | 0.754 |
| XGBoost | 4.97 | | 0.336 |

For contrast, on a random split of 192 designs: TabPFN-3.5 0.31 ($R^2$ 0.998), XGBoost 0.22 ($R^2$ 0.999), averaged over three split seeds.

### Per geometry, MAE (TabPFN averaged over seeds)

| Geometry | TabPFN-3.5 | TabPFN-3.5, short | TabPFN-2 | XGBoost | Mixed | Two-stage |
|---|---|---|---|---|---|---|
| 0.98 / 7.0 m | 1.06 | 2.86 | 1.81 | 3.96 | 1.97 | 2.13 |
| 0.90 / 7.0 m | 4.31 | 2.76 | 2.64 | 3.10 | 2.60 | 1.25 |
| 0.86 / 7.0 m | 2.53 | 1.22 | 4.12 | 2.99 | 2.93 | 0.76 |
| 0.82 / 7.0 m | 5.76 | 7.19 | 7.93 | 13.03 | 8.42 | 7.96 |
| 0.79 / 7.0 m | 10.67 | 9.56 | 12.74 | 3.16 | 7.00 | 11.39 |
| 0.76 / 7.0 m | 2.52 | 3.67 | 3.58 | 23.72 | 1.97 | 0.66 |
| 0.74 / 3.5 m | 1.07 | 1.21 | 2.11 | 0.39 | 1.00 | 1.08 |
| 0.71 / 3.5 m | 0.56 | 1.51 | 0.32 | 0.45 | 0.44 | 0.92 |
| 0.69 / 3.5 m | 1.36 | 1.55 | 0.45 | 0.44 | 0.66 | 0.81 |
| 0.66 / 3.5 m | 1.54 | 1.73 | 1.72 | 3.78 | 1.33 | 0.97 |
| 0.64 / 3.5 m | 0.91 | 2.75 | 2.57 | 2.33 | 2.51 | 3.12 |
| 0.62 / 3.5 m | 2.76 | 2.42 | 3.14 | 2.33 | 1.34 | 2.54 |

Geometries are named by relative compactness and overall height. The 0.82 geometry is predicted too high and the 0.79 geometry too low by every arm.

### Bands, nominal 80% (TabPFN seed 0)

| Band | Overall coverage | Mean width | Geometries covered at 0.80 or more | 0.82 / 7.0 m | 0.79 / 7.0 m |
|---|---|---|---|---|---|
| TabPFN-3.5 | 0.82 | 10.9 | 9 of 12 | 0.44 | 0.00 |
| TabPFN-3.5, group-calibrated | 0.76 | 10.4 | 9 of 12 | 0.28 | 0.00 |
| Mixed effects | 0.83 | 8.3 | 10 of 12 | 0.00 | 0.00 |
| Two-stage | 0.83 | 9.5 | 10 of 12 | 0.00 | 0.00 |

### Differences over geometries (per-geometry MAE, first arm minus second)

The critical value of the max-t family is 3.72.

| Contrast | Difference | Simultaneous 95% interval |
|---|---|---|
| TabPFN-3.5 − XGBoost | −2.05 | −9.48 to 5.38 |
| TabPFN-3.5 − mixed effects | +0.24 | −1.51 to 1.99 |
| TabPFN-3.5 − two-stage | +0.12 | −1.59 to 1.84 |
| TabPFN-3.5 − TabPFN-2 | −0.67 | −1.95 to 0.60 |
| TabPFN-3.5, short − TabPFN-3.5 | +0.28 | −0.99 to 1.56 |

## What this says

**A random split answers a different question and picks a different winner.** On random designs every geometry is in the context, and both models fill in missing glazing setups almost perfectly; XGBoost is slightly better. On unseen geometries the errors are nine (TabPFN) and twenty-three (XGBoost) times larger, and XGBoost is the worst arm. The three geometries held out in [01](01-unseen-geometry-transfer.md) are among the easier ones: over all twelve, TabPFN-3.5's MAE is about three times what it was there.

**The hard geometries are the ones whose level the shape does not predict.** Two neighbouring two-storey geometries, compactness 0.82 and 0.79, differ by 13 units in mean heating load. Whichever is held out, every arm carries over something like the other's level: the 0.82 geometry is predicted too high by all six arms, the 0.79 geometry too low. In the mixed model their offsets $u_g$ are about −2.1 and +1.6 times $\tau$. Nothing in the five shape columns flags them, and TabPFN's band is no wider for them than for geometries it predicts well.

**Models built around the grouping are as good as TabPFN here, not better.** A linear mixed-effects model and a two-stage model, each a few lines, are level with TabPFN-3.5 on average and better on some geometries, worse on others. What they add is a decomposition: the mixed model's $\tau = 3.2$ against $\sigma = 0.65$ says that two designs on the same geometry can be compared to within about one unit, two designs on two new geometries only to within about five. For an unseen geometry its random effect cannot be estimated, so its prediction is the fixed part alone.

**Gradient boosting can fail on a new group for a reason that has nothing to do with smoothness.** With the 0.76 two-storey geometry held out, compactness separates the context's one-storey from its two-storey geometries as cleanly as height does, and XGBoost predicts the held-out geometry like a one-storey building, an error of 23.7 units. When two columns separate the context's groups equally well, which one a tree splits on is arbitrary, and a new group can fall on the wrong side.

**Coverage must be judged per group.** Every band's overall coverage is near the nominal 0.80, and every band covers the 0.79 geometry not at all and the 0.82 geometry at most 44% of the time. A band that is right 80% of the time on average is wrong for a whole building form at once. The group calibration barely moves TabPFN's band: with most geometries covered comfortably, the 80% quantile of the other geometries' scores is slightly negative, so the band narrows a little and still misses the hard geometries. Calibrating against groups needs more than twelve of them, or a target stated as a share of groups.

**Twelve groups cannot separate these models.** No simultaneous interval excludes zero. TabPFN-3.5's point advantage over XGBoost, two units per geometry, rests mostly on XGBoost's two catastrophic geometries; against TabPFN-2 the unadjusted interval just excludes zero, the simultaneous one does not. The compact two-column shape description is slightly worse on average and not distinguishable.

## Settings and reproducibility

- **Data:** UCI Energy Efficiency, 768 designs in 12 geometries of 64; geometries numbered in the order they first appear in the workbook.
- **Parity:** seed 0 of TabPFN-3.5 reproduces the notebook's leave-one-geometry-out MAE, 2.780874, to six decimals.
- **Software:** TabPFN 9.2.0 as a library, checkpoints `tabpfn-v3.5-20260909` and `tabpfn-v2-regressor`; XGBoost 3.4.2, statsmodels 0.15.0, scikit-learn 1.9.1, PyTorch 2.14.1, Python 3.13.
- **Hardware:** laptop CPU (Darwin arm64). The report runs in about 30 minutes, almost all of it about 110 TabPFN fits; peak memory 2.8 GB.
