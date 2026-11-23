# Models that use the grouping

[`02_grouped_structure.ipynb`](../../02_grouped_structure.ipynb) compares TabPFN and XGBoost, which see only rows of columns, with two models that are told the rows come in groups: a linear mixed-effects model and a two-stage model. The two did as well as TabPFN when predicting building geometries they had never seen, and they did it with a few lines of code each. These notes explain how the two models work, from the ground up, on the table they were used on: 768 simulated buildings in twelve geometries of 64 designs, with the heating load to predict.

| Note | What it covers |
|---|---|
| [01 · The linear mixed-effects model](01-mixed-effects.md) | Complete, no and partial pooling; the random-intercept model and its two variances; generalized least squares and REML; why least squares is too sure of between-group effects and too unsure of within-group ones; the estimated offsets and shrinkage; one prediction for an unseen geometry worked by hand; what the band covers; which comparisons the estimates support; how a few observed designs of the new geometry remove most of the error. |
| [02 · The two-stage model](02-two-stage.md) | The level-plus-shift decomposition behind the 99% of variance it explains; the shift table; Gaussian process regression on eleven geometry means, its kernel and length scales; what the process learns about shape; why its noise term plays the role of the mixed model's between-group variance; the same worked prediction; where it is the best model in the notebook and where it is the worst. |
| [03 · The two models side by side](03-comparison.md) | The same structure in two descriptions; geometry-by-geometry levels; the within-geometry part; what each model's uncertainty is worth and why both bands fail on the same two geometries; a joint model that contains both; where TabPFN sits; how to choose. |

The notes read in order, and each one defines its symbols where they first appear. They assume ordinary linear regression and the normal distribution; the mixed model and Gaussian process regression are built up from there.

## The numbers and figures

Headline results, such as each model's error over all twelve geometries and the coverage of its bands, come from the [results ledger](../results/index.md). The notes add quantities that explain how the models work: coefficients and standard errors, the estimated offsets, each fold's variance estimates, the Gaussian process's length scales and the error after observing a few designs of a new geometry. [`figures/make_figures.py`](figures/make_figures.py) computes all of them, draws the four figures and writes the numbers to [`figures/numbers.json`](figures/numbers.json). It runs in a few seconds with the `tabicllab` environment and needs no TabPFN fit:

```bash
python notebooks/grouped_transfer/docs/group_models/figures/make_figures.py
```
