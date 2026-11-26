# 01 · Effects with a known answer, over ten draws and two scenarios

*Part of the [treatment-effects results ledger](index.md). Notebook: [`01_effects_with_a_known_answer.ipynb`](../../01_effects_with_a_known_answer.ipynb). Every number on this page comes from [`01_effects_with_a_known_answer_report.py`](../../01_effects_with_a_known_answer_report.py), whose docstring fixed the protocol before the full run, and is saved in [`reports/01-effects-with-a-known-answer.json`](reports/01-effects-with-a-known-answer.json), with one report per draw in [`reports/runs/`](reports/runs/).*

## The question

When a treatment was not assigned at random, how accurately do the standard effect estimators recover the average effect and each person's effect, and does it help to build them on TabPFN instead of linear models or tuned gradient boosting?

The notebook answers this on one simulated care program, in which coaches call members with a high HbA1c and high app use, the same members who improve without a call. One draw cannot separate real differences from the variation between draws, and its simulator suits linear models: the coaches' decisions are exactly a logistic function of three covariates and the outcome without a call is nearly linear. This page repeats the notebook's analysis over ten independent draws, in the notebook's scenario and in a harder one.

## How it was run

| Setting | Value |
|---|---|
| Scenarios | **smooth**: the notebook's program. **rules**: coaches decide by thresholds and combinations of covariates (for example, HbA1c of at least 9, or app use of at least 15 days for members under 55), and the outcome and the effect depend on interactions, so no part of it can be represented without interaction terms. Both are defined in `helper.simulate_care_program`. |
| Draws | seeds 0 to 9; a seed fixes the simulated members, the fold split and every model's random state |
| Size | 2,000 members, 5 cross-fitting folds, selection strength 1 for the main analysis |
| Base models | linear and logistic regression without interactions; XGBoost tuned by a 3-fold grid search over depth, learning rate and number of trees; TabPFN-3.5 with default settings |
| Overlap sweep | selection strengths 0, 1, 2 and 3 on the same members |
| Decision rule | TabPFN minus each reference, paired by draw, with intervals that hold simultaneously at 95% (Bonferroni over the 16 contrasts for individual effects and over 4 for the average effect); a difference counts only when its interval lies wholly on one side of zero |
| Environment | NVIDIA A40; Python 3.12.3, torch 2.7.0a0 (NGC 25.02), TabPFN 9.2.0 at upstream commit `3f2e9f9` with the v3.5 weights, scikit-learn 1.9.1, XGBoost 2.1.3; about two minutes per draw |
| Parity | seed 0 of the smooth scenario reproduces every number the notebook prints, to its printed decimals, including the TabPFN and gradient-boosting ones |

The scenarios differ in size of effect as well as in shape. The true average effect is 0.470 in the smooth scenario and 0.292 in the rules scenario, and the members' effects vary with a standard deviation of 0.266 and 0.300. In both, the plain comparison of called and uncalled members is badly off, by 0.507 and 0.426 on average: about twice the true effect.

## How well each base model fits the three nuisances

Root mean squared error against the truth, mean over ten draws with the range: the outcome without a call ($\mu_0$) and with a call ($\mu_1$) in HbA1c points, the propensity $e$ in probability units.

| Scenario | Base model | $\mu_0$ | $\mu_1$ | $e$ |
|---|---|---|---|---|
| smooth | linear | **0.064** (0.058 to 0.076) | 0.122 (0.116 to 0.130) | **0.031** (0.019 to 0.044) |
| smooth | gradient boosting | 0.148 (0.134 to 0.181) | 0.165 (0.142 to 0.201) | 0.079 (0.072 to 0.087) |
| smooth | TabPFN | 0.069 (0.059 to 0.080) | **0.090** (0.075 to 0.126) | 0.038 (0.031 to 0.047) |
| rules | linear | 0.284 (0.269 to 0.298) | 0.365 (0.357 to 0.372) | 0.130 (0.127 to 0.135) |
| rules | gradient boosting | 0.186 (0.163 to 0.249) | 0.180 (0.167 to 0.188) | **0.096** (0.087 to 0.104) |
| rules | TabPFN | **0.160** (0.137 to 0.195) | **0.159** (0.148 to 0.163) | **0.096** (0.081 to 0.109) |

In the smooth scenario the linear models are right or nearly right for two of the three problems and win them, with TabPFN close; in the rules scenario they are wrong for all three and fall far behind. TabPFN is the most accurate, or level with the most accurate, on every problem in the rules scenario and on the one smooth problem where the linear model is wrong. Gradient boosting is the least accurate in the smooth scenario and second in the rules scenario, where it ties TabPFN on the propensity.

## The average effect

Error against the draw's true average effect: mean over ten draws with the range, the root mean squared error over draws, the mean reported standard error, and how many of the ten 95% intervals contain the truth.

| Scenario | Base model | Estimator | Mean error (range) | RMSE | Mean SE | Covering |
|---|---|---|---|---|---|---|
| smooth | linear | regression adjustment | −0.002 (−0.046 to 0.044) | 0.025 | | |
| smooth | linear | IPW | 0.018 (−0.033 to 0.095) | 0.044 | 0.049 | 9 |
| smooth | linear | AIPW | 0.015 (−0.033 to 0.070) | 0.031 | 0.031 | 9 |
| smooth | gradient boosting | regression adjustment | 0.070 (0.020 to 0.103) | 0.074 | | |
| smooth | gradient boosting | IPW | 0.108 (0.022 to 0.186) | 0.117 | 0.044 | 3 |
| smooth | gradient boosting | AIPW | 0.028 (−0.017 to 0.079) | 0.039 | 0.028 | 8 |
| smooth | TabPFN | regression adjustment | 0.022 (−0.026 to 0.077) | 0.035 | | |
| smooth | TabPFN | IPW | 0.026 (−0.016 to 0.106) | 0.045 | 0.052 | 9 |
| smooth | TabPFN | AIPW | 0.019 (−0.021 to 0.080) | 0.034 | 0.032 | 9 |
| rules | linear | regression adjustment | 0.049 (−0.017 to 0.077) | 0.057 | | |
| rules | linear | IPW | 0.044 (−0.016 to 0.072) | 0.052 | 0.042 | 10 |
| rules | linear | AIPW | 0.074 (0.001 to 0.111) | 0.081 | 0.030 | **3** |
| rules | gradient boosting | regression adjustment | 0.056 (0.001 to 0.085) | 0.061 | | |
| rules | gradient boosting | IPW | 0.094 (0.037 to 0.132) | 0.098 | 0.041 | 2 |
| rules | gradient boosting | AIPW | 0.032 (−0.033 to 0.064) | 0.043 | 0.026 | 8 |
| rules | TabPFN | regression adjustment | 0.046 (−0.017 to 0.084) | 0.053 | | |
| rules | TabPFN | IPW | 0.048 (−0.011 to 0.088) | 0.056 | 0.043 | 8 |
| rules | TabPFN | AIPW | 0.028 (−0.035 to 0.059) | 0.039 | 0.027 | 9 |

Paired contrasts of the absolute AIPW error, TabPFN minus reference, simultaneous 95%: in the rules scenario TabPFN's error is lower than the linear models' by 0.039 (0.012 to 0.066); the other three contrasts are not resolved (smooth: +0.002 against linear, −0.007 against gradient boosting; rules: −0.004 against gradient boosting).

**When both nuisance models are wrong, the doubly robust estimator is not robust.** With linear models in the rules scenario, AIPW is the worst estimator in its block: a bias of 0.074, a quarter of the true effect, and an interval that covers the truth in 3 draws of 10, because its standard error of 0.030 reflects sampling noise only, not the bias. IPW with the same, equally wrong, propensity model covers in all ten draws, by being less precise rather than more accurate. Double robustness promises a consistent estimate when one of the two models is right, and here neither is. With gradient boosting or TabPFN, whose nuisances are much closer, AIPW has the smallest error in its block in both scenarios and covers in eight or nine draws.

**In the smooth scenario the base model barely matters for AIPW.** All three base models give a mean error under 0.03 and cover in eight or nine of ten draws. It matters a great deal for IPW, which inherits gradient boosting's flattened propensities: a bias of 0.108 and three intervals of ten covering.

## Each member's effect

PEHE, the root mean squared error of the estimated individual effects against the true ones, mean over ten draws with the range. For scale, an estimator that gave every member the true average effect would have a PEHE equal to the effects' standard deviation, 0.266 (smooth) and 0.300 (rules).

| Scenario | Base model | S-learner | T-learner | X-learner | DR-learner |
|---|---|---|---|---|---|
| smooth | linear | 0.269 (0.263 to 0.282) | 0.124 (0.106 to 0.140) | 0.121 (0.101 to 0.137) | 0.133 (0.109 to 0.155) |
| smooth | gradient boosting | 0.151 (0.125 to 0.188) | 0.195 (0.177 to 0.213) | 0.120 (0.105 to 0.144) | 0.160 (0.137 to 0.218) |
| smooth | TabPFN | 0.098 (0.074 to 0.125) | 0.102 (0.082 to 0.120) | **0.084** (0.062 to 0.106) | 0.094 (0.068 to 0.125) |
| rules | linear | 0.305 (0.298 to 0.317) | 0.245 (0.212 to 0.277) | 0.242 (0.210 to 0.273) | 0.226 (0.202 to 0.252) |
| rules | gradient boosting | 0.164 (0.130 to 0.207) | 0.220 (0.203 to 0.247) | 0.147 (0.127 to 0.164) | 0.177 (0.129 to 0.222) |
| rules | TabPFN | **0.128** (0.108 to 0.155) | 0.181 (0.160 to 0.208) | **0.129** (0.113 to 0.152) | 0.144 (0.128 to 0.166) |

Paired contrasts of PEHE, TabPFN minus reference for the same meta-learner, with simultaneous 95% intervals over the 16 contrasts:

| Scenario | Meta-learner | Against linear | Against gradient boosting |
|---|---|---|---|
| smooth | S-learner | −0.171 (−0.191 to −0.152) | −0.053 (−0.072 to −0.034) |
| smooth | T-learner | −0.022 (−0.033 to −0.012) | −0.093 (−0.112 to −0.075) |
| smooth | X-learner | −0.037 (−0.052 to −0.022) | −0.037 (−0.047 to −0.026) |
| smooth | DR-learner | −0.039 (−0.055 to −0.024) | −0.066 (−0.103 to −0.030) |
| rules | S-learner | −0.177 (−0.191 to −0.163) | −0.036 (−0.066 to −0.006) |
| rules | T-learner | −0.064 (−0.077 to −0.050) | −0.039 (−0.053 to −0.026) |
| rules | X-learner | −0.113 (−0.138 to −0.088) | −0.018 (−0.032 to −0.004) |
| rules | DR-learner | −0.082 (−0.094 to −0.070) | −0.034 (−0.072 to 0.005), not resolved |

Slope of estimated on true effects (1 is full size, below 1 is shrinkage) and the targeting score at a 30% budget (0 is no better than calling at random, 1 is as good as knowing every effect), means over ten draws:

| Scenario | Base model | Slope S / T / X / DR | Targeting S / T / X / DR |
|---|---|---|---|
| smooth | linear | 0.00 / 1.04 / 1.04 / 0.91 | 0.00 / 0.97 / 0.98 / 0.93 |
| smooth | gradient boosting | 0.60 / 1.06 / 0.79 / 0.76 | 0.91 / 0.85 / 0.94 / 0.90 |
| smooth | TabPFN | 0.73 / 1.02 / 0.93 / 0.96 | 0.99 / 0.96 / 0.99 / 0.98 |
| rules | linear | 0.00 / 0.91 / 0.91 / 0.62 | −0.01 / 0.84 / 0.84 / 0.80 |
| rules | gradient boosting | 0.62 / 1.06 / 0.71 / 0.72 | 0.90 / 0.85 / 0.94 / 0.90 |
| rules | TabPFN | 0.81 / 1.08 / 0.78 / 0.89 | 0.93 / 0.91 / 0.94 / 0.93 |

**TabPFN gives the most accurate individual effects, and the result holds in the harder scenario.** Its PEHE is lower than the reference's in 15 of the 16 contrasts, all with simultaneous intervals wholly below zero; the sixteenth, the DR-learner against gradient boosting in the rules scenario, points the same way but is not resolved. The margin depends on how well the reference's assumptions fit: against linear models it is large where they are wrong (the rules scenario, and every S-learner) and modest where they are nearly right; against gradient boosting it is largest where boosting is noisiest (the T-learner in the smooth scenario, 0.093) and smallest for the X-learner in the rules scenario, 0.018.

**The notebook's draw was the most favourable of the ten for TabPFN.** There, TabPFN's X-learner had a PEHE of 0.062, about half the references'. Over ten draws its mean is 0.084, against 0.121 and 0.120: about 30% lower, not half. Seed 0 is the lowest value in TabPFN's range.

**TabPFN's S-learner shrinks effects in every draw,** reproducing differences between members at 73% of their size in the smooth scenario and 81% in the rules scenario, against about 60% for gradient boosting; the linear S-learner gives everyone the same effect. Shrinkage is not confined to the S-learner in the harder scenario: there TabPFN's X-learner keeps 78% and its DR-learner 89%. Only the T-learner, which fits the two arms separately, keeps differences at full size or above, at the price of more noise.

**Ranking members is easier than estimating their effects.** Most estimators capture 90% or more of the attainable targeting gain in both scenarios; TabPFN's lie between 0.91 and 0.99. Linear models fall to 0.80 to 0.84 in the rules scenario, and the linear S-learner, with no ranking at all, captures nothing.

## When the arms stop overlapping

Mean error over ten draws as selection strengthens, with the number of draws (of ten) whose 95% interval covers the truth in brackets; at strength 3, 42% (smooth) and 28% (rules) of members have a true propensity below 0.05 or above 0.95.

| Scenario | Strength | Uncalled arm, effective members | Plain | Linear IPW | Linear AIPW | Boosting IPW | Boosting AIPW | TabPFN IPW | TabPFN AIPW |
|---|---|---|---|---|---|---|---|---|---|
| smooth | 0 | 1,201 | 0.009 | −0.002 [10] | −0.001 [10] | 0.003 [10] | −0.002 [9] | 0.007 [9] | −0.000 [10] |
| smooth | 1 | 856 | 0.507 | 0.018 [9] | 0.015 [9] | 0.108 [3] | 0.028 [8] | 0.026 [9] | 0.019 [9] |
| smooth | 2 | 336 | 0.687 | 0.057 [9] | −0.005 [10] | 0.160 [2] | 0.047 [7] | 0.046 [9] | 0.008 [10] |
| smooth | 3 | 175 | 0.751 | 0.163 [5] | 0.001 [10] | 0.264 [1] | 0.098 [3] | 0.161 [6] | 0.034 [8] |
| rules | 0 | 1,338 | 0.015 | −0.000 [10] | −0.000 [10] | 0.004 [10] | −0.004 [10] | 0.007 [10] | 0.001 [10] |
| rules | 1 | 705 | 0.426 | 0.044 [10] | 0.074 [3] | 0.094 [2] | 0.032 [8] | 0.048 [8] | 0.028 [9] |
| rules | 2 | 191 | 0.555 | 0.086 [4] | 0.131 [0] | 0.148 [2] | 0.079 [4] | 0.086 [5] | 0.060 [4] |
| rules | 3 | 113 | 0.592 | 0.109 [5] | 0.163 [0] | 0.169 [3] | 0.122 [2] | 0.161 [4] | 0.090 [4] |

In the smooth scenario the single draw's picture holds: IPW fails with every base model at strength 3, and AIPW holds with linear models (whose propensity model is exactly right) and, slightly less well, with TabPFN. In the rules scenario nothing holds once overlap is poor. At strength 2 and above every estimator is biased by 0.06 or more and covers the truth in at most five draws of ten; TabPFN's AIPW is the least biased, 0.060 and 0.090, but its intervals still cover in only four. With a misspecified or merely imperfect outcome model and members who have no comparable people in the other arm, no estimator here gives an interval that can be trusted, and the overlap table is the warning.

## What comes next

- **Effect intervals for each member,** and a check of their calibration over draws, now that the point estimates are characterized.
- **Real data:** notebook 02 on the NHEFS observational study (data staged), where only a published average is known.
- **A trial as the ground truth:** notebook 03 builds observational studies from a randomized trial, so estimators can be checked against a real, not simulated, answer.
