# 01 · Prediction and uncertainty, against references

*Part of [the results ledger](index.md), which holds what the characterizations add up to, the rules they follow and the settings they share. Notebook: [`01_prediction_and_uncertainty.ipynb`](../../01_prediction_and_uncertainty.ipynb).*

**Question.** With its default settings and no tuning, how good are TabPFN's predictions on three small tables of different kinds, measured against the references a practitioner would try first, and is the uncertainty it reports calibrated?

**Answer.** On the two tables whose test rows are drawn like the context rows, TabPFN-3.5 is at least as good as every reference and calibrated. Its heart-disease probabilities have a lower log loss than untuned gradient boosting (0.362 against 0.486, a difference whose simultaneous interval lies wholly below zero), and they are level with a cross-validated logistic regression and with TabPFN-2. On house prices it has the lowest error of all arms, $16,079 against $18,552 for gradient boosting, though 160 houses cannot resolve that difference, and its 80% prediction intervals cover exactly 80% of the houses under every seed. On the electricity forecast, where the 48 test hours lie after the context in time, the point forecasts are good, 7.7 EUR/MWh with market features against 8.1 for gradient boosting and 10.3 for repeating last week, but the intervals are not: the nominal 80% bands cover 56% of the hours with calendar features and 42% with market features. **The calibration holds where the test rows resemble the context and fails on a future horizon.**

## What was measured

For a test row $i$ with label $y_i$, a classifier predicts $p_i$, its probability that $y_i = 1$, and a regressor predicts a mean $\hat y_i$ and the 10th and 90th percentiles $q_{0.1,i}$ and $q_{0.9,i}$ of its predictive distribution. Each task has one primary score, fixed before the run:

- **Heart disease: the log loss** $\frac{1}{n}\sum_i -\left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$. It scores the probabilities themselves, so a model that ranks patients well but states its probabilities badly is penalized; a constant prediction of the base rate scores 0.687 here.
- **House prices: the mean absolute error** $\frac{1}{n}\sum_i \lvert \hat y_i - y_i \rvert$, in dollars.
- **Electricity: the mean absolute error** over the 48 forecast hours, in EUR/MWh.

The **coverage** of the 80% interval is the fraction of rows with $q_{0.1,i} \le y_i \le q_{0.9,i}$; a calibrated interval covers about 0.80. Accuracy, precision, recall, ROC AUC, the Brier score, $R^2$ and RMSE are reported beside the primary scores without a verdict.

## What was compared

| Arm | What it is |
|---|---|
| **TabPFN-3.5** | the lab's main model, default constructor, no tuning; seeds 0, 1 and 2 (seed 0 is what the notebook prints) |
| **TabPFN-2** | the earlier version, the same way |
| **LightGBM** | gradient boosting with its default settings, the untuned counterpart of an untuned TabPFN; missing values left to it, text columns as categories |
| **Linear** | logistic regression for heart disease, ridge regression for house prices, after median imputation, standardization and one-hot encoding of text; penalty chosen by 5-fold cross-validation on the context rows |
| **Prior** | the training class frequency or mean price: the floor every model must clear |
| **Naive forecasts** | electricity only: the last 24 context hours repeated (daily naive), and each hour's price one week earlier (weekly naive) |

Heart and housing comparisons are paired over test rows: for each row, TabPFN-3.5's loss (averaged over its seeds) minus the other arm's. Intervals are simultaneous 95% intervals over the four contrasts of a task, from a joint paired bootstrap with max-t adjustment. A difference counts only when its interval lies wholly on one side of zero. The forecast gets point values only: its 48 hourly errors are strongly autocorrelated and span two days, so an interval over hours would claim more precision than the data hold.

## Results

### Heart disease: 700 context patients, 220 test patients

| Arm | Log loss | Brier | ROC AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| **TabPFN-3.5** | **0.362** | 0.113 | 0.918 | 0.838 | 0.853 | 0.855 |
| TabPFN-2 | 0.356 | 0.111 | 0.920 | 0.850 | 0.858 | 0.874 |
| Linear | 0.400 | 0.125 | 0.902 | 0.818 | 0.848 | 0.820 |
| LightGBM | 0.486 | 0.144 | 0.888 | 0.791 | 0.817 | 0.803 |
| Prior | 0.687 | 0.247 | 0.500 | 0.555 | 0.555 | 1.000 |

| TabPFN-3.5 minus | Log-loss difference | Simultaneous 95% interval | Resolved |
|---|---|---|---|
| TabPFN-2 | +0.006 | [−0.009, +0.020] | no |
| Linear | −0.038 | [−0.090, +0.014] | no |
| LightGBM | −0.124 | [−0.220, −0.029] | **yes, TabPFN better** |
| Prior | −0.325 | [−0.426, −0.225] | yes |

The seeds barely matter: TabPFN-3.5's log loss is 0.363, 0.361 and 0.362 and its AUC 0.918, 0.919 and 0.918 across seeds 0, 1 and 2. Untuned LightGBM's deficit is larger in its probabilities than in its ranking of patients: its AUC is 0.03 lower, its log loss 0.12 higher.

### House prices: 1,300 context houses, 160 test houses

| Arm | MAE | RMSE | $R^2$ | 80% interval coverage | Mean interval width |
|---|---|---|---|---|---|
| **TabPFN-3.5** | **$16,079** | $26,372 | 0.897 | 0.800 | $51,885 |
| TabPFN-2 | $17,157 | $31,242 | 0.856 | 0.785 | $53,660 |
| LightGBM | $18,552 | $32,763 | 0.841 | | |
| Linear | $20,824 | $33,514 | 0.834 | | |
| Prior | $54,752 | $82,249 | 0.000 | | |

| TabPFN-3.5 minus | MAE difference | Simultaneous 95% interval | Resolved |
|---|---|---|---|
| TabPFN-2 | −$1,078 | [−$8,334, +$6,178] | no |
| LightGBM | −$2,473 | [−$9,429, +$4,484] | no |
| Linear | −$4,745 | [−$13,807, +$4,317] | no |
| Prior | −$38,672 | [−$61,737, −$15,608] | yes |

TabPFN-3.5's MAE across seeds is $15,994, $16,225 and $16,018, and its interval covers exactly 128 of 160 houses under each seed. A perfectly calibrated 80% interval would cover between about 74% and 86% of 160 houses by chance alone, so TabPFN-2's 78.5% is also consistent with calibration. The intervals on the differences are wide because a few expensive houses dominate the absolute errors, which is also why RMSE is so much larger than MAE for every arm.

### Electricity: 336 context hours, 48 forecast hours (18 and 19 March 2024)

| Arm | Features | MAE (EUR/MWh) | RMSE | 80% interval coverage | Mean interval width |
|---|---|---|---|---|---|
| TabPFN-3.5 | clock | 18.16 | 26.63 | 0.708 | 48.2 |
| TabPFN-3.5 | calendar | 8.69 | 11.66 | 0.563 | 18.7 |
| **TabPFN-3.5** | **calendar + market** | **7.71** | 10.28 | 0.424 | 13.3 |
| TabPFN-2 | clock | 20.05 | 28.05 | 0.840 | 83.2 |
| TabPFN-2 | calendar | 8.98 | 11.67 | 0.458 | 16.8 |
| TabPFN-2 | calendar + market | 8.45 | 11.55 | 0.479 | 12.8 |
| LightGBM | calendar + market | 8.15 | 11.47 | | |
| Weekly naive | | 10.30 | 13.54 | | |
| Daily naive | | 19.77 | 24.09 | | |

The mean price over the forecast hours is 81.4 EUR/MWh. The columns decide almost everything: from the clock alone TabPFN does no better than repeating yesterday, calendar features halve the error, and the day-ahead load and renewables forecasts cut it by another tenth. With the market features, TabPFN-3.5's MAE is 7.88, 7.73 and 7.52 across seeds, against 8.15 for LightGBM on the same columns; on 48 correlated hours that gap is an illustration, not a finding.

## What this says

- **On a small table with exchangeable rows, untuned TabPFN is a strong reference.** No reference beat it on either primary score, and it beat untuned gradient boosting on the probabilities that matter for a risk decision. A method in this lab that claims to improve on TabPFN has to beat these numbers, not a weak baseline.
- **Its intervals are calibrated in that setting.** Coverage of 0.800 on house prices under every seed, from a model never told what calibration is.
- **They are not calibrated under a shift in time.** On the forecast horizon the bands are narrow and cover 42% to 56% of the hours. The forecast window is spikier than the context: 5 of its 48 hours are above 110 EUR/MWh, against 7 of the 336 context hours, and its peak, 146, exceeds anything in the context (135). The predictive distribution, inferred from the context, does not anticipate that. The clock-only forecasts cover more only because they are wide and wrong. This is the clearest opening these tables show for a method on top of TabPFN: recalibrating its predictive distribution when the query rows are not drawn like the context, for example by conformal calibration on recent hours.
- **TabPFN-3.5 and TabPFN-2 are close here.** Version 3.5 is nominally better on house prices and forecasting and nominally worse on heart disease; none of the differences is resolved.

## Settings and reproducibility

- **Script:** [`01_prediction_and_uncertainty_report.py`](../../01_prediction_and_uncertainty_report.py), with the protocol in its docstring, run after a one-seed development run. Full report, with per-row losses, per-seed scores and every contrast: [`01-prediction-and-uncertainty.json`](reports/01-prediction-and-uncertainty.json).
- **Parity with the notebook.** Before scoring, the run checked that TabPFN-3.5 with seed 0 reproduces every value the notebook prints (precision 0.85, recall 0.84, MAE $15,994, $R^2$ 0.90, coverage 80%, forecast MAEs 8.69 and 7.88) and would have stopped otherwise.
- **Data:** splits from [`helper.py`](../../helper.py). UCI heart disease, four studies, 920 patients, stratified 700 / 220 with seed 0; zero cholesterol recoded as missing. Ames housing, OpenML 42165, nine features, 1,300 / 160 with seed 0. Hourly DE-LU day-ahead prices and the day-ahead load, solar and wind forecasts from the Energy-Charts API (SMARD and the transmission operators, CC BY 4.0), 4 to 19 March 2024. The MD5 of every downloaded file is in the report.
- **Models:** TabPFN 9.2.0, checkpoints `tabpfn-v3.5-20260909` (classifier and regressor), `tabpfn-v2-classifier-finetuned-zk73skhh` and `tabpfn-v2-regressor`; default constructors. LightGBM 4.7.0 with default settings; scikit-learn 1.9.1.
- **Statistics:** 10,000 bootstrap resamples per task, generator seed 0, one resample shared by every contrast of a task; max-t simultaneous 95% intervals with critical values 2.60 (heart) and 4.72 (housing).
- **Run:** a laptop CPU (Apple silicon), 467 seconds for all three tasks, peak memory 2.8 GB.
- **Limits.** One split per table; a different split would move every number, and the paired intervals describe uncertainty over test rows, not over splits. The references are untuned or lightly tuned on purpose, to match an untuned TabPFN; a tuned gradient-boosting model would be a stronger reference. Forty-eight forecast hours from one fortnight are an illustration of the calibration failure, not a measurement of its typical size.
