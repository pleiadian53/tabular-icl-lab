# The results ledger: TabPFN out of the box

*An append-only record of what TabPFN does with its default settings and no tuning, before any method of this lab is added to it. This page holds what the characterizations add up to, the rules they follow, the settings they share and what they suggest next. Each characterization has its own page, which says what was asked, how it was run and what was measured, with a link from every number to the script or notebook and the report that produced it.*

> **What this ledger covers.** General-purpose public tables, small enough to run on a laptop: clinical, real-estate and energy data. The numbers here are the out-of-the-box reference for the lab's methods; they are not claims about any application, which keeps its own ledger, such as [perturbation response](../../../../examples/perturbation_response/docs/results/index.md).

---

## The characterizations

| # | Question | Answer | Page |
|---|---|---|---|
| 01 | Untuned, how good are TabPFN's predictions against the references a practitioner would try first, and is its uncertainty calibrated? | At least as good as every reference on heart disease and house prices, and better than untuned gradient boosting on heart-disease probabilities (log loss 0.362 against 0.486). Its 80% intervals cover exactly 80% of test houses, but only 42% to 56% of the hours of an electricity forecast, where the test rows lie after the context in time. | [prediction and uncertainty](01-prediction-and-uncertainty.md) |
| 02 | Do standard post-hoc explanations work on an in-context learner, and are they sensible? | Yes, from one run on ten patients: SHAP decomposes each prediction exactly, ranks diagnostic test results above routine risk factors, and every partial dependence curve runs in the clinically expected direction. Stability across seeds and backgrounds is not yet measured. | [explaining an in-context learner](02-interpreting-predictions.md) |

---

## What they add up to

**On a small table whose test rows are drawn like its context rows, untuned TabPFN is a strong reference.** Across a classification table with missing values and a regression table with text columns, no reference beat it, untuned gradient boosting lost to it on the quality of its probabilities, and its prediction intervals were calibrated. A method that claims to improve on TabPFN has to clear that bar.

**Its calibration does not survive a shift.** When the rows to predict lie after the context in time, as in a forecast, the predictive distribution is too narrow: nominal 80% intervals covered 42% to 56% of the hours. Recalibrating TabPFN's predictive distribution for query rows that are not drawn like the context is the clearest opening these tables show for a method built on top of it.

**What the model attends to can be read with standard tools.** An in-context learner exposes no parameters, but model-agnostic explanations need none; on a clinical table they gave exact decompositions and clinically coherent rankings and response curves.

---

## The rules every characterization follows

- **The model is used as shipped.** Default constructors, no tuning, the version and checkpoint recorded. TabPFN-3.5 is the main model; TabPFN-2 is reported beside it where a script runs both.
- **Every number comes from a saved report.** A report script, whose docstring fixes the protocol before the full run, or a notebook cell that writes the report, as with notebook 02. A report script first checks that it reproduces the numbers its notebook prints, and stops if it does not.
- **References are the ones a practitioner would try first**, at a comparable level of effort: gradient boosting with its default settings, regularized linear models with their penalty chosen by internal cross-validation on the context rows, and the trivial predictor every model must beat.
- **Comparisons are paired over test rows**, with a joint paired bootstrap, 10,000 resamples shared by every contrast of a task, and max-t simultaneous 95% intervals. A difference counts only when its interval lies wholly on one side of zero. Where rows are not exchangeable, as for consecutive forecast hours, only point values are reported.
- **Several seeds where the model is seeded**, three by default, with losses averaged over seeds per row before comparing.

---

## The settings every characterization shares

- **Code:** [`helper.py`](../../helper.py) holds the loaders, the splits (seed 0), the model setup and the plots.
- **Data:** downloaded once into the shared data lake and filed by domain. UCI heart disease (four studies, 920 patients), Statlog heart disease (OpenML 53, 270 patients), Ames housing (OpenML 42165, 1,460 sales), and hourly DE-LU day-ahead electricity prices with the day-ahead load and renewables forecasts (Energy-Charts API, SMARD and the transmission operators, CC BY 4.0), 4 to 19 March 2024.
- **Software:** TabPFN 9.2.0 used as a library, checkpoint `tabpfn-v3.5-20260909`; Python 3.13, PyTorch 2.14, scikit-learn 1.9, LightGBM 4.7, SHAP 0.52.
- **Hardware:** a laptop CPU. TabPFN's guard against more than 1,000 context rows on a CPU is lifted for the 1,300-row housing table, which runs in under a minute.

---

## What comes next

Three follow-ups are open, none yet committed to:

- **Calibration under shift.** Measure how TabPFN's interval coverage degrades as query rows move away from the context, over many forecast windows rather than one, and test whether a conformal correction fitted on recent rows restores it.
- **Stability of explanations.** Repeat notebook 02 over several model seeds and background samples on all 70 test patients, and report how much the feature ranking and the curves move. It is about nine times the compute of one run, so it belongs on the GPU pod.
- **Cost of the cached context.** Measure the speed and memory of `fit_with_cache` against the default fit as the context grows, the budget every repeated-query method in the lab will spend.
