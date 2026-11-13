# 02 · Explaining an in-context learner

*Part of [the results ledger](index.md), which holds what the characterizations add up to, the rules they follow and the settings they share. Notebook: [`02_interpreting_predictions.ipynb`](../../02_interpreting_predictions.ipynb).*

**Question.** TabPFN has no fitted parameters for a dataset: whatever it learns from the context exists only during one forward pass. Do the standard post-hoc explanations still work on such a model, and is what they say about its predictions sensible?

**Answer, from one run on ten explained patients: yes on both counts, with the evidence still thin.** Permutation SHAP decomposes each predicted risk exactly, to within $6 \times 10^{-8}$, so the method behaves as it does on any classifier. The explanations rank diagnostic test results first (the thallium scan, the number of vessels seen narrowed on fluoroscopy, the chest-pain type and the ST depression on exercise) and routine or demographic measurements last, and every partial dependence curve runs in the clinically expected direction. The model itself discriminates well on this table, test AUC 0.916. How stable the ranking is across seeds, background samples and more patients is not yet measured.

## What was measured

The table is Statlog heart disease: 270 patients, 13 measurements, complete, 120 with heart disease, split 200 for context and 70 for testing. TabPFN-3.5 is fitted with the context cached (`fit_mode="fit_with_cache"`), so the thousands of predictions the explanations request do not re-encode the context each time, and told which seven columns are category codes.

**SHAP.** For a patient $x$ with predicted risk $f(x)$, SHAP gives each of the 13 features a contribution $\phi_j(x)$ with

$$
f(x) = \phi_0 + \sum_{j=1}^{13} \phi_j(x),
$$

where the base value $\phi_0$ is the mean predicted risk over a background set. A feature's contribution is its average effect on the prediction when added to coalitions of the other features, with absent features filled in from the background patients. The estimator is SHAP's permutation explainer, with 10 background patients drawn from the context with seed 0, applied to the first 10 test patients. A feature's importance is its mean absolute contribution over those 10 patients.

**Partial dependence.** For a feature $j$ and a value $v$, $\mathrm{PD}_j(v)$ is the predicted risk averaged over the 200 context patients with feature $j$ set to $v$ and everything else kept. It is drawn over the 5th to 95th percentiles of the feature for the four continuous features with the highest SHAP importance.

## Results

| | Value |
|---|---|
| Test AUC, 70 patients | 0.916 |
| Base value $\phi_0$, mean risk over the background | 0.402 |
| Largest gap between $\phi_0 + \sum_j \phi_j(x)$ and $f(x)$ | $5.9 \times 10^{-8}$ |
| Highest-risk explained patient | predicted 0.975, has heart disease |
| Explained patient closest to 0.5 | predicted 0.484, has heart disease |

**Feature importance**, mean absolute SHAP contribution to the predicted risk over the 10 explained patients:

| Rank | Feature | What it records | Importance |
|---|---|---|---|
| 1 | `thalassemia` | thallium stress scan: normal, fixed defect or reversible defect | 0.105 |
| 2 | `n_major_vessels` | major vessels (0 to 3) coloured by fluoroscopy | 0.101 |
| 3 | `chest_pain_type` | typical angina, atypical angina, non-anginal pain or none | 0.079 |
| 4 | `st_depression` | ST-segment depression induced by exercise, relative to rest | 0.069 |
| 5 | `sex` | | 0.060 |
| 6 | `exercise_angina` | angina induced by exercise | 0.046 |
| 7 | `st_slope` | slope of the peak-exercise ST segment | 0.044 |
| 8 | `max_heart_rate` | maximum heart rate achieved | 0.030 |
| 9 | `resting_ecg` | resting ECG result | 0.027 |
| 10 | `cholesterol` | serum cholesterol | 0.024 |
| 11 | `resting_bp` | resting blood pressure | 0.020 |
| 12 | `age` | | 0.017 |
| 13 | `fasting_blood_sugar` | fasting blood sugar above 120 mg/dl | 0.010 |

**Partial dependence** of the mean predicted risk, from the low to the high end of each feature's range:

| Feature | Range swept | Mean predicted risk | Direction |
|---|---|---|---|
| `st_depression` | 0 to 3.1 mm | 0.39 to 0.55 | rises |
| `max_heart_rate` | 111 to 181 bpm | 0.50 to 0.41 | falls |
| `cholesterol` | 177 to 330 mg/dl | 0.39 to 0.50 | rises |
| `resting_bp` | 108 to 160 mmHg | 0.42 to 0.47 | rises |

## What this says

- **The explanation machinery works unchanged.** An in-context learner is, to a model-agnostic explainer, just a function from rows to probabilities. Exact additivity and smooth partial dependence curves are what one would see for a fitted classifier: three of the four curves are monotone at every grid step, and resting blood pressure rises overall with three dips of at most 0.001.
- **The model leans on the diagnostic tests.** The four most important features are results of cardiac investigations; age, blood pressure, cholesterol and fasting blood sugar, the routine risk factors, rank last. On a table where every patient underwent these cardiac tests, it is unsurprising that their results carry more information about the diagnosis than the routine risk factors do.
- **The directions agree with clinical expectation.** More exercise-induced ST depression raises the predicted risk; a higher achievable heart rate, a sign of better exercise capacity, lowers it; higher cholesterol and blood pressure raise it modestly. The shared vertical axis shows the effect sizes: ST depression and cholesterol move the average risk by about 0.1 to 0.15 over their ranges, blood pressure by 0.05.
- **These are statements about the model, not the patients.** A large attribution says the model relies on a measurement to separate the outcomes in this context; it does not say the measurement causes the disease or that changing it would change the risk.

## Settings and reproducibility

- **Notebook:** [`02_interpreting_predictions.ipynb`](../../02_interpreting_predictions.ipynb), executed end to end; its last cell writes [`02-interpreting-predictions.json`](reports/02-interpreting-predictions.json), the source of every number above, including the partial dependence curves on their full grids. Data and plotting code: [`helper.py`](../../helper.py).
- **Data:** Statlog heart, OpenML 53, 270 patients, stratified split 200 / 70 with seed 0; seven columns passed to TabPFN as categorical, the vessel count kept numeric as an ordinal count.
- **Model:** TabPFN 9.2.0, checkpoint `tabpfn-v3.5-20260909`, default settings, `fit_mode="fit_with_cache"`, random state 0. SHAP 0.52.0, permutation explainer with default evaluation budget; 10 background patients and 10 explained patients.
- **Run:** a laptop CPU (Apple silicon), 16 minutes end to end, about 1 minute per explained patient and 5 minutes for the four partial dependence curves.
- **Limits.** One model seed, one background sample and ten patients: the top two features, at 0.105 and 0.101, stand apart from the third at 0.079, but from the third rank down neighbouring importances differ by 0.01 or less in most places, and their order could change with another seed, background or set of patients. Partial dependence holds every other feature fixed, so it averages over combinations that may not occur together, for example a maximum heart rate of 180 in the oldest patients.
