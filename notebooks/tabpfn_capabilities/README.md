# TabPFN capabilities: the out-of-the-box reference

**Question.** What does TabPFN do on a small table with its default settings and no tuning: how accurate are its predictions, how trustworthy is its stated uncertainty, and do the standard explanations of a prediction still work on a model that learns in context?

**Why it matters.** Every method in this lab builds on a pretrained tabular foundation model. A claim that a method improves on the model needs a measured starting point, on tables where the answer can be checked, against the references a practitioner would try first. These notebooks establish that starting point. They are deliberately general-purpose (clinical, real-estate and energy tables) so that the reference does not depend on any one application.

## Notebooks

| Notebook | What it measures |
|---|---|
| [`01_prediction_and_uncertainty.ipynb`](01_prediction_and_uncertainty.ipynb) | Classification with missing values (heart disease, 920 patients), regression with text columns and the calibration of its 80% prediction intervals (Ames housing), and forecasting recast as regression, where only the columns describing each hour change (German day-ahead electricity prices). |
| [`02_interpreting_predictions.ipynb`](02_interpreting_predictions.ipynb) | SHAP attributions for single predictions and across patients, and partial dependence curves, on a model with a cached context (Statlog heart disease, 270 patients). |

## Results

Each notebook has a page in the [results ledger](docs/results/index.md), which reruns or records its numbers with the settings that produced them. Notebook 01's page adds three seeds, both TabPFN versions and reference models, with paired intervals; its numbers come from [`01_prediction_and_uncertainty_report.py`](01_prediction_and_uncertainty_report.py), whose docstring fixes the protocol.

## Running

Use the `tabicllab` environment and its Jupyter kernel (see the repository README). `helper.py` downloads the datasets once into the shared data lake (`$SSLLAB_DATA_ROOT`, by default `~/work/data`) and selects TabPFN-3.5 when its license has been accepted, TabPFN-2 otherwise. Everything runs on a laptop CPU: notebook 01 in about four minutes, notebook 02 in about twenty (the SHAP values take most of it), and the notebook 01 report in about ten.
