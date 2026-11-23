# Grouped transfer: predicting for groups the model has never seen

**Question.** When the rows of a table come in groups, and the rows to predict belong to a group that is absent from the context, how well does TabPFN transfer, where does its error come from, how should such a transfer be evaluated, and do models that represent the grouping explicitly do better or at least say more honestly how uncertain they are?

**Why it matters.** Most tables this lab cares about are made of groups. In the main application a perturbation is a group of cells, and the question that matters is always a perturbation the model has not seen; the same shape appears with patients from one hospital or readings from one machine. Rows inside a group share most of what determines the target, so a model can be excellent on new rows of a known group and far off on a new group, and an evaluation that mixes the two reports the easy answer. These notebooks work through the problem on a table small and clean enough that every number can be checked: 768 simulated buildings, each a combination of one of twelve geometries, four orientations and sixteen glazing setups, with the heating load to predict. Each geometry is a group of 64 designs.

## Notebooks

| Notebook | What it measures |
|---|---|
| [`01_unseen_geometry_transfer.ipynb`](01_unseen_geometry_transfer.ipynb) | Which columns are admissible inputs (a sibling outcome of the same simulation is not), TabPFN against XGBoost on three held-out geometries, the error split by geometry into a shared offset and scatter, which design decisions the estimates can support, and what removing two redundant columns does. |
| [`02_grouped_structure.ipynb`](02_grouped_structure.ipynb) | How much of the variance the grouping explains and what that does to the effective sample size, what a random split hides, leave-one-geometry-out over all twelve geometries, a mixed-effects and a two-stage model as references that use the grouping, the calibration of 80% bands for a new geometry with a group-calibrated correction, and a bootstrap over geometries for the differences between models. |

## Results

Each notebook has a page in the [results ledger](docs/results/index.md), which repeats its runs with three seeds and both TabPFN versions and records every number with the settings that produced it. The numbers come from [`01_unseen_geometry_transfer_report.py`](01_unseen_geometry_transfer_report.py) and [`02_grouped_structure_report.py`](02_grouped_structure_report.py), whose docstrings fix the protocols.

## Notes on the models that use the grouping

[`docs/group_models/`](docs/group_models/README.md) explains the two reference models of notebook 02 from the ground up, on this table. [The first note](docs/group_models/01-mixed-effects.md) covers the linear mixed-effects model: pooling, the two variances, REML, shrinkage, a prediction for an unseen geometry worked by hand, and what a few observed designs of a new geometry change. [The second](docs/group_models/02-two-stage.md) covers the two-stage model: the shift table and Gaussian process regression on the geometry means. [The third](docs/group_models/03-comparison.md) compares the two geometry by geometry and asks what each one's uncertainty is worth.

## Data

The UCI Energy Efficiency data: Athanasios Tsanas and Angeliki Xifara (2012), *Energy Efficiency* [dataset], UCI Machine Learning Repository, DOI [10.24432/C51307](https://doi.org/10.24432/C51307), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `helper.py` downloads the workbook once into the shared data lake, under `energy/uci_energy_efficiency`, checks it against a fixed SHA-256, renames the columns and numbers the designs and geometries in the workbook's order; no value is changed.

## Running

Use the `tabicllab` environment and its Jupyter kernel (see the repository README). `helper.py` selects TabPFN-3.5 when its license has been accepted and TabPFN-2 otherwise, and runs on the CPU on a Mac. On a laptop CPU, notebook 01 runs in about a minute and notebook 02 in about five minutes, most of it TabPFN's twelve leave-one-out fits; the two reports take about three and thirty minutes.
