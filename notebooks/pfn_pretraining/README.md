# Pretraining a tabular in-context learner: the prior as the lever

**Question.** A prior-fitted network such as TabPFN learns everything it knows from the synthetic datasets it was pretrained on. How much do that prior, and the model's scale, each determine what it can infer from a handful of labeled rows, including on problems outside the family it was pretrained on?

**Why it matters.** The prior is where knowledge about a domain can enter such a model. This lab's route to a model of its own is a prior that generates tasks from perturbation biology, for example from a simulator of gene regulatory networks under intervention, so that the pretrained learner's inference is tuned to that domain. The TabPFN library provides the trained model, its inference and fine-tuning, but not the prior or the pretraining loop. This folder builds the smallest complete version of both, so each part can be measured and then replaced.

## Notebooks

| Notebook | What it measures |
|---|---|
| [`01_pretraining_from_scratch.ipynb`](01_pretraining_from_scratch.ipynb) | A transformer with attention across the features and across the rows of a table, pretrained on synthetic two-feature classification tables. Three runs: a small model on a prior of straight class boundaries, the same model on a richer prior of wavy boundaries, and a larger model trained longer on the richer prior. Each is scored on three fixed problems it never trained on, one of them outside both priors, against logistic regression and a random forest fitted on the same 40 labeled points. |

The model and the training loop follow [nanoTabPFN](https://github.com/automl/nanoTabPFN) (Apache-2.0; [arXiv:2511.03634](https://arxiv.org/abs/2511.03634)) and are written out in the notebook; `helper.py` holds the evaluation problems, the classical learners and the plots.

## Results

The [results ledger](docs/results/index.md) records the numbers. [`01_pretraining_from_scratch_report.py`](01_pretraining_from_scratch_report.py) runs the notebook once per pretraining seed, each run writing its scores on every evaluation draw, and aggregates them with paired intervals; its docstring fixes the protocol.

## Running

Use the `tabicllab` environment and its Jupyter kernel (see the repository README). The notebook needs only PyTorch and scikit-learn, not TabPFN. On an idle laptop CPU it takes about 40 minutes, most of it the larger model's 2,500 training steps; on a GPU it takes a few minutes.
