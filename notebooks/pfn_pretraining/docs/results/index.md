# The results ledger: pretraining a tabular in-context learner

*An append-only record of what a prior-fitted network learns from the prior it is pretrained on, measured on a model small enough to pretrain from scratch. This page holds what the measurements add up to, the rules they follow, the settings they share and what they suggest next. Each measurement has its own page, which says what was asked, how it was run and what was measured, with a link from every number to the notebook and the report that produced it.*

> **What this ledger covers.** Two-feature synthetic classification problems with a known boundary, where the best achievable accuracy can be computed and every model sees exactly the same points. The numbers set the scale of what a prior and model size can change; they are not claims about TabPFN, which was pretrained on a far richer prior, or about any application.

---

## The measurements

| # | Question | Answer | Page |
|---|---|---|---|
| 01 | How much do the prior and the model's scale each determine what a pretrained in-context learner infers from 40 labeled points, inside and outside the family of problems it was pretrained on? | The prior decides the shape of what the model can infer: pretrained on straight boundaries it draws straight boundaries everywhere, and a prior that adds waves raises accuracy on curved problems by about 0.09 at a cost of 0.013 on the straight one. Scale adds a further 0.03 on both curved problems. The larger model beats a random forest on all three problems and logistic regression on the two curved ones. Three pretraining seeds agree. | [pretraining from scratch](01-pretraining-from-scratch.md) |

---

## What they add up to

**The prior is the lever that moves inference most.** With the architecture, size and training budget held fixed, changing only the family of synthetic tables the model is pretrained on changed its accuracy on curved boundaries by about three times as much as moving to a model with nearly six times as many parameters, trained for more than three times as many steps. This is the property the lab's own direction depends on: a prior built from a domain is expected to shape what the model can infer in that domain, and this is the smallest setting where that is visible.

**A matched prior gives Bayesian-quality inference from few points.** On the problem inside its prior, the model pretrained on straight boundaries reached 0.961 accuracy from 40 labeled points, 0.011 below the best achievable accuracy on those draws, 0.972, where logistic regression fitted to the same points reached 0.940. Pretraining amortizes the inference a matched Bayesian model would do, which is what the PFN objective promises.

Both hold in each of three pretraining seeds. In every seed the prior moved accuracy on the curved problems 2.5 to 5 times as much as scale did, about three times on average, and the line-prior model came within 0.015 of the best achievable accuracy on the line. No model's accuracy on a problem moves by more than 0.014 from one seed to another.

---

## The rules every measurement follows

- **A protocol fixed before the full run.** The report script's docstring states the arms, the contrasts, the statistics and the expectations before the numbers exist.
- **Every number comes from a saved report.** The notebook's last cell writes the accuracy of every model on every evaluation draw; the report script aggregates those files.
- **Comparisons are paired over evaluation draws.** Every model and every classical learner is scored on the same 20 fixed draws of each problem. Contrasts use a joint paired bootstrap over draws, 10,000 resamples shared by every contrast, with max-t simultaneous 95% intervals; a difference counts only when its interval lies wholly on one side of zero.
- **Several pretraining seeds**, three by default. Each pretraining run starts from the seed alone, so runs do not depend on what ran before; per-draw accuracy is averaged over seeds before comparing, and the spread over seeds is reported beside the intervals.
- **The best achievable accuracy is stated.** The evaluation problems are synthetic with known boundaries, so every accuracy can be read against the accuracy of a classifier that knows the true boundary.

---

## The settings every measurement shares

- **Code:** [`helper.py`](../../helper.py) holds the evaluation problems, the classical learners and the plots; the model, the priors and the training loop are written out in the notebook, following [nanoTabPFN](https://github.com/automl/nanoTabPFN) (Apache-2.0; [arXiv:2511.03634](https://arxiv.org/abs/2511.03634)).
- **Evaluation:** three problems, 20 draws each of 40 labeled and 200 query points, generator seed 1234.
- **Software and hardware:** TabPFN is not used. Seed 0 ran on a laptop CPU with four threads, the fastest setting measured for tensors this small, under Python 3.13, PyTorch 2.14 and scikit-learn 1.9; seeds 1 and 2 ran on an NVIDIA A40 GPU through the lab's pod task, under Python 3.12, PyTorch 2.7 and scikit-learn 1.5. Every per-seed report records its environment. Training times in the reports are wall-clock times, and seed 0's include a period when the laptop was busy with other work, so they are recorded but are not a measurement of the models.

---

## What comes next

- **A harder out-of-prior problem.** The problem meant to lie outside both priors adds a fast wave of amplitude 0.6 to a slow sine, and with points scattered around the boundary with spread 1, 40 labeled points barely resolve it: both wave-prior models score about as well on it as on the plain sine. A problem whose departure from the prior is resolvable from the context, such as a boundary with a sharp step, would test how far pretraining carries beyond its family.
- **A prior from a scientific simulator.** Replace the synthetic boundaries with tasks generated from a simulator of gene regulatory networks under intervention, keep the model and the loop, and measure the result against TabPFN on the lab's own endpoints.
