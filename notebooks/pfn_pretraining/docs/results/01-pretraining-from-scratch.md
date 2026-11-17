# 01 · Pretraining from scratch: the prior and the scale

*Part of [the results ledger](index.md), which holds what the measurements add up to, the rules they follow and the settings they share. Notebook: [`01_pretraining_from_scratch.ipynb`](../../01_pretraining_from_scratch.ipynb), which shows the run with pretraining seed 0. Report script: [`01_pretraining_from_scratch_report.py`](../../01_pretraining_from_scratch_report.py), whose docstring fixes the protocol. Report: [`reports/01-pretraining-from-scratch.json`](reports/01-pretraining-from-scratch.json), aggregated over pretraining seeds 0, 1 and 2.*

**Question.** A prior-fitted network learns everything it can infer from the synthetic tables it is pretrained on. With 40 labeled points as context, how much do the prior and the model's scale each determine its accuracy, on problems inside and outside the family it was pretrained on, and how does it compare with classical learners fitted to the same points?

**Answer.** The prior decides the shape of what the model can infer, and it moves accuracy about three times as much as scale does. Pretrained on straight boundaries, the model draws a straight boundary on every problem, curved or not. Pretraining the same model for the same number of steps on boundaries with added waves raises its accuracy on the two curved problems by 0.087 and 0.086 and costs 0.013 on the straight one. A model with nearly six times as many parameters, trained on the wave prior for more than three times as many steps, adds 0.028 and 0.029 on the curved problems and nothing measurable on the straight one. That larger model beats a random forest on all three problems, by 0.07 to 0.09, and logistic regression on the two curved problems; on the straight problem it ties logistic regression. On the problem inside its prior, the model pretrained on straight boundaries comes within 0.011 of the best achievable accuracy. All three pretraining seeds tell the same story: the seed-to-seed spread of any model's accuracy is at most 0.014.

## What was measured

**The priors.** A synthetic table has 50 rows and two features. A boundary curve $f$ is drawn, then each row gets a first feature $x_1 \sim \mathcal{N}(0, 1)$ and a second feature $x_2 = f(x_1) + u$ with $u \sim \mathcal{N}(0, 1)$, so points scatter around the curve. The label is $y = 1$ when $u + e > 0$ and $y = 0$ otherwise, where $e \sim \mathcal{N}(0, 0.1^2)$ is label noise that blurs the labels of points close to the boundary. The two priors differ only in the curve:

- **line prior:** $f(x_1) = a + b x_1$, with intercept $a$ and slope $b$ each drawn from $\mathcal{N}(0, 1)$;
- **wave prior:** $f(x_1) = a + b x_1 + A \sin(\omega x_1 + \varphi)$, with the same $a$ and $b$, amplitude $A$ uniform on $[0, 2]$, angular frequency $\omega$ uniform on $[0.5, 5]$ and phase $\varphi$ uniform on $[0, 2\pi]$.

The first $n$ rows of a table are the labeled context and the rest are queries, with $n$ drawn between 10 and 47 for each batch.

**The model.** A transformer that embeds every cell of the table, with the label as one more column, and alternates attention across the features of a row with attention from every row to the labeled rows of the same column. It follows [nanoTabPFN](https://github.com/automl/nanoTabPFN), and the notebook writes it out in full. It is pretrained with the cross-entropy of the query labels, over batches of 32 fresh tables, with AdamW at learning rate $10^{-3}$ decaying to 0 along a cosine schedule. Each pretraining run starts from the seed alone.

| Arm | Embedding width | Layers | Parameters | Steps | Prior |
|---|---|---|---|---|---|
| small, line prior | 48 | 2 | 85,538 | 750 | line |
| small, wave prior | 48 | 2 | 85,538 | 750 | wave |
| large, wave prior | 96 | 3 | 486,242 | 2,500 | wave |

**The evaluation problems.** Three fixed boundaries, generated the same way as the priors, that no model trains on:

| Problem | Boundary $f(x_1)$ | Inside the line prior | Inside the wave prior |
|---|---|---|---|
| line | $0.3 + 0.8 x_1$ | yes | yes |
| sine | $1.5 \sin(1.5 x_1)$ | no | yes |
| sine + fast wave | $\sin(1.5 x_1) + 0.6 \sin(4 x_1)$ | no | no |

Each problem has 20 fixed draws of 40 labeled and 200 query points, generator seed 1234. A model's accuracy on a draw is the fraction of the 200 query points it classifies correctly. For a pretrained model it is averaged over the three pretraining seeds draw by draw, and its accuracy on a problem is then the mean over the 20 draws. Every model and learner sees exactly the same points.

**The references.** Logistic regression and a random forest of 100 trees, each fitted to the 40 labeled points of every draw. The **ceiling** is the classifier that knows the true boundary: it errs only where label noise flipped a point, which happens with probability $\arctan(0.1)/\pi \approx 0.032$, so its expected accuracy is about 0.968 on every problem; on these particular draws it is 0.972, 0.972 and 0.969.

## Results

**Accuracy**, mean over the 20 draws of each problem, pretrained models averaged over three seeds:

| | line | sine | sine + fast wave | mean |
|---|---|---|---|---|
| small, line prior | **0.961** | 0.805 | 0.801 | 0.856 |
| small, wave prior | 0.948 | 0.893 | 0.887 | 0.909 |
| large, wave prior | 0.952 | **0.921** | **0.916** | **0.930** |
| logistic regression | 0.940 | 0.776 | 0.798 | 0.838 |
| random forest | 0.862 | 0.827 | 0.846 | 0.845 |
| ceiling: the true boundary | 0.972 | 0.972 | 0.969 | 0.971 |

**Spread over pretraining seeds**, each seed's accuracy on a problem, for seeds 0, 1 and 2:

| | line | sine | sine + fast wave |
|---|---|---|---|
| small, line prior | 0.960, 0.958, 0.964 | 0.805, 0.806, 0.805 | 0.799, 0.802, 0.802 |
| small, wave prior | 0.949, 0.951, 0.944 | 0.888, 0.902, 0.889 | 0.885, 0.890, 0.886 |
| large, wave prior | 0.951, 0.952, 0.954 | 0.920, 0.919, 0.922 | 0.917, 0.915, 0.915 |

**Contrasts**, paired over the 20 draws, with max-t simultaneous 95% intervals over the twelve contrasts (critical value 3.49). A difference counts when its interval excludes zero. The intervals describe uncertainty over evaluation draws; the seed spread above is the uncertainty over pretraining runs.

| Contrast | Problem | Difference | 95% simultaneous interval | Counts |
|---|---|---|---|---|
| prior: small wave − small line | line | −0.013 | [−0.021, −0.004] | yes |
| | sine | +0.087 | [+0.057, +0.118] | yes |
| | sine + fast wave | +0.086 | [+0.065, +0.108] | yes |
| scale: large wave − small wave | line | +0.004 | [−0.003, +0.011] | no |
| | sine | +0.028 | [+0.015, +0.041] | yes |
| | sine + fast wave | +0.029 | [+0.010, +0.047] | yes |
| large wave − random forest | line | +0.091 | [+0.061, +0.120] | yes |
| | sine | +0.094 | [+0.063, +0.125] | yes |
| | sine + fast wave | +0.070 | [+0.038, +0.101] | yes |
| large wave − logistic regression | line | +0.013 | [−0.007, +0.033] | no |
| | sine | +0.145 | [+0.111, +0.179] | yes |
| | sine + fast wave | +0.117 | [+0.092, +0.143] | yes |

The effect of the prior is the same in every seed: on the sine problem the wave prior gained 0.083, 0.095 and 0.084. The effect of scale varies more, because it is the difference between two accuracies that each move a little: on the sine problem the larger model gained 0.033, 0.018 and 0.033, and on the sine plus fast wave 0.033, 0.024 and 0.029. Every seed agrees on the sign of every contrast that counts.

**What the decision surfaces show.** The notebook plots each model's predicted probability over the plane for the first draw of every problem. The model pretrained on straight boundaries draws a straight boundary through the sine and the fast wave alike, so its errors there are those of the best straight line. Both wave-prior models bend with the sine; the larger one follows it more tightly and picks up part of the fast wave near the centre of the data, where the labeled points are dense.

## Against the expectations stated before the run

- *The wave prior helps on the sine problem and not on the line.* Confirmed, with a small, resolved cost on the line: a prior that spreads its probability over wavy boundaries gives a little less weight to straight ones.
- *Scale helps most where the prior already covers the problem.* Not confirmed in this form: scale added the same amount on the sine, inside the wave prior, as on the sine plus fast wave, outside it, and nothing measurable on the line, where the small model is already within 0.024 of the ceiling.
- *On the problem outside both priors, nothing was predicted.* Both wave-prior models scored as well on it as on the plain sine. That says less about generalization than it seems: the fast wave has amplitude 0.6 against a scatter of 1 around the boundary, so 40 labeled points barely resolve it, and a boundary close to the slow sine alone already classifies most points correctly.

## What this does and does not show

It shows, in the smallest setting where it can be measured, that the prior moves a pretrained in-context learner's inference about three times as much as its scale does on the curved problems. It also shows that a matched prior gives near-optimal inference from few points. On the line problem, the line-prior model reached 0.961 against the ceiling of 0.972 and logistic regression's 0.940; logistic regression is the right model family there but must estimate its two parameters from 40 points without a prior. That comparison is a point value outside the pre-stated contrasts.

It does not show how far pretraining carries beyond its family, for which the fast wave is too weak a test, or how the two effects grow with a richer prior and a larger model. Nor does it say anything about TabPFN, whose prior and scale are of a different order.

## Runs and environments

Seed 0 ran on a laptop CPU, PyTorch 2.14 and scikit-learn 1.9; it is the run saved in the notebook. Seeds 1 and 2 ran on an NVIDIA A40 GPU through the lab's pod task, `ops/skypilot/r4_3.yaml`, with PyTorch 2.7 and scikit-learn 1.5; on the GPU each seed's whole notebook took about 2.3 minutes. Each per-seed report records its environment. The classical learners are scored from seed 0's report: logistic regression gives identical scores in both environments, while the random forest differs on 2 of the 60 draws between the two scikit-learn versions, which moves its mean on one problem by less than 0.001.

The training times in the reports are wall-clock times. Seed 0's include a period when the laptop was busy with other work, so they are recorded but are not a measurement of the models.

## Reproducing

From `notebooks/pfn_pretraining/`, in the `tabicllab` environment:

```bash
python 01_pretraining_from_scratch_report.py --run    # pretrain with seeds 0, 1 and 2, then aggregate
python 01_pretraining_from_scratch_report.py          # aggregate the per-seed reports
```

Per-seed reports: [seed 0](reports/01-pretraining-from-scratch-seed0.json), [seed 1](reports/01-pretraining-from-scratch-seed1.json), [seed 2](reports/01-pretraining-from-scratch-seed2.json). On a GPU pod: `sky launch -c ticl-r4 ops/skypilot/r4_3.yaml`, as described in the repository's `ops/README.md`.
