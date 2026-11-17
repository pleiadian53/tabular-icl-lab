# 01 · Pretraining from scratch: the prior and the scale

*Part of [the results ledger](index.md), which holds what the measurements add up to, the rules they follow and the settings they share. Notebook: [`01_pretraining_from_scratch.ipynb`](../../01_pretraining_from_scratch.ipynb). Report script: [`01_pretraining_from_scratch_report.py`](../../01_pretraining_from_scratch_report.py), whose docstring fixes the protocol.*

> **Status: provisional, one pretraining seed of three.** Every number below comes from the run with seed 0, aggregated into [`reports/01-pretraining-from-scratch.json`](reports/01-pretraining-from-scratch.json), which records `"complete": false`. The intervals describe uncertainty over evaluation draws for this one pretraining run, not over pretraining seeds. Seeds 1 and 2 will replace these numbers with seed-averaged ones.

**Question.** A prior-fitted network learns everything it can infer from the synthetic tables it is pretrained on. With 40 labeled points as context, how much do the prior and the model's scale each determine its accuracy, on problems inside and outside the family it was pretrained on, and how does it compare with classical learners fitted to the same points?

**Answer, from seed 0.** The prior decides the shape of what the model can infer. Pretrained on straight boundaries, the model draws a straight boundary on every problem, curved or not. Pretraining the same model for the same number of steps on boundaries with added waves raises its accuracy on the two curved problems by 0.083 and 0.086 and costs 0.011 on the straight one. A model with nearly six times as many parameters, trained on the wave prior for more than three times as many steps, adds 0.033 on each curved problem and nothing on the straight one. That larger model beats a random forest on all three problems, by 0.07 to 0.09, and logistic regression on the two curved problems; on the straight problem it ties logistic regression. On the problem inside its prior, the model pretrained on straight boundaries comes within 0.012 of the best achievable accuracy.

## What was measured

**The priors.** A synthetic table has 50 rows and two features. A boundary curve $f$ is drawn, then each row gets a first feature $x_1 \sim \mathcal{N}(0, 1)$ and a second feature $x_2 = f(x_1) + u$ with $u \sim \mathcal{N}(0, 1)$, so points scatter around the curve. The label is $y = 1$ when $u + e > 0$ and $y = 0$ otherwise, where $e \sim \mathcal{N}(0, 0.1^2)$ is label noise that blurs the labels of points close to the boundary. The two priors differ only in the curve:

- **line prior:** $f(x_1) = a + b x_1$, with intercept $a$ and slope $b$ each drawn from $\mathcal{N}(0, 1)$;
- **wave prior:** $f(x_1) = a + b x_1 + A \sin(\omega x_1 + \varphi)$, with the same $a$ and $b$, amplitude $A$ uniform on $[0, 2]$, angular frequency $\omega$ uniform on $[0.5, 5]$ and phase $\varphi$ uniform on $[0, 2\pi]$.

The first $n$ rows of a table are the labeled context and the rest are queries, with $n$ drawn between 10 and 47 for each batch.

**The model.** A transformer that embeds every cell of the table, with the label as one more column, and alternates attention across the features of a row with attention from every row to the labeled rows of the same column. It follows [nanoTabPFN](https://github.com/automl/nanoTabPFN), and the notebook writes it out in full. It is pretrained with the cross-entropy of the query labels, over batches of 32 fresh tables, with AdamW at learning rate $10^{-3}$ decaying to 0 along a cosine schedule.

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

Each problem has 20 fixed draws of 40 labeled and 200 query points, generator seed 1234. A model's accuracy on a draw is the fraction of the 200 query points it classifies correctly, and its accuracy on a problem is the mean over the 20 draws. Every model and learner sees exactly the same points.

**The references.** Logistic regression and a random forest of 100 trees, each fitted to the 40 labeled points of every draw. The **ceiling** is the classifier that knows the true boundary: it errs only where label noise flipped a point, which happens with probability $\arctan(0.1)/\pi \approx 0.032$, so its expected accuracy is about 0.968 on every problem; on these particular draws it is 0.972, 0.972 and 0.969.

## Results

**Accuracy**, mean over the 20 draws of each problem:

| | line | sine | sine + fast wave | mean |
|---|---|---|---|---|
| small, line prior | **0.960** | 0.805 | 0.799 | 0.855 |
| small, wave prior | 0.949 | 0.888 | 0.885 | 0.907 |
| large, wave prior | 0.951 | **0.920** | **0.917** | **0.929** |
| logistic regression | 0.940 | 0.776 | 0.798 | 0.838 |
| random forest | 0.862 | 0.827 | 0.846 | 0.845 |
| ceiling: the true boundary | 0.972 | 0.972 | 0.969 | 0.971 |

**Contrasts**, paired over the 20 draws, with max-t simultaneous 95% intervals over the twelve contrasts (critical value 3.39). A difference counts when its interval excludes zero:

| Contrast | Problem | Difference | 95% simultaneous interval | Counts |
|---|---|---|---|---|
| prior: small wave − small line | line | −0.011 | [−0.020, −0.002] | yes |
| | sine | +0.083 | [+0.051, +0.115] | yes |
| | sine + fast wave | +0.086 | [+0.061, +0.110] | yes |
| scale: large wave − small wave | line | +0.002 | [−0.009, +0.012] | no |
| | sine | +0.033 | [+0.015, +0.050] | yes |
| | sine + fast wave | +0.033 | [+0.011, +0.055] | yes |
| large wave − random forest | line | +0.089 | [+0.061, +0.117] | yes |
| | sine | +0.094 | [+0.065, +0.123] | yes |
| | sine + fast wave | +0.071 | [+0.040, +0.102] | yes |
| large wave − logistic regression | line | +0.011 | [−0.009, +0.031] | no |
| | sine | +0.145 | [+0.112, +0.178] | yes |
| | sine + fast wave | +0.119 | [+0.095, +0.143] | yes |

**What the decision surfaces show.** The notebook plots each model's predicted probability over the plane for the first draw of every problem. The model pretrained on straight boundaries draws a straight boundary through the sine and the fast wave alike, so its errors there are those of the best straight line. Both wave-prior models bend with the sine; the larger one follows it more tightly and picks up part of the fast wave near the centre of the data, where the labeled points are dense.

## Against the expectations stated before the run

- *The wave prior helps on the sine problem and not on the line.* Confirmed, with a small, resolved cost on the line: a prior that spreads its probability over wavy boundaries gives a little less weight to straight ones.
- *Scale helps most where the prior already covers the problem.* Not confirmed in this form: scale added the same 0.033 on the sine, inside the wave prior, and on the sine plus fast wave, outside it, and nothing on the line, where the small model is already within 0.023 of the ceiling.
- *On the problem outside both priors, nothing was predicted.* Both wave-prior models scored as well on it as on the plain sine. That says less about generalization than it seems: the fast wave has amplitude 0.6 against a scatter of 1 around the boundary, so 40 labeled points barely resolve it, and a boundary close to the slow sine alone already classifies most points correctly.

## What this does and does not show

It shows, in the smallest setting where it can be measured, that the prior moves a pretrained in-context learner's inference more than its scale does: about two and a half times as much on the curved problems. It also shows that a matched prior gives near-optimal inference from few points. On the line problem, the line-prior model reached 0.960 against the ceiling of 0.972 and logistic regression's 0.940; logistic regression is the right model family there but must estimate its two parameters from 40 points without a prior. That comparison is a point value outside the pre-stated contrasts.

It does not show how these effects vary across pretraining runs, which seeds 1 and 2 will measure, or how far pretraining carries beyond its family, for which the fast wave is too weak a test. Nor does it say anything about TabPFN, whose prior and scale are of a different order.

## Reproducing

From `notebooks/pfn_pretraining/`, in the `tabicllab` environment:

```bash
python 01_pretraining_from_scratch_report.py --run 1 2 --no-aggregate   # pretrain with seeds 1 and 2
python 01_pretraining_from_scratch_report.py                            # aggregate seeds 0, 1 and 2
python 01_pretraining_from_scratch_report.py --seeds 0                  # this page's provisional report
```

Per-seed reports: [`reports/01-pretraining-from-scratch-seed0.json`](reports/01-pretraining-from-scratch-seed0.json). The training times recorded in it are wall-clock times that include a period when the laptop was busy with other work; on an idle 8-core laptop CPU the notebook takes about 40 minutes.
