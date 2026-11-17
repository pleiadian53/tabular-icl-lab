"""Evaluation problems, baselines and plots for pretraining a tabular in-context learner.

The notebook in this folder pretrains a small prior-fitted network (PFN) from scratch:
a transformer that reads a table of labeled rows and predicts the labels of new rows in
one forward pass, trained only on synthetic tables drawn from a prior. This module holds
everything around the model that the notebook should not have to repeat: the device,
three fixed evaluation problems the model never sees during pretraining, two classical
learners fitted on the same labeled points as a reference, and the plots.

The evaluation problems. Each is a two-feature binary classification problem of the kind
the prior generates: a point (x1, x2) has x1 drawn from a standard normal, x2 scattered
around a boundary curve f(x1) with standard-normal spread, and label 1 when it lies above
the curve (with a little label noise near it, as in the prior). The three curves are

    line                  f(x1) = 0.3 + 0.8 x1
    sine                  f(x1) = 1.5 sin(1.5 x1)
    sine + fast wave      f(x1) = sin(1.5 x1) + 0.6 sin(4 x1)

The line lies inside both priors in the notebook; the sine lies inside the wave prior but
not the line prior; the sum of two sines lies outside both, so it measures how far what
the model learned carries beyond the family it was pretrained on.

Each problem is scored over ``N_DRAWS`` independent draws of ``N_CONTEXT`` labeled points
and ``N_QUERY`` query points, with a fixed generator seed, so every model and baseline sees
exactly the same points and a single lucky draw cannot decide a comparison.

Device: CUDA when available, otherwise the CPU with four threads. Apple's MPS backend is not
used by default.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# On a CPU the tensors here are tiny (32 tables of 50 rows and 3 cells), so thread-pool
# overhead dominates: on an 8-core laptop a training step of the larger model took 0.74 s
# with 4 threads, 1.0 s with 8 and 1.2 s with 1. Four threads is the measured best.
if device.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))

N_CONTEXT = 40
N_QUERY = 200
N_DRAWS = 20
NOISE_STD = 0.1
EVAL_SEED = 1234

PROBLEMS = {
    "line": lambda t: 0.3 + 0.8 * t,
    "sine": lambda t: 1.5 * torch.sin(1.5 * t),
    "sine + fast wave": lambda t: torch.sin(1.5 * t) + 0.6 * torch.sin(4 * t),
}


def draw_problem(boundary, n_points: int, generator: torch.Generator):
    """Points scattered around ``boundary``, labeled 1 above it (with label noise)."""
    x1 = torch.randn(n_points, generator=generator)
    x2 = boundary(x1) + torch.randn(n_points, generator=generator)
    noise = NOISE_STD * torch.randn(n_points, generator=generator)
    y = (x2 + noise > boundary(x1)).float()
    return torch.stack([x1, x2], dim=-1), y


def evaluation_sets(seed: int = EVAL_SEED) -> dict[str, list[tuple]]:
    """For every problem, ``N_DRAWS`` fixed (x_context, y_context, x_query, y_query) draws."""
    generator = torch.Generator().manual_seed(seed)
    sets = {}
    for name, boundary in PROBLEMS.items():
        draws = []
        for _ in range(N_DRAWS):
            x, y = draw_problem(boundary, N_CONTEXT + N_QUERY, generator)
            draws.append((x[:N_CONTEXT], y[:N_CONTEXT], x[N_CONTEXT:], y[N_CONTEXT:]))
        sets[name] = draws
    return sets


_EVAL_SETS = evaluation_sets()


@torch.no_grad()
def model_probabilities(model, x_context, y_context, x_query) -> np.ndarray:
    """P(y = 1) for the query rows, from a model with the notebook's call signature."""
    model.eval()
    logits = model(
        x_context[None].to(device), y_context[None].to(device), x_query[None].to(device)
    )
    model.train()
    return torch.softmax(logits[0], dim=-1)[:, 1].cpu().numpy()


def accuracy_per_draw(model) -> dict[str, list[float]]:
    """Query accuracy on every fixed draw of every evaluation problem."""
    return {
        name: [
            float(((model_probabilities(model, xc, yc, xq) > 0.5) == yq.numpy().astype(bool)).mean())
            for xc, yc, xq, yq in draws
        ]
        for name, draws in _EVAL_SETS.items()
    }


def accuracy_by_problem(model) -> dict[str, float]:
    """Mean query accuracy over the fixed draws of each evaluation problem."""
    return {name: float(np.mean(accs)) for name, accs in accuracy_per_draw(model).items()}


def evaluate_model(model) -> float:
    """Mean accuracy over the three evaluation problems: the number logged during pretraining."""
    return float(np.mean(list(accuracy_by_problem(model).values())))


def ceiling_accuracy_per_draw() -> dict[str, list[float]]:
    """Query accuracy of the classifier that knows the true boundary: the best achievable.

    It labels a point 1 when it lies above the boundary and still errs where the label
    noise flipped a point near the boundary. With the point's offset from the boundary
    u ~ N(0, 1) and label noise e ~ N(0, NOISE_STD^2), it errs with probability
    P(sign(u + e) != sign(u)) = arctan(NOISE_STD) / pi, about 0.032, so its expected
    accuracy is about 0.968 on every problem; on the fixed draws it varies a little.
    """
    return {
        name: [
            float(((xq[:, 1] > PROBLEMS[name](xq[:, 0])).float() == yq).float().mean())
            for _, _, xq, yq in draws
        ]
        for name, draws in _EVAL_SETS.items()
    }


BASELINES = {
    "logistic regression": lambda: LogisticRegression(),
    "random forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
}


def baseline_accuracy_per_draw(name: str) -> dict[str, list[float]]:
    """Query accuracy of a classical learner fitted on each draw's labeled points."""
    make = BASELINES[name]
    scores = {}
    for problem, draws in _EVAL_SETS.items():
        accs = []
        for xc, yc, xq, yq in draws:
            if len(torch.unique(yc)) < 2:  # a draw with one class: predict it
                accs.append(float((yq == yc[0]).float().mean()))
                continue
            clf = make().fit(xc.numpy(), yc.numpy())
            accs.append(float((clf.predict(xq.numpy()) == yq.numpy()).mean()))
        scores[problem] = accs
    return scores


def compare_with_baseline(models: dict) -> pd.DataFrame:
    """Accuracy per problem for each pretrained model and two classical learners.

    The learners are fitted on the same 40 labeled points the pretrained models read as
    context: logistic regression (a straight boundary) and a random forest with 100 trees.
    """
    rows = {name: accuracy_by_problem(model) for name, model in models.items()}
    for name in BASELINES:
        rows[name] = {p: float(np.mean(a)) for p, a in baseline_accuracy_per_draw(name).items()}
    table = pd.DataFrame(rows).T
    table["mean"] = table.mean(axis=1)
    return table.round(3)


# ----------------------------------------------------------------------- plotting


def _scatter(ax, x, y):
    x, y = x.cpu().numpy(), y.cpu().numpy().astype(bool)
    ax.scatter(x[y, 0], x[y, 1], s=14, c="crimson", marker="o", label="class 1")
    ax.scatter(x[~y, 0], x[~y, 1], s=18, c="royalblue", marker="x", label="class 0")


def plot_prior_samples(x, y, boundary) -> None:
    """A few synthetic tables from the prior, each with its true boundary (dashed)."""
    n = x.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharex=True, sharey=True)
    grid = torch.linspace(-3, 3, 200)
    curves = boundary(grid.expand(n, -1))
    for i, ax in enumerate(np.atleast_1d(axes)):
        _scatter(ax, x[i], y[i])
        ax.plot(grid, curves[i], "k--", lw=1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x1")
    np.atleast_1d(axes)[0].set_ylabel("x2")
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def plot_predictions(model, resolution: int = 80) -> None:
    """The model's P(class 1) over the plane for the first draw of each evaluation problem.

    Red is class 1, blue class 0 and white an even split; the dashed line is the true
    boundary and the markers are the 40 labeled points the model reads as context.
    """
    g1, g2 = torch.meshgrid(torch.linspace(-3, 3, resolution), torch.linspace(-5, 5, resolution), indexing="xy")
    grid = torch.stack([g1.flatten(), g2.flatten()], dim=-1)
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(3.6 * len(PROBLEMS), 3.4), sharey=True)
    for ax, (name, boundary) in zip(axes, PROBLEMS.items()):
        xc, yc, _, _ = _EVAL_SETS[name][0]
        proba = model_probabilities(model, xc, yc, grid).reshape(resolution, resolution)
        ax.imshow(proba, origin="lower", extent=(-3, 3, -5, 5), aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75)
        t = torch.linspace(-3, 3, 200)
        ax.plot(t, boundary(t), "k--", lw=1)
        _scatter(ax, xc, yc)
        ax.set_title(name)
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.tight_layout()
    plt.show()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

