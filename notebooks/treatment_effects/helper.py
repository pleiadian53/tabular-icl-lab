"""The simulated care program, the base learners and the plots for the treatment-effect notebooks.

The question these notebooks ask is how well treatment effects can be estimated from
data in which the treatment was not assigned at random, and how much a strong tabular
predictor such as TabPFN helps when it is used as the model inside the standard effect
estimators. The estimators themselves are general and live in
``tabicllab.treatment_effects``; this module holds what is particular to the notebooks.

The simulated care program. A program for adults with type 2 diabetes offers coaching
calls. Coaches decide whom to call, and they favour members with a high HbA1c (a
measure of average blood sugar over the past three months, in percent) and members
who are active in the program's app. The outcome is the reduction in HbA1c over six
months, in percentage points, so a positive value is an improvement. The simulator
draws each member's covariates, both potential outcomes (the reduction without
coaching and with it), the probability that coaches call them, and whether they were
called. Real data would show only the outcome of the arm a member was in; here every
estimate can be scored against each member's true effect. ``simulate_care_program``
documents every equation.

The base learners. ``base_learners`` returns three pairs of factories, one regressor
and one classifier each: a linear model, gradient boosting tuned by cross-validated
grid search, and TabPFN, untuned. Every estimator in the notebooks is run with each of
the three, so a difference between rows of a results table is a difference between
the models, not between the estimators.

Compute. TabPFN on a laptop CPU is slow, and the notebooks fit it dozens of times, so
they are executed on a GPU pod (see ``run_te_01_pod.sh``). With the environment
variable ``TE_SMOKE=1`` they run on a small table with a light TabPFN and a one-point
grid for gradient boosting, which checks that every cell works without the cost.
``TE_MEMBERS`` and ``TE_FOLDS`` set the table size and the number of cross-fitting folds
directly, for either mode; nothing else in the notebooks assumes a size.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 0
SMOKE = os.environ.get("TE_SMOKE", "") not in ("", "0")
N_MEMBERS = int(os.environ.get("TE_MEMBERS") or (300 if SMOKE else 2000))
"""Members simulated per table: 2,000 for the full run, 300 in a smoke run, or ``TE_MEMBERS``."""
N_FOLDS = int(os.environ.get("TE_FOLDS") or (2 if SMOKE else 5))
"""Cross-fitting folds: 5 for the full run, 2 in a smoke run, or ``TE_FOLDS``."""

COVARIATES = [
    "age",
    "baseline_hba1c",
    "bmi",
    "years_since_diagnosis",
    "on_insulin",
    "app_days_last_month",
    "sex",
    "ldl",
]
"""What the program knows about a member before deciding whether to call them. The last
two have no part in the simulation: real tables carry columns that matter for nothing."""

TRUTH = ["y0", "y1", "expected_y0", "expected_y1", "true_effect", "true_propensity"]
"""Columns only a simulator has. They are never given to a model."""

_PLOTS = True


# --------------------------------------------------------------------------- setup


def configure_plots(plots: bool = True) -> None:
    """Quiet the routine warnings and set the plot style; ``plots=False`` skips drawing."""
    global _PLOTS
    _PLOTS = plots
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="IProgress not found")
    pd.set_option("display.width", 140)
    pd.set_option("display.precision", 3)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def init_tabpfn(version: str | None = None) -> str:
    """Select the local TabPFN weights and keep the model off Apple's MPS backend.

    With ``version=None`` the choice is TabPFN-3.5 when a license token is found or its
    weights are already in the model cache (as on the pod's network volume), and
    TabPFN-2 otherwise. TabPFN then runs on a CUDA GPU when one is present and on the
    CPU otherwise. Returns the version.
    """
    from tabpfn.browser_auth import get_cached_token
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings

    if version is None:
        cache = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", Path.home() / "Library" / "Caches" / "tabpfn"))
        has_weights = any(cache.glob("tabpfn-v3.5-*.safetensors")) if cache.exists() else False
        has_token = bool(os.environ.get("TABPFN_TOKEN") or get_cached_token())
        version = "v3.5" if (has_token or has_weights) else "v2"
    settings.tabpfn.model_version = ModelVersion(version)
    os.environ.setdefault("TABPFN_EXCLUDE_DEVICES", "mps")
    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"TabPFN {version} weights, running on {device}" + (" (smoke run)" if SMOKE else ""))
    return version


# ----------------------------------------------------------------- the simulated program


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def simulate_care_program(n: int = 2000, selection: float = 1.0, seed: int = SEED) -> pd.DataFrame:
    """Draw ``n`` members of the simulated care program.

    Covariates, drawn independently except where stated:

    - ``age``: uniform on 30 to 75 years.
    - ``baseline_hba1c``: normal with mean 8.0 and standard deviation 1.2, kept within
      6 to 13 percent.
    - ``bmi``: normal with mean 34 and standard deviation 5, kept within 22 to 55.
    - ``years_since_diagnosis``: exponential with mean 7, at most 30.
    - ``on_insulin``: 1 with probability ``sigmoid(-1.2 + 0.6 (hba1c - 8) + 0.1 (years - 7))``,
      so members with higher HbA1c and a longer history are more often on insulin.
    - ``app_days_last_month``: days the member used the app in the past 30, binomial with
      a member-specific rate drawn from Beta(2, 3), so about 12 days on average.
    - ``sex`` and ``ldl``: drawn but used nowhere below.

    Potential outcomes, the HbA1c reduction over six months without coaching (``y0``)
    and with it (``y1 = y0 + true_effect``), with ``h`` the baseline HbA1c, ``a`` the
    app days and ``u`` the insulin indicator:

    - ``mu0 = 0.35 (h - 8) + 0.25 tanh((a - 12) / 6) - 0.15 u - 0.02 (bmi - 34) - 0.005 (age - 52.5)``.
      Members with a high HbA1c improve more without help (a high reading is partly a
      bad stretch that passes, and partly prompts other care), and engaged members
      improve more.
    - ``true_effect = 0.2 + 0.3 max(h - 7.5, 0) + 0.25 sigmoid((a - 15) / 3) - 0.2 u``.
      Coaching helps everyone a little, helps more when HbA1c is high and when the
      member is engaged, and helps less for members on insulin, whose levels depend
      more on dosing than on habits.
    - ``y0 = mu0 + noise``, with normal noise of standard deviation 0.5. The frame keeps
      ``expected_y0 = mu0`` and ``expected_y1 = mu0 + true_effect``, the averages an
      outcome model should recover, so outcome models can be scored too.

    Who gets coached: coaches call a member with probability
    ``true_propensity = sigmoid(-0.4 + selection (0.9 z_h + 0.7 z_a - 0.3 z_age))``,
    where ``z_h = (h - 8) / 1.2``, ``z_a = (a - 12) / 6`` and ``z_age = (age - 52.5) / 13``
    put the three on a common scale. ``selection`` sets how strongly coaches choose:
    0 is a coin flip with the same chance for everyone (a randomized trial), 1 is the
    default, and larger values make the two arms ever more different.

    The coached members therefore have higher HbA1c and more app days: the same two
    things that make members improve without coaching. That is confounding, and it is
    the reason a plain comparison of the two arms overstates the effect.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(30, 75, n)
    h = np.clip(rng.normal(8.0, 1.2, n), 6.0, 13.0)
    bmi = np.clip(rng.normal(34, 5, n), 22, 55)
    years = np.minimum(rng.exponential(7, n), 30)
    insulin = rng.binomial(1, _sigmoid(-1.2 + 0.6 * (h - 8) + 0.1 * (years - 7)))
    app_days = rng.binomial(30, rng.beta(2, 3, n))
    sex = rng.binomial(1, 0.5, n)
    ldl = np.clip(rng.normal(100, 30, n), 40, 250)

    mu0 = 0.35 * (h - 8) + 0.25 * np.tanh((app_days - 12) / 6) - 0.15 * insulin - 0.02 * (bmi - 34) - 0.005 * (age - 52.5)
    effect = 0.2 + 0.3 * np.maximum(h - 7.5, 0) + 0.25 * _sigmoid((app_days - 15) / 3) - 0.2 * insulin
    y0 = mu0 + rng.normal(0, 0.5, n)
    y1 = y0 + effect
    z = 0.9 * (h - 8) / 1.2 + 0.7 * (app_days - 12) / 6 - 0.3 * (age - 52.5) / 13
    propensity = _sigmoid(-0.4 + selection * z)
    coached = rng.binomial(1, propensity)

    frame = pd.DataFrame({
        "age": age.round(1),
        "baseline_hba1c": h.round(2),
        "bmi": bmi.round(1),
        "years_since_diagnosis": years.round(1),
        "on_insulin": insulin,
        "app_days_last_month": app_days,
        "sex": sex,
        "ldl": ldl.round(0),
        "coached": coached,
        "hba1c_reduction": np.where(coached == 1, y1, y0),
        "y0": y0,
        "y1": y1,
        "expected_y0": mu0,
        "expected_y1": mu0 + effect,
        "true_effect": effect,
        "true_propensity": propensity,
    })
    frame.index = pd.Index([f"M{i:05d}" for i in range(1, n + 1)], name="member")
    return frame


# ------------------------------------------------------------------------ base learners


def base_learners(seed: int = SEED) -> dict:
    """Three base models, each a ``(regressor factory, classifier factory)`` pair.

    - ``linear``: ordinary least squares and logistic regression on standardized
      covariates, no interaction terms; what an analyst would fit first.
    - ``gradient boosting``: XGBoost tuned by 3-fold cross-validated grid search over
      tree depth (2, 3, 5), learning rate (0.03, 0.1) and number of trees (100, 300),
      refitted on all rows it is given; the strong conventional reference.
    - ``TabPFN``: TabPFN's regressor and classifier with their defaults, untuned.

    Each factory returns a fresh, unfitted model, so the estimators in
    ``tabicllab.treatment_effects`` can fit one per fold and per arm.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from xgboost import XGBClassifier, XGBRegressor

    grid = {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1], "n_estimators": [100, 300]}
    if SMOKE:
        grid = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [50]}
    pfn = {"n_estimators": 1} if SMOKE else {}
    jobs = 2 if SMOKE else min(12, os.cpu_count() or 1)   # grid points fitted in parallel

    return {
        "linear": (
            lambda: make_pipeline(StandardScaler(), LinearRegression()),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        ),
        "gradient boosting": (
            lambda: GridSearchCV(XGBRegressor(tree_method="hist", random_state=seed, n_jobs=1), grid, cv=3,
                                 scoring="neg_mean_squared_error", n_jobs=jobs),
            lambda: GridSearchCV(XGBClassifier(tree_method="hist", random_state=seed, n_jobs=1), grid, cv=3,
                                 scoring="neg_log_loss", n_jobs=jobs),
        ),
        "TabPFN": (
            lambda: TabPFNRegressor(random_state=seed, **pfn),
            lambda: TabPFNClassifier(random_state=seed, **pfn),
        ),
    }


# --------------------------------------------------------------------------- the plots


def _finish(fig) -> None:
    fig.tight_layout()
    plt.show()


def plot_who_gets_coached(frame: pd.DataFrame) -> None:
    """Baseline HbA1c and app days of coached and uncoached members, and the true
    propensity by arm: what the coaches' choices look like."""
    if not _PLOTS:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    arms = {"not coached": frame["coached"] == 0, "coached": frame["coached"] == 1}
    colours = {"not coached": "tab:blue", "coached": "tab:orange"}
    for ax, column, label in zip(axes, ["baseline_hba1c", "app_days_last_month", "true_propensity"],
                                 ["baseline HbA1c, %", "app days in the last month", "true probability of a call"]):
        bins = np.linspace(frame[column].min(), frame[column].max(), 31)
        for arm, mask in arms.items():
            ax.hist(frame.loc[mask, column], bins=bins, alpha=0.55, color=colours[arm], label=arm, density=True)
        ax.set_xlabel(label)
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    _finish(fig)


def plot_propensities(frame: pd.DataFrame, estimated: dict) -> None:
    """Each base model's estimated propensity against the true one, one panel per model."""
    if not _PLOTS:
        return
    fig, axes = plt.subplots(1, len(estimated), figsize=(4 * len(estimated), 3.6), sharey=True)
    for ax, (name, e) in zip(np.atleast_1d(axes), estimated.items()):
        ax.scatter(frame["true_propensity"], e, s=4, alpha=0.4, color="tab:blue")
        ax.plot([0, 1], [0, 1], color="black", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("true probability of a call")
    np.atleast_1d(axes)[0].set_ylabel("estimated probability")
    _finish(fig)


def plot_effect_estimates(frame: pd.DataFrame, estimates: dict, learners: list[str], methods: list[str]) -> None:
    """Estimated against true individual effect: one row per estimator, one column per
    base model. Points on the diagonal are exact; a flat cloud is an estimator that
    gives everyone nearly the same effect."""
    if not _PLOTS:
        return
    truth = frame["true_effect"].to_numpy()
    low, high = np.percentile(truth, [0.5, 99.5])
    pad = 0.6 * (high - low)
    fig, axes = plt.subplots(len(methods), len(learners), figsize=(3.4 * len(learners), 3.0 * len(methods)),
                             sharex=True, sharey=True, squeeze=False)
    for i, method in enumerate(methods):
        for j, learner in enumerate(learners):
            ax = axes[i, j]
            ax.scatter(truth, estimates[(learner, method)], s=3, alpha=0.3, color="tab:blue")
            ax.plot([low, high], [low, high], color="black", lw=1)
            ax.set_ylim(low - pad, high + pad)
            if i == 0:
                ax.set_title(learner, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{method}\nestimated effect")
            if i == len(methods) - 1:
                ax.set_xlabel("true effect")
    _finish(fig)


def plot_selection_sweep(sweep: pd.DataFrame) -> None:
    """Error of each average-effect estimator as the coaches' selection grows, one panel
    per base model, with the share of members whose true propensity is extreme."""
    if not _PLOTS:
        return
    learners = list(dict.fromkeys(sweep["learner"]))
    fig, axes = plt.subplots(1, len(learners), figsize=(4 * len(learners), 3.4), sharey=True)
    for ax, learner in zip(np.atleast_1d(axes), learners):
        part = sweep[sweep["learner"] == learner]
        for method, style in [("naive", ":"), ("IPW", "--"), ("AIPW", "-")]:
            rows = part[part["method"] == method]
            ax.plot(rows["selection"], rows["error"], style, marker="o", ms=4, label=method)
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(learner, fontsize=10)
        ax.set_xlabel("selection strength")
    np.atleast_1d(axes)[0].set_ylabel("estimate minus true average effect")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _finish(fig)
