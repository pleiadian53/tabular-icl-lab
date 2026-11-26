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

The observational study. ``load_nhefs`` reads the NHEFS data that Hernán and Robins use
throughout *Causal Inference: What If*: 1,629 adult smokers examined in 1971 and again
in 1982, whether they quit smoking in between, and how their weight changed. The file is
downloaded once from the book's site into the shared data lake (``$DATA_LAKE_ROOT``, by
default ``~/work/data``) under ``epidemiology/nhefs`` and checked against a fixed
SHA-256. ``textbook_design`` builds the covariate terms of the book's models, with
squares and category indicators, which the linear base model uses there.

The base learners. ``base_learners`` returns three pairs of factories, one regressor
and one classifier each: a linear model, gradient boosting tuned by cross-validated
grid search, and TabPFN, untuned. Every estimator in the notebooks is run with each of
the three, so a difference between rows of a results table is a difference between
the models, not between the estimators.

Compute. TabPFN on a laptop CPU is slow, and the notebooks fit it dozens of times, so
they are executed on a GPU pod (see ``run_te_pod.sh``). With the environment
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

DATA_ROOT = Path(os.environ.get("DATA_LAKE_ROOT") or os.environ.get("SSLLAB_DATA_ROOT") or Path.home() / "work" / "data")
NHEFS_DIR = DATA_ROOT / "epidemiology" / "nhefs"
_NHEFS_URL = "https://miguelhernan.org/s/nhefs.csv"
_NHEFS_SHA256 = "6726827bfcebf7324d786211a517d0515532c2405cc71c5a1fab68d7116ea16c"

NHEFS_COVARIATES = ["sex", "race", "age", "education", "smokeintensity", "smokeyrs", "exercise", "active", "wt71"]
"""The confounders of the book's analysis, measured in 1971: sex (1 female), race (1 not
white), age in years, education (5 levels, 1 to 5), cigarettes per day, years of
smoking, recreational exercise (0 much, 1 moderate, 2 little or none), daily activity
(0 very active, 1 moderately, 2 inactive) and weight in kg."""

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


SCENARIOS = ("smooth", "rules")
"""The two versions of the program: ``smooth`` (the notebook's) and ``rules`` (harder for
models without interactions; see ``simulate_care_program``)."""


def simulate_care_program(n: int = 2000, selection: float = 1.0, seed: int = SEED,
                          scenario: str = "smooth") -> pd.DataFrame:
    """Draw ``n`` members of the simulated care program.

    ``scenario`` chooses how outcomes and the coaches' decisions depend on the
    covariates; the covariates themselves, the noise and the random draws are the same
    in both, so a seed gives the same members in either scenario. ``smooth``, the
    default and the notebook's, is described first; ``rules`` follows.

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

    The ``rules`` scenario keeps that structure but makes every part depend on
    thresholds and combinations of covariates, which a model without interaction terms
    cannot represent. With ``[...]`` an indicator, 1 when the condition holds and 0
    otherwise:

    - ``mu0 = 0.1 + 0.3 (h - 8) + 0.35 [h >= 9.5] + 0.3 tanh((a - 12) / 5) (1 + [h >= 9])
      - 0.3 u [bmi >= 36] - 0.25 ((age - 52.5) / 13)^2``. Engagement matters twice as
      much for members with a high HbA1c, insulin users with a high BMI improve less, and
      improvement peaks in middle age.
    - ``true_effect = 0.1 + 0.45 [h >= 8.5] [a >= 12] + 0.25 max(h - 7.5, 0) (1 - u)
      + 0.15 [age < 45] [a >= 18] - 0.15 u [bmi >= 36]``. The call helps most when a high
      HbA1c meets an engaged member, its slope in HbA1c applies only to members not on
      insulin, and it slightly harms insulin users with a high BMI.
    - ``true_propensity = sigmoid(-0.7 + selection (1.5 [h >= 9] + 1.1 [a >= 15] [age < 55]
      + 0.9 [h >= 8] [a >= 10] - 1.0 u [bmi >= 36]))``: coaches follow rules of thumb,
      calling members with a high HbA1c, young engaged members, and members who are both
      above 8 and active, and avoiding insulin users with a high BMI.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    age = rng.uniform(30, 75, n)
    h = np.clip(rng.normal(8.0, 1.2, n), 6.0, 13.0)
    bmi = np.clip(rng.normal(34, 5, n), 22, 55)
    years = np.minimum(rng.exponential(7, n), 30)
    insulin = rng.binomial(1, _sigmoid(-1.2 + 0.6 * (h - 8) + 0.1 * (years - 7)))
    app_days = rng.binomial(30, rng.beta(2, 3, n))
    sex = rng.binomial(1, 0.5, n)
    ldl = np.clip(rng.normal(100, 30, n), 40, 250)

    if scenario == "smooth":
        mu0 = 0.35 * (h - 8) + 0.25 * np.tanh((app_days - 12) / 6) - 0.15 * insulin - 0.02 * (bmi - 34) - 0.005 * (age - 52.5)
        effect = 0.2 + 0.3 * np.maximum(h - 7.5, 0) + 0.25 * _sigmoid((app_days - 15) / 3) - 0.2 * insulin
        z = 0.9 * (h - 8) / 1.2 + 0.7 * (app_days - 12) / 6 - 0.3 * (age - 52.5) / 13
        logit = -0.4 + selection * z
    else:
        heavy_insulin = insulin * (bmi >= 36)
        mu0 = (0.1 + 0.3 * (h - 8) + 0.35 * (h >= 9.5) + 0.3 * np.tanh((app_days - 12) / 5) * (1 + (h >= 9))
               - 0.3 * heavy_insulin - 0.25 * ((age - 52.5) / 13) ** 2)
        effect = (0.1 + 0.45 * (h >= 8.5) * (app_days >= 12) + 0.25 * np.maximum(h - 7.5, 0) * (1 - insulin)
                  + 0.15 * (age < 45) * (app_days >= 18) - 0.15 * heavy_insulin)
        rules = (1.5 * (h >= 9) + 1.1 * (app_days >= 15) * (age < 55) + 0.9 * (h >= 8) * (app_days >= 10)
                 - 1.0 * heavy_insulin)
        logit = -0.7 + selection * rules
    y0 = mu0 + rng.normal(0, 0.5, n)
    y1 = y0 + effect
    propensity = _sigmoid(logit)
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


# ------------------------------------------------------------------ the observational study


def load_nhefs(complete_outcome: bool = True) -> pd.DataFrame:
    """The NHEFS data, downloaded once into the data lake and checked against its SHA-256.

    ``qsmk`` is 1 for people who quit smoking between 1971 and 1982 and ``wt82_71`` is
    their weight change over that time in kg. With ``complete_outcome`` (the default)
    the 63 people whose 1982 weight is missing are dropped, leaving 1,566, the sample
    on which the book reports its unweighted-for-censoring estimates. The book's
    sequence number ``seqn`` is the index.
    """
    import hashlib
    import urllib.request

    path = NHEFS_DIR / "nhefs.csv"
    if not path.exists():
        NHEFS_DIR.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(_NHEFS_URL, headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _NHEFS_SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {_NHEFS_SHA256}; the upstream file changed")
    frame = pd.read_csv(path).set_index("seqn")
    return frame.dropna(subset=["wt82_71"]) if complete_outcome else frame


def textbook_design(frame: pd.DataFrame) -> pd.DataFrame:
    """The covariate terms of the book's propensity and outcome models.

    Sex, race, age and its square, indicators for education levels 2 to 5, cigarettes
    per day and its square, years of smoking and its square, indicators for exercise
    levels 1 and 2 and activity levels 1 and 2, and weight and its square: the terms of
    Program 12.2's propensity model in *What If*.
    """
    out = frame[["sex", "race", "age", "smokeintensity", "smokeyrs", "wt71"]].astype(float).copy()
    for column in ("age", "smokeintensity", "smokeyrs", "wt71"):
        out[f"{column}^2"] = frame[column].astype(float) ** 2
    for column, levels in (("education", [2, 3, 4, 5]), ("exercise", [1, 2]), ("active", [1, 2])):
        for level in levels:
            out[f"{column}={level}"] = (frame[column] == level).astype(float)
    return out


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


def plot_overlap(t, propensities: dict) -> None:
    """Estimated propensity by arm, one panel per base model."""
    if not _PLOTS:
        return
    t = np.asarray(t)
    fig, axes = plt.subplots(1, len(propensities), figsize=(4 * len(propensities), 3.0), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (name, e) in zip(np.atleast_1d(axes), propensities.items()):
        ax.hist(e[t == 0], bins=bins, alpha=0.55, density=True, color="tab:blue", label="did not quit")
        ax.hist(e[t == 1], bins=bins, alpha=0.55, density=True, color="tab:orange", label="quit")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("estimated probability of quitting")
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _finish(fig)


def plot_balance(balance: pd.DataFrame) -> None:
    """Standardized mean differences per covariate, before weighting and after weighting
    by each base model's propensities; the band marks the usual 0.1 threshold."""
    if not _PLOTS:
        return
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(balance) + 1.2))
    rows = np.arange(len(balance))
    markers = {"unweighted": ("o", "black")}
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for k, column in enumerate(balance.columns):
        marker, colour = markers.get(column, ("s", palette[k % len(palette)]))
        ax.scatter(balance[column], rows, marker=marker, color=colour, s=28, label=column, zorder=3)
    ax.axvspan(-0.1, 0.1, color="gray", alpha=0.15, lw=0)
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(rows, balance.index)
    ax.invert_yaxis()
    ax.set_xlabel("standardized mean difference, quit minus did not quit")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    _finish(fig)


def plot_gates(groups: dict) -> None:
    """Doubly robust average effect in each fifth of people ranked by estimated effect,
    with 95% intervals, one panel per base model."""
    if not _PLOTS:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.2), sharey=True)
    for ax, (name, rows) in zip(np.atleast_1d(axes), groups.items()):
        x = [row["group"] for row in rows]
        y = [row["average"].estimate for row in rows]
        err = [1.96 * row["average"].se for row in rows]
        ax.errorbar(x, y, yerr=err, fmt="o", color="tab:blue", capsize=3, label="doubly robust average")
        ax.plot(x, [row["mean_estimate"] for row in rows], "x", color="black", label="mean estimate")
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("fifth of estimated effect (1 lowest)")
    np.atleast_1d(axes)[0].set_ylabel("effect on weight change, kg")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _finish(fig)


def plot_effect_by_covariate(frame: pd.DataFrame, effects: dict, columns: list[str], labels: dict) -> None:
    """Each person's estimated effect against a covariate: one row per base model, one
    column per covariate."""
    if not _PLOTS:
        return
    fig, axes = plt.subplots(len(effects), len(columns), figsize=(3.4 * len(columns), 2.8 * len(effects)),
                             sharey=True, squeeze=False)
    for i, (name, tau) in enumerate(effects.items()):
        for j, column in enumerate(columns):
            ax = axes[i, j]
            ax.scatter(frame[column], tau, s=5, alpha=0.35, color="tab:blue")
            ax.axhline(np.mean(tau), color="black", lw=1)
            if i == len(effects) - 1:
                ax.set_xlabel(labels.get(column, column))
            if j == 0:
                ax.set_ylabel(f"{name}\nestimated effect, kg")
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
