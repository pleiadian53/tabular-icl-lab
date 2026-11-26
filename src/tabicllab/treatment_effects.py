"""Estimators of average and individual treatment effects, built on any prediction model.

A treatment effect is a difference between two outcomes of the same person: the outcome
with the treatment and the outcome without it. Only one of the two is ever observed, so
no effect can be read off the data directly. The estimators here recover effects from
ordinary prediction problems, and every one of them takes the prediction model as an
argument: a function that returns a fresh scikit-learn-style estimator, so the same
estimator can be built on a linear model, gradient boosting or a tabular in-context
learner such as TabPFN, and the choice of model can be compared on its own.

Notation, used in every docstring below. For person ``i``: covariates ``X[i]`` measured
before treatment, treatment ``t[i]`` (1 treated, 0 not), observed outcome ``y[i]``.
``mu0(x)`` and ``mu1(x)`` are the expected outcome without and with treatment for
covariates ``x``; ``e(x)`` is the propensity, the probability of treatment given ``x``;
``tau(x) = mu1(x) - mu0(x)`` is the conditional average treatment effect (CATE), the
individual-level effect these estimators target, and its average over people is the
average treatment effect (ATE). The three functions ``mu0``, ``mu1`` and ``e`` are called
nuisances: they are not what is wanted, but every estimator needs some of them.

Cross-fitting. Every prediction a person receives comes from a model that did not see
that person: the rows are split into folds, and each fold is predicted by models fitted
on the other folds. Without it, a flexible model that memorizes its training rows makes
residuals look smaller than they are and biases the doubly robust estimators.

The estimators:

- ``s_learner``: one model of the outcome with the treatment as one more column; the
  effect is its prediction with the treatment set to 1 minus with it set to 0.
- ``t_learner``: one outcome model per arm; the effect is the difference of the two.
- ``x_learner`` (Künzel et al., 2019): imputes each person's effect from the other arm's
  model, fits a model to the imputed effects in each arm, and blends the two by the
  propensity.
- ``dr_learner`` (Kennedy, 2023): regresses the doubly robust pseudo-outcome on the
  covariates.
- ``naive_difference``, ``ipw_ate`` and ``aipw_ate``: the average effect as a plain
  difference of means, by inverse propensity weighting, and by the augmented
  (doubly robust) version, which is consistent when either the outcome models or the
  propensity are right.

``cate_scores`` scores estimated individual effects against true ones, which only a
simulation or a semi-synthetic benchmark provides.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

Factory = Callable[[], object]
"""A function returning a fresh, unfitted estimator with ``fit`` and ``predict`` (or
``predict_proba`` for the propensity)."""


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its standard error."""

    estimate: float
    se: float

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Normal-approximation confidence interval."""
        z = stats.norm.ppf(0.5 + level / 2)
        return self.estimate - z * self.se, self.estimate + z * self.se


@dataclass
class Nuisances:
    """Cross-fitted nuisance predictions, one value per person.

    ``mu0`` and ``mu1`` are the predicted outcomes without and with treatment, ``e``
    the predicted propensity after clipping to ``[clip, 1 - clip]``, and ``e_raw`` the
    propensity before clipping.
    """

    mu0: np.ndarray
    mu1: np.ndarray
    e: np.ndarray
    e_raw: np.ndarray


def _arrays(X, t, y=None):
    X = np.asarray(X, dtype=float)
    t = np.asarray(t).astype(int)
    if not set(np.unique(t)) <= {0, 1}:
        raise ValueError("t must be 0 or 1")
    if y is None:
        return X, t
    return X, t, np.asarray(y, dtype=float)


def fold_ids(t, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold number per person, stratified by treatment so every fold has both arms."""
    t = np.asarray(t).astype(int)
    folds = np.empty(len(t), dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(t)), t)):
        folds[test] = k
    return folds


def fit_nuisances(X, t, y, outcome: Factory, propensity: Factory, folds: np.ndarray,
                  clip: float = 0.01) -> Nuisances:
    """Cross-fitted ``mu0``, ``mu1`` and ``e``.

    For each fold, ``mu0`` is fitted on the untreated rows of the other folds, ``mu1``
    on their treated rows and ``e`` on all their rows, and the fold is predicted by all
    three. The propensity is clipped to ``[clip, 1 - clip]`` for use in weights, because
    a weight of one over a propensity near zero lets a single person dominate the
    estimate; ``Nuisances.e_raw`` keeps the unclipped values for diagnostics.
    """
    X, t, y = _arrays(X, t, y)
    mu0, mu1, e = (np.empty(len(y)) for _ in range(3))
    for k in np.unique(folds):
        test, train = folds == k, folds != k
        mu0[test] = outcome().fit(X[train & (t == 0)], y[train & (t == 0)]).predict(X[test])
        mu1[test] = outcome().fit(X[train & (t == 1)], y[train & (t == 1)]).predict(X[test])
        e[test] = propensity().fit(X[train], t[train]).predict_proba(X[test])[:, 1]
    return Nuisances(mu0=mu0, mu1=mu1, e=np.clip(e, clip, 1 - clip), e_raw=e)


def s_learner(X, t, y, model: Factory, folds: np.ndarray) -> np.ndarray:
    """Individual effects from one outcome model with the treatment as a column.

    The simplest estimator, and the one most at risk of shrinking effects: when the
    treatment is one column among many and its effect is small next to the other
    columns', a model can learn to give it little weight, and then the predicted
    effect is pulled towards zero for everyone. A linear model with no interaction
    terms gives the same effect to every person.
    """
    X, t, y = _arrays(X, t, y)
    tau = np.empty(len(y))
    for k in np.unique(folds):
        test, train = folds == k, folds != k
        fitted = model().fit(np.column_stack([X[train], t[train]]), y[train])
        ones, zeros = np.ones(test.sum()), np.zeros(test.sum())
        tau[test] = fitted.predict(np.column_stack([X[test], ones])) - fitted.predict(np.column_stack([X[test], zeros]))
    return tau


def t_learner(nuisances: Nuisances) -> np.ndarray:
    """Individual effects as the difference of the two cross-fitted outcome models.

    Each arm gets its own model, so the treatment cannot be ignored; the price is that
    each model sees only its own arm, and the errors of two separate fits add up in
    their difference.
    """
    return nuisances.mu1 - nuisances.mu0


def x_learner(X, t, y, nuisances: Nuisances, model: Factory, folds: np.ndarray) -> np.ndarray:
    """Individual effects by the X-learner.

    Each person's effect is first imputed with the other arm's model: a treated person's
    as their outcome minus the predicted untreated outcome, ``y - mu0(x)``, an untreated
    person's as the predicted treated outcome minus their outcome, ``mu1(x) - y``. A
    model is fitted to the imputed effects in each arm, giving ``tau1`` from the treated
    and ``tau0`` from the untreated, and the two are blended as
    ``e(x) * tau0(x) + (1 - e(x)) * tau1(x)``: where few people are treated the blend
    leans on ``tau1``, which was imputed with the well-estimated ``mu0``, and the other
    way round.
    """
    X, t, y = _arrays(X, t, y)
    imputed = np.where(t == 1, y - nuisances.mu0, nuisances.mu1 - y)
    tau = np.empty(len(y))
    for k in np.unique(folds):
        test, train = folds == k, folds != k
        tau1 = model().fit(X[train & (t == 1)], imputed[train & (t == 1)]).predict(X[test])
        tau0 = model().fit(X[train & (t == 0)], imputed[train & (t == 0)]).predict(X[test])
        e = nuisances.e[test]
        tau[test] = e * tau0 + (1 - e) * tau1
    return tau


def dr_pseudo_outcome(t, y, nuisances: Nuisances) -> np.ndarray:
    """The doubly robust pseudo-outcome, whose average is the AIPW estimate of the ATE.

    ``mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)``: the outcome models'
    predicted effect, corrected by each person's weighted residual in the arm they were
    in. Its expectation given ``x`` is ``tau(x)`` when either the outcome models or the
    propensity are correct.
    """
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    nu = nuisances
    return nu.mu1 - nu.mu0 + t * (y - nu.mu1) / nu.e - (1 - t) * (y - nu.mu0) / (1 - nu.e)


def dr_learner(X, t, y, nuisances: Nuisances, model: Factory, folds: np.ndarray) -> np.ndarray:
    """Individual effects by regressing the doubly robust pseudo-outcome on ``X``.

    The pseudo-outcomes come from cross-fitted nuisances, and the final regression is
    cross-fitted too. Pseudo-outcomes are noisy, especially where the propensity is near
    0 or 1, so the final model has to average a lot of noise away.
    """
    X, t, y = _arrays(X, t, y)
    phi = dr_pseudo_outcome(t, y, nuisances)
    tau = np.empty(len(y))
    for k in np.unique(folds):
        test, train = folds == k, folds != k
        tau[test] = model().fit(X[train], phi[train]).predict(X[test])
    return tau


def naive_difference(t, y) -> Estimate:
    """Mean outcome of the treated minus mean outcome of the untreated, with Welch's SE.

    An estimate of the average effect only when treatment was assigned at random.
    """
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    return Estimate(y1.mean() - y0.mean(), float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))))


def ipw_ate(t, y, e) -> Estimate:
    """Average effect by inverse propensity weighting, with normalized weights.

    Each treated person is weighted by ``1 / e(x)`` and each untreated person by
    ``1 / (1 - e(x))``, so the weighted treated and the weighted untreated each resemble
    the whole population; the weights are normalized within each arm (the Hajek form),
    which is more stable than dividing by ``n``. The standard error is from the
    estimator's influence function and treats the propensity as known.
    """
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    w1, w0 = t / e, (1 - t) / (1 - e)
    m1, m0 = np.sum(w1 * y) / np.sum(w1), np.sum(w0 * y) / np.sum(w0)
    influence = w1 * (y - m1) / w1.mean() - w0 * (y - m0) / w0.mean()
    return Estimate(m1 - m0, float(influence.std(ddof=1) / np.sqrt(len(y))))


def aipw_ate(t, y, nuisances: Nuisances) -> Estimate:
    """Average effect by augmented inverse propensity weighting (doubly robust).

    The mean of the doubly robust pseudo-outcome, with the standard error of that mean.
    With cross-fitted nuisances this is the double machine learning estimator of the
    ATE for a binary treatment.
    """
    phi = dr_pseudo_outcome(t, y, nuisances)
    return Estimate(float(phi.mean()), float(phi.std(ddof=1) / np.sqrt(len(phi))))


def overlap_summary(t, e) -> dict:
    """How well the two arms overlap, from the (unclipped) propensities.

    ``share_extreme`` is the share of people with a propensity below 0.05 or above 0.95,
    for whom one arm offers almost no comparable people. ``effective_n_treated`` and
    ``effective_n_control`` are the effective sample sizes of the inverse propensity
    weights in each arm, ``(sum w)^2 / sum w^2``: how many equally weighted people the
    weighted arm is worth.
    """
    t = np.asarray(t).astype(int)
    e = np.asarray(e, dtype=float)
    w1, w0 = 1 / e[t == 1], 1 / (1 - e[t == 0])
    return {
        "propensity_min": float(e.min()),
        "propensity_max": float(e.max()),
        "share_extreme": float(np.mean((e < 0.05) | (e > 0.95))),
        "effective_n_treated": float(w1.sum() ** 2 / np.sum(w1**2)),
        "effective_n_control": float(w0.sum() ** 2 / np.sum(w0**2)),
        "n_treated": int(t.sum()),
        "n_control": int((1 - t).sum()),
    }


def cate_scores(tau_hat, tau, budget: float = 0.3) -> dict:
    """Score estimated individual effects against the true ones.

    - ``pehe``: the root mean squared difference between estimated and true effects
      (precision in estimating heterogeneous effects, the standard benchmark metric).
    - ``ate_error``: mean estimated effect minus mean true effect.
    - ``correlation``: how well the estimates order people; not defined when the
      estimates are all equal.
    - ``slope``: the slope of the estimates regressed on the true effects. 1 means
      differences between people are reproduced at full size; below 1 means they are
      shrunk towards a common value; 0 means every person gets the same effect.
    - ``targeting``: treat the ``budget`` share of people with the largest estimated
      effect. The value is the gain over treating a random set of the same size, as a
      share of the gain of the best possible set: 0 is no better than random, 1 is as
      good as knowing every true effect. Larger effects are taken to be better.
    """
    tau_hat = np.asarray(tau_hat, dtype=float)
    tau = np.asarray(tau, dtype=float)
    k = max(1, int(round(budget * len(tau))))
    chosen = np.argsort(-tau_hat, kind="stable")[:k]
    best = np.sort(tau)[::-1][:k]
    attainable = best.mean() - tau.mean()
    spread = tau_hat.std()
    return {
        "pehe": float(np.sqrt(np.mean((tau_hat - tau) ** 2))),
        "ate_error": float(tau_hat.mean() - tau.mean()),
        "correlation": float(np.corrcoef(tau_hat, tau)[0, 1]) if spread > 1e-12 else float("nan"),
        "slope": float(np.cov(tau_hat, tau)[0, 1] / tau.var(ddof=1)),
        "targeting": float((tau[chosen].mean() - tau.mean()) / attainable) if attainable > 0 else float("nan"),
    }


# ------------------------------------------------------------ checks without ground truth


def standardized_differences(X, t, weights=None, names=None) -> dict:
    """Standardized mean difference of each covariate between treated and untreated.

    For each column, the difference between the (weighted) means of the two arms
    divided by the pooled unweighted standard deviation, ``sqrt((s1^2 + s0^2) / 2)``.
    Dividing by the same unweighted spread before and after weighting makes the two
    comparable. A common rule of thumb reads an absolute value below 0.1 as balanced.
    ``weights`` are typically inverse propensity weights; ``None`` gives the raw
    imbalance. Returns ``{name: difference}``.
    """
    X, t = _arrays(X, t)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    names = list(names) if names is not None else [f"x{j}" for j in range(X.shape[1])]
    out = {}
    for j, name in enumerate(names):
        x = X[:, j]
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        spread = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
        out[name] = float((m1 - m0) / spread) if spread > 0 else 0.0
    return out


def blp_test(pseudo_outcome, tau_hat) -> dict:
    """Best linear predictor test of estimated individual effects (Chernozhukov et al.).

    Regresses the doubly robust pseudo-outcome on a constant and the centred estimate,
    ``phi = b1 + b2 (tau_hat - mean(tau_hat)) + error``, with heteroskedasticity-robust
    (HC1) standard errors. ``b1`` estimates the average effect. ``b2`` asks whether the
    estimates carry real differences between people: 0 means they predict nothing about
    who benefits more, 1 means they are right on average about how much more, and a
    value between says the differences are real but overstated. A ``b2`` interval that
    excludes 0 is evidence of heterogeneity. The estimates should be cross-fitted, so
    that a person's estimate did not see their own pseudo-outcome.
    """
    phi = np.asarray(pseudo_outcome, dtype=float)
    centred = np.asarray(tau_hat, dtype=float) - np.mean(tau_hat)
    design = np.column_stack([np.ones(len(phi)), centred])
    coef, *_ = np.linalg.lstsq(design, phi, rcond=None)
    resid = phi - design @ coef
    bread = np.linalg.inv(design.T @ design)
    meat = (design * resid[:, None] ** 2).T @ design
    cov = bread @ meat @ bread * len(phi) / (len(phi) - 2)
    se = np.sqrt(np.diag(cov))
    return {"average": Estimate(float(coef[0]), float(se[0])), "heterogeneity": Estimate(float(coef[1]), float(se[1]))}


def gates(pseudo_outcome, tau_hat, n_groups: int = 5) -> list[dict]:
    """Group average treatment effects by quantile of the estimated effect.

    People are split into ``n_groups`` equal groups by their estimated effect, and in
    each group the average effect is estimated as the mean of the doubly robust
    pseudo-outcome, with its standard error. If the estimates carry real differences,
    the groups' averages rise from the lowest group to the highest; if they do not,
    the averages are flat. Each entry gives the group number (1 is the lowest
    estimates), the mean estimate in the group, and the doubly robust average.
    """
    phi = np.asarray(pseudo_outcome, dtype=float)
    tau_hat = np.asarray(tau_hat, dtype=float)
    order = np.argsort(tau_hat, kind="stable")
    out = []
    for g, members in enumerate(np.array_split(order, n_groups), start=1):
        out.append({
            "group": g,
            "n": int(len(members)),
            "mean_estimate": float(tau_hat[members].mean()),
            "average": Estimate(float(phi[members].mean()), float(phi[members].std(ddof=1) / np.sqrt(len(members)))),
        })
    return out
