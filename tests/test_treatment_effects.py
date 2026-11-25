"""Unit tests for the treatment-effect estimators, on linear data with known effects."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from tabicllab.treatment_effects import (
    aipw_ate,
    cate_scores,
    dr_learner,
    fit_nuisances,
    fold_ids,
    ipw_ate,
    naive_difference,
    overlap_summary,
    s_learner,
    t_learner,
    x_learner,
)


def confounded(n: int = 4000, seed: int = 0):
    """x0 raises both the chance of treatment and the outcome; the effect is 1 + x1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-1.2 * X[:, 0]))
    t = rng.binomial(1, e)
    tau = 1 + X[:, 1]
    y = 2 * X[:, 0] + X[:, 2] + t * tau + rng.normal(scale=0.5, size=n)
    return X, t, y, tau, e


def linear():
    return LinearRegression()


def logistic():
    return LogisticRegression(max_iter=1000)


def test_folds_are_stratified_by_treatment():
    t = np.r_[np.ones(30, int), np.zeros(70, int)]
    folds = fold_ids(t, n_folds=5, seed=0)
    for k in range(5):
        assert t[folds == k].sum() == 6 and (folds == k).sum() == 20


def test_naive_difference_is_biased_and_doubly_robust_estimators_are_not():
    X, t, y, tau, _ = confounded()
    truth = tau.mean()
    assert naive_difference(t, y).estimate - truth > 1.0          # x0 pushes treated outcomes up
    folds = fold_ids(t, seed=0)
    nu = fit_nuisances(X, t, y, linear, logistic, folds)
    for estimate in (ipw_ate(t, y, nu.e), aipw_ate(t, y, nu)):
        low, high = estimate.interval(0.99)
        assert low < truth < high


def test_aipw_survives_a_wrong_propensity_model():
    X, t, y, tau, _ = confounded(seed=1)
    folds = fold_ids(t, seed=1)

    class Constant:                                                # ignores x: the propensity is wrong
        def fit(self, X, t):
            self.p = t.mean()
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])

    nu = fit_nuisances(X, t, y, linear, Constant, folds)
    assert abs(aipw_ate(t, y, nu).estimate - tau.mean()) < 0.1     # right outcome models suffice
    assert abs(ipw_ate(t, y, nu.e).estimate - tau.mean()) > 1.0    # IPW has nothing else to lean on


def test_meta_learners_recover_a_linear_effect():
    X, t, y, tau, _ = confounded(seed=2)
    folds = fold_ids(t, seed=2)
    nu = fit_nuisances(X, t, y, linear, logistic, folds)
    for estimate in (t_learner(nu), x_learner(X, t, y, nu, linear, folds), dr_learner(X, t, y, nu, linear, folds)):
        scores = cate_scores(estimate, tau)
        assert scores["pehe"] < 0.15 and abs(scores["slope"] - 1) < 0.1


def test_a_linear_s_learner_gives_everyone_the_same_effect():
    X, t, y, tau, _ = confounded(seed=3)
    estimate = s_learner(X, t, y, linear, fold_ids(t, seed=3))
    scores = cate_scores(estimate, tau)
    assert abs(scores["ate_error"]) < 0.1                          # the average is right
    assert abs(scores["slope"]) < 0.05                             # but nobody differs
    assert estimate.std() < 0.05


def test_cate_scores_on_perfect_and_random_estimates():
    rng = np.random.default_rng(4)
    tau = rng.normal(size=5000)
    perfect = cate_scores(tau, tau)
    assert perfect["pehe"] == 0
    assert np.isclose(perfect["slope"], 1) and np.isclose(perfect["targeting"], 1)
    noise = cate_scores(rng.normal(size=5000), tau)
    assert abs(noise["targeting"]) < 0.05 and abs(noise["slope"]) < 0.05


def test_overlap_summary_counts_extreme_propensities():
    t = np.array([1, 1, 0, 0])
    e = np.array([0.5, 0.99, 0.5, 0.02])
    summary = overlap_summary(t, e)
    assert summary["share_extreme"] == 0.5
    assert 1 <= summary["effective_n_treated"] <= 2


def test_every_estimator_runs_on_a_tiny_table():
    X, t, y, tau, _ = confounded(n=40, seed=5)
    folds = fold_ids(t, n_folds=2, seed=5)
    nu = fit_nuisances(X, t, y, linear, logistic, folds)
    for estimate in (s_learner(X, t, y, linear, folds), t_learner(nu), x_learner(X, t, y, nu, linear, folds),
                     dr_learner(X, t, y, nu, linear, folds)):
        assert estimate.shape == (40,) and np.isfinite(estimate).all()
    for estimate in (naive_difference(t, y), ipw_ate(t, y, nu.e), aipw_ate(t, y, nu)):
        assert np.isfinite(estimate.estimate) and np.isfinite(estimate.se)
    assert np.isfinite(cate_scores(t_learner(nu), tau)["pehe"])
