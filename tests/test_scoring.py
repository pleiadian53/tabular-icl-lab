"""Unit tests for the per-pair scores on synthetic profiles."""

from __future__ import annotations

import math

import numpy as np

from tabicllab.scoring import pair_metrics, pearson


def _profiles(seed: int = 0, g: int = 50):
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(size=g)
    da, db = rng.normal(size=g), rng.normal(size=g)
    inter = 0.3 * rng.normal(size=g)
    true_pair = ctrl + da + db + inter
    return ctrl, da, db, inter, true_pair, np.arange(20)


def test_additive_prediction_scores_like_the_interaction() -> None:
    ctrl, da, db, inter, true_pair, idx = _profiles()
    m = pair_metrics(pred_pair=ctrl + da + db, true_pair=true_pair, control_mean=ctrl,
                     delta_a=da, delta_b=db, top_idx=idx)
    assert math.isclose(m["rel_err"], m["rel_gi"])          # additivity's error is the interaction
    assert math.isclose(m["log_r"], 0.0, abs_tol=1e-12)     # and its ratio to itself is 1
    assert math.isclose(m["scale"], 1.0)


def test_perfect_prediction() -> None:
    ctrl, da, db, inter, true_pair, idx = _profiles()
    m = pair_metrics(pred_pair=true_pair, true_pair=true_pair, control_mean=ctrl,
                     delta_a=da, delta_b=db, top_idx=idx)
    assert m["rel_err"] == 0.0
    assert math.isclose(m["delta_r"], 1.0)
    assert m["log_r"] < -20                                # floored, finite


def test_zero_interaction_leaves_ratio_undefined() -> None:
    ctrl, da, db, _, _, idx = _profiles()
    true_pair = ctrl + da + db
    m = pair_metrics(pred_pair=true_pair + 0.1, true_pair=true_pair, control_mean=ctrl,
                     delta_a=da, delta_b=db, top_idx=idx)
    assert math.isnan(m["log_r"])
    assert math.isfinite(m["rel_err"])


def test_pearson_constant_is_nan() -> None:
    assert math.isnan(pearson(np.ones(5), np.arange(5)))
