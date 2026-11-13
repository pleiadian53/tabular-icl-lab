"""Unit tests for the joint paired bootstrap with max-t simultaneous intervals."""

from __future__ import annotations

import numpy as np

from tabicllab.bootstrap import maxt_family, percentile_interval, resample_indices


def test_simultaneous_intervals_are_wider_than_unadjusted_and_cover_the_mean() -> None:
    rng = np.random.default_rng(0)
    diffs = {f"c{i}": rng.normal(loc=-0.05 * i, scale=0.1, size=20) for i in range(4)}
    idx = resample_indices(20, 4000, 0)
    fam = maxt_family(diffs, idx)
    assert fam["critical_value"] > 1.96                 # a family of four needs a wider band
    for k, c in fam["contrasts"].items():
        lo, hi = c["ci_simultaneous"]
        assert lo < c["diff"] < hi
        assert hi - lo > c["ci_unadjusted"][1] - c["ci_unadjusted"][0]
        assert np.isclose(c["diff"], diffs[k].mean())


def test_a_clear_effect_lies_wholly_on_one_side() -> None:
    d = np.full(20, -0.1) + np.random.default_rng(2).normal(scale=0.01, size=20)
    fam = maxt_family({"x": d}, resample_indices(20, 2000, 0))
    assert fam["contrasts"]["x"]["ci_simultaneous"][1] < 0


def test_resample_is_shared_and_reproducible() -> None:
    np.testing.assert_array_equal(resample_indices(5, 10, 3), resample_indices(5, 10, 3))
    pi = percentile_interval(np.arange(10.0), resample_indices(10, 2000, 0))
    assert pi["ci"][0] < 4.5 < pi["ci"][1]
