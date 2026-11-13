"""Unit tests for the residual learner's table and the no-interaction references."""

from __future__ import annotations

import numpy as np

from tabicllab.residual_table import FEATURES, global_scale, row_features, shrink


def _singles(seed: int = 0, n: int = 30):
    rng = np.random.default_rng(seed)
    da, db = rng.normal(size=n), rng.normal(size=n)
    se2a, se2b = rng.uniform(0.01, 0.1, n), rng.uniform(0.01, 0.1, n)
    cm, cv = rng.uniform(0, 2, n), rng.uniform(0.1, 1, n)
    tgt = np.zeros(n, dtype=bool)
    tgt[3] = True
    return da, db, se2a, se2b, cm, cv, tgt


def test_rows_do_not_depend_on_which_gene_is_named_first() -> None:
    da, db, se2a, se2b, cm, cv, tgt = _singles()
    x_ab = row_features(da, db, se2a, se2b, cm, cv, tgt)
    x_ba = row_features(db, da, se2b, se2a, cm, cv, tgt)
    assert x_ab.shape == (30, len(FEATURES))
    np.testing.assert_array_equal(x_ab, x_ba)


def test_larger_single_comes_first_with_its_own_error() -> None:
    da, db, se2a, se2b, cm, cv, tgt = _singles()
    x = row_features(da, db, se2a, se2b, cm, cv, tgt)
    assert np.all(np.abs(x[:, 0]) >= np.abs(x[:, 1]))
    a_big = np.abs(da) >= np.abs(db)
    np.testing.assert_allclose(x[a_big, 2], np.sqrt(se2a[a_big]))
    np.testing.assert_allclose(x[~a_big, 2], np.sqrt(se2b[~a_big]))
    np.testing.assert_allclose(x[:, 5], np.sqrt(cv))
    assert x[3, 6] == 1.0 and x[:, 6].sum() == 1.0


def test_shrink_zeroes_noise_and_spares_precise_effects() -> None:
    delta = np.array([0.05, -0.05, 2.0, 0.0])
    se2 = np.array([0.01, 0.01, 0.01, 0.01])          # SE = 0.1
    out = shrink(delta, se2)
    assert out[0] == 0.0 and out[1] == 0.0 and out[3] == 0.0
    np.testing.assert_allclose(out[2], 2.0 * (1 - 0.01 / 4.0))
    assert np.all(np.abs(out) <= np.abs(delta))


def test_global_scale_recovers_a_known_beta() -> None:
    rng = np.random.default_rng(1)
    s = [rng.normal(size=20) for _ in range(10)]
    d = [0.9 * x for x in s]
    assert np.isclose(global_scale(d, s), 0.9)
