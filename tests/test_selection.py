"""The fast Wilcoxon z and BH adjustment agree with scipy and statsmodels-style definitions."""

from __future__ import annotations

import numpy as np
from scipy.stats import false_discovery_control, mannwhitneyu

from tabicllab.selection import bh_adjust, select_top_genes, wilcoxon_z


def test_wilcoxon_z_matches_scipy_u_with_ties() -> None:
    rng = np.random.default_rng(0)
    ref = np.round(rng.exponential(size=(300, 40)), 1)          # rounded: many ties, like log counts
    grp = np.round(rng.exponential(size=(60, 40)) + 0.2 * np.arange(40) / 40, 1)
    z = wilcoxon_z(grp, np.sort(ref, axis=0))
    n1, n2 = grp.shape[0], ref.shape[0]
    u = mannwhitneyu(grp, ref, axis=0, method="asymptotic", use_continuity=False).statistic
    z_ref = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    np.testing.assert_allclose(z, z_ref, rtol=1e-12, atol=1e-12)


def test_bh_matches_scipy() -> None:
    p = np.random.default_rng(1).uniform(size=500) ** 3
    np.testing.assert_allclose(bh_adjust(p), false_discovery_control(p, method="bh"), rtol=1e-12)


def test_selection_respects_guards() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(500, 30))
    grp = rng.normal(size=(100, 30))
    grp[:, :5] += 2.0                                   # five genes truly shifted
    grp[:, 5] = 0.0                                     # never detected
    top = select_top_genes(grp, np.sort(ref, axis=0), top_k=20)
    assert set(range(5)) <= set(top.tolist())
    assert 5 not in top
