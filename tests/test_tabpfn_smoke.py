"""Smoke test: TabPFN is installed, its weights load, and it predicts sensibly.

Runs the TabPFN-3.5 regressor (the lab's main model, non-commercial license,
gated) and the TabPFN-2 regressor (Apache-2.0-based license with attribution,
no account needed once cached) on CPU, on a small synthetic table whose target is
a known nonlinear function of two of five features. A working install should
explain most of the variance from 200 context rows. If the weights cannot be
loaded (no network, no cached checkpoint), the test is skipped with the reason
rather than failing, the same preflight idea as agentic-spliceai's M4 benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_table(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.1 * rng.normal(size=n)
    return x, y


@pytest.mark.parametrize("version", ["v3.5", "v2"])
def test_tabpfn_regressor_fits_on_cpu(version: str) -> None:
    """TabPFN-3.5 is the lab's main model; TabPFN-2 is the optional robustness arm.

    TabPFN-3.5's weights are gated: the test skips until its license has been
    accepted once (browser flow or ``TABPFN_TOKEN``).
    """
    tabpfn = pytest.importorskip("tabpfn")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    x_ctx, y_ctx = _make_table(200, seed=0)
    x_qry, y_qry = _make_table(100, seed=1)

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion(version), device="cpu", random_state=0, n_estimators=2
    )
    try:
        model.fit(x_ctx, y_ctx)
    except Exception as exc:  # weights download or license gate
        pytest.skip(f"TabPFN {tabpfn.__version__} {version} weights unavailable: {exc}")

    y_hat = model.predict(x_qry)
    assert y_hat.shape == y_qry.shape
    r2 = 1.0 - np.sum((y_qry - y_hat) ** 2) / np.sum((y_qry - y_qry.mean()) ** 2)
    assert r2 > 0.8, f"R^2 {r2:.3f} on a nearly noise-free target"
