"""Smoke test: the package imports and reports its version."""

import tabicllab


def test_version() -> None:
    assert tabicllab.__version__ == "0.1.0"
