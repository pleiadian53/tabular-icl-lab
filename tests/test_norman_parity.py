"""Parity with ssl-lab: the additive reference scores exactly as ssl-lab published it.

Reads the shared Norman cache (skipped if it is not linked at ``data/norman2019``)
and checks that this lab's data access and scoring reproduce ssl-lab's per-pair
Δ-correlation and size-aware error for the additive reference on the twenty
held-out pairs. The expected numbers were copied from ssl-lab's saved reports
into ``tests/fixtures/ssllab_additive_reference.json``. The size-aware error and
the mean Δ-correlation match Round 8's float64 computation to 1e-12; per-pair
Δ-correlations match ssl-lab's script 18 to 1e-8, the difference being that script's
float32 subtraction of the control mean. If this test fails, the
two projects no longer score the same thing, and no comparison between them holds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tabicllab.norman import load_norman, pair_genes
from tabicllab.scoring import pair_metrics

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "norman2019"
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ssllab_additive_reference.json").read_text())


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def cache():
    if not (CACHE / "tokens_meta.npz").exists():
        pytest.skip(f"Norman cache not linked at {CACHE}")
    return load_norman(CACHE)


def test_cache_is_the_one_ssllab_scored(cache) -> None:
    assert _md5(CACHE / "de_genes.json") == FIXTURE["data"]["de_genes_md5"]
    assert _md5(CACHE / "splits.json") == FIXTURE["data"]["splits_md5"]


def test_additive_reference_matches_ssllab(cache) -> None:
    got_r, got_e = [], []
    for pair, expected in FIXTURE["per_pair"].items():
        a, b = pair_genes(pair)
        da, db = cache.single_delta(a), cache.single_delta(b)
        assert cache.pair_split(pair) == "test"
        m = pair_metrics(
            pred_pair=cache.control_mean + da + db, true_pair=cache.pair_mean(pair),
            control_mean=cache.control_mean, delta_a=da, delta_b=db,
            top_idx=cache.scored_genes(pair, FIXTURE["top_k"]),
        )
        # Script 18 subtracted the control in float32; this lab, like Round 8, works in float64.
        assert m["delta_r"] == pytest.approx(expected["delta_r"], abs=1e-8), pair
        assert m["rel_err"] == pytest.approx(expected["rel_err"], abs=1e-12), pair
        got_r.append(m["delta_r"])
        got_e.append(m["rel_err"])
    assert float(np.mean(got_r)) == pytest.approx(FIXTURE["mean_delta_r_round8_float64"], abs=1e-12)
    assert float(np.mean(got_e)) == pytest.approx(FIXTURE["mean_rel_err"], abs=1e-12)
