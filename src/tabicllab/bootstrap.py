"""Joint paired bootstrap over pairs, with max-t simultaneous intervals.

Every arm of a round gives one number per held-out pair (training seeds averaged
first), and every comparison is a paired difference over the same pairs. This module
reimplements the procedure the sibling project ssl-lab uses for the same comparisons
(its ``src/ssllab/eval/bootstrap.py``), so that an interval here and an interval
there mean the same thing:

- **One resample for everything.** The pair indices are drawn once per bootstrap
  iteration and reused for every contrast and every metric, so the dependence
  between contrasts that share an arm or share pairs is kept.
- **Max-t for a family of contrasts.** Each contrast's bootstrap mean is studentized
  and centred; the per-iteration maximum of :math:`|t|` over the family gives one
  critical value :math:`c`, and :math:`\\bar d \\pm c \\cdot \\mathrm{se}` then holds
  simultaneously, at the stated level, for every contrast in the family.
- **Secondaries get plain percentile intervals** and no verdict.

The intervals describe uncertainty from the finite set of pairs only, not from
training seeds, and the pairs share genes, so they are not strictly independent.
"""

from __future__ import annotations

import numpy as np


def resample_indices(n: int, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    """Bootstrap index matrix ``(n_boot, n)``, drawn once and shared by every contrast."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def contrast_stats(d: np.ndarray, idx: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """For paired differences ``d`` (n,): observed mean, its standard error, the bootstrap
    means ``(n_boot,)`` and the centred studentized statistics ``(n_boot,)``."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    bs = d[idx]
    bmean = bs.mean(1)
    bse = bs.std(axis=1, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(bse > 1e-12, (bmean - obs) / bse, 0.0)
    return obs, se, bmean, t


def maxt_family(diffs: dict[str, np.ndarray], idx: np.ndarray, alpha: float = 0.05) -> dict:
    """Simultaneous intervals for a family of paired contrasts on one shared resample.

    ``diffs`` maps a contrast's name to its per-pair differences. Returns
    ``{"critical_value": c, "contrasts": {name: {diff, se, ci_unadjusted, ci_simultaneous}}}``.
    """
    stats = {k: contrast_stats(d, idx) for k, d in diffs.items()}
    t_abs = np.vstack([np.abs(s[3]) for s in stats.values()])
    crit = float(np.percentile(t_abs.max(axis=0), 100 * (1 - alpha)))
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k, (obs, se, bmean, _) in stats.items():
        ulo, uhi = np.percentile(bmean, [lo_pct, hi_pct])
        out[k] = {"diff": obs, "se": se,
                  "ci_unadjusted": [float(ulo), float(uhi)],
                  "ci_simultaneous": [obs - crit * se, obs + crit * se]}
    return {"critical_value": crit, "contrasts": out}


def percentile_interval(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> dict:
    """Mean of ``values`` (n,) with a plain percentile interval on the shared resample."""
    v = np.asarray(values, dtype=np.float64)
    b = v[idx].mean(1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "ci": [float(lo), float(hi)]}
