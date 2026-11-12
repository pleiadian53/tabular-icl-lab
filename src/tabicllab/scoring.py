"""Per-pair scores for two-gene combinations, identical to ssl-lab's.

This lab compares its predictions with the sibling project ssl-lab's on the same
twenty held-out Norman pairs, so it must score them the same way. Rather than
installing ssl-lab (which would couple the two projects and let a change there
silently move a score here), the two scores are reimplemented from ssl-lab's
public source (``src/ssllab/eval/effect_size.py`` and
``src/ssllab/eval/interaction.py``), with the same numerical rules, and a parity
test reproduces ssl-lab's published per-pair numbers for the additive reference.

Notation, for one pair :math:`p = A{+}B`, restricted to its scored genes :math:`T`
(its top 20 differentially expressed genes), in log1p-CP10K expression:

- :math:`\\Delta_p` = the pair's measured mean minus the control mean (the truth);
- :math:`\\hat\\Delta_p` = the predicted mean minus the control mean;
- :math:`\\Delta_A, \\Delta_B` = the singles' measured effects;
- :math:`I_p = \\Delta_p - \\Delta_A - \\Delta_B`, the measured interaction.

Scores:

- ``delta_r``: Pearson correlation of :math:`\\hat\\Delta_p` and :math:`\\Delta_p` over
  :math:`T`, the standing benchmark;
- ``rel_err`` (:math:`e_p`, Round 8's size-aware endpoint):
  :math:`\\lVert \\hat\\Delta_p - \\Delta_p \\rVert / \\lVert \\Delta_p \\rVert`;
- ``rel_gi``: :math:`\\lVert I_p \\rVert / \\lVert \\Delta_p \\rVert`, which is the
  additive reference's :math:`e_p`;
- ``log_r``: :math:`\\log(\\lVert \\hat\\Delta_p - \\Delta_p \\rVert / \\lVert I_p \\rVert)`,
  the error relative to additivity's (negative means closer than additivity);
- ``lam`` and ``scale``: the least-squares coefficient of the truth and of the
  prediction on the additive direction :math:`a_p = \\Delta_A + \\Delta_B`.

Summaries across pairs are arithmetic means of the per-pair values, as ssl-lab
reports them (mean ``delta_r`` 0.961 and mean ``rel_err`` 0.289 for additivity).
"""

from __future__ import annotations

import math

import numpy as np

NUMERATOR_FLOOR = 1e-12
DENOMINATOR_REL_FLOOR = 1e-6


def pearson(a, b) -> float:
    """Pearson correlation; NaN when either vector is (numerically) constant, as in ssl-lab."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pair_metrics(*, pred_pair, true_pair, control_mean, delta_a, delta_b, top_idx) -> dict[str, float]:
    """All per-pair scores for one combination, from mean profiles (G,) and the scored genes."""
    idx = np.asarray(top_idx, dtype=int)
    ctrl = np.asarray(control_mean, dtype=np.float64)[idx]
    d_hat = np.asarray(pred_pair, dtype=np.float64)[idx] - ctrl
    d = np.asarray(true_pair, dtype=np.float64)[idx] - ctrl
    s = np.asarray(delta_a, dtype=np.float64)[idx] + np.asarray(delta_b, dtype=np.float64)[idx]
    inter = d - s

    err = float(np.linalg.norm(d_hat - d))
    n_delta = float(np.linalg.norm(d))
    n_inter = float(np.linalg.norm(inter))
    ratio_ok = n_inter > DENOMINATOR_REL_FLOOR * n_delta and n_inter > 0.0
    nan = float("nan")
    ss = float(s @ s)
    return {
        "delta_r": pearson(d_hat, d),
        "rel_err": err / n_delta if n_delta >= NUMERATOR_FLOOR else nan,
        "rel_gi": n_inter / n_delta if n_delta >= NUMERATOR_FLOOR else nan,
        "log_r": (math.log(max(err, NUMERATOR_FLOOR)) - math.log(n_inter)) if ratio_ok else nan,
        "lam": float(d @ s) / ss if ss > NUMERATOR_FLOOR else nan,
        "scale": float(d_hat @ s) / ss if ss > NUMERATOR_FLOOR else nan,
        "err_norm": err,
        "delta_norm": n_delta,
        "interaction_norm": n_inter,
    }
