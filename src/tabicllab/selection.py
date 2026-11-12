"""Choosing a pair's scored genes the way ssl-lab does, on any subset of cells.

ssl-lab scores each perturbation on its top 20 differentially expressed genes,
chosen by scanpy's Wilcoxon rank-sum test of the perturbation's cells against all
control cells (``rank_genes_groups(method="wilcoxon")``, ranked by ``|z|``), with
two guards: the gene is detected in at least 10% of the perturbation's cells, and
its Benjamini–Hochberg adjusted p-value is below 0.05. Those genes are chosen on
all of the pair's cells, which is what the residual-ceiling analysis (R1.1) must
avoid when it estimates noise: a gene chosen because its measured change came out
large is a gene whose noise tends to push the same way in every subset of those
cells. So the selection has to be repeatable on a subset, and that is this module.

The statistic. For one gene, with :math:`n_1` cells in the group and :math:`n_2` in
the reference, and average ranks over the combined sample,

.. math::

    U = \\sum_{i \\in \\text{group}} \\big( L_i + \\tfrac{1}{2} E_i \\big), \\qquad
    z = \\frac{U - n_1 n_2 / 2}{\\sqrt{n_1 n_2 (n_1 + n_2 + 1) / 12}},

where :math:`L_i` and :math:`E_i` count the reference cells below and equal to group
cell :math:`i`. This is scanpy's score without tie correction (its default). It is
computed by sorting the reference once and counting with binary search, so no
combined ranking of thousands of cells is ever built.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

MIN_DETECT_FRAC = 0.10
MAX_PVAL_ADJ = 0.05


def wilcoxon_z(group: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Wilcoxon rank-sum z (G,) of ``group`` (n1, G) against a column-sorted reference (n2, G)."""
    n1, g = group.shape
    n2 = ref_sorted.shape[0]
    u = np.empty(g, dtype=np.float64)
    for j in range(g):
        col = ref_sorted[:, j]
        x = group[:, j]
        lo = np.searchsorted(col, x, side="left")
        hi = np.searchsorted(col, x, side="right")
        u[j] = lo.sum() + 0.5 * (hi - lo).sum()
    return (u - n1 * n2 / 2.0) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (as statsmodels ``fdr_bh``, which scanpy uses)."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def select_top_genes(group: np.ndarray, ref_sorted: np.ndarray, top_k: int = 20,
                     min_detect_frac: float = MIN_DETECT_FRAC,
                     max_pval_adj: float = MAX_PVAL_ADJ) -> np.ndarray:
    """The group's top-``k`` genes by ``|z|`` among those passing ssl-lab's two guards.

    ``group`` is the group's normalized expression (n1, G); a gene counts as detected
    in a cell when its normalized value is above zero, which is when its count is.
    Never pads with genes that fail a guard, so fewer than ``top_k`` may come back.
    """
    z = wilcoxon_z(group, ref_sorted)
    p_adj = bh_adjust(2.0 * norm.sf(np.abs(z)))
    detect = (group > 0).mean(axis=0)
    ok = (detect >= min_detect_frac) & (p_adj < max_pval_adj)
    rank = np.where(ok, -np.abs(z), np.inf)
    order = np.argsort(rank, kind="stable")[:top_k]
    return order[np.isfinite(rank[order])]
