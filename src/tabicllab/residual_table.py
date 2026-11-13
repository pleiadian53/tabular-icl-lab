"""The table a residual learner sees: one row per (two-gene pair, scored gene).

Route 1 asks whether a learner can predict the part of a two-gene combination's
effect that the additive prediction misses. For a perturbation :math:`p` and gene
:math:`g`, :math:`\\Delta_g(p)` is the change in mean log1p-CP10K expression of
:math:`g` between the cells carrying :math:`p` and the control cells. For a pair
:math:`AB`, the additive prediction is :math:`a_g = \\Delta_g(A) + \\Delta_g(B)` and
the residual, the learner's label, is

.. math::

    \\varepsilon_g(AB) = \\Delta_g(AB) - \\Delta_g(A) - \\Delta_g(B).

A row is one gene :math:`g` of one pair's scored genes (its top differentially
expressed genes, the set every score is computed on). Its seven features describe
how the two singles moved that gene, how precisely those moves were measured, and
the gene's level in control cells:

``delta_big``, ``delta_small``
    the two single-gene effects :math:`\\Delta_g(A)` and :math:`\\Delta_g(B)`, ordered by
    magnitude so a row is the same whichever gene of the pair is named first;
``se_big``, ``se_small``
    their standard errors, :math:`\\sqrt{s^2_{p,g}/n_p + s^2_{c,g}/n_c}`, where
    :math:`s^2_{p,g}` is the variance of :math:`g` over the :math:`n_p` cells of group
    :math:`p` and :math:`c` is the control group;
``control_mean``, ``control_sd``
    the mean and standard deviation of :math:`g` across control cells;
``is_target``
    1 when :math:`g` is :math:`A` or :math:`B` itself, whose expression a CRISPRa guide
    drives directly.

No feature uses a pair's own cells, so a held-out pair's features are available in
exactly the way the additive reference's inputs are. The design and its reasons are
recorded with round 1 (R1.3); gene identity is deliberately absent (that is R1.5).

The same statistics give the two *no-interaction references*, which a learner must
beat before it is credited with learning interactions: a global scale
:math:`\\beta (\\Delta(A) + \\Delta(B))`, and each single's effect shrunk toward zero
by its own noise, :func:`shrink`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tabicllab.norman import NormanCache, pair_genes

FEATURES = ("delta_big", "delta_small", "se_big", "se_small",
            "control_mean", "control_sd", "is_target")


@dataclass(frozen=True)
class GroupStats:
    """Per-gene mean and variance (ddof 1) over a group of cells, and the group's size."""

    mean: np.ndarray
    var: np.ndarray
    n: int

    @property
    def se2(self) -> np.ndarray:
        """Squared standard error of the group mean, per gene."""
        return self.var / self.n


def group_stats(cache: NormanCache, rows: np.ndarray, chunk: int = 2048) -> GroupStats:
    """Mean (the cache's float32 arithmetic, cast to float64) and variance of these cells.

    The variance is accumulated chunk by chunk around that mean, so even the 11,726
    control cells never sit in memory as float64 at once.
    """
    rows = np.sort(np.asarray(rows))
    if rows.size < 2:
        raise ValueError("need at least two cells for a variance")
    mean = cache.mean(rows)
    ss = np.zeros(cache.n_genes)
    for i in range(0, rows.size, chunk):
        x = cache.rows(rows[i:i + chunk]).astype(np.float64) - mean
        ss += np.einsum("ij,ij->j", x, x)
    return GroupStats(mean=mean, var=ss / (rows.size - 1), n=int(rows.size))


def shrink(delta: np.ndarray, se2: np.ndarray) -> np.ndarray:
    """Each effect shrunk toward zero by its own noise: :math:`\\Delta \\max(0, 1 - \\mathrm{SE}^2/\\Delta^2)`.

    The factor :math:`(\\Delta^2 - \\mathrm{SE}^2)/\\Delta^2` is the plug-in estimate of
    how much of the measured :math:`\\Delta^2` is signal (the expected square of a
    noisy estimate is the true square plus its variance), clipped at zero. An effect
    no larger than its standard error is set to zero; a large, precise one is barely
    touched. It uses no pair and fits nothing.
    """
    delta = np.asarray(delta, dtype=np.float64)
    d2 = delta * delta
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(d2 > 0, np.clip(1.0 - np.asarray(se2) / d2, 0.0, 1.0), 0.0)
    return delta * factor


def global_scale(deltas: list[np.ndarray], additive: list[np.ndarray]) -> float:
    """Least-squares :math:`\\beta` in :math:`\\Delta(AB) \\approx \\beta (\\Delta(A) + \\Delta(B))`, pooled over pairs."""
    num = sum(float(d @ s) for d, s in zip(deltas, additive))
    den = sum(float(s @ s) for s in additive)
    return num / den if den > 0 else 1.0


def row_features(da: np.ndarray, db: np.ndarray, se2a: np.ndarray, se2b: np.ndarray,
                 control_mean: np.ndarray, control_var: np.ndarray,
                 is_target: np.ndarray) -> np.ndarray:
    """The seven features (n, 7) for n genes of one pair, in the order of :data:`FEATURES`.

    The single with the larger :math:`|\\Delta_g|` comes first (ties broken by the larger
    value), so swapping the two genes' names leaves every row unchanged.
    """
    da, db = np.asarray(da, dtype=np.float64), np.asarray(db, dtype=np.float64)
    swap = (np.abs(db) > np.abs(da)) | ((np.abs(db) == np.abs(da)) & (db > da))
    big, small = np.where(swap, db, da), np.where(swap, da, db)
    se_big = np.sqrt(np.where(swap, se2b, se2a))
    se_small = np.sqrt(np.where(swap, se2a, se2b))
    return np.column_stack([big, small, se_big, se_small, control_mean,
                            np.sqrt(control_var), np.asarray(is_target, dtype=np.float64)])


@dataclass
class PairRows:
    """One pair's block of rows, with what is needed to score it (never as features)."""

    name: str
    genes: np.ndarray          # scored gene indices T_p, (T,)
    X: np.ndarray              # features (T, 7)
    additive: np.ndarray       # a_g = Delta_g(A) + Delta_g(B) on T_p
    shrunk: np.ndarray         # shrink(Delta_g(A)) + shrink(Delta_g(B)) on T_p
    true_delta: np.ndarray     # Delta_g(AB) on T_p, from the pair's own cells
    pair_se2: np.ndarray       # s^2_{AB,g} / n_AB on T_p, for the e_p floor
    n_cells: int

    @property
    def eps(self) -> np.ndarray:
        """The label: the measured residual on T_p."""
        return self.true_delta - self.additive


@dataclass
class ResidualData:
    """Group statistics for the control, every single and every pair, computed once."""

    cache: NormanCache
    top_k: int = 20
    control: GroupStats = field(init=False)
    singles: dict[str, GroupStats] = field(init=False, default_factory=dict)
    gene_index: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.control = group_stats(self.cache, self.cache.control_cells())
        self.gene_index = {str(g): i for i, g in enumerate(self.cache.gene_ids)}

    def single(self, gene: str) -> GroupStats:
        if gene not in self.singles:
            self.singles[gene] = group_stats(self.cache, self.cache.cells(gene, "train"))
        return self.singles[gene]

    def delta(self, gene: str) -> np.ndarray:
        """A single's effect (G,): its training cells' mean minus the control mean."""
        return self.single(gene).mean - self.control.mean

    def delta_se2(self, gene: str) -> np.ndarray:
        """Squared standard error of that effect (G,): the single's and the control's."""
        return self.single(gene).se2 + self.control.se2

    def pair_rows(self, name: str) -> PairRows:
        a, b = pair_genes(name)
        idx = self.cache.scored_genes(name, self.top_k)
        da, db = self.delta(a)[idx], self.delta(b)[idx]
        se2a, se2b = self.delta_se2(a)[idx], self.delta_se2(b)[idx]
        targets = {self.gene_index.get(a, -1), self.gene_index.get(b, -1)}
        is_target = np.isin(idx, list(targets))
        pair = group_stats(self.cache, self.cache.cells(name, self.cache.pair_split(name)))
        return PairRows(
            name=name, genes=idx,
            X=row_features(da, db, se2a, se2b, self.control.mean[idx], self.control.var[idx], is_target),
            additive=da + db, shrunk=shrink(da, se2a) + shrink(db, se2b),
            true_delta=pair.mean[idx] - self.control.mean[idx],
            pair_se2=pair.se2[idx], n_cells=pair.n,
        )


def stack(blocks: list[PairRows]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features (n, 7), labels (n,) and each row's block number (n,), for a list of pairs."""
    X = np.vstack([b.X for b in blocks])
    y = np.concatenate([b.eps for b in blocks])
    groups = np.concatenate([np.full(len(b.genes), i) for i, b in enumerate(blocks)])
    return X, y, groups
