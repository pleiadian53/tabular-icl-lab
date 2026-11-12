"""R1.1: is the genetic-interaction residual reproducible, or mostly measurement noise?

Context. On the twenty held-out Norman pairs, the additive prediction (the sum of the
two single-gene effects) is the reference every learned model has failed to beat in
the sibling project ssl-lab. A model can only beat it by predicting the residual

    eps_g(AB) = Delta_g(AB) - Delta_g(A) - Delta_g(B),

where Delta_g(p) is the change in mean log1p-CP10K expression of gene g between the
cells carrying perturbation p and the control cells. Before any model is built, this
script measures how much of the measured residual is reproducible signal. If it is
mostly noise, no model can beat additivity on these pairs and the route ends here.

Why the obvious measurement is wrong. Writing m_p for a group's measured mean and c for
the control mean, eps = m_AB - m_A - m_B + c, and each mean carries its own sampling
error. Splitting only the pair's cells into halves leaves the singles' and the control's
errors in both halves, so the halves agree partly because they share noise. And the
scored genes were chosen by a Wilcoxon test on all of the pair's cells, which favours
genes whose noise happens to push the same way in every subset of those cells. Both
inflate any agreement between halves. So every estimate below uses halves that share
no cells, in all four groups, and the primary one re-selects the genes on one half and
measures on the other.

PROTOCOL (fixed before any number was computed)
-----------------------------------------------
Data: the shared Norman cache (ssl-lab's, read-only), combination split. Singles use their
training cells; control uses all control cells; each pair uses all its cells. Scored genes:
each pair's top 20 by ssl-lab's criterion (``tabicllab.selection``). Repeats: 50 random
splits of every group into two disjoint halves (seed 0).

Primary, per held-out pair: the cross-fitted signal fraction phi_p. In each repeat and
direction, choose the pair's top-20 genes on half 1 (its pair cells against half 1 of the
control), then on half 2 compute the residual eps_2 over those genes and its noise
variance from the cell variances, SE_g^2 = s_AB^2/n_AB + s_A^2/n_A + s_B^2/n_B + s_c^2/n_c.
The signal energy is S = ||eps_2||^2 - sum_g SE_g^2; at the full sample size the noise
halves, so phi = S / (S + sum_g SE_g^2 / 2), clipped to [0, 1] for the count below. phi_p
is the mean over repeats and both directions.

Decision rule for Route 1:
  - continue to R1.2 and R1.3 if at least 10 of the 20 pairs have phi_p >= 0.5
    (the interaction carries at least as much signal as noise at full sample size);
  - stop, and write up the negative result, if at most 4 pairs do;
  - in between, continue, and R1.3 reports the pairs with phi_p >= 0.5 as a stratum
    named here, before any model is scored.

Secondary, all reported, none decisive:
  1. split-half correlation of eps over the fixed scored genes, all four groups split,
     stepped up to the full sample by Spearman-Brown, 2 rho / (1 + rho): an upper bound,
     inflated by gene selection;
  2. the same with only the pair's cells split (singles and control shared): the naive
     estimate, which quantifies the shared-noise inflation;
  3. the analytic signal fraction on the fixed scored genes and full data, which carries
     the selection bias;
  4. the e_p floor: the error a perfect predictor of eps would still show, because the
     pair's own cells are never seen, ||e_AB|| / ||Delta(AB)|| on the scored genes, from
     s_AB^2 / n_AB (and ssl-lab's split-half estimate of the same, for comparison); the
     headroom for R1.3 on e_p is 0.289 minus this floor;
  5. on the 91 training pairs only (no held-out pair is used), the share of the residual
     energy explained by one global scale, eps ~ (beta - 1)(Delta(A) + Delta(B)), with
     beta fitted leaving each pair out: generic sub- or super-additivity, not pair-specific
     interaction.

Limit: disjoint halves of one experiment still share its guides, lanes and processing.

Usage
-----
    python examples/perturbation_response/01_residual_ceiling.py              # full run
    python examples/perturbation_response/01_residual_ceiling.py --limit 2 --reps 2   # smoke

Output: ``output/r1_1_residual_ceiling/report.json`` (per-pair numbers, summaries,
settings, data fingerprints and the decision).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tabicllab.norman import NormanCache, load_norman, pair_genes  # noqa: E402
from tabicllab.scoring import pearson  # noqa: E402
from tabicllab.selection import select_top_genes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("r1.1")

PHI_THRESHOLD = 0.5
CONTINUE_IF_AT_LEAST = 10
STOP_IF_AT_MOST = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "norman2019"))
    p.add_argument("--out", default=str(ROOT / "output" / "r1_1_residual_ceiling" / "report.json"))
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="score only the first N held-out pairs (smoke)")
    return p.parse_args()


def spearman_brown(rho: float) -> float:
    return 2 * rho / (1 + rho) if np.isfinite(rho) and rho > -1 else float("nan")


def halves(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    return perm[: n // 2], perm[n // 2:]


def mean_var(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean (float32 arithmetic, as the cache) cast to float64, per-gene variance (ddof 1), n."""
    return x.mean(0).astype(np.float64), x.astype(np.float64).var(0, ddof=1), x.shape[0]


def signal_fraction(eps: np.ndarray, se2: np.ndarray) -> float:
    """Full-sample signal fraction from a half-sample residual and its noise variances."""
    s = float(eps @ eps - se2.sum())
    denom = s + se2.sum() / 2.0
    return s / denom if denom > 0 else 0.0


def global_scale(cache: NormanCache, top_k: int, deltas: dict[str, np.ndarray]) -> dict:
    """Share of the training pairs' residual energy explained by one global scale (leave-one-pair-out)."""
    rows = []
    for q in cache.pairs("train"):
        a, b = pair_genes(q)
        if a not in deltas or b not in deltas:
            continue
        idx = cache.scored_genes(q, top_k)
        d = (cache.pair_mean(q) - cache.control_mean)[idx]
        s = (deltas[a] + deltas[b])[idx]
        rows.append((q, d, s))
    num = np.array([float(d @ s) for _, d, s in rows])
    den = np.array([float(s @ s) for _, d, s in rows])
    resid_add = sum(float((d - s) @ (d - s)) for _, d, s in rows)
    resid_scaled = 0.0
    for i, (_, d, s) in enumerate(rows):
        beta = (num.sum() - num[i]) / (den.sum() - den[i])
        resid_scaled += float((d - beta * s) @ (d - beta * s))
    beta_all = float(num.sum() / den.sum())
    return {"n_pairs": len(rows), "beta": beta_all,
            "share_explained_loo": 1.0 - resid_scaled / resid_add,
            "lambda_per_pair": {q: float(n / dd) for (q, _, _), n, dd in zip(rows, num, den)}}


def main() -> None:
    args = parse_args()
    t0 = time.time()
    cache = load_norman(args.data_dir)
    rng = np.random.default_rng(args.seed)
    pairs = cache.pairs("test")[: args.limit] if args.limit else cache.pairs("test")
    singles = sorted({g for p in pairs for g in pair_genes(p)} | {g for q in cache.pairs("train") for g in pair_genes(q)})

    log.info("loading control and %d pairs' groups (memory-mapped, rows read on demand)", len(pairs))
    ctrl = cache.rows(cache.control_cells())
    single_rows = {g: cache.rows(cache.cells(g, "train")) for g in {g for p in pairs for g in pair_genes(p)}}
    pair_rows = {p: cache.rows(cache.cells(p, cache.pair_split(p))) for p in pairs}
    deltas = {g: cache.single_delta(g) for g in singles}               # full-data single effects
    pair_delta = {p: cache.pair_mean(p) - cache.control_mean for p in pairs}

    per = {p: {"phi": [], "phi_fixed_half": [], "rho_full_split": [], "rho_pair_only": [], "floor_split": []} for p in pairs}
    for r in range(args.reps):
        c_h = halves(ctrl.shape[0], rng)
        ctrl_half = [mean_var(ctrl[h]) for h in c_h]
        ctrl_sorted = [np.sort(ctrl[h], axis=0) for h in c_h]
        s_h = {g: halves(x.shape[0], rng) for g, x in single_rows.items()}
        for p in pairs:
            a, b = pair_genes(p)
            ph = halves(pair_rows[p].shape[0], rng)
            parts = []
            for k in (0, 1):
                xab = pair_rows[p][ph[k]]
                mab, vab, nab = mean_var(xab)
                ma, va, na = mean_var(single_rows[a][s_h[a][k]])
                mb, vb, nb = mean_var(single_rows[b][s_h[b][k]])
                mc, vc, nc = ctrl_half[k]
                parts.append({"x": xab, "mab": mab, "eps": mab - ma - mb + mc,
                              "se2": vab / nab + va / na + vb / nb + vc / nc})
            fixed = cache.scored_genes(p, args.top_k)
            # primary: select on one half, measure on the other, both directions
            for k in (0, 1):
                sel = select_top_genes(parts[k]["x"], ctrl_sorted[k], args.top_k)
                o = parts[1 - k]
                per[p]["phi"].append(signal_fraction(o["eps"][sel], o["se2"][sel]))
                per[p]["phi_fixed_half"].append(signal_fraction(o["eps"][fixed], o["se2"][fixed]))
            # secondary 1: all four groups split, fixed genes
            per[p]["rho_full_split"].append(pearson(parts[0]["eps"][fixed], parts[1]["eps"][fixed]))
            # secondary 2: only the pair's cells split
            naive = [(pt["mab"] - cache.control_mean - deltas[a] - deltas[b])[fixed] for pt in parts]
            per[p]["rho_pair_only"].append(pearson(naive[0], naive[1]))
            # secondary 4 (ssl-lab's split-half floor): ||mean_1 - mean_2|| / 2 over ||Delta||
            d = pair_delta[p][fixed]
            per[p]["floor_split"].append(float(np.linalg.norm(parts[0]["mab"][fixed] - parts[1]["mab"][fixed])) / 2 / float(np.linalg.norm(d)))
        if (r + 1) % 10 == 0 or r == 0:
            log.info("repeat %d/%d done (%.0fs)", r + 1, args.reps, time.time() - t0)

    out_pairs = {}
    for p in pairs:
        a, b = pair_genes(p)
        fixed = cache.scored_genes(p, args.top_k)
        x = pair_rows[p]
        mab, vab, nab = mean_var(x)
        ma, va, na = mean_var(single_rows[a])
        mb, vb, nb = mean_var(single_rows[b])
        mc, vc, nc = mean_var(ctrl)
        eps = (mab - ma - mb + mc)[fixed]
        se2 = (vab / nab + va / na + vb / nb + vc / nc)[fixed]
        delta = (mab - mc)[fixed]
        s_full = float(eps @ eps - se2.sum())
        rho_fs = float(np.nanmean(per[p]["rho_full_split"]))
        rho_po = float(np.nanmean(per[p]["rho_pair_only"]))
        out_pairs[p] = {
            "n_cells": {"pair": nab, "A": na, "B": nb, "control": nc},
            "phi": float(np.mean(per[p]["phi"])),
            "phi_fixed_genes_half": float(np.mean(per[p]["phi_fixed_half"])),
            "rho_full_split_sb": spearman_brown(rho_fs),
            "rho_pair_only_sb": spearman_brown(rho_po),
            "phi_fixed_genes_full_data": s_full / float(eps @ eps) if eps @ eps > 0 else float("nan"),
            "rel_gi": float(np.linalg.norm(eps) / np.linalg.norm(delta)),
            "ep_floor_analytic": float(np.sqrt((vab / nab)[fixed].sum()) / np.linalg.norm(delta)),
            "ep_floor_split": float(np.mean(per[p]["floor_split"])),
        }
    phis = np.array([v["phi"] for v in out_pairs.values()])
    n_reliable = int((phis >= PHI_THRESHOLD).sum())
    n_total = len(phis)
    if args.limit:
        decision = "smoke run: no decision"
    elif n_reliable >= CONTINUE_IF_AT_LEAST:
        decision = "continue"
    elif n_reliable <= STOP_IF_AT_MOST:
        decision = "stop: write up the negative result"
    else:
        decision = "continue with the reliable pairs as a named stratum"

    report = {
        "protocol": "R1.1, fixed in this script's docstring before any number was computed",
        "settings": {"reps": args.reps, "seed": args.seed, "top_k": args.top_k, "limit": args.limit,
                     "phi_threshold": PHI_THRESHOLD, "continue_if_at_least": CONTINUE_IF_AT_LEAST,
                     "stop_if_at_most": STOP_IF_AT_MOST},
        "data": {f: hashlib.md5((Path(args.data_dir) / f).read_bytes()).hexdigest()
                 for f in ("de_genes.json", "splits.json")},
        "summary": {
            "n_pairs": n_total, "n_reliable": n_reliable, "reliable_pairs": sorted(p for p, v in out_pairs.items() if v["phi"] >= PHI_THRESHOLD),
            "decision": decision,
            **{f"median_{k}": float(np.nanmedian([v[k] for v in out_pairs.values()]))
               for k in ("phi", "phi_fixed_genes_half", "rho_full_split_sb", "rho_pair_only_sb",
                         "phi_fixed_genes_full_data", "rel_gi", "ep_floor_analytic", "ep_floor_split")},
            "mean_rel_gi": float(np.mean([v["rel_gi"] for v in out_pairs.values()])),
            "mean_ep_floor_analytic": float(np.mean([v["ep_floor_analytic"] for v in out_pairs.values()])),
        },
        "global_scale_training_pairs": global_scale(cache, args.top_k, deltas),
        "per_pair": out_pairs,
        "runtime_s": round(time.time() - t0, 1),
        # ru_maxrss is bytes on macOS and kilobytes on Linux
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             / (1e9 if sys.platform == "darwin" else 1e6), 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    s = report["summary"]
    log.info("phi >= %.1f on %d/%d pairs (median phi %.2f); decision: %s", PHI_THRESHOLD, n_reliable, n_total, s["median_phi"], decision)
    log.info("wrote %s (%.0fs, peak %.2f GB)", out, report["runtime_s"], report["peak_rss_gb"])


if __name__ == "__main__":
    main()
