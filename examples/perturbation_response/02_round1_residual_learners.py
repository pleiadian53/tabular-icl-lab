"""R1.3, round 1: can an in-context learner predict the genetic-interaction residual?

Context. On the twenty held-out Norman (2019) pairs, the additive prediction, the sum
of the two single-gene effects, is the reference every learned model in the sibling
project ssl-lab has failed to beat. R1.1 showed that what additivity misses is mostly
reproducible signal (18 of 20 pairs at least half signal) and that the room to improve
is in the size of the change: additivity's size-aware error is 0.289, and noise in the
pair's own cells sets a floor near 0.08 for any predictor. This round asks whether
TabPFN-3.5, given the training pairs as examples in its context and no tuning, predicts
enough of the residual to close part of that gap; and whether a gain, if any, is
learning interactions or only correcting the singles' noise, which two no-interaction
references measure.

Notation. Delta_g(p) is the change in mean log1p-CP10K expression of gene g between the
cells of perturbation p and the control cells. For a pair AB, the additive prediction is
a_g = Delta_g(A) + Delta_g(B), the residual is eps_g = Delta_g(AB) - a_g, and an arm
predicts Delta_hat_g = a_g + eps_hat_g. T_p is the pair's scored genes (its top 20
differentially expressed genes, ssl-lab's set). The size-aware error is
e_p = ||Delta_hat - Delta|| / ||Delta|| on T_p.

PROTOCOL (fixed in this docstring before any learner was scored)
---------------------------------------------------------------
Data: the shared Norman cache (ssl-lab's, read-only), combination split. Singles use their
training cells, control uses all control cells, each pair uses all its cells.

Table (R1.2, ``tabicllab.residual_table``): one row per (pair, gene in T_p); seven features
from the single and control cells only (the two singles' effects ordered by magnitude,
their standard errors, the gene's control mean and standard deviation, whether the gene
is one of the pair's targets); label eps_g. Context: the 91 training pairs (about 1,800
rows). Queries: the 20 test pairs (primary) and the 20 validation pairs (a second readout).

Arms, each predicting eps_hat on T_p:
  additive   eps_hat = 0, the reference.
  scaled     eps_hat = (beta - 1) a, one global beta fitted by least squares on the
             training pairs: generic sub- or super-additivity, no interaction.
  shrunk     each single's effect shrunk toward zero by its own noise,
             Delta * max(0, 1 - SE^2 / Delta^2), then summed: no interaction, no fitting.
  ridge      a residual learner on the same table: standardize, polynomial features of
             degree 1 or 2, standardize, ridge; degree and penalty tuned.
  gbm        LightGBM on the same table; leaves, learning rate, trees and minimum leaf
             size tuned.
  tabpfn     TabPFN-3.5 regressor (checkpoint tabpfn-v3.5-20260909), default settings for
             that version, no tuning; fitted on the context rows; seeds 0, 1, 2.
Tuning for ridge and gbm: 5-fold cross-validation over training *pairs* (each pair's rows
stay together), folds drawn with seed 0, criterion the mean e_p over the held-out pairs;
the grids are RIDGE_GRID (2 degrees x 19 penalties, 1e-3 to 1e6) and GBM_GRID (72 points)
below. Nothing is tuned on validation or test pairs. Development runs (``--smoke``) use
training and validation pairs only; no test pair is scored before the full run.

Aggregation (Round 8's order): per pair and seed compute e_p; average the seeds per pair
(arithmetic; deterministic arms have one); an arm's score is the mean over pairs.

Primary family, on the 20 test pairs: per-pair paired differences of seed-averaged e_p,
    tabpfn - additive, tabpfn - scaled, tabpfn - shrunk, tabpfn - ridge, tabpfn - gbm;
joint paired bootstrap over pairs, 10,000 resamples, generator seed 0, one resample shared
by every family and secondary on that set; max-t simultaneous 95% intervals
(``tabicllab.bootstrap``, the procedure of ssl-lab's Round 8).

Decision rule, read from the primary family:
  - TabPFN beats additivity if tabpfn - additive lies wholly below 0; wholly above 0 means
    additivity is better; otherwise inconclusive (a tie).
  - It is credited with learning interactions only if tabpfn - additive, tabpfn - scaled
    and tabpfn - shrunk all lie wholly below 0. Beating additivity alone is reported as a
    gain that a no-interaction rule may explain.
  - Against the tuned learners: wholly below 0 means TabPFN is better than that learner,
    wholly above 0 worse, otherwise a tie.
  - Route 1's rule: Route 4's second gate opens only if TabPFN is credited with learning
    interactions; a tie with additivity is reported as a strong tuning-free reference and
    the route is reviewed.
  A verdict needs at least 18 of the 20 test pairs with finite e_p for every arm and seed;
  otherwise the primary is reported as not evaluable. Excluded pairs are listed.

Secondary family, same resample, max-t, no verdict: ridge - additive, gbm - additive,
scaled - additive, shrunk - additive.

Reported beside, no verdict:
  1. per arm: mean e_p with a percentile interval; mean Delta-correlation on T_p (the
     standing benchmark, additivity 0.961); the geometric-mean error ratio to additivity
     (exp of the mean seed-averaged log of e_p over additivity's e_p, Round 8's cross-read);
     the number of pairs where the arm's e_p is below additivity's; interaction recovery,
     the Pearson correlation on T_p of eps_hat with eps (undefined for additivity);
  2. per-pair numbers for every arm, and each pair's e_p floor from its own cells,
     sqrt(sum_g s^2_AB,g / n_AB) / ||Delta||;
  3. the TabPFN seed ensemble (predictions averaged over the three seeds, scored once);
  4. strata, descriptive with percentile intervals: the test pairs R1.1 found reliable
     (phi_p >= 0.5, 18 pairs) and the pairs with at least 100 cells (16 pairs);
  5. the validation pairs, every arm fitted exactly as for the test pairs, both families
     with simultaneous intervals, no verdict;
  6. cross-validation over the 91 training pairs, 5 outer folds by pair (seed 1); in each
     fold every arm is refitted on the other folds (ridge and gbm re-tuned by inner 5-fold
     CV, beta refitted), so each training pair is predicted once; both families, no
     verdict. With 91 pairs it has more power than the test set and reads as a check.

Parity: with the full test set, additivity's mean e_p must reproduce ssl-lab's published
0.289096 to 1e-12, or the script stops.

Usage
-----
    python examples/perturbation_response/02_round1_residual_learners.py           # full (pod)
    python examples/perturbation_response/02_round1_residual_learners.py --smoke   # tiny, local

Output: ``output/r1_3_round1/report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tabicllab.bootstrap import maxt_family, percentile_interval, resample_indices  # noqa: E402
from tabicllab.norman import load_norman  # noqa: E402
from tabicllab.residual_table import FEATURES, PairRows, ResidualData, global_scale, stack  # noqa: E402
from tabicllab.scoring import pair_metrics, pearson  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("r1.3")

TABPFN_CHECKPOINT = "tabpfn-v3.5-20260909"
SSLLAB_ADDITIVE_MEAN_EP = 0.2890959583925055
SEEDS = (0, 1, 2)
RIDGE_GRID = [{"degree": d, "alpha": float(a)} for d in (1, 2) for a in np.logspace(-3, 6, 19)]
GBM_GRID = [{"num_leaves": nl, "learning_rate": lr, "n_estimators": ne, "min_child_samples": mc}
            for nl in (4, 8, 16) for lr in (0.01, 0.03, 0.1) for ne in (30, 100, 300, 1000)
            for mc in (10, 30)]
SMOKE_RIDGE_GRID = RIDGE_GRID[::7]
SMOKE_GBM_GRID = GBM_GRID[::18]
PRIMARY = [("tabpfn", "additive"), ("tabpfn", "scaled"), ("tabpfn", "shrunk"),
           ("tabpfn", "ridge"), ("tabpfn", "gbm")]
SECONDARY = [("ridge", "additive"), ("gbm", "additive"), ("scaled", "additive"), ("shrunk", "additive")]
MIN_PAIRS_FOR_VERDICT = 18
R1_1_REPORT = (ROOT / "examples" / "perturbation_response" / "docs" / "results" / "reports"
               / "round-00-residual-ceiling.json")


def default_jobs() -> int:
    """Workers for the grids. A pod's ``os.cpu_count()`` can report the host's cores, not
    the container's, so use the CPUs this process may run on and cap the pool."""
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 2
    return max(1, min(8, usable - 1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "norman2019"))
    p.add_argument("--out", default=str(ROOT / "output" / "r1_3_round1" / "report.json"))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="auto", help="TabPFN device: auto, cuda or cpu")
    p.add_argument("--jobs", type=int, default=default_jobs(),
                   help="parallel workers for the ridge and gbm grids (default: usable CPUs - 1, at most 8)")
    p.add_argument("--no-tabpfn", action="store_true", help="skip the TabPFN arm (development only)")
    p.add_argument("--skip-cv", action="store_true", help="skip readout 6, CV over training pairs")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 12 training pairs, and 6 validation pairs standing in for both query "
                        "sets (no test pair is touched), small grids, TabPFN with 2 estimators on CPU, "
                        "200 resamples")
    return p.parse_args()


# ----------------------------------------------------------------------------- learners
def make_ridge(params: dict):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    return make_pipeline(StandardScaler(), PolynomialFeatures(params["degree"], include_bias=False),
                         StandardScaler(), Ridge(alpha=params["alpha"]))


def make_gbm(params: dict):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**params, subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0,
                         random_state=0, deterministic=True, force_col_wise=True,
                         n_jobs=1, verbose=-1)


LEARNERS = {"ridge": make_ridge, "gbm": make_gbm}


def pair_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Fold number (n,) for each pair, a random balanced assignment."""
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.empty(n, dtype=int)
    folds[perm] = np.arange(n) % k
    return folds


def block_ep(eps_hat: np.ndarray, block: PairRows) -> float:
    """e_p from a residual prediction: ||a + eps_hat - Delta|| = ||eps_hat - eps||."""
    return float(np.linalg.norm(eps_hat - block.eps) / np.linalg.norm(block.true_delta))


def fit_predict(make, params: dict, train: list[PairRows], query: list[PairRows]) -> list[np.ndarray]:
    X, y, _ = stack(train)
    model = make(params).fit(X, y)
    return [model.predict(b.X) for b in query]


def cv_mean_ep(name: str, params: dict, blocks: list[PairRows], folds: np.ndarray) -> float:
    eps = []
    for f in np.unique(folds):
        train = [b for b, g in zip(blocks, folds) if g != f]
        held = [b for b, g in zip(blocks, folds) if g == f]
        for b, pred in zip(held, fit_predict(LEARNERS[name], params, train, held)):
            eps.append(block_ep(pred, b))
    return float(np.mean(eps))


def tune(name: str, grid: list[dict], blocks: list[PairRows], k: int, seed: int, jobs: int) -> dict:
    """Pick the grid point with the lowest mean held-out e_p over pair-grouped folds."""
    from joblib import Parallel, delayed
    folds = pair_folds(len(blocks), k, seed)
    scores = Parallel(n_jobs=jobs)(delayed(cv_mean_ep)(name, p, blocks, folds) for p in grid)
    best = int(np.argmin(scores))
    return {"best": grid[best], "best_cv_mean_ep": float(scores[best]),
            "grid_cv_mean_ep": [[p, float(s)] for p, s in zip(grid, scores)]}


def tabpfn_predict(train: list[PairRows], query: list[PairRows], seeds, device: str,
                   n_estimators) -> dict[int, list[np.ndarray]]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    X, y, _ = stack(train)
    Xq = np.vstack([b.X for b in query])
    cuts = np.cumsum([len(b.genes) for b in query])[:-1]
    out = {}
    for s in seeds:
        overrides = {"device": device, "random_state": s}
        if n_estimators is not None:
            overrides["n_estimators"] = n_estimators
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, **overrides)
        model.fit(X, y)
        out[s] = np.split(model.predict(Xq).astype(np.float64), cuts)
    return out


def all_arms(train: list[PairRows], queries: dict[str, list[PairRows]], cfg: dict) -> tuple[dict, dict]:
    """Residual predictions {set: {arm: {seed: [eps_hat per pair]}}} and the fitted settings."""
    beta = global_scale([b.true_delta for b in train], [b.additive for b in train])
    fitted = {"beta": beta}
    preds = {q: {"additive": {0: [np.zeros(len(b.genes)) for b in blocks]},
                 "scaled": {0: [(beta - 1.0) * b.additive for b in blocks]},
                 "shrunk": {0: [b.shrunk - b.additive for b in blocks]}}
             for q, blocks in queries.items()}
    for name, grid in (("ridge", cfg["ridge_grid"]), ("gbm", cfg["gbm_grid"])):
        t0 = time.time()
        fitted[name] = tune(name, grid, train, cfg["folds"], cfg["tune_seed"], cfg["jobs"])
        for q, blocks in queries.items():
            preds[q][name] = {0: fit_predict(LEARNERS[name], fitted[name]["best"], train, blocks)}
        log.info("  %s tuned in %.0fs: %s (CV mean e_p %.4f)", name, time.time() - t0,
                 fitted[name]["best"], fitted[name]["best_cv_mean_ep"])
    if cfg["tabpfn"]:
        t0 = time.time()
        names = list(queries)
        allq = [b for q in names for b in queries[q]]
        by_seed = tabpfn_predict(train, allq, cfg["seeds"], cfg["device"], cfg["tabpfn_estimators"])
        cut = np.cumsum([len(queries[q]) for q in names])[:-1]
        for q, sl in zip(names, np.split(np.arange(len(allq)), cut)):
            preds[q]["tabpfn"] = {s: [by_seed[s][i] for i in sl] for s in cfg["seeds"]}
            preds[q]["tabpfn_ensemble"] = {0: [np.mean([by_seed[s][i] for s in cfg["seeds"]], 0) for i in sl]}
        log.info("  tabpfn: %d seeds in %.0fs on %s", len(cfg["seeds"]), time.time() - t0, cfg["device"])
    return preds, fitted


# ----------------------------------------------------------------------------- scoring
def score(blocks: list[PairRows], preds: dict) -> dict:
    """Per-pair, seed-averaged metrics {arm: {metric: (n_pairs,)}}."""
    out = {}
    for arm, by_seed in preds.items():
        rows = {"rel_err": [], "delta_r": [], "log_r": [], "recovery": []}
        for i, b in enumerate(blocks):
            per_seed = []
            for eps_hat in (by_seed[s][i] for s in by_seed):
                m = pair_metrics(pred_pair=b.additive + eps_hat, true_pair=b.true_delta,
                                 control_mean=np.zeros_like(b.additive), delta_a=b.additive,
                                 delta_b=np.zeros_like(b.additive), top_idx=np.arange(len(b.genes)))
                per_seed.append((m["rel_err"], m["delta_r"], m["log_r"], pearson(eps_hat, b.eps)))
            for key, v in zip(rows, np.mean(np.array(per_seed, dtype=np.float64), 0)):
                rows[key].append(float(v))
        out[arm] = {k: np.array(v) for k, v in rows.items()}
    return out


def summarize(blocks: list[PairRows], scores: dict, n_boot: int, verdict: bool) -> dict:
    arms = list(scores)
    finite = np.all([np.isfinite(scores[a]["rel_err"]) for a in arms], axis=0)
    excluded = [b.name for b, ok in zip(blocks, finite) if not ok]
    keep = np.flatnonzero(finite)
    idx = resample_indices(len(keep), n_boot, 0)
    E = {a: scores[a]["rel_err"][keep] for a in arms}
    has = set(arms)
    prim = {f"{a}-{b}": E[a] - E[b] for a, b in PRIMARY if a in has and b in has}
    sec = {f"{a}-{b}": E[a] - E[b] for a, b in SECONDARY if a in has and b in has}
    add = E["additive"]
    per_arm = {}
    for a in arms:
        lr = scores[a]["log_r"][keep]
        rec = scores[a]["recovery"][keep]
        per_arm[a] = {
            "mean_ep": percentile_interval(E[a], idx),
            "mean_delta_r": float(np.mean(scores[a]["delta_r"][keep])),
            "geo_mean_ratio_to_additive": float(np.exp(np.nanmean(lr))) if np.isfinite(lr).any() else None,
            "pairs_below_additive": int(np.sum(E[a] < add - 1e-15)),
            "mean_interaction_recovery": float(np.nanmean(rec)) if np.isfinite(rec).any() else None,
        }
    floor = np.array([np.sqrt(b.pair_se2.sum()) / np.linalg.norm(b.true_delta) for b in blocks])
    out = {"n_pairs": len(blocks), "n_scored": int(keep.size), "excluded": excluded,
           "mean_ep_floor": float(floor[keep].mean()), "arms": per_arm,
           "primary_family": maxt_family(prim, idx) if prim else None,
           "secondary_family": maxt_family(sec, idx) if sec else None,
           "per_pair": {b.name: {"n_cells": b.n_cells, "n_genes": int(len(b.genes)),
                                 "ep_floor": float(floor[i]),
                                 **{a: {k: float(scores[a][k][i]) for k in ("rel_err", "delta_r", "recovery")}
                                    for a in arms}}
                        for i, b in enumerate(blocks)}}
    if verdict:
        out["verdict"] = decide(out["primary_family"], keep.size)
    return out


def decide(fam: dict | None, n_scored: int) -> dict:
    if fam is None:
        return {"evaluable": False, "reason": "TabPFN arm not run"}
    if n_scored < MIN_PAIRS_FOR_VERDICT:
        return {"evaluable": False, "reason": f"only {n_scored} pairs scored, need {MIN_PAIRS_FOR_VERDICT}"}
    c = fam["contrasts"]

    def side(k: str) -> str:
        lo, hi = c[k]["ci_simultaneous"]
        return "below" if hi < 0 else "above" if lo > 0 else "contains 0"

    refs = {k: side(f"tabpfn-{k}") for k in ("additive", "scaled", "shrunk")}
    if all(v == "below" for v in refs.values()):
        main = "learns interactions: beats additivity and both no-interaction references"
    elif refs["additive"] == "below":
        failed = [k for k in ("scaled", "shrunk") if refs[k] != "below"]
        main = f"beats additivity, but not {' or '.join(failed)}: a gain a no-interaction rule may explain"
    elif refs["additive"] == "above":
        main = "additivity better"
    else:
        main = "inconclusive: a tie with additivity"
    tuned = {k: {"below": "TabPFN better", "above": "TabPFN worse", "contains 0": "tie"}[side(f"tabpfn-{k}")]
             for k in ("ridge", "gbm")}
    route = ("Route 4's second gate opens" if main.startswith("learns")
             else "review Route 1; TabPFN reported as a tuning-free reference" if main.startswith("inconclusive")
             else "review Route 1")
    return {"evaluable": True, "tabpfn": main, "interval_sides": refs, "vs_tuned_learners": tuned,
            "route_1": route}


def strata(blocks: list[PairRows], scores: dict, n_boot: int) -> dict:
    """Descriptive: TabPFN minus each reference on the reliable and the larger pairs."""
    if "tabpfn" not in scores:
        return {}
    phi = {}
    if R1_1_REPORT.exists():
        phi = {k: v["phi"] for k, v in json.loads(R1_1_REPORT.read_text())["per_pair"].items()}
    masks = {"r1_1_reliable_phi_ge_0.5": np.array([phi.get(b.name, np.nan) >= 0.5 for b in blocks]),
             "at_least_100_cells": np.array([b.n_cells >= 100 for b in blocks])}
    out = {}
    for name, m in masks.items():
        sel = np.flatnonzero(m)
        if sel.size < 2:
            continue
        idx = resample_indices(sel.size, n_boot, 0)
        out[name] = {"n_pairs": int(sel.size), "pairs": [blocks[i].name for i in sel],
                     **{f"tabpfn-{r}": percentile_interval(scores["tabpfn"]["rel_err"][sel]
                                                           - scores[r]["rel_err"][sel], idx)
                        for r in ("additive", "scaled", "shrunk", "ridge", "gbm") if r in scores}}
    return out


def cross_validate_training(train: list[PairRows], cfg: dict) -> dict:
    """Readout 6: every training pair predicted once by arms fitted on the other folds."""
    folds = pair_folds(len(train), cfg["folds"], 1)
    preds: dict = {}
    fitted = []
    for f in range(cfg["folds"]):
        tr = [b for b, g in zip(train, folds) if g != f]
        held = [b for b, g in zip(train, folds) if g == f]
        log.info(" outer fold %d: %d context pairs, %d held out", f, len(tr), len(held))
        p, fit = all_arms(tr, {"held": held}, cfg)
        fitted.append({"beta": fit["beta"], **{k: fit[k]["best"] for k in ("ridge", "gbm")}})
        for arm, by_seed in p["held"].items():
            dst = preds.setdefault(arm, {s: [None] * len(train) for s in by_seed})
            pos = np.flatnonzero(folds == f)
            for s, lst in by_seed.items():
                for j, v in zip(pos, lst):
                    dst[s][j] = v
    return {"folds_seed": 1, "fitted_per_fold": fitted,
            **summarize(train, score(train, preds), cfg["n_boot"], verdict=False)}


# ----------------------------------------------------------------------------- main
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def versions() -> dict:
    import lightgbm
    import sklearn
    out = {"numpy": np.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}
    try:
        import tabpfn
        import torch
        out.update(tabpfn=tabpfn.__version__, torch=torch.__version__)
    except ImportError:
        pass
    return out


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    args = parse_args()
    t_start = time.time()
    data_dir = Path(args.data_dir)
    cache = load_norman(data_dir)
    data = ResidualData(cache, top_k=args.top_k)

    train_names = cache.pairs("train")
    test_names, val_names = cache.pairs("test"), cache.pairs("val")
    if args.smoke:
        # The smoke never touches a test pair: validation pairs stand in for both query sets.
        train_names, test_names, val_names = train_names[:12], val_names[:3], val_names[3:6]
    log.info("reading groups: %d training, %d test, %d validation pairs",
             len(train_names), len(test_names), len(val_names))
    train = [data.pair_rows(n) for n in train_names]
    test = [data.pair_rows(n) for n in test_names]
    val = [data.pair_rows(n) for n in val_names]
    log.info("table: %d context rows, %d test rows, %d validation rows (data read in %.0fs)",
             sum(len(b.genes) for b in train), sum(len(b.genes) for b in test),
             sum(len(b.genes) for b in val), time.time() - t_start)

    if not args.smoke:
        add_ep = np.mean([np.linalg.norm(b.eps) / np.linalg.norm(b.true_delta) for b in test])
        if abs(add_ep - SSLLAB_ADDITIVE_MEAN_EP) > 1e-12:
            raise SystemExit(f"PARITY FAILED: additive mean e_p {add_ep!r} != {SSLLAB_ADDITIVE_MEAN_EP!r}")
        log.info("parity: additive mean e_p %.6f matches ssl-lab", add_ep)

    cfg = {"folds": args.folds, "tune_seed": 0, "jobs": args.jobs, "seeds": SEEDS,
           "device": "cpu" if args.smoke else resolve_device(args.device),
           "tabpfn": not args.no_tabpfn, "tabpfn_estimators": 2 if args.smoke else None,
           "ridge_grid": SMOKE_RIDGE_GRID if args.smoke else RIDGE_GRID,
           "gbm_grid": SMOKE_GBM_GRID if args.smoke else GBM_GRID,
           "n_boot": 200 if args.smoke else args.n_boot}

    log.info("fitting every arm on the training pairs")
    preds, fitted = all_arms(train, {"test": test, "val": val}, cfg)
    log.info("scoring")
    test_scores = score(test, preds["test"])
    report = {
        "protocol": "R1.3 round 1; see the docstring of 02_round1_residual_learners.py",
        "settings": {"top_k": args.top_k, "features": list(FEATURES), "seeds": list(SEEDS),
                     "folds": args.folds, "n_boot": cfg["n_boot"], "device": cfg["device"],
                     "smoke": args.smoke, "tabpfn_checkpoint": TABPFN_CHECKPOINT,
                     "tabpfn_settings": "create_default_for_version(V3_5), n_estimators "
                                        + ("2 (smoke)" if args.smoke else "'auto' (the version default)"),
                     "ridge_grid_size": len(cfg["ridge_grid"]), "gbm_grid_size": len(cfg["gbm_grid"]),
                     "versions": versions()},
        "data": {"de_genes_md5": md5(data_dir / "de_genes.json"), "splits_md5": md5(data_dir / "splits.json"),
                 "context_pairs": len(train), "context_rows": int(sum(len(b.genes) for b in train))},
        "fitted": fitted,
        "test": {**summarize(test, test_scores, cfg["n_boot"], verdict=True),
                 "strata": strata(test, test_scores, cfg["n_boot"])},
        "val": summarize(val, score(val, preds["val"]), cfg["n_boot"], verdict=False),
    }
    if not args.skip_cv:
        log.info("readout 6: cross-validation over the training pairs")
        report["cv_training_pairs"] = cross_validate_training(train, cfg)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["runtime_s"] = round(time.time() - t_start, 1)
    report["peak_rss_gb"] = round(rss / (1e9 if sys.platform == "darwin" else 1e6), 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    t = report["test"]
    log.info("TEST (%d pairs scored): mean e_p floor %.3f", t["n_scored"], t["mean_ep_floor"])
    for a, s in t["arms"].items():
        log.info("  %-16s e_p %.4f [%.4f, %.4f]  Δr %.4f  below additive %d/%d", a, s["mean_ep"]["mean"],
                 *s["mean_ep"]["ci"], s["mean_delta_r"], s["pairs_below_additive"], t["n_scored"])
    for fam in ("primary_family", "secondary_family"):
        if t[fam]:
            for k, c in t[fam]["contrasts"].items():
                log.info("  %-20s %+.4f  simultaneous [%+.4f, %+.4f]", k, c["diff"], *c["ci_simultaneous"])
    log.info("VERDICT: %s", json.dumps(t["verdict"]))
    log.info("wrote %s (%.0fs, peak %.2f GB)", out, report["runtime_s"], report["peak_rss_gb"])


if __name__ == "__main__":
    main()
