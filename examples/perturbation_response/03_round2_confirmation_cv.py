"""R1.6, round 2: confirm round 1 with cross-validation over all 131 pairs.

Context. Round 1 (R1.3) asked whether TabPFN-3.5, given the training combinations as
examples and never tuned, predicts the genetic-interaction residual well enough to beat
additivity, the sum of the two single-gene effects. On its primary readout, the twenty
held-out test pairs, the answer was a tie: the size-aware error fell from 0.289 to 0.260
and TabPFN did better on 13 of 20 pairs, but the simultaneous interval included zero. Its
secondary readouts were clearer: on the twenty validation pairs the error fell from 0.318
to 0.221, and across the 91 training pairs, cross-validated, from 0.312 to 0.233, with
TabPFN ahead of every other arm. Twenty pairs, one of them dominated by an interaction
the inputs cannot foresee, gave the primary too little power to decide.

This round asks the same question with the power to answer it: every one of the 131
combinations is predicted by models fitted without it, and the primary readout is all
131 pairs.

What this round can and cannot confirm. Every pair here was already scored once in
round 1, the test pairs as the primary, the validation pairs and the training pairs as
secondary readouts. So this is not a test on unseen data. It is a test, with everything
fixed in advance and nothing changed in response to round 1, of whether round 1's
pattern holds when the whole dataset is the readout: the table, the arms, the tuning
grids and the decision rule are round 1's, and the code that fits and scores them is
round 1's script, loaded unchanged. A confirmation on independent data needs another
combinatorial screen.

Notation, as in round 1. Delta_g(p) is the change in mean log1p-CP10K expression of gene g
between the cells of perturbation p and the control cells. For a pair AB, the additive
prediction is a_g = Delta_g(A) + Delta_g(B), the residual eps_g = Delta_g(AB) - a_g, and
an arm predicts Delta_hat_g = a_g + eps_hat_g on the pair's scored genes T_p (its top 20
differentially expressed genes). The size-aware error is
e_p = ||Delta_hat - Delta|| / ||Delta|| on T_p.

PROTOCOL (fixed in this docstring before the full run)
-----------------------------------------------------
Data and table: round 1's (``tabicllab.residual_table``): one row per (pair, gene in T_p),
seven features from the single and control cells only, label eps_g. Singles use their
training cells, control all control cells, each pair all its cells.

Pairs: all 131 two-gene combinations of the combination split (91 training, 20
validation, 20 test), pooled.

Cross-validation: 5 folds over pairs (each pair's rows stay together), repeated 3 times
with fold assignments drawn with seeds 0, 1 and 2 (balanced random assignment, round 1's
``pair_folds``). In every outer fold, every arm is fitted on the other four folds only:
beta refitted, ridge and gbm re-tuned by inner 5-fold cross-validation over the context
pairs (inner folds seed 0, criterion mean e_p, round 1's grids), TabPFN-3.5 fitted with
seeds 0, 1, 2. Every pair is therefore predicted three times, once per repeat, by models
that never saw it.

Arms: round 1's six, unchanged: additive (the reference), scaled additive, shrunk
additive, ridge, gbm, TabPFN-3.5 (checkpoint tabpfn-v3.5-20260909, version defaults, no
tuning).

Aggregation: per pair, e_p is averaged over the 3 repeats and, for TabPFN, the 3 seeds
(9 predictions; deterministic arms 3); an arm's score is the mean over pairs.

Primary family, over all 131 pairs: per-pair paired differences of averaged e_p,
    tabpfn - additive, tabpfn - scaled, tabpfn - shrunk, tabpfn - ridge, tabpfn - gbm;
joint paired bootstrap over pairs, 10,000 resamples, generator seed 0, max-t simultaneous
95% intervals (round 1's procedure).

Decision rule, round 1's:
  - TabPFN beats additivity if tabpfn - additive lies wholly below 0; wholly above means
    additivity is better; otherwise a tie.
  - It is credited with learning interactions only if tabpfn - additive, tabpfn - scaled
    and tabpfn - shrunk all lie wholly below 0.
  - Against the tuned learners: wholly below 0, TabPFN better; wholly above, worse;
    otherwise a tie.
  - Route 1: if TabPFN is credited with learning interactions, Route 4's second gate
    opens and the gene descriptors (R1.5) proceed with this round's TabPFN arm as their
    reference; otherwise the route is reviewed again.
  A verdict needs at least 125 of the 131 pairs with finite e_p for every arm.

Secondary family, same resample, max-t, no verdict: ridge - additive, gbm - additive,
scaled - additive, shrunk - additive.

Reported beside, no verdict:
  1. per arm: mean e_p with a percentile interval, mean Delta-correlation, geometric-mean
     error ratio to additivity, pairs below additivity, interaction recovery (round 1's
     definitions);
  2. strata, each with every arm's mean e_p and TabPFN minus each other arm with
     percentile intervals: the 20 test pairs (commensurable with ssl-lab and round 1),
     the 20 validation pairs, the 91 training pairs, and the pairs with at least 100 cells;
  3. stability over fold assignments: each repeat's mean e_p per arm and its primary
     differences, scored alone;
  4. per-pair numbers for every arm, and each pair's e_p floor from its own cells.

Parity: additivity's mean e_p over the 20 test pairs must reproduce ssl-lab's published
0.289096 to 1e-12, or the script stops.

Limits: the bootstrap treats pairs as independent, but pairs share single genes (73 genes
make the 131 pairs), so the intervals are somewhat narrower than they should be. A pair's
singles appear in other pairs in its context, as they do for the test pairs in round 1.

Usage
-----
    python examples/perturbation_response/03_round2_confirmation_cv.py           # full (pod)
    python examples/perturbation_response/03_round2_confirmation_cv.py --smoke   # tiny, local

Output: ``output/r1_6_round2/report.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tabicllab.bootstrap import percentile_interval, resample_indices  # noqa: E402
from tabicllab.norman import load_norman  # noqa: E402
from tabicllab.residual_table import FEATURES, ResidualData  # noqa: E402


def _load_round1():
    """Round 1's script as a module, so this round fits and scores with the same code."""
    path = Path(__file__).with_name("02_round1_residual_learners.py")
    spec = importlib.util.spec_from_file_location("round1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r1 = _load_round1()
log = logging.getLogger("r1.6")

REPEAT_SEEDS = (0, 1, 2)
MIN_PAIRS_FOR_VERDICT = 125
MAIN_ARMS = ("additive", "scaled", "shrunk", "ridge", "gbm", "tabpfn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "norman2019"))
    p.add_argument("--out", default=str(ROOT / "output" / "r1_6_round2" / "report.json"))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="auto", help="TabPFN device: auto, cuda or cpu")
    p.add_argument("--jobs", type=int, default=r1.default_jobs())
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 24 training pairs only, 2 repeats, small grids, TabPFN with "
                        "2 estimators on CPU, 200 resamples")
    return p.parse_args()


def cross_validated_predictions(blocks: list, cfg: dict, repeats) -> tuple[dict, list]:
    """{arm: {(repeat, seed): [eps_hat per pair]}}: every pair predicted once per repeat."""
    preds: dict = {}
    fitted = []
    for rep in repeats:
        folds = r1.pair_folds(len(blocks), cfg["folds"], rep)
        for f in range(cfg["folds"]):
            ctx = [b for b, g in zip(blocks, folds) if g != f]
            held_pos = np.flatnonzero(folds == f)
            held = [blocks[i] for i in held_pos]
            log.info("repeat %d, fold %d: %d context pairs, %d held out", rep, f, len(ctx), len(held))
            p, fit = r1.all_arms(ctx, {"held": held}, cfg)
            fitted.append({"repeat": rep, "fold": f, "beta": fit["beta"],
                           **{k: fit[k]["best"] for k in ("ridge", "gbm")}})
            for arm, by_seed in p["held"].items():
                for s, lst in by_seed.items():
                    dst = preds.setdefault(arm, {}).setdefault((rep, s), [None] * len(blocks))
                    for j, v in zip(held_pos, lst):
                        dst[j] = v
    return preds, fitted


def stratum(blocks: list, scores: dict, sel: np.ndarray, n_boot: int) -> dict:
    idx = resample_indices(sel.size, n_boot, 0)
    out = {"n_pairs": int(sel.size),
           "mean_ep": {a: float(scores[a]["rel_err"][sel].mean()) for a in scores},
           "pairs_below_additive": {a: int(np.sum(scores[a]["rel_err"][sel] < scores["additive"]["rel_err"][sel] - 1e-15))
                                    for a in scores}}
    if "tabpfn" in scores:
        for r in ("additive", "scaled", "shrunk", "ridge", "gbm"):
            out[f"tabpfn-{r}"] = percentile_interval(scores["tabpfn"]["rel_err"][sel] - scores[r]["rel_err"][sel], idx)
    return out


def main() -> None:
    args = parse_args()
    t_start = time.time()
    data_dir = Path(args.data_dir)
    cache = load_norman(data_dir)
    data = ResidualData(cache, top_k=args.top_k)

    split_of = {}
    names = []
    for split in ("train", "val", "test"):
        for n in cache.pairs(split):
            names.append(n)
            split_of[n] = split
    repeats = REPEAT_SEEDS
    if args.smoke:          # development never touches a validation or test pair
        names = [n for n in names if split_of[n] == "train"][:24]
        repeats = REPEAT_SEEDS[:2]
    blocks = [data.pair_rows(n) for n in names]
    log.info("table: %d pairs, %d rows (data read in %.0fs)",
             len(blocks), sum(len(b.genes) for b in blocks), time.time() - t_start)

    if not args.smoke:
        test = [b for b in blocks if split_of[b.name] == "test"]
        add_ep = np.mean([np.linalg.norm(b.eps) / np.linalg.norm(b.true_delta) for b in test])
        if abs(add_ep - r1.SSLLAB_ADDITIVE_MEAN_EP) > 1e-12:
            raise SystemExit(f"PARITY FAILED: additive mean e_p {add_ep!r} != {r1.SSLLAB_ADDITIVE_MEAN_EP!r}")
        log.info("parity: additive mean e_p on the test pairs %.6f matches ssl-lab", add_ep)

    cfg = {"folds": args.folds, "tune_seed": 0, "jobs": args.jobs, "seeds": r1.SEEDS,
           "device": "cpu" if args.smoke else r1.resolve_device(args.device),
           "tabpfn": True, "tabpfn_estimators": 2 if args.smoke else None,
           "ridge_grid": r1.SMOKE_RIDGE_GRID if args.smoke else r1.RIDGE_GRID,
           "gbm_grid": r1.SMOKE_GBM_GRID if args.smoke else r1.GBM_GRID,
           "n_boot": 200 if args.smoke else args.n_boot}

    preds, fitted = cross_validated_predictions(blocks, cfg, repeats)
    log.info("scoring")
    scores = r1.score(blocks, preds)
    r1.MIN_PAIRS_FOR_VERDICT = MIN_PAIRS_FOR_VERDICT     # round 1's decide(), at this round's size
    main_scores = {a: scores[a] for a in MAIN_ARMS}
    primary = r1.summarize(blocks, scores, cfg["n_boot"], verdict=True)

    split_arr = np.array([split_of[b.name] for b in blocks])
    cells = np.array([b.n_cells for b in blocks])
    strata = {}
    for name, sel in (("test_pairs", np.flatnonzero(split_arr == "test")),
                      ("validation_pairs", np.flatnonzero(split_arr == "val")),
                      ("training_pairs", np.flatnonzero(split_arr == "train")),
                      ("at_least_100_cells", np.flatnonzero(cells >= 100))):
        if sel.size >= 2:
            strata[name] = {"pairs": [blocks[i].name for i in sel],
                            **stratum(blocks, main_scores, sel, cfg["n_boot"])}

    per_repeat = {}
    for rep in repeats:
        sub = {a: {k: v for k, v in by.items() if k[0] == rep} for a, by in preds.items() if a in MAIN_ARMS}
        sc = r1.score(blocks, sub)
        per_repeat[str(rep)] = {
            "mean_ep": {a: float(sc[a]["rel_err"].mean()) for a in sc},
            **{f"tabpfn-{r}": float((sc["tabpfn"]["rel_err"] - sc[r]["rel_err"]).mean())
               for r in ("additive", "scaled", "shrunk", "ridge", "gbm")}}

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "protocol": "R1.6 round 2; see the docstring of 03_round2_confirmation_cv.py",
        "settings": {"top_k": args.top_k, "features": list(FEATURES), "tabpfn_seeds": list(r1.SEEDS),
                     "repeat_seeds": list(repeats), "folds": args.folds, "n_boot": cfg["n_boot"],
                     "device": cfg["device"], "smoke": args.smoke, "tabpfn_checkpoint": r1.TABPFN_CHECKPOINT,
                     "tabpfn_settings": "create_default_for_version(V3_5), n_estimators "
                                        + ("2 (smoke)" if args.smoke else "'auto' (the version default)"),
                     "ridge_grid_size": len(cfg["ridge_grid"]), "gbm_grid_size": len(cfg["gbm_grid"]),
                     "min_pairs_for_verdict": MIN_PAIRS_FOR_VERDICT, "versions": r1.versions()},
        "data": {"de_genes_md5": r1.md5(data_dir / "de_genes.json"), "splits_md5": r1.md5(data_dir / "splits.json"),
                 "n_pairs": len(blocks), "n_rows": int(sum(len(b.genes) for b in blocks)),
                 "split_of_pair": split_of if not args.smoke else {n: split_of[n] for n in names}},
        "fitted_per_fold": fitted,
        "all_pairs": primary,
        "strata": strata,
        "per_repeat": per_repeat,
        "runtime_s": round(time.time() - t_start, 1),
        "peak_rss_gb": round(rss / (1e9 if sys.platform == "darwin" else 1e6), 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    t = report["all_pairs"]
    log.info("ALL PAIRS (%d scored): mean e_p floor %.3f", t["n_scored"], t["mean_ep_floor"])
    for a, s in t["arms"].items():
        log.info("  %-16s e_p %.4f [%.4f, %.4f]  Δr %.4f  below additive %d/%d", a, s["mean_ep"]["mean"],
                 *s["mean_ep"]["ci"], s["mean_delta_r"], s["pairs_below_additive"], t["n_scored"])
    for fam in ("primary_family", "secondary_family"):
        for k, c in t[fam]["contrasts"].items():
            log.info("  %-20s %+.4f  simultaneous [%+.4f, %+.4f]", k, c["diff"], *c["ci_simultaneous"])
    for name, s in strata.items():
        log.info("  stratum %-20s n=%3d  tabpfn %.4f additive %.4f", name, s["n_pairs"],
                 s["mean_ep"]["tabpfn"], s["mean_ep"]["additive"])
    log.info("VERDICT: %s", json.dumps(t["verdict"]))
    log.info("wrote %s (%.0fs, peak %.2f GB)", out, report["runtime_s"], report["peak_rss_gb"])


if __name__ == "__main__":
    main()
