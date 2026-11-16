"""R1.7, round 3: does the learned combination rule transfer to genes never seen in combination?

Context. Round 2 (R1.6) cross-validated every arm over all 131 two-gene combinations of the
Norman (2019) screen and credited TabPFN-3.5 with learning interactions: its size-aware
error was 0.227 against additivity's 0.309, and it beat a globally scaled additivity, an
additivity with noise-shrunk singles, a tuned ridge and a tuned LightGBM, every
simultaneous interval below zero. But round 2's folds were drawn at random over pairs, so a
held-out pair's two genes usually appeared in other pairs of the context. CEBPE, for
instance, is in eleven of the 131 pairs, and the largest gains were on CEBP pairs. A row's
features are the two singles' effects on one gene, and a gene's single effect is the same
vector in every pair that contains it, so those values can act as an implicit identity of
the gene. Part of round 2's gain may therefore come from having seen a gene's other
combinations, not from a rule that holds for any two genes.

This round removes that possibility. Each pair is predicted by models whose context
contains no other combination of either of its genes. If TabPFN still beats additivity
and both no-interaction references, it has learned a combination rule that transfers to
genes it has never seen combined; if it does not, round 2's gain depended on familiarity
with the genes.

Notation, as in rounds 1 and 2. Delta_g(p) is the change in mean log1p-CP10K expression of
gene g between the cells of perturbation p and the control cells. For a pair AB, the
additive prediction is a_g = Delta_g(A) + Delta_g(B), the residual eps_g = Delta_g(AB) - a_g,
and an arm predicts Delta_hat_g = a_g + eps_hat_g on the pair's scored genes T_p. The
size-aware error is e_p = ||Delta_hat - Delta|| / ||Delta|| on T_p.

PROTOCOL (fixed in this docstring before the full run)
-----------------------------------------------------
Data, table and arms: round 2's, unchanged, fitted and scored by round 1's script loaded
unchanged (``02_round1_residual_learners.py``): seven features per (pair, scored gene) row
from single and control cells only; arms additive, scaled additive, shrunk additive, ridge
and LightGBM (both tuned by inner 5-fold cross-validation over the context pairs, seed 0,
round 1's grids, criterion mean e_p), TabPFN-3.5 (checkpoint tabpfn-v3.5-20260909, version
defaults, seeds 0, 1, 2).

Pairs: all 131 combinations (91 training, 20 validation, 20 test), pooled.

Primary readout, leave-both-genes-out: for each pair AB, the context is every other pair
that contains neither A nor B. Every arm is fitted on that context (beta refitted, ridge and
LightGBM re-tuned, TabPFN given it as context) and predicts AB alone. Each pair is
predicted once; TabPFN's e_p is averaged over its three seeds.

Primary family, over all 131 pairs: per-pair paired differences of e_p,
    tabpfn - additive, tabpfn - scaled, tabpfn - shrunk, tabpfn - ridge, tabpfn - gbm;
joint paired bootstrap over pairs, 10,000 resamples, generator seed 0, max-t simultaneous
95% intervals.

Decision rule, round 2's:
  - TabPFN beats additivity if tabpfn - additive lies wholly below 0; wholly above means
    additivity is better; otherwise a tie.
  - It is credited with a combination rule that transfers to unseen genes only if
    tabpfn - additive, tabpfn - scaled and tabpfn - shrunk all lie wholly below 0.
  - Otherwise round 2's credit is read as depending, at least in part, on familiarity with
    the genes' other combinations.
  - Against the tuned learners: wholly below 0, TabPFN better; wholly above, worse;
    otherwise a tie.
  A verdict needs at least 125 of the 131 pairs with finite e_p for every arm.

Secondary family, same resample, max-t, no verdict: ridge - additive, gbm - additive,
scaled - additive, shrunk - additive.

Reported beside, no verdict:
  1. per arm: mean e_p with a percentile interval, mean Delta-correlation, geometric-mean
     error ratio to additivity, pairs below additivity, interaction recovery;
  2. retention: per pair, this round's e_p minus round 2's for TabPFN and for LightGBM (read
     from round 2's committed report), and the share of round 2's mean gain over additivity
     that survives here;
  3. leave-one-gene-out: for each of the 73 genes, every pair containing it is held out
     together and predicted by arms fitted on the rest; each pair is predicted twice, once
     per gene, and its e_p is averaged over both (and TabPFN's seeds); both families,
     no verdict. Its context may contain the pair's other gene, so it sits between round 2
     and the primary;
  4. strata: the 20 test pairs, the 20 validation pairs, the 91 training pairs, the pairs
     with at least 100 cells, and the pairs containing CEBPA, CEBPB or CEBPE;
  5. per-pair numbers and context sizes.

Parity: additivity's mean e_p over the 20 test pairs must reproduce ssl-lab's published
0.289096 to 1e-12, or the script stops.

Limits: the context is smaller than round 2's (each pair loses every pair that shares a gene
with it), so some loss is expected from fewer examples alone; the context sizes are
reported. Inner tuning folds are drawn at random over the context pairs, not by gene. The
bootstrap treats pairs as independent, but pairs share genes. Every pair was scored before,
in rounds 1 and 2.

Usage
-----
    python examples/perturbation_response/04_round3_leave_genes_out.py           # full (pod)
    python examples/perturbation_response/04_round3_leave_genes_out.py --smoke   # tiny, local

Output: ``output/r1_7_round3/report.json``.
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
from tabicllab.norman import load_norman, pair_genes  # noqa: E402
from tabicllab.residual_table import FEATURES, ResidualData  # noqa: E402


def _load_round1():
    """Round 1's script as a module, so this round fits and scores with the same code."""
    path = Path(__file__).with_name("02_round1_residual_learners.py")
    spec = importlib.util.spec_from_file_location("round1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r1 = _load_round1()
log = logging.getLogger("r1.7")

MIN_PAIRS_FOR_VERDICT = 125
MAIN_ARMS = ("additive", "scaled", "shrunk", "ridge", "gbm", "tabpfn")
CEBP = {"CEBPA", "CEBPB", "CEBPE"}
ROUND2_REPORT = (ROOT / "examples" / "perturbation_response" / "docs" / "results" / "reports"
                 / "round-02-confirmation-cv.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "norman2019"))
    p.add_argument("--out", default=str(ROOT / "output" / "r1_7_round3" / "report.json"))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--folds", type=int, default=5, help="inner tuning folds")
    p.add_argument("--device", default="auto", help="TabPFN device: auto, cuda or cpu")
    p.add_argument("--jobs", type=int, default=r1.default_jobs())
    p.add_argument("--skip-one-gene", action="store_true", help="skip readout 3 (development only)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 30 training pairs only, small grids, TabPFN with 2 estimators "
                        "(on CPU unless --device names one), 200 resamples")
    return p.parse_args()


def held_out_predictions(blocks: list, plan: list[tuple[list[int], list[tuple[int, int]]]], cfg: dict) -> dict:
    """Predictions {arm: {(slot, seed): [eps_hat per pair]}} for a list of (context, held).

    Each entry fits every arm once on its context pairs and predicts its held pairs, given as
    (pair index, slot); every pair must be predicted exactly once per slot.
    """
    n_slots = 1 + max(slot for _, held in plan for _, slot in held)
    preds: dict = {}
    t0 = time.time()
    for k, (ctx_idx, held_slots) in enumerate(plan):
        ctx = [blocks[i] for i in ctx_idx]
        held = [blocks[i] for i, _ in held_slots]
        p, _ = r1.all_arms(ctx, {"held": held}, cfg)
        for arm, by_seed in p["held"].items():
            for s, lst in by_seed.items():
                dst = preds.setdefault(arm, {})
                for sl in range(n_slots):
                    dst.setdefault((sl, s), [None] * len(blocks))
                for (j, slot), v in zip(held_slots, lst):
                    dst[(slot, s)][j] = v
        if (k + 1) % 10 == 0 or k + 1 == len(plan):
            log.info("%d of %d contexts fitted (%.0fs)", k + 1, len(plan), time.time() - t0)
    missing = [(a, key) for a, by in preds.items() for key, lst in by.items() if any(v is None for v in lst)]
    if missing:
        raise RuntimeError(f"pairs left unpredicted: {missing[:3]}")
    return preds


def stratum(scores: dict, sel: np.ndarray, n_boot: int) -> dict:
    idx = resample_indices(sel.size, n_boot, 0)
    out = {"n_pairs": int(sel.size),
           "mean_ep": {a: float(scores[a]["rel_err"][sel].mean()) for a in scores},
           "pairs_below_additive": {a: int(np.sum(scores[a]["rel_err"][sel] < scores["additive"]["rel_err"][sel] - 1e-15))
                                    for a in scores}}
    for r in ("additive", "scaled", "shrunk", "ridge", "gbm"):
        out[f"tabpfn-{r}"] = percentile_interval(scores["tabpfn"]["rel_err"][sel] - scores[r]["rel_err"][sel], idx)
    return out


def retention(blocks: list, scores: dict, n_boot: int) -> dict:
    """Readout 2: how much of round 2's per-pair result survives without the genes' other pairs."""
    if not ROUND2_REPORT.exists():
        return {"available": False}
    r2 = json.loads(ROUND2_REPORT.read_text())["all_pairs"]["per_pair"]
    names = [b.name for b in blocks]
    if not all(n in r2 for n in names):
        return {"available": False, "reason": "round 2 report does not cover these pairs"}
    idx = resample_indices(len(names), n_boot, 0)
    out = {"available": True}
    for arm in ("tabpfn", "gbm"):
        here = scores[arm]["rel_err"]
        there = np.array([r2[n][arm]["rel_err"] for n in names])
        add = scores["additive"]["rel_err"]
        gain_here, gain_there = float((add - here).mean()), float((add - there).mean())
        out[arm] = {"this_round_minus_round2": percentile_interval(here - there, idx),
                    "mean_gain_over_additive_round2": gain_there,
                    "mean_gain_over_additive_here": gain_here,
                    "share_of_round2_gain_retained": gain_here / gain_there if gain_there else None}
    return out


def main() -> None:
    args = parse_args()
    t_start = time.time()
    data_dir = Path(args.data_dir)
    cache = load_norman(data_dir)
    data = ResidualData(cache, top_k=args.top_k)

    split_of, names = {}, []
    for split in ("train", "val", "test"):
        for n in cache.pairs(split):
            names.append(n)
            split_of[n] = split
    if args.smoke:          # development never touches a validation or test pair
        names = [n for n in names if split_of[n] == "train"][:30]
    blocks = [data.pair_rows(n) for n in names]
    genes_of = [set(pair_genes(n)) for n in names]
    all_genes = sorted(set().union(*genes_of))
    log.info("table: %d pairs over %d genes, %d rows (data read in %.0fs)",
             len(blocks), len(all_genes), sum(len(b.genes) for b in blocks), time.time() - t_start)

    if not args.smoke:
        test = [b for b in blocks if split_of[b.name] == "test"]
        add_ep = np.mean([np.linalg.norm(b.eps) / np.linalg.norm(b.true_delta) for b in test])
        if abs(add_ep - r1.SSLLAB_ADDITIVE_MEAN_EP) > 1e-12:
            raise SystemExit(f"PARITY FAILED: additive mean e_p {add_ep!r} != {r1.SSLLAB_ADDITIVE_MEAN_EP!r}")
        log.info("parity: additive mean e_p on the test pairs %.6f matches ssl-lab", add_ep)

    cfg = {"folds": args.folds, "tune_seed": 0, "jobs": args.jobs, "seeds": r1.SEEDS,
           "device": "cpu" if args.smoke and args.device == "auto" else r1.resolve_device(args.device),
           "tabpfn": True, "tabpfn_estimators": 2 if args.smoke else None,
           "ridge_grid": r1.SMOKE_RIDGE_GRID if args.smoke else r1.RIDGE_GRID,
           "gbm_grid": r1.SMOKE_GBM_GRID if args.smoke else r1.GBM_GRID,
           "n_boot": 200 if args.smoke else args.n_boot}
    r1.MIN_PAIRS_FOR_VERDICT = MIN_PAIRS_FOR_VERDICT     # round 1's decide(), at this round's size

    # Primary: for each pair, a context with no pair sharing either of its genes.
    strict_plan = []
    ctx_sizes = {}
    for i, gi in enumerate(genes_of):
        ctx = [j for j, gj in enumerate(genes_of) if j != i and not (gi & gj)]
        strict_plan.append((ctx, [(i, 0)]))
        ctx_sizes[names[i]] = len(ctx)
    log.info("PRIMARY leave-both-genes-out: %d contexts, %d to %d context pairs",
             len(strict_plan), min(ctx_sizes.values()), max(ctx_sizes.values()))
    strict_scores = r1.score(blocks, held_out_predictions(blocks, strict_plan, cfg))
    primary = r1.summarize(blocks, strict_scores, cfg["n_boot"], verdict=True)
    for n in primary["per_pair"]:
        primary["per_pair"][n]["context_pairs"] = ctx_sizes[n]

    main_scores = {a: strict_scores[a] for a in MAIN_ARMS}
    split_arr = np.array([split_of[b.name] for b in blocks])
    cells = np.array([b.n_cells for b in blocks])
    cebp = np.array([bool(g & CEBP) for g in genes_of])
    strata = {}
    for name, sel in (("test_pairs", np.flatnonzero(split_arr == "test")),
                      ("validation_pairs", np.flatnonzero(split_arr == "val")),
                      ("training_pairs", np.flatnonzero(split_arr == "train")),
                      ("at_least_100_cells", np.flatnonzero(cells >= 100)),
                      ("cebp_pairs", np.flatnonzero(cebp))):
        if sel.size >= 2:
            strata[name] = {"pairs": [blocks[i].name for i in sel], **stratum(main_scores, sel, cfg["n_boot"])}

    report = {
        "protocol": "R1.7 round 3; see the docstring of 04_round3_leave_genes_out.py",
        "settings": {"top_k": args.top_k, "features": list(FEATURES), "tabpfn_seeds": list(r1.SEEDS),
                     "inner_folds": args.folds, "n_boot": cfg["n_boot"], "device": cfg["device"],
                     "smoke": args.smoke, "tabpfn_checkpoint": r1.TABPFN_CHECKPOINT,
                     "tabpfn_settings": "create_default_for_version(V3_5), n_estimators "
                                        + ("2 (smoke)" if args.smoke else "'auto' (the version default)"),
                     "ridge_grid_size": len(cfg["ridge_grid"]), "gbm_grid_size": len(cfg["gbm_grid"]),
                     "min_pairs_for_verdict": MIN_PAIRS_FOR_VERDICT, "versions": r1.versions()},
        "data": {"de_genes_md5": r1.md5(data_dir / "de_genes.json"), "splits_md5": r1.md5(data_dir / "splits.json"),
                 "n_pairs": len(blocks), "n_genes": len(all_genes),
                 "n_rows": int(sum(len(b.genes) for b in blocks)), "split_of_pair": {n: split_of[n] for n in names},
                 "context_pairs": {"min": min(ctx_sizes.values()), "median": float(np.median(list(ctx_sizes.values()))),
                                   "max": max(ctx_sizes.values())}},
        "all_pairs_leave_both_genes_out": primary,
        "strata": strata,
        "retention_vs_round2": retention(blocks, main_scores, cfg["n_boot"]) if not args.smoke else {"available": False},
    }

    if not args.skip_one_gene:
        plan = []
        for g in all_genes:     # one context per gene; slot 0 holds the pair's first gene out, slot 1 its second
            held = [(i, sorted(gi).index(g)) for i, gi in enumerate(genes_of) if g in gi]
            ctx = [j for j, gj in enumerate(genes_of) if g not in gj]
            plan.append((ctx, held))
        log.info("READOUT 3 leave-one-gene-out: %d contexts", len(plan))
        one_scores = r1.score(blocks, held_out_predictions(blocks, plan, cfg))
        report["all_pairs_leave_one_gene_out"] = r1.summarize(blocks, one_scores, cfg["n_boot"], verdict=False)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["runtime_s"] = round(time.time() - t_start, 1)
    report["peak_rss_gb"] = round(rss / (1e9 if sys.platform == "darwin" else 1e6), 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    for key in ("all_pairs_leave_both_genes_out", "all_pairs_leave_one_gene_out"):
        if key not in report:
            continue
        t = report[key]
        log.info("%s (%d scored): mean e_p floor %.3f", key.upper(), t["n_scored"], t["mean_ep_floor"])
        for a, s in t["arms"].items():
            log.info("  %-16s e_p %.4f [%.4f, %.4f]  Δr %.4f  below additive %d/%d", a, s["mean_ep"]["mean"],
                     *s["mean_ep"]["ci"], s["mean_delta_r"], s["pairs_below_additive"], t["n_scored"])
        for fam in ("primary_family", "secondary_family"):
            for k, c in t[fam]["contrasts"].items():
                log.info("  %-20s %+.4f  simultaneous [%+.4f, %+.4f]", k, c["diff"], *c["ci_simultaneous"])
    ret = report["retention_vs_round2"]
    if ret.get("available"):
        for arm in ("tabpfn", "gbm"):
            log.info("  retention %s: gain over additive %.4f here vs %.4f in round 2 (share %.2f)", arm,
                     ret[arm]["mean_gain_over_additive_here"], ret[arm]["mean_gain_over_additive_round2"],
                     ret[arm]["share_of_round2_gain_retained"])
    log.info("VERDICT: %s", json.dumps(report["all_pairs_leave_both_genes_out"]["verdict"]))
    log.info("wrote %s (%.0fs, peak %.2f GB)", out, report["runtime_s"], report["peak_rss_gb"])


if __name__ == "__main__":
    main()
