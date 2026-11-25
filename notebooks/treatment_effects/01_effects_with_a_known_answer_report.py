"""Report for notebook 01: effect estimators on a simulated care program, over seeds and two scenarios.

Context. The notebook ``01_effects_with_a_known_answer.ipynb`` simulates one care
program with seed 0 and compares effect estimators built on three base models: linear
models, tuned gradient boosting and TabPFN. One draw cannot say whether the
differences it shows are larger than the variation from one simulated program to the
next, and its simulator suits linear models: the coaches' decisions are exactly
logistic and the outcome without a call nearly linear. This script repeats the
notebook's analysis over ten draws of the program, in the notebook's scenario and in a
harder one, records every number in JSON reports, and checks that seed 0 of the
notebook's scenario reproduces the notebook.

PROTOCOL (fixed in this docstring before the full run)
-------------------------------------------------------
Scenarios (``helper.simulate_care_program``).
  smooth  the notebook's program: logistic propensity in three covariates, outcomes
          smooth in HbA1c and app use, the effect with one kink at HbA1c 7.5.
  rules   coaches decide by thresholds and combinations of covariates, and outcomes
          and effects depend on interactions; models without interaction terms cannot
          represent any of the three.

Draws. Seeds 0 to 9. A seed fixes the simulated members, the fold split and every
model's random state. 2,000 members, 5 cross-fitting folds, selection strength 1 for
the main analysis.

Arms. The notebook's three base models (``helper.base_learners``): linear and logistic
regression without interactions; XGBoost tuned by a 3-fold grid search over depth,
learning rate and number of trees; TabPFN-3.5 with default settings.

Endpoints, per scenario and draw.
  Nuisances: RMSE of the cross-fitted mu0, mu1 and propensity against the truth.
  Average effect: the plain comparison, and per base model regression adjustment, IPW
    and AIPW, each with its error against the draw's true ATE; IPW and AIPW also with
    their standard error and whether the 95% interval covers the truth.
  Individual effects: per base model and meta-learner (S, T, X, DR), PEHE, correlation,
    slope of estimated on true effects, error of the mean, and the targeting score at
    a 30% budget (``tabicllab.treatment_effects.cate_scores``).
  Overlap sweep: selection strengths 0, 1, 2 and 3, the same members' covariates and
    potential outcomes; per strength the overlap summary of the true propensity, and
    per base model the error of the plain comparison, IPW and AIPW.

Aggregates. Means over the ten draws with their range; for IPW and AIPW also the
root mean squared error over draws, the mean reported standard error, and how many of
the ten intervals cover the truth.

Contrasts and the decision rule. The primary question is whether TabPFN gives more
accurate individual effects than each reference. Contrast: PEHE of TabPFN minus PEHE
of the reference, for the same meta-learner and draw; 2 scenarios x 4 meta-learners x
2 references = 16 contrasts. Each gets a paired t interval over the ten draws, at the
Bonferroni level 0.05 / 16, so the sixteen hold simultaneously at 95%. A contrast
counts as a difference only when its interval lies wholly on one side of zero. The
same is reported for the absolute AIPW error, TabPFN against each reference, as a
secondary family of 4 contrasts at 0.05 / 4.

What is not done. DoubleML, the notebook's cross-check of the AIPW implementation, is
not repeated. The sweep is not repeated at strengths 0.5 and 1.5. Results describe
these two simulated programs only.

Parity. Seed 0 of the smooth scenario repeats the notebook exactly. The plain
comparison and every linear-model number are deterministic and must match the
notebook's printed values to their printed decimals; TabPFN and gradient boosting
numbers are recorded against the notebook's, since GPU arithmetic and the grid
search's parallel order can move them in the last decimals.

Usage:
    python 01_effects_with_a_known_answer_report.py --smoke        # tiny, no report
    python 01_effects_with_a_known_answer_report.py --run 0 1 2    # per-draw reports
    python 01_effects_with_a_known_answer_report.py --aggregate    # combine them
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "docs" / "results" / "reports"
RUNS = REPORTS / "runs"
AGGREGATE = REPORTS / "01-effects-with-a-known-answer.json"

SEEDS = list(range(10))
STRENGTHS = [0.0, 1.0, 2.0, 3.0]
METHODS = ["S-learner", "T-learner", "X-learner", "DR-learner"]
REFERENCES = ["linear", "gradient boosting"]

# The notebook's printed values for seed 0, smooth scenario (executed 2026-11-25).
NOTEBOOK = {
    "plain": 1.004,
    "linear_aipw": 0.472,
    "linear_ipw": 0.496,
    "linear_pehe_x": 0.115,
    "tabpfn_aipw": 0.483,
    "tabpfn_pehe_x": 0.062,
    "boosting_aipw": 0.505,
}


def run_one(scenario: str, seed: int, n: int, n_folds: int, strengths: list[float]) -> dict:
    """The notebook's analysis for one scenario and seed, plus the overlap sweep."""
    import helper
    from tabicllab.treatment_effects import (
        aipw_ate, cate_scores, dr_learner, fit_nuisances, fold_ids, ipw_ate,
        naive_difference, overlap_summary, s_learner, t_learner, x_learner,
    )

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def covered(estimate, truth):
        low, high = estimate.interval()
        return bool(low <= truth <= high)

    frame = helper.simulate_care_program(n, selection=1.0, seed=seed, scenario=scenario)
    X = frame[helper.COVARIATES].to_numpy(dtype=float)
    t, y, tau = frame["coached"].to_numpy(), frame["hba1c_reduction"].to_numpy(), frame["true_effect"].to_numpy()
    true_ate = float(tau.mean())
    learners = helper.base_learners(seed=seed)
    folds = fold_ids(t, n_folds=n_folds, seed=seed)
    plain = naive_difference(t, y)
    out = {
        "scenario": scenario, "seed": seed, "n": n, "n_folds": n_folds,
        "true_ate": true_ate, "true_att": float(tau[t == 1].mean()), "true_effect_sd": float(tau.std()),
        "overlap": overlap_summary(t, frame["true_propensity"]),
        "plain": {"estimate": plain.estimate, "se": plain.se, "error": plain.estimate - true_ate},
        "learners": {},
    }
    for name, (regressor, classifier) in learners.items():
        start = time.perf_counter()
        nu = fit_nuisances(X, t, y, regressor, classifier, folds)
        nuisance_seconds = time.perf_counter() - start
        ipw, aipw = ipw_ate(t, y, nu.e), aipw_ate(t, y, nu)
        start = time.perf_counter()
        estimates = {
            "S-learner": s_learner(X, t, y, regressor, folds),
            "T-learner": t_learner(nu),
            "X-learner": x_learner(X, t, y, nu, regressor, folds),
            "DR-learner": dr_learner(X, t, y, nu, regressor, folds),
        }
        out["learners"][name] = {
            "nuisance_rmse": {
                "mu0": rmse(nu.mu0, frame["expected_y0"]),
                "mu1": rmse(nu.mu1, frame["expected_y1"]),
                "propensity": rmse(nu.e_raw, frame["true_propensity"]),
            },
            "average": {
                "regression adjustment": {"estimate": float(t_learner(nu).mean()), "error": float(t_learner(nu).mean()) - true_ate},
                "IPW": {"estimate": ipw.estimate, "se": ipw.se, "error": ipw.estimate - true_ate, "covered": covered(ipw, true_ate)},
                "AIPW": {"estimate": aipw.estimate, "se": aipw.se, "error": aipw.estimate - true_ate, "covered": covered(aipw, true_ate)},
            },
            "individual": {method: cate_scores(value, tau, budget=0.3) for method, value in estimates.items()},
            "seconds": {"nuisances": nuisance_seconds, "meta_learners": time.perf_counter() - start},
        }
        print(f"  {scenario} seed {seed} {name}: AIPW error {aipw.estimate - true_ate:+.3f}, "
              f"X-learner PEHE {out['learners'][name]['individual']['X-learner']['pehe']:.3f}", flush=True)

    sweep = []
    for strength in strengths:
        drawn = helper.simulate_care_program(n, selection=strength, seed=seed, scenario=scenario)
        Xs = drawn[helper.COVARIATES].to_numpy(dtype=float)
        ts, ys = drawn["coached"].to_numpy(), drawn["hba1c_reduction"].to_numpy()
        truth = float(drawn["true_effect"].mean())
        fs = fold_ids(ts, n_folds=n_folds, seed=seed)
        row = {"selection": strength, "overlap": overlap_summary(ts, drawn["true_propensity"]),
               "naive_error": naive_difference(ts, ys).estimate - truth, "learners": {}}
        for name, (regressor, classifier) in learners.items():
            nu = fit_nuisances(Xs, ts, ys, regressor, classifier, fs)
            ipw, aipw = ipw_ate(ts, ys, nu.e), aipw_ate(ts, ys, nu)
            row["learners"][name] = {
                "IPW": {"error": ipw.estimate - truth, "se": ipw.se, "covered": covered(ipw, truth)},
                "AIPW": {"error": aipw.estimate - truth, "se": aipw.se, "covered": covered(aipw, truth)},
            }
        sweep.append(row)
    out["sweep"] = sweep
    return out


def environment() -> dict:
    import sklearn
    import tabpfn
    import torch
    import xgboost

    return {
        "python": platform.python_version(), "torch": torch.__version__, "tabpfn": tabpfn.__version__,
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def paired_interval(differences, level: float) -> dict:
    """Paired t interval for the mean difference, at confidence ``level``."""
    from scipy import stats

    d = np.asarray(differences, dtype=float)
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    low, high = d.mean() - half, d.mean() + half
    verdict = "lower" if high < 0 else ("higher" if low > 0 else "not resolved")
    return {"mean": float(d.mean()), "low": float(low), "high": float(high), "verdict": verdict, "n": int(len(d))}


def aggregate(runs: list[dict]) -> dict:
    by_scenario: dict[str, list[dict]] = {}
    for run in runs:
        by_scenario.setdefault(run["scenario"], []).append(run)
    result = {"scenarios": {}, "contrasts": {"pehe": [], "aipw_abs_error": []}}
    n_pehe = len(by_scenario) * len(METHODS) * len(REFERENCES)
    n_aipw = len(by_scenario) * len(REFERENCES)
    for scenario, group in sorted(by_scenario.items()):
        group = sorted(group, key=lambda r: r["seed"])
        learners = list(group[0]["learners"])
        block = {
            "seeds": [r["seed"] for r in group],
            "true_ate": summary([r["true_ate"] for r in group]),
            "true_att": summary([r["true_att"] for r in group]),
            "true_effect_sd": summary([r["true_effect_sd"] for r in group]),
            "plain_error": summary([r["plain"]["error"] for r in group]),
            "nuisance_rmse": {}, "average": {}, "individual": {}, "sweep": {},
        }
        for name in learners:
            block["nuisance_rmse"][name] = {
                part: summary([r["learners"][name]["nuisance_rmse"][part] for r in group]) for part in ("mu0", "mu1", "propensity")
            }
            block["average"][name] = {}
            for method in ("regression adjustment", "IPW", "AIPW"):
                errors = np.array([r["learners"][name]["average"][method]["error"] for r in group])
                entry = {"error": summary(errors), "rmse": float(np.sqrt(np.mean(errors**2)))}
                if method != "regression adjustment":
                    entry["mean_se"] = float(np.mean([r["learners"][name]["average"][method]["se"] for r in group]))
                    entry["covered"] = int(sum(r["learners"][name]["average"][method]["covered"] for r in group))
                block["average"][name][method] = entry
            block["individual"][name] = {
                method: {metric: summary([r["learners"][name]["individual"][method][metric] for r in group])
                         for metric in ("pehe", "correlation", "slope", "ate_error", "targeting")}
                for method in METHODS
            }
        for i, strength in enumerate(group[0]["sweep"]):
            rows = [r["sweep"][i] for r in group]
            entry = {
                "share_extreme": summary([row["overlap"]["share_extreme"] for row in rows]),
                "effective_n_control": summary([row["overlap"]["effective_n_control"] for row in rows]),
                "effective_n_treated": summary([row["overlap"]["effective_n_treated"] for row in rows]),
                "naive_error": summary([row["naive_error"] for row in rows]),
            }
            for name in learners:
                for method in ("IPW", "AIPW"):
                    errors = np.array([row["learners"][name][method]["error"] for row in rows])
                    entry[f"{name} {method}"] = {"error": summary(errors), "rmse": float(np.sqrt(np.mean(errors**2))),
                                                 "covered": int(sum(row["learners"][name][method]["covered"] for row in rows))}
            block["sweep"][str(strength["selection"])] = entry
        result["scenarios"][scenario] = block

        for method in METHODS:
            for reference in REFERENCES:
                diff = [r["learners"]["TabPFN"]["individual"][method]["pehe"] - r["learners"][reference]["individual"][method]["pehe"]
                        for r in group]
                result["contrasts"]["pehe"].append({"scenario": scenario, "method": method, "reference": reference,
                                                    **paired_interval(diff, 1 - 0.05 / n_pehe)})
        for reference in REFERENCES:
            diff = [abs(r["learners"]["TabPFN"]["average"]["AIPW"]["error"]) - abs(r["learners"][reference]["average"]["AIPW"]["error"])
                    for r in group]
            result["contrasts"]["aipw_abs_error"].append({"scenario": scenario, "reference": reference,
                                                          **paired_interval(diff, 1 - 0.05 / n_aipw)})
    result["contrasts"]["levels"] = {"pehe": f"Bonferroni over {n_pehe}", "aipw_abs_error": f"Bonferroni over {n_aipw}"}
    return result


def parity(runs: list[dict]) -> dict:
    base = [r for r in runs if r["scenario"] == "smooth" and r["seed"] == 0]
    if not base:
        return {"checked": False}
    r = base[0]
    observed = {
        "plain": r["plain"]["estimate"],
        "linear_aipw": r["learners"]["linear"]["average"]["AIPW"]["estimate"],
        "linear_ipw": r["learners"]["linear"]["average"]["IPW"]["estimate"],
        "linear_pehe_x": r["learners"]["linear"]["individual"]["X-learner"]["pehe"],
        "tabpfn_aipw": r["learners"]["TabPFN"]["average"]["AIPW"]["estimate"],
        "tabpfn_pehe_x": r["learners"]["TabPFN"]["individual"]["X-learner"]["pehe"],
        "boosting_aipw": r["learners"]["gradient boosting"]["average"]["AIPW"]["estimate"],
    }
    deterministic = ("plain", "linear_aipw", "linear_ipw", "linear_pehe_x")
    return {
        "checked": True,
        "notebook": NOTEBOOK,
        "observed": observed,
        "deterministic_match": all(round(observed[k], 3) == NOTEBOOK[k] for k in deterministic),
        "differences": {k: observed[k] - NOTEBOOK[k] for k in NOTEBOOK},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="one tiny draw per scenario with a light TabPFN; writes nothing")
    parser.add_argument("--run", nargs="*", type=int, help="seeds to run (default 0 to 9); one report per scenario and seed")
    parser.add_argument("--scenarios", nargs="*", default=["smooth", "rules"])
    parser.add_argument("--aggregate", action="store_true", help="combine the per-draw reports into the results report")
    parser.add_argument("--overwrite", action="store_true", help="rerun draws whose report already exists")
    args = parser.parse_args()

    if args.smoke:
        os.environ["TE_SMOKE"] = "1"
    sys.path.insert(0, str(HERE))
    import helper

    if args.smoke or args.run is not None:
        helper.configure_plots(plots=False)
        version = helper.init_tabpfn()
        env = {**environment(), "tabpfn_weights": version}
    if args.smoke:
        runs = [run_one(s, 99, helper.N_MEMBERS, helper.N_FOLDS, [0.0, 2.0]) for s in args.scenarios]
        print(json.dumps(aggregate(runs + [dict(r, seed=98) for r in runs])["scenarios"][args.scenarios[0]]["average"]["TabPFN"], indent=1)[:400])
        print("smoke done")
        return
    if args.run is not None:
        RUNS.mkdir(parents=True, exist_ok=True)
        for seed in (args.run or SEEDS):
            for scenario in args.scenarios:
                path = RUNS / f"01-{scenario}-seed{seed}.json"
                if path.exists() and not args.overwrite:
                    print(f"skip {path.name} (exists)")
                    continue
                start = time.perf_counter()
                result = run_one(scenario, seed, helper.N_MEMBERS, helper.N_FOLDS, STRENGTHS)
                result["seconds"] = time.perf_counter() - start
                result["environment"] = env
                path.write_text(json.dumps(result, indent=1))
                print(f"wrote {path.name} in {result['seconds']:.0f} s", flush=True)
    if args.aggregate:
        runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("01-*-seed*.json"))]
        report = {"protocol": "see the docstring of 01_effects_with_a_known_answer_report.py",
                  "n_runs": len(runs), "environment": runs[0]["environment"] if runs else {},
                  "parity": parity(runs), **aggregate(runs)}
        AGGREGATE.write_text(json.dumps(report, indent=1))
        print(f"wrote {AGGREGATE} from {len(runs)} runs; parity: {report['parity'].get('deterministic_match')}")


if __name__ == "__main__":
    main()
