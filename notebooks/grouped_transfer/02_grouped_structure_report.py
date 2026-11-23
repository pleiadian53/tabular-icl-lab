"""Report for notebook 02: transfer to each of the twelve geometries, with intervals.

Context. Notebook 01 held out three geometries, so its geometry-level conclusions
rest on three groups. Notebook 02 holds out each of the twelve geometries in turn
(leave one group out), compares TabPFN with references that do and do not use the
grouping, contrasts the grouped evaluation with a random split, and asks whether the
prediction bands are calibrated when the whole geometry is new. It runs one seed.
This script repeats every arm with three seeds where the model has one, adds TabPFN-2
and a second description of the shape, attaches bootstrap intervals over geometries
to the differences between arms, and writes a JSON report for the results page.

PROTOCOL (fixed in this docstring before the full run)
-------------------------------------------------------
Data: UCI Energy Efficiency from ``helper.py``, 768 designs, 12 geometries of 64.

Evaluations.
  logo     leave one geometry out: for each geometry g, the context is the other 11
           geometries (704 designs) and the 64 designs of g are predicted. Every
           design is predicted exactly once.
  random   192 designs drawn at random (split seed = model seed), the other 576 as
           context; geometries appear on both sides. TabPFN-3.5 and XGBoost only.

Arms (logo).
  tabpfn_v3.5        TabPFN-3.5, eight design columns, random_state 0, 1, 2.
  tabpfn_v3.5_short  TabPFN-3.5 with the shape described by relative compactness and
                     overall height only (plus orientation and the two glazing
                     columns): is a compact description of the geometry easier to
                     transfer? random_state 0, 1, 2.
  tabpfn_v2          TabPFN-2, eight columns, random_state 0, 1, 2.
  xgboost            XGBoost defaults, one thread, eight columns.
  mixed              linear mixed-effects model, random intercept per geometry,
                     ``helper.MIXED_FORMULA``; prediction = fixed part.
  two_stage          Gaussian process over the five shape columns for the geometry
                     level, plus the context's average within-geometry shift for the
                     design's height and glazing setup.

Endpoints. Per geometry: MAE and mean signed error (seed-averaged predictions'
errors averaged over seeds for TabPFN, i.e. MAE per seed then the mean). Overall:
MAE and R^2 over all 768 designs. Bands (10% to 90%, nominal 80%): coverage and mean
width per geometry and overall, for TabPFN-3.5 (seed 0), the mixed model and the
two-stage model, and for TabPFN-3.5's band after group calibration
(``helper.group_calibrated_band``).

Inference. The unit is the geometry (n = 12). Primary family, with max-t
simultaneous 95% intervals from one shared bootstrap over geometries (10,000
resamples): per-geometry MAE differences tabpfn_v3.5 minus each of xgboost, mixed,
two_stage, tabpfn_v2 and tabpfn_v3.5_short. A difference is called only when its
simultaneous interval excludes 0. Twelve units make the bootstrap coarse; the
intervals are reported as they come and read with that in mind.

Variance structure. Shares of heating-load variance between geometries and with a
glazing shift, the intraclass correlation and the effective sample size, from
``helper.variance_shares`` on all 768 designs.

Parity. Seed 0 of tabpfn_v3.5 must reproduce the notebook's overall leave-one-out
MAE to its six printed decimals.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import helper

from tabicllab.bootstrap import maxt_family, resample_indices

HERE = Path(__file__).resolve().parent

REPORT = HERE / "docs" / "results" / "reports" / "02-grouped-structure.json"
SEEDS = [0, 1, 2]
SHORT = ["relative_compactness", "overall_height", "orientation", "glazing_area", "glazing_distribution"]
# Seed 0 TabPFN-3.5 leave-one-out MAE as printed by the notebook, six decimals.
NOTEBOOK_LOGO_MAE = 2.780874


def tabpfn_logo(frame, features, seeds, start):
    runs = []
    for seed in seeds:
        logo = helper.leave_one_group_out(frame, helper.fit_predict_tabpfn, features=features, seed=seed)
        runs.append(logo)
        print(f"  seed {seed}: MAE {helper.scores(logo['actual'], logo['mean'])['mae']:.3f}, {time.time() - start:.0f} s",
              flush=True)
    return runs


def describe(runs: list[pd.DataFrame]) -> dict:
    """Overall and per-geometry errors, averaged over seeds, with the seed range."""
    per_seed = [helper.scores(r["actual"], r["mean"]) for r in runs]
    by_geo = [helper.errors_by_geometry(r["geometry"], r["actual"], {"m": r["mean"]})["m"] for r in runs]
    mae = pd.concat([b["MAE"] for b in by_geo], axis=1).mean(axis=1)
    signed = pd.concat([b["signed"] for b in by_geo], axis=1).mean(axis=1)
    seconds = [r.groupby("geometry")["seconds"].first().sum() for r in runs]
    return {
        "mae": float(np.mean([s["mae"] for s in per_seed])),
        "mae_range": [float(min(s["mae"] for s in per_seed)), float(max(s["mae"] for s in per_seed))],
        "r2": float(np.mean([s["r2"] for s in per_seed])),
        "by_geometry": {int(g): {"mae": float(mae[g]), "signed": float(signed[g])} for g in mae.index},
        "seconds_all_folds": float(np.mean(seconds)),
    }


def band(logo, low, high) -> dict:
    table = helper.coverage_by_geometry(logo["geometry"], logo["actual"], low, high)
    return {str(g): {"coverage": float(r["coverage"]), "width": float(r["width"])} for g, r in table.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="seed 0, TabPFN-3.5 eight columns only, no report")
    args = parser.parse_args()
    seeds, n_boot = ([0], 2000) if args.smoke else (SEEDS, 10_000)

    warnings.filterwarnings("ignore")
    helper.configure_plots(plots=False)
    start = time.time()
    frame = helper.load_energy_efficiency()

    arms: dict[str, list[pd.DataFrame]] = {}
    helper.init_tabpfn("v3.5")
    print("tabpfn_v3.5", flush=True)
    arms["tabpfn_v3.5"] = tabpfn_logo(frame, helper.DESIGN_COLUMNS, seeds, start)
    if not args.smoke:
        print("tabpfn_v3.5_short", flush=True)
        arms["tabpfn_v3.5_short"] = tabpfn_logo(frame, SHORT, seeds, start)
        helper.init_tabpfn("v2")
        print("tabpfn_v2", flush=True)
        arms["tabpfn_v2"] = tabpfn_logo(frame, helper.DESIGN_COLUMNS, seeds, start)
        helper.init_tabpfn("v3.5")
    arms["xgboost"] = [helper.leave_one_group_out(frame, helper.fit_predict_xgboost, features=helper.DESIGN_COLUMNS)]
    mixed = helper.leave_one_group_out(frame, helper.fit_predict_mixed)
    two_stage = helper.leave_one_group_out(frame, helper.fit_predict_two_stage)
    arms["mixed"], arms["two_stage"] = [mixed], [two_stage]

    results = {name: describe(runs) for name, runs in arms.items()}

    # Random split: the optimistic contrast.
    random = {"tabpfn_v3.5": [], "xgboost": []}
    for seed in seeds:
        history, current, answers = helper.random_split(frame, seed=seed)
        random["tabpfn_v3.5"].append(helper.scores(answers, helper.fit_predict_tabpfn(history, current, helper.DESIGN_COLUMNS, seed=seed)["mean"]))
        random["xgboost"].append(helper.scores(answers, helper.fit_predict_xgboost(history, current, helper.DESIGN_COLUMNS, seed=seed)["mean"]))
    random_summary = {k: {"mae": float(np.mean([r["mae"] for r in v])), "r2": float(np.mean([r["r2"] for r in v]))}
                      for k, v in random.items()}

    # Bands, seed 0.
    tab0 = arms["tabpfn_v3.5"][0]
    cal_low, cal_high = helper.group_calibrated_band(tab0)
    bands = {
        "tabpfn_v3.5": band(tab0, tab0["low"], tab0["high"]),
        "tabpfn_v3.5_group_calibrated": band(tab0, cal_low, cal_high),
        "mixed": band(mixed, mixed["low"], mixed["high"]),
        "two_stage": band(two_stage, two_stage["low"], two_stage["high"]),
    }

    # Bootstrap over geometries.
    per_geo = {name: pd.concat([helper.per_geometry_mae(r) for r in runs], axis=1).mean(axis=1) for name, runs in arms.items()}
    # Each contrast is first arm minus second arm, per geometry; negative favours the first.
    pairs = [("tabpfn_v3.5", other) for other in ("xgboost", "mixed", "two_stage", "tabpfn_v2")]
    pairs.append(("tabpfn_v3.5_short", "tabpfn_v3.5"))
    contrasts = {f"{a} - {b}": (per_geo[a] - per_geo[b]).to_numpy() for a, b in pairs if a in per_geo and b in per_geo}
    family = maxt_family(contrasts, resample_indices(12, n_boot, seed=0))

    mixed_fit = helper.fit_predict_mixed(frame, frame)
    logo_mae = helper.scores(tab0["actual"], tab0["mean"])["mae"]
    report = {
        "protocol": "see the docstring of 02_grouped_structure_report.py",
        "settings": {"seeds": seeds, "n_boot": n_boot, "short_features": SHORT, "mixed_formula": helper.MIXED_FORMULA,
                     "band": "10% to 90%", "device": "cpu", "machine": f"{platform.system()} {platform.machine()}"},
        "data_sha256": {"energy/uci_energy_efficiency/ENB2012_data.xlsx": helper._WORKBOOK_SHA256},
        "parity_with_notebook": {"notebook_mae": NOTEBOOK_LOGO_MAE, "report_mae": logo_mae,
                                 "match": None if NOTEBOOK_LOGO_MAE is None else bool(round(logo_mae, 6) == NOTEBOOK_LOGO_MAE)},
        "geometry_names": {int(g): n for g, n in helper.geometry_names(frame).items()},
        "variance": helper.variance_shares(frame),
        "mixed_full_table": {"tau": mixed_fit["tau"], "sigma": mixed_fit["sigma"]},
        "logo": results,
        "random_split": random_summary,
        "bands": bands,
        "bootstrap_over_geometries": family,
        "runtime_s": round(time.time() - start, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
    }
    if args.smoke:
        print(json.dumps({k: report[k] for k in ("variance", "mixed_full_table", "random_split", "parity_with_notebook")}, indent=1))
        for name, r in results.items():
            print(name, round(r["mae"], 3), round(r["r2"], 3), {g: round(v["mae"], 2) for g, v in r["by_geometry"].items()})
        for name, b in bands.items():
            print("band", name, {g: (round(v["coverage"], 2), round(v["width"], 1)) for g, v in b.items()})
        for k, v in family["contrasts"].items():
            print(k, round(v["diff"], 3), [round(x, 3) for x in v["ci_simultaneous"]])
        print("runtime", report["runtime_s"])
        return
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"wrote {REPORT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
