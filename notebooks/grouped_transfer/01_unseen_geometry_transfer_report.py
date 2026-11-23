"""Report for notebook 01: TabPFN on three building geometries it has never seen.

Context. The notebook ``01_unseen_geometry_transfer.ipynb`` fits TabPFN once, with
seed 0, on the 576 designs of nine geometries and predicts the heating load of the
192 designs of the three held-out geometries, then compares it with XGBoost and
repeats the run without two input columns. One seed and one model version cannot say
whether the differences it shows are larger than the run-to-run variation of
TabPFN's ensemble. This script repeats the same split with three seeds and both
TabPFN versions, records every number in a JSON report that the results page cites,
and checks that seed 0 reproduces the notebook.

PROTOCOL (fixed in this docstring before the full run)
-------------------------------------------------------
Data and split: exactly the notebook's, from ``helper.py``. UCI Energy Efficiency,
768 designs; geometries 2, 6 and 10 (first-seen order in the workbook) held out, 192
designs; the other nine geometries, 576 designs, are the context.

Feature sets.
  all_eight    the eight design columns, in the data-dictionary order.
  trimmed_six  without ``surface_area`` (exactly wall area plus twice roof area)
               and ``orientation`` (no consistent effect in the context).
Cooling load and the design identifier are never inputs.

Arms.
  tabpfn_v3.5  TabPFN-3.5, default constructor, random_state 0, 1, 2.
  tabpfn_v2    TabPFN-2, the same way.
  xgboost      XGBoost, default settings, one thread, random_state 0. Its fit is
               deterministic, so one seed suffices.

Endpoints. Over the 192 held-out designs: MAE, R^2 and mean signed error
(prediction minus truth); per held-out geometry: MAE and mean signed error; for
TabPFN, coverage and mean width of the 10% to 90% predictive band, overall and per
geometry; fit-plus-predict seconds. Seed-averaged values are reported with the range
over seeds.

What is not done. With three held-out geometries, a bootstrap over geometries has
three units and says nothing useful, so no interval is attached to a difference
between arms here; notebook 02 repeats the comparison over all twelve geometries,
where it can be. The seed range describes TabPFN's own variability only.

Parity. Seed 0 of TabPFN-3.5 on ``all_eight`` must reproduce the MAE the notebook
prints, to its six decimals; the script records the check.
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

HERE = Path(__file__).resolve().parent
REPORT = HERE / "docs" / "results" / "reports" / "01-unseen-geometry-transfer.json"
SEEDS = [0, 1, 2]
VERSIONS = ["v3.5", "v2"]
FEATURE_SETS = {
    "all_eight": helper.DESIGN_COLUMNS,
    "trimmed_six": [c for c in helper.DESIGN_COLUMNS if c not in ("surface_area", "orientation")],
}
# Seed 0, TabPFN-3.5, all eight columns: the MAE the notebook prints, to six decimals.
NOTEBOOK_MAE = 0.873092


def summarize(geometry, answers, out: dict) -> dict:
    """Scores, per-geometry errors and, when a band is present, its coverage."""
    record = helper.scores(answers, out["mean"])
    table = helper.errors_by_geometry(geometry, answers, {"m": out["mean"]})["m"]
    record["by_geometry"] = {int(g): {"mae": float(r["MAE"]), "signed": float(r["signed"])} for g, r in table.iterrows()}
    if "low" in out:
        cover = helper.coverage_by_geometry(geometry, answers, out["low"], out["high"])
        record["band"] = {str(g): {"coverage": float(r["coverage"]), "width": float(r["width"])} for g, r in cover.iterrows()}
    record["seconds"] = float(out["seconds"])
    return record


def seed_summary(runs: list[dict]) -> dict:
    """Mean and range over seeds of the headline numbers and the per-geometry MAE."""
    def stat(values):
        values = np.asarray(values, float)
        return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}

    out = {k: stat([r[k] for r in runs]) for k in ("mae", "r2", "signed", "seconds")}
    out["by_geometry_mae"] = {g: stat([r["by_geometry"][g]["mae"] for r in runs]) for g in runs[0]["by_geometry"]}
    out["by_geometry_signed"] = {g: stat([r["by_geometry"][g]["signed"] for r in runs]) for g in runs[0]["by_geometry"]}
    if "band" in runs[0]:
        out["band_coverage"] = stat([r["band"]["all"]["coverage"] for r in runs])
        out["band_width"] = stat([r["band"]["all"]["width"] for r in runs])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="one seed, TabPFN-3.5 only, no report")
    args = parser.parse_args()
    seeds, versions = ([0], ["v3.5"]) if args.smoke else (SEEDS, VERSIONS)

    warnings.filterwarnings("ignore")
    helper.configure_plots(plots=False)
    start = time.time()

    frame = helper.load_energy_efficiency()
    history, current, answers = helper.grouped_split(frame)
    geometry = current["geometry"].to_numpy()

    results: dict = {"xgboost": {}, "tabpfn": {}}
    for name, columns in FEATURE_SETS.items():
        results["xgboost"][name] = summarize(geometry, answers, helper.fit_predict_xgboost(history, current, columns))
    for version in versions:
        helper.init_tabpfn(version)
        results["tabpfn"][version] = {}
        for name, columns in FEATURE_SETS.items():
            runs = []
            for seed in seeds:
                out = helper.fit_predict_tabpfn(history, current, columns, seed=seed)
                runs.append(summarize(geometry, answers, out))
                print(f"{version} {name} seed {seed}: MAE {runs[-1]['mae']:.3f}, {time.time() - start:.0f} s", flush=True)
            results["tabpfn"][version][name] = {"per_seed": runs, "over_seeds": seed_summary(runs)}

    reproduced = results["tabpfn"]["v3.5"]["all_eight"]["per_seed"][0]["mae"]
    parity = {
        "notebook_mae": NOTEBOOK_MAE,
        "report_mae": reproduced,
        "match": bool(round(reproduced, 6) == NOTEBOOK_MAE),
    }

    import sklearn
    import tabpfn
    import torch
    import xgboost
    from tabpfn.model_loading import resolve_model_path

    report = {
        "protocol": "see the docstring of 01_unseen_geometry_transfer_report.py",
        "settings": {
            "seeds": seeds,
            "held_out_geometries": list(helper.HELD_OUT_GEOMETRIES),
            "feature_sets": FEATURE_SETS,
            "band": "10% to 90% predictive quantiles",
            "device": "cpu",
            "tabpfn_settings": "default constructor, no tuning; version set by helper.init_tabpfn",
            "checkpoints": {f"{v}_regressor": resolve_model_path(None, "regressor", v)[2][0] for v in versions},
            "versions": {
                "python": platform.python_version(),
                "tabpfn": tabpfn.__version__,
                "torch": torch.__version__,
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "machine": f"{platform.system()} {platform.machine()}",
        },
        "data_sha256": {"energy/uci_energy_efficiency/ENB2012_data.xlsx": helper._WORKBOOK_SHA256},
        "parity_with_notebook": parity,
        "geometry_names": {int(g): n for g, n in helper.geometry_names(frame).items()},
        "results": results,
        "runtime_s": round(time.time() - start, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
    }
    if args.smoke:
        print(json.dumps({k: report[k] for k in ("parity_with_notebook", "runtime_s", "peak_rss_gb")}, indent=1))
        return
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"wrote {REPORT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
