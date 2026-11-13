"""Report for notebook 01: how good are TabPFN's out-of-the-box predictions, against references?

Context. The notebook ``01_prediction_and_uncertainty.ipynb`` runs TabPFN
once on each of three small tables and prints a handful of numbers: precision and recall
for heart disease, MAE, R^2 and the coverage of a 10 to 90 percent band for house prices,
and a forecast plot for electricity prices. Printed once and alone, those numbers cannot
say whether TabPFN did well. This script re-runs the same three tasks on the same splits
with several seeds, adds references a practitioner would reach for first, and writes every
number to a JSON report that the results pages cite.

Notation. For a test row i with label y_i, a classifier gives p_i = P(y_i = 1) and a
regressor gives a point prediction yhat_i (the predictive mean) and the 10th and 90th
percentiles q10_i and q90_i of its predictive distribution. The per-row losses are the
log loss -[y_i log p_i + (1 - y_i) log(1 - p_i)], the Brier score (p_i - y_i)^2, and the
absolute error |yhat_i - y_i|. A band covers row i when q10_i <= y_i <= q90_i; its width is
q90_i - q10_i.

PROTOCOL (fixed in this docstring before the full run)
-------------------------------------------------------
Data and splits: exactly the notebook's, from ``helper.py``. Heart disease: UCI, four
studies, 920 patients, stratified split 700 context / 220 test, seed 0. Housing: Ames
(OpenML 42165), nine features, 1,300 / 160, seed 0. Electricity: hourly DE-LU day-ahead
price with day-ahead load and renewables forecasts, 2024-03-04 to 2024-03-19, the first
336 hours as context and the last 48 as the horizon.

Arms.
  tabpfn_v3.5   TabPFN-3.5, the notebook's model: default constructor, no tuning;
                random_state 0, 1, 2 (0 is the notebook's default).
  tabpfn_v2     TabPFN-2, the same way; the lab's robustness arm.
  lightgbm      LightGBM with its default settings and random_state 0: the untuned
                gradient-boosting reference, the fair comparison with an untuned
                TabPFN. Missing values are left to LightGBM; text columns become
                pandas categories.
  linear        logistic regression (classification) or ridge regression (housing)
                after median imputation and standardization, text columns one-hot
                encoded; penalty chosen by internal cross-validation on the context
                rows only (LogisticRegressionCV, RidgeCV, 5 folds).
  prior         the training class frequency (classification) or training mean
                (housing): the floor any model must clear.
Electricity arms: TabPFN (both versions, three seeds) on three feature sets, as in the
notebook: "clock" (hours elapsed since the first context hour), "calendar" (hour of day,
day of week) and "market" (calendar plus the two day-ahead forecasts, the notebook's
last section); LightGBM on "market"; and two naive forecasts that need no model, "daily
naive" (the last 24 context hours, repeated) and "weekly naive" (each hour's value 168
hours earlier).

Aggregation. For a seeded arm, per-row losses are averaged over its seeds first; an
arm's score is the mean over test rows. Point metrics computed on whole predictions
(accuracy, precision, recall and ROC AUC; R^2) are averaged over seeds.

Primary readouts, one per task:
  heart        mean log loss (it scores the predicted probabilities themselves),
  housing      MAE in dollars,
  electricity  MAE in EUR/MWh over the 48 hours.
Comparisons, heart and housing: per-row paired differences tabpfn_v3.5 minus each other
arm, joint paired bootstrap over test rows (10,000 resamples, generator seed 0, one
resample per task shared by every contrast), max-t simultaneous 95% intervals
(``tabicllab.bootstrap``). A difference counts only when its simultaneous interval lies
wholly on one side of zero. Electricity: point values only. Its 48 hourly errors are
strongly autocorrelated and span two days, so a bootstrap over hours would overstate the
precision; the forecasting numbers are illustrations, not tests.

Secondary readouts, reported with no verdict: Brier score, accuracy, precision, recall,
ROC AUC; R^2, RMSE; for TabPFN, the 10 to 90 percent band's coverage (nominal 0.80) and
mean width on housing and electricity.

Parity. Before scoring anything, the run checks that TabPFN-3.5 with random_state 0
reproduces the values saved in the notebook at the precision the notebook prints
(precision 0.85, recall 0.84; MAE $15,994, R^2 0.90, band coverage 80%; calendar MAE
8.69 and calendar-plus-market MAE 7.88 EUR/MWh), and stops if it does not.

Run: from this directory, ``python 02_first_predictions_report.py`` in the tabicllab
environment (CPU; about 15 to 25 minutes on a laptop). ``--smoke`` runs one seed and 200
bootstrap resamples and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import helper

HERE = Path(__file__).resolve().parent
REPORT = HERE / "docs" / "results" / "reports" / "01-prediction-and-uncertainty.json"
SEEDS = [0, 1, 2]
VERSIONS = ["v3.5", "v2"]
QUANTILES = [0.1, 0.5, 0.9]
EPS = 1e-15

# What the notebook prints for TabPFN-3.5, random_state 0, and at what precision.
NOTEBOOK_PARITY = {
    "heart_precision": (0.85, 2),
    "heart_recall": (0.84, 2),
    "housing_mae": (15994, 0),
    "housing_r2": (0.90, 2),
    "housing_coverage": (0.80, 2),
    "electricity_calendar_mae": (8.69, 2),
    "electricity_market_mae": (7.88, 2),
}


# ----------------------------------------------------------------------- utilities


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def tabpfn(kind: str, seed: int):
    """A default TabPFN estimator for the version ``helper.init_tabpfn`` selected last."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    cls = TabPFNClassifier if kind == "classifier" else TabPFNRegressor
    return cls(random_state=seed)


def as_categories(train: pd.DataFrame, test: pd.DataFrame):
    """Text columns as pandas categories with one shared category list, for LightGBM."""
    train, test = train.copy(), test.copy()
    for col in train.columns:
        if not pd.api.types.is_numeric_dtype(train[col]):
            cats = sorted(set(train[col].dropna()) | set(test[col].dropna()))
            train[col] = pd.Categorical(train[col], categories=cats)
            test[col] = pd.Categorical(test[col], categories=cats)
    return train, test


def linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute and standardize numbers; one-hot encode text."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    text = [c for c in X.columns if c not in numeric]
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric)]
    if text:
        parts.append(("txt", OneHotEncoder(handle_unknown="ignore"), text))
    return ColumnTransformer(parts)


def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(per_row: dict[str, np.ndarray], reference: str, n_boot: int) -> dict:
    """Paired max-t family: ``reference`` minus every other arm, on per-row losses."""
    from tabicllab.bootstrap import maxt_family, resample_indices

    n = len(per_row[reference])
    idx = resample_indices(n, n_boot=n_boot, seed=0)
    diffs = {
        f"{reference} - {arm}": per_row[reference] - loss
        for arm, loss in per_row.items()
        if arm != reference
    }
    return maxt_family(diffs, idx)


# --------------------------------------------------------------------------- tasks


def heart_task(seeds: list[int], n_boot: int) -> tuple[dict, dict]:
    X_tr, X_te, y_tr, y_te = helper.load_heart_disease_full(train_size=700, test_size=220)
    y = y_te.to_numpy()
    probs: dict[str, list[np.ndarray]] = {}

    for version in VERSIONS:
        helper.init_tabpfn(version)
        for seed in seeds:
            model = tabpfn("classifier", seed).fit(X_tr, y_tr)
            probs.setdefault(f"tabpfn_{version}", []).append(model.predict_proba(X_te)[:, 1])
            if version == "v3.5" and seed == 0:
                hard = model.predict(X_te)
                parity = {
                    "heart_precision": precision_score(y, hard),
                    "heart_recall": recall_score(y, hard),
                }

    from lightgbm import LGBMClassifier

    gbm_tr, gbm_te = as_categories(X_tr, X_te)
    gbm = LGBMClassifier(random_state=0, verbose=-1).fit(gbm_tr, y_tr)
    probs["lightgbm"] = [gbm.predict_proba(gbm_te)[:, 1]]
    linear = make_pipeline(linear_preprocessor(X_tr), LogisticRegressionCV(cv=5, max_iter=5000))
    probs["linear"] = [linear.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]]
    probs["prior"] = [np.full(len(y), y_tr.mean())]

    arms, log_loss = {}, {}
    for arm, runs in probs.items():
        log_loss[arm] = np.mean([log_loss_rows(y, p) for p in runs], axis=0)
        brier = np.mean([(p - y) ** 2 for p in runs], axis=0)
        point = {
            "accuracy": [accuracy_score(y, p >= 0.5) for p in runs],
            "precision": [precision_score(y, p >= 0.5, zero_division=0) for p in runs],
            "recall": [recall_score(y, p >= 0.5) for p in runs],
            "roc_auc": [roc_auc_score(y, p) if np.ptp(p) > 0 else 0.5 for p in runs],
        }
        arms[arm] = {
            "n_seeds": len(runs),
            "log_loss": float(log_loss[arm].mean()),
            "brier": float(brier.mean()),
            **{k: float(np.mean(v)) for k, v in point.items()},
            "roc_auc_by_seed": [float(v) for v in point["roc_auc"]],
            "log_loss_by_seed": [float(log_loss_rows(y, p).mean()) for p in runs],
        }
    report = {
        "n_context": len(X_tr),
        "n_test": len(X_te),
        "test_positive_rate": float(y.mean()),
        "primary": "log_loss",
        "arms": arms,
        "comparisons_log_loss": compare(log_loss, "tabpfn_v3.5", n_boot),
        "per_row_log_loss": {k: v.round(6).tolist() for k, v in log_loss.items()},
    }
    return report, parity


def housing_task(seeds: list[int], n_boot: int) -> tuple[dict, dict]:
    X_tr, X_te, y_tr, y_te = helper.load_housing(train_size=1300, test_size=160)
    y = y_te.to_numpy()
    preds: dict[str, list[np.ndarray]] = {}
    bands: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for version in VERSIONS:
        helper.init_tabpfn(version)
        for seed in seeds:
            model = tabpfn("regressor", seed).fit(X_tr, y_tr)
            out = model.predict(X_te, output_type="main", quantiles=QUANTILES)
            preds.setdefault(f"tabpfn_{version}", []).append(out["mean"])
            bands.setdefault(f"tabpfn_{version}", []).append((out["quantiles"][0], out["quantiles"][2]))
            if version == "v3.5" and seed == 0:
                low, high = out["quantiles"][0], out["quantiles"][2]
                parity = {
                    "housing_mae": float(np.mean(np.abs(out["mean"] - y))),
                    "housing_r2": r2_score(y, out["mean"]),
                    "housing_coverage": float(np.mean((y >= low) & (y <= high))),
                }

    from lightgbm import LGBMRegressor

    gbm_tr, gbm_te = as_categories(X_tr, X_te)
    preds["lightgbm"] = [LGBMRegressor(random_state=0, verbose=-1).fit(gbm_tr, y_tr).predict(gbm_te)]
    linear = make_pipeline(linear_preprocessor(X_tr), RidgeCV(alphas=np.logspace(-3, 4, 15), cv=5))
    preds["linear"] = [linear.fit(X_tr, y_tr).predict(X_te)]
    preds["prior"] = [np.full(len(y), y_tr.mean())]

    arms, abs_err = {}, {}
    for arm, runs in preds.items():
        abs_err[arm] = np.mean([np.abs(p - y) for p in runs], axis=0)
        arms[arm] = {
            "n_seeds": len(runs),
            "mae": float(abs_err[arm].mean()),
            "rmse": float(np.mean([np.sqrt(np.mean((p - y) ** 2)) for p in runs])),
            "r2": float(np.mean([r2_score(y, p) for p in runs])),
            "mae_by_seed": [float(np.mean(np.abs(p - y))) for p in runs],
        }
        if arm in bands:
            arms[arm]["band_coverage"] = float(np.mean([np.mean((y >= lo) & (y <= hi)) for lo, hi in bands[arm]]))
            arms[arm]["band_width_mean"] = float(np.mean([np.mean(hi - lo) for lo, hi in bands[arm]]))
            arms[arm]["band_coverage_by_seed"] = [float(np.mean((y >= lo) & (y <= hi))) for lo, hi in bands[arm]]
    report = {
        "n_context": len(X_tr),
        "n_test": len(X_te),
        "test_price_mean": float(y.mean()),
        "primary": "mae",
        "arms": arms,
        "comparisons_abs_error": compare(abs_err, "tabpfn_v3.5", n_boot),
        "per_row_abs_error": {k: v.round(2).tolist() for k, v in abs_err.items()},
    }
    return report, parity


def electricity_task(seeds: list[int]) -> tuple[dict, dict]:
    history, future = helper.load_electricity(horizon=48)
    y = future["price_eur"].to_numpy()

    def features(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if name == "clock":
            elapsed = (df["hour"] - history["hour"].iloc[0]) / pd.Timedelta(hours=1)
            return elapsed.to_frame("hour")
        out = pd.DataFrame({"hour_of_day": df["hour"].dt.hour, "day_of_week": df["hour"].dt.day_of_week})
        if name == "market":
            out[["load_forecast", "renewables_forecast"]] = df[["load_forecast", "renewables_forecast"]]
        return out

    preds: dict[str, list[np.ndarray]] = {}
    bands: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    parity = {}
    for version in VERSIONS:
        helper.init_tabpfn(version)
        for feature_set in ("clock", "calendar", "market"):
            arm = f"tabpfn_{version}_{feature_set}"
            for seed in seeds:
                model = tabpfn("regressor", seed).fit(features(history, feature_set), history["price_eur"])
                out = model.predict(features(future, feature_set), output_type="main", quantiles=QUANTILES)
                preds.setdefault(arm, []).append(out["mean"])
                bands.setdefault(arm, []).append((out["quantiles"][0], out["quantiles"][2]))
                if version == "v3.5" and seed == 0 and feature_set != "clock":
                    parity[f"electricity_{feature_set}_mae"] = float(np.mean(np.abs(out["mean"] - y)))

    from lightgbm import LGBMRegressor

    gbm = LGBMRegressor(random_state=0, verbose=-1).fit(features(history, "market"), history["price_eur"])
    preds["lightgbm_market"] = [gbm.predict(features(future, "market"))]
    past = history["price_eur"].to_numpy()
    preds["daily_naive"] = [np.tile(past[-24:], 2)]
    full = np.concatenate([past, y])
    preds["weekly_naive"] = [full[len(past) - 168 : len(past) - 168 + 48]]

    arms = {}
    for arm, runs in preds.items():
        arms[arm] = {
            "n_seeds": len(runs),
            "mae": float(np.mean([np.mean(np.abs(p - y)) for p in runs])),
            "rmse": float(np.mean([np.sqrt(np.mean((p - y) ** 2)) for p in runs])),
            "mae_by_seed": [float(np.mean(np.abs(p - y))) for p in runs],
        }
        if arm in bands:
            arms[arm]["band_coverage"] = float(np.mean([np.mean((y >= lo) & (y <= hi)) for lo, hi in bands[arm]]))
            arms[arm]["band_width_mean"] = float(np.mean([np.mean(hi - lo) for lo, hi in bands[arm]]))
    report = {
        "n_context_hours": len(history),
        "horizon_hours": len(future),
        "horizon": [str(future["hour"].iloc[0]), str(future["hour"].iloc[-1])],
        "test_price_mean": float(y.mean()),
        "primary": "mae",
        "arms": arms,
        "intervals": "none: 48 autocorrelated hours over two days (see the protocol)",
    }
    return report, parity


# ---------------------------------------------------------------------------- main


def check_parity(observed: dict) -> dict:
    """Stop unless TabPFN-3.5, seed 0, reproduces the notebook's printed values."""
    rows = {}
    for key, (expected, digits) in NOTEBOOK_PARITY.items():
        got = round(observed[key], digits)
        rows[key] = {"notebook": expected, "observed": observed[key], "match": got == expected}
    bad = {k: v for k, v in rows.items() if not v["match"]}
    if bad:
        sys.exit(f"Parity with the notebook failed, stopping: {bad}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="one seed, 200 resamples, no report")
    args = parser.parse_args()
    seeds, n_boot = ([0], 200) if args.smoke else (SEEDS, 10_000)

    warnings.filterwarnings("ignore")
    helper.configure_plots(plots=False)
    start = time.time()

    heart, p1 = heart_task(seeds, n_boot)
    print(f"heart done, {time.time() - start:.0f} s", flush=True)
    housing, p2 = housing_task(seeds, n_boot)
    print(f"housing done, {time.time() - start:.0f} s", flush=True)
    electricity, p3 = electricity_task(seeds)
    print(f"electricity done, {time.time() - start:.0f} s", flush=True)
    parity = check_parity({**p1, **p2, **p3})

    import lightgbm
    import sklearn
    import tabpfn
    import torch
    from tabpfn.model_loading import resolve_model_path

    checkpoints = {
        f"{v}_{which}": resolve_model_path(None, which, v)[2][0]
        for v in VERSIONS
        for which in ("classifier", "regressor")
    }
    data_files = sorted(
        p for d in (helper.HEART_DIR, helper.HOUSING_DIR, helper.ELECTRICITY_DIR) for p in d.iterdir() if p.is_file()
    )
    report = {
        "protocol": "see the docstring of 01_prediction_and_uncertainty_report.py",
        "settings": {
            "seeds": seeds,
            "n_boot": n_boot,
            "quantiles": QUANTILES,
            "device": "cpu",
            "tabpfn_settings": "default constructor, no tuning; version set by helper.init_tabpfn",
            "checkpoints": checkpoints,
            "versions": {
                "python": platform.python_version(),
                "tabpfn": tabpfn.__version__,
                "torch": torch.__version__,
                "sklearn": sklearn.__version__,
                "lightgbm": lightgbm.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "machine": f"{platform.system()} {platform.machine()}",
        },
        "data_md5": {f"{p.parent.parent.name}/{p.parent.name}/{p.name}": md5(p) for p in data_files},
        "parity_with_notebook": parity,
        "heart": heart,
        "housing": housing,
        "electricity": electricity,
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
