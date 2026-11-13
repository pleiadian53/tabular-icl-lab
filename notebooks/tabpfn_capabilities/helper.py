"""Shared data, model setup and plots for the TabPFN capability notebooks.

The notebooks in this folder measure what TabPFN does out of the box, before any method
of this lab is added to it: how good its predictions are on small tables, how well its
uncertainty is calibrated, and what its explanations say. Those measurements are the
reference the lab's own methods have to beat. Everything runs on local TabPFN, used as a
library, so every number is tied to a known checkpoint and no data leaves the machine.

The model. ``init_tabpfn`` selects the weights: TabPFN-3.5, the lab's main model, when a
license token is available (cached by the one-time acceptance described in the README,
or ``TABPFN_TOKEN``), and otherwise TabPFN-2, whose weights download without an account.
It also keeps TabPFN off Apple's MPS backend, so a laptop runs on the CPU while a pod
with CUDA still uses its GPUs.

The data come from public sources. Two clinical tables: the UCI heart-disease collection
of four hospital studies (920 patients, many measurements missing) and the complete
Statlog heart table (270 patients). The Ames housing table from OpenML (id 42165), with
nine features chosen here. Sixteen days of hourly German day-ahead electricity prices
from SMARD with the day-ahead load and renewables forecasts published by the
transmission operators, both served by the Energy-Charts API of Fraunhofer ISE.
Downloads are cached in the shared data lake (``$SSLLAB_DATA_ROOT``, by default
``~/work/data``), filed by domain: ``clinical/uci_heart_disease``,
``clinical/statlog_heart``, ``real_estate/ames_housing`` and
``energy/de_day_ahead_power``. Only the first run needs the network.
"""

from __future__ import annotations

import json
import os
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 0
DATA_ROOT = Path(os.environ.get("SSLLAB_DATA_ROOT", Path.home() / "work" / "data"))
HEART_DIR = DATA_ROOT / "clinical" / "uci_heart_disease"
STATLOG_DIR = DATA_ROOT / "clinical" / "statlog_heart"
HOUSING_DIR = DATA_ROOT / "real_estate" / "ames_housing"
ELECTRICITY_DIR = DATA_ROOT / "energy" / "de_day_ahead_power"

# Set by configure_plots. When FIGURE_DIR is a path, figures are written there
# instead of shown, which is how the notebook is checked headlessly.
_PLOTS = True
_FIGURE_DIR: Path | None = None
_figure_count = 0


# --------------------------------------------------------------------------- setup


def configure_plots(plots: bool = True, figure_dir: str | Path | None = None) -> None:
    """Quiet the routine warnings and choose how figures are handled.

    With ``plots=False`` every plotting helper returns without drawing. With a
    ``figure_dir`` the figures are saved there as numbered PNG files, otherwise they
    are shown inline.
    """
    global _PLOTS, _FIGURE_DIR
    _PLOTS = plots
    _FIGURE_DIR = Path(figure_dir) if figure_dir else None
    if _FIGURE_DIR:
        _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 120)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def init_tabpfn(version: str | None = None) -> str:
    """Select the local TabPFN weights and keep the model off MPS. Returns the version.

    The notebook's ``TabPFNClassifier()`` and ``TabPFNRegressor()`` take their weights
    from TabPFN's global default, which this sets. With ``version=None`` the choice is
    TabPFN-3.5 when a license token is found and TabPFN-2 otherwise; pass ``"v3.5"`` or
    ``"v2"`` to force one.

    Local TabPFN refuses to run on the CPU with more than 1000 context rows unless
    told otherwise, a guard against slow runs. The housing example uses 1300 rows,
    which takes well under a minute on a laptop, so the guard is lifted here.
    """
    from tabpfn.browser_auth import get_cached_token
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings

    if version is None:
        has_token = bool(os.environ.get("TABPFN_TOKEN") or get_cached_token())
        version = "v3.5" if has_token else "v2"
    settings.tabpfn.model_version = ModelVersion(version)
    settings.tabpfn.allow_cpu_large_dataset = True
    # Read by TabPFN each time it picks a device, so this applies to models built later.
    os.environ.setdefault("TABPFN_EXCLUDE_DEVICES", "mps")
    print(f"TabPFN {version} weights, running locally")
    return version


# ------------------------------------------------------------------------ datasets


def _download(url: str, directory: Path, name: str) -> Path:
    """Fetch ``url`` once into ``directory`` and return the cached path."""
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "tabular-icl-lab"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


_HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
_HEART_SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "long_beach": "processed.va.data",
}
_HEART_COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_blood_sugar",
    "resting_ecg", "max_heart_rate", "exercise_angina", "st_depression", "st_slope",
    "n_major_vessels", "thalassemia", "num",
]


def load_heart_disease_full(train_size: int = 700, test_size: int = 220):
    """UCI heart disease, four studies pooled (920 patients), stratified split.

    The target is 1 when the angiographic diagnosis ``num`` is 1 to 4 (narrowing in at
    least one major vessel) and 0 when it is 0. Missing measurements, coded ``?`` in
    the source, stay as NaN. A cholesterol of 0, which the Swiss and Long Beach files
    use for "not measured", is also set to NaN.
    """
    frames = []
    for site, filename in _HEART_SITES.items():
        path = _download(_HEART_URL + filename, HEART_DIR, filename)
        frame = pd.read_csv(path, header=None, names=_HEART_COLUMNS, na_values="?")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data.loc[data["cholesterol"] == 0, "cholesterol"] = np.nan
    y = (data.pop("num") > 0).astype(int).rename("heart_disease")
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, train_size=train_size, test_size=test_size, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


# Statlog's OpenML column names, mapped to the names used for the UCI table above.
_STATLOG_COLUMNS = {
    "age": "age",
    "sex": "sex",
    "chest": "chest_pain_type",
    "resting_blood_pressure": "resting_bp",
    "serum_cholestoral": "cholesterol",
    "fasting_blood_sugar": "fasting_blood_sugar",
    "resting_electrocardiographic_results": "resting_ecg",
    "maximum_heart_rate_achieved": "max_heart_rate",
    "exercise_induced_angina": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "number_of_major_vessels": "n_major_vessels",
    "thal": "thalassemia",
}
# Measurements on a continuous scale; the rest are codes for categories, except the
# vessel count (0 to 3), which is ordinal and treated as a number.
CONTINUOUS = ["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"]
CATEGORICAL = ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg",
               "exercise_angina", "st_slope", "thalassemia"]


def load_statlog_heart(train_size: int = 200, test_size: int = 70):
    """Statlog heart disease (OpenML 53): 270 patients, no missing values.

    The same thirteen measurements as the UCI table, under the same names, from a single
    study and complete, which makes it the cleaner table for explanations: an attribution
    method never has to decide what a missing value contributed. The target is 1 when
    heart disease is present (120 of 270). Stratified split with seed 0.
    """
    path = STATLOG_DIR / "heart_statlog_openml_53.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        STATLOG_DIR.mkdir(parents=True, exist_ok=True)
        fetch_openml(data_id=53, as_frame=True, parser="auto").frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    y = (frame.pop("class") == "present").astype(int).rename("heart_disease")
    X = frame.rename(columns=_STATLOG_COLUMNS)[list(_STATLOG_COLUMNS.values())].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def categorical_indices(feature_names: list[str]) -> list[int]:
    """Positions of the categorical codes, to pass to TabPFN as categorical features."""
    return [i for i, name in enumerate(feature_names) if name in CATEGORICAL]


_HOUSING_FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt",
    "FullBath", "LotArea", "KitchenQual", "Neighborhood",
]


def load_housing(train_size: int = 1300, test_size: int = 160):
    """Ames housing (OpenML 42165), nine features including raw text columns.

    ``Neighborhood`` and ``KitchenQual`` are left as strings, so the example exercises
    TabPFN's handling of text categories. The target is the sale price in dollars.
    """
    path = HOUSING_DIR / "ames_openml_42165.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        HOUSING_DIR.mkdir(parents=True, exist_ok=True)
        fetch_openml(data_id=42165, as_frame=True, parser="auto").frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    X = frame[_HOUSING_FEATURES]
    y = frame["SalePrice"].astype(float).rename("sale_price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


# A fixed window of whole days, chosen to avoid a daylight-saving change (31 March
# 2024) and the 15-minute day-ahead products introduced in October 2025.
_ELECTRICITY_START = "2024-03-04"
_ELECTRICITY_DAYS = 16
_ENERGY_CHARTS = "https://api.energy-charts.info"


def _energy_charts_hourly(endpoint: str, value_key: str, start: str, end: str, name: str):
    """One Energy-Charts series as an hourly mean on Berlin local time."""
    url = f"{_ENERGY_CHARTS}/{endpoint}&start={start}&end={end}"
    payload = json.loads(_download(url, ELECTRICITY_DIR, name).read_text())
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin")
    series = pd.Series(payload[value_key], index=index, dtype=float)
    return series.resample("1h").mean()


def load_electricity(horizon: int = 48):
    """Hourly German day-ahead prices with day-ahead load and renewables forecasts.

    Returns ``history`` and ``future`` tables with columns ``hour`` (timestamp,
    Berlin local time), ``price_eur`` (EUR/MWh, DE-LU bidding zone),
    ``load_forecast`` and ``renewables_forecast`` (MW, the day-ahead forecasts of
    load and of solar plus onshore and offshore wind). The last ``horizon`` hours are
    the ``future`` to forecast. The forecasts are published before the hours they
    describe, so using them as features does not leak the future.
    """
    start = pd.Timestamp(_ELECTRICITY_START)
    end = start + pd.Timedelta(days=_ELECTRICITY_DAYS)
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    tag = f"{s}_{_ELECTRICITY_DAYS}d"
    price = _energy_charts_hourly("price?bzn=DE-LU", "price", s, e, f"price_{tag}.json")
    forecasts = {
        kind: _energy_charts_hourly(
            f"public_power_forecast?country=de&production_type={kind}&forecast_type=day-ahead",
            "forecast_values", s, e, f"forecast_{kind}_{tag}.json",
        )
        for kind in ("load", "solar", "wind_onshore", "wind_offshore")
    }
    table = pd.DataFrame({
        "price_eur": price,
        "load_forecast": forecasts["load"],
        "renewables_forecast": forecasts["solar"] + forecasts["wind_onshore"] + forecasts["wind_offshore"],
    })
    table.index = table.index.tz_localize(None)
    table = table.loc[str(start):str(end - pd.Timedelta(hours=1))].dropna()
    table = table.rename_axis("hour").reset_index()
    return table.iloc[:-horizon].reset_index(drop=True), table.iloc[-horizon:].reset_index(drop=True)


# ------------------------------------------------------------------------ plotting


def _finish(fig) -> None:
    """Show the figure inline, or save it when a figure directory is configured."""
    global _figure_count
    fig.tight_layout()
    if _FIGURE_DIR:
        _figure_count += 1
        fig.savefig(_FIGURE_DIR / f"figure_{_figure_count:02d}.png")
        plt.close(fig)
    else:
        plt.show()


def plot_actual_vs_predicted(actual, predicted, unit: str = "target") -> None:
    """Scatter of predicted against actual values, with the identity line."""
    if not _PLOTS:
        return
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, s=14, alpha=0.7)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.set_xlabel(f"actual {unit}")
    ax.set_ylabel(f"predicted {unit}")
    _finish(fig)


def plot_predicted_distribution(values, actual=None, unit: str = "target") -> None:
    """Histogram of sampled or binned predictive values, with the actual value marked."""
    if not _PLOTS:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.asarray(values), bins=40, alpha=0.8)
    if actual is not None:
        ax.axvline(actual, color="black", ls="--", label="actual")
        ax.legend()
    ax.set_xlabel(unit)
    _finish(fig)


def plot_quantile_band(actual, low, median, high, ylabel: str, unit: str = "rows") -> None:
    """Each test row's 10 to 90 percent band and median, sorted by median, with the truth."""
    if not _PLOTS:
        return
    order = np.argsort(np.asarray(median))
    actual, low, median, high = (np.asarray(a)[order] for a in (actual, low, median, high))
    x = np.arange(len(order))
    inside = (actual >= low) & (actual <= high)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x, low, high, alpha=0.25, label="10 to 90 percent band")
    ax.plot(x, median, lw=1, label="median")
    ax.scatter(x[inside], actual[inside], s=10, color="black", label="actual, inside")
    ax.scatter(x[~inside], actual[~inside], s=14, color="crimson", label="actual, outside")
    ax.set_xlabel(f"{unit}, sorted by predicted median")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    _finish(fig)


def plot_electricity(history) -> None:
    """The price history the forecast will read as context."""
    if not _PLOTS:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(history["hour"], history["price_eur"], lw=1)
    ax.set_ylabel("price (EUR/MWh)")
    ax.set_title("Hourly German day-ahead price, history")
    _finish(fig)


def plot_forecast_comparison(
    history, future, forecasts: dict, time_col: str = "hour", target_col: str = "price_eur",
    ylabel: str = "value", title: str = "", band=None, context_hours: int = 96,
) -> None:
    """The tail of the history, the held-out truth, and one line per forecast.

    ``band`` is an optional ``(low, high)`` pair drawn around the forecast horizon.
    The MAE of each forecast against the truth is shown in its legend entry.
    """
    if not _PLOTS:
        return
    tail = history.iloc[-context_hours:]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tail[time_col], tail[target_col], color="gray", lw=1, label="history")
    ax.plot(future[time_col], future[target_col], color="black", lw=1.5, label="actual")
    if band is not None:
        ax.fill_between(future[time_col], band[0], band[1], alpha=0.2, label="10 to 90 percent band")
    truth = future[target_col].to_numpy()
    for name, forecast in forecasts.items():
        mae = np.mean(np.abs(truth - np.asarray(forecast)))
        ax.plot(future[time_col], forecast, lw=1.5, label=f"{name} (MAE {mae:.1f})")
    ax.axvline(future[time_col].iloc[0], color="gray", ls=":", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    _finish(fig)


# ------------------------------------------------------------------ interpretation


def make_shap_explainer(model, X_background: pd.DataFrame, n_background: int = 10, seed: int = SEED):
    """A permutation SHAP explainer for the predicted probability of class 1.

    SHAP needs a reference population to stand in for "feature unknown": a feature is
    removed from a patient by replacing its value with values from these background
    rows. ``n_background`` rows are sampled from ``X_background`` with ``seed``. The
    model must have been fitted on a plain array with the same column order.
    """
    import shap

    background = X_background.sample(n_background, random_state=seed).to_numpy(dtype=float)

    def risk(X):
        return model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    return shap.PermutationExplainer(risk, background, feature_names=list(X_background.columns), seed=seed)


def top_continuous_features(shap_values, feature_names: list[str], continuous: list[str], k: int = 4) -> list[str]:
    """The ``k`` continuous features with the largest mean absolute SHAP value."""
    importance = np.abs(shap_values.values).mean(axis=0)
    ranked = [feature_names[i] for i in np.argsort(importance)[::-1]]
    return [name for name in ranked if name in continuous][:k]


def plot_patient_waterfall(shap_values, index: int, title: str, max_display: int = 10) -> None:
    """One patient's attributions, from the background average to their predicted risk."""
    if not _PLOTS:
        return
    import shap

    shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    _finish(fig)


def plot_beeswarm(shap_values, max_display: int = 13) -> None:
    """Every explained patient's attribution for every feature, features ranked by impact."""
    if not _PLOTS:
        return
    import shap

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    _finish(plt.gcf())


def partial_dependence(model, X: np.ndarray, column: int, grid_points: int = 20):
    """Average predicted risk as one column sweeps its range, all other values kept.

    The grid is the column's 5th to 95th percentiles in ``grid_points`` steps (duplicates
    removed). All grid copies of ``X`` go to the model in one call.
    """
    grid = np.unique(np.quantile(X[:, column], np.linspace(0.05, 0.95, grid_points)))
    stacked = np.repeat(X[None, :, :], len(grid), axis=0)
    stacked[:, :, column] = grid[:, None]
    risk = model.predict_proba(stacked.reshape(-1, X.shape[1]))[:, 1]
    return grid, risk.reshape(len(grid), len(X)).mean(axis=1)


def plot_partial_dependence(model, X: np.ndarray, features: list[str], feature_names: list[str]):
    """Partial dependence of predicted risk on each feature, on one shared risk axis.

    Returns ``{feature: (grid, mean risk)}`` so the curves can be inspected as numbers.
    """
    curves = {f: partial_dependence(model, X, feature_names.index(f)) for f in features}
    if _PLOTS:
        fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.2), sharey=True)
        for ax, (name, (grid, risk)) in zip(np.atleast_1d(axes), curves.items()):
            ax.plot(grid, risk, lw=2)
            column = X[:, feature_names.index(name)]
            ax.plot(column, np.full_like(column, ax.get_ylim()[0]), "|", color="gray", alpha=0.4)
            ax.set_xlabel(name)
        np.atleast_1d(axes)[0].set_ylabel("mean predicted risk")
        _finish(fig)
    return curves
