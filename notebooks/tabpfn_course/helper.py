"""Support module for the TabPFN first-predictions notebook, run on local TabPFN.

The notebook comes from Prior Labs' TabPFN course, where it imports a course-provided
``helper`` module and talks to TabPFN through ``tabpfn_client``, the hosted API that
needs a Prior Labs account. This lab uses TabPFN as a local library instead (decision
D1), so this module rebuilds the same functions under the same names, with two
differences worth knowing before reading the results.

The model runs locally. ``init_tabpfn`` selects the weights: TabPFN-3.5, the lab's
main model, when a license token is available (cached by the one-time acceptance
described in the README, or ``TABPFN_TOKEN``), and otherwise TabPFN-2, whose weights
download without an account. It also keeps TabPFN off Apple's MPS backend, so a laptop
runs on the CPU while a pod with CUDA still uses its GPUs.

The data are rebuilt from public sources, so the numbers will not match the course's
exactly. The heart-disease table is the UCI collection of four hospital studies (920
patients); the housing table is the Ames dataset from OpenML (id 42165) with nine
features chosen here; the electricity table is sixteen days of German day-ahead prices
from SMARD with the day-ahead load and renewables forecasts published by the
transmission operators, both served by the Energy-Charts API of Fraunhofer ISE.
Downloads are cached in the shared data lake (``$SSLLAB_DATA_ROOT``, by default
``~/work/data``), filed by domain like the lake's other datasets:
``clinical/uci_heart_disease``, ``real_estate/ames_housing`` and
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
HOUSING_DIR = DATA_ROOT / "real_estate" / "ames_housing"
ELECTRICITY_DIR = DATA_ROOT / "energy" / "de_day_ahead_power"

# Set by configure_lesson. When FIGURE_DIR is a path, figures are written there
# instead of shown, which is how the notebook is checked headlessly.
_PLOTS = True
_FIGURE_DIR: Path | None = None
_figure_count = 0


# --------------------------------------------------------------------------- setup


def configure_lesson(plots: bool = True, figure_dir: str | Path | None = None) -> None:
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
