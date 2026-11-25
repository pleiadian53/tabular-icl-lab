"""Data, splits, reference models and plots for the grouped-transfer notebooks.

The question these notebooks ask is what happens when the rows a model must predict
belong to a group it has never seen. Many tables are made of groups: cells from the
same perturbation, patients from the same hospital, readings from the same machine,
designs that share one building geometry. Rows inside a group share most of what
determines the target, so a model can do well on new rows of a known group and still
be far off on a new group. A random split of rows puts rows of every group into both
the context and the test set, so every test row belongs to a group the model has
already seen. The score then measures the easier task, filling in new rows of known
groups, while it is easily read as a score for new groups. Only a split that holds
out whole groups measures the harder task.

The table. The UCI Energy Efficiency data (Tsanas and Xifara, 2012; DOI
10.24432/C51307; CC BY 4.0) holds 768 simulated residential buildings. Each one is a
combination of three design choices made before any simulation runs:

- one of 12 **geometries**, each described by five shape columns: relative
  compactness, surface area, wall area, roof area and overall height. All twelve
  enclose the same volume, 771.75 cubic metres, arranged differently;
- one of 4 **orientations**, coded 2 to 5;
- one of 16 **glazing setups**: no windows, or a window area of 10%, 25% or 40% of the
  floor area spread over the facades in one of five ways.

The simulation then gives two outcomes per building, the heating load and the
cooling load. Every combination appears exactly once, 12 x 4 x 16 = 768, so the table
is a full factorial design and each geometry is a group of 64 designs with identical
shape columns.

The split. ``grouped_split`` holds out whole geometries. By default it holds out
``HELD_OUT_GEOMETRIES``, the third, seventh and eleventh in the order they first
appear in the source workbook, 192 designs, with the other 576 as the context TabPFN
learns from; together they are one two-storey geometry inside the range of the
context, one one-storey geometry inside it and one just outside it. The default is
fixed so that every number in the notebooks' text refers to the same three unseen
geometries. Any other set can be passed, by name or drawn by ``random_geometries``,
and ``geometry_table`` and ``plot_geometry_levels`` take the same set so that tables
and figures mark the geometries actually held out. ``leave_one_group_out`` repeats
the split for each of the twelve geometries in turn, which is the systematic answer
to how the results depend on the choice. Design identifiers follow the workbook's row
order (``E0001`` to ``E0768``).

The model. ``init_tabpfn`` selects local weights: TabPFN-3.5, the lab's main model,
when a license token is available, and TabPFN-2 otherwise. It keeps TabPFN off Apple's
MPS backend, so a laptop runs on the CPU while a pod with CUDA uses its GPUs.

The data come from the UCI repository and are cached once in the shared data lake
(``$DATA_LAKE_ROOT``, or the older ``$SSLLAB_DATA_ROOT``, by default ``~/work/data``)
under ``energy/uci_energy_efficiency``. The workbook is checked against a fixed
SHA-256, so a changed upstream file stops the notebooks instead of silently moving
their numbers.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 0
DATA_ROOT = Path(
    os.environ.get("DATA_LAKE_ROOT") or os.environ.get("SSLLAB_DATA_ROOT") or Path.home() / "work" / "data"
)
ENERGY_DIR = DATA_ROOT / "energy" / "uci_energy_efficiency"
_ENERGY_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
_WORKBOOK = "ENB2012_data.xlsx"
_WORKBOOK_SHA256 = "0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4"

SHAPE_COLUMNS = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height"]
DESIGN_COLUMNS = SHAPE_COLUMNS + ["orientation", "glazing_area", "glazing_distribution"]
TARGET = "heating_load"
SIBLING_OUTCOME = "cooling_load"
HELD_OUT_GEOMETRIES = (2, 6, 10)
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Set by configure_plots. When _FIGURE_DIR is a path, figures are written there
# instead of shown, which is how the notebooks are checked headlessly.
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
    warnings.filterwarnings("ignore", message="IProgress not found")
    pd.set_option("display.width", 120)
    pd.set_option("display.precision", 3)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def init_tabpfn(version: str | None = None) -> str:
    """Select the local TabPFN weights and keep the model off MPS. Returns the version.

    ``TabPFNRegressor()`` takes its weights from TabPFN's global default, which this
    sets. With ``version=None`` the choice is TabPFN-3.5 when a license token is found
    and TabPFN-2 otherwise; pass ``"v3.5"`` or ``"v2"`` to force one.
    """
    from tabpfn.browser_auth import get_cached_token
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings

    if version is None:
        has_token = bool(os.environ.get("TABPFN_TOKEN") or get_cached_token())
        version = "v3.5" if has_token else "v2"
    settings.tabpfn.model_version = ModelVersion(version)
    # Read by TabPFN each time it picks a device, so this applies to models built later.
    os.environ.setdefault("TABPFN_EXCLUDE_DEVICES", "mps")
    print(f"TabPFN {version} weights, running locally")
    return version


# ------------------------------------------------------------------------ the data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workbook() -> Path:
    """Download the UCI archive once, keep only the workbook, and check its hash."""
    path = ENERGY_DIR / _WORKBOOK
    if not path.exists():
        ENERGY_DIR.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(_ENERGY_URL, headers={"User-Agent": "tabular-icl-lab"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.write_bytes(archive.read(_WORKBOOK))
    if _sha256(path) != _WORKBOOK_SHA256:
        raise RuntimeError(f"{path} does not match the expected SHA-256; delete it to download again")
    return path


def load_energy_efficiency() -> pd.DataFrame:
    """All 768 designs, indexed by ``design_id``, with a ``geometry`` group label.

    Columns: the eight design columns, the two simulated outcomes ``heating_load`` and
    ``cooling_load`` (kWh per square metre), and ``geometry``, an integer from 0 to 11
    numbering the distinct combinations of the five shape columns in the order they
    first appear in the workbook. Values are exactly the workbook's; nothing is
    scaled or imputed.
    """
    raw = pd.read_excel(_workbook()).dropna(how="all").dropna(axis=1, how="all")
    frame = raw.iloc[:, :10].copy()
    frame.columns = DESIGN_COLUMNS + [TARGET, SIBLING_OUTCOME]
    frame.index = pd.Index([f"E{i + 1:04d}" for i in range(len(frame))], name="design_id")
    first_seen = {key: code for code, key in enumerate(dict.fromkeys(map(tuple, frame[SHAPE_COLUMNS].to_numpy())))}
    frame["geometry"] = [first_seen[tuple(row)] for row in frame[SHAPE_COLUMNS].to_numpy()]
    return frame


def geometry_table(frame: pd.DataFrame, held_out=HELD_OUT_GEOMETRIES) -> pd.DataFrame:
    """One row per geometry: its shape columns, design count, mean heating load, and
    whether it is among the ``held_out`` geometries."""
    table = frame.groupby("geometry").agg(
        **{c: (c, "first") for c in SHAPE_COLUMNS},
        designs=(TARGET, "size"),
        mean_heating_load=(TARGET, "mean"),
    )
    table["held_out"] = table.index.isin(held_out)
    return table


def geometry_names(frame: pd.DataFrame) -> pd.Series:
    """Short names such as ``"0.86 / 7.0 m"``: relative compactness and overall height.

    Those two columns are enough to tell the twelve geometries apart, so they serve as
    readable names in tables and plots.
    """
    table = frame.groupby("geometry")[["relative_compactness", "overall_height"]].first()
    return pd.Series(
        [f"{rc:.2f} / {h:.1f} m" for rc, h in table.to_numpy()], index=table.index, name="geometry_name"
    )


def random_geometries(frame: pd.DataFrame, n: int = 3, seed: int | None = None,
                      stratify: str | None = "overall_height") -> tuple[int, ...]:
    """Draw ``n`` geometries at random to hold out, as a sorted tuple of geometry numbers.

    With ``stratify`` set to a column that is constant within each geometry, the draw
    first takes one geometry from each value of that column and fills the rest from
    the remaining geometries. The default, ``"overall_height"``, guarantees at least
    one one-storey and one two-storey geometry. A plain draw of three from twelve
    puts all three at the same height in 2 of every 11 draws, and such a split says
    nothing about buildings of the other height. Pass ``stratify=None`` for a plain
    draw. ``seed`` makes the
    draw repeatable; ``None`` gives a new draw each call.
    """
    rng = np.random.default_rng(seed)
    geometries = frame.groupby("geometry").first()
    if not 1 <= n < len(geometries):
        raise ValueError(f"n must be between 1 and {len(geometries) - 1}, leaving some geometries as context")
    chosen: list[int] = []
    if stratify is not None:
        levels = geometries.groupby(stratify).groups
        if n < len(levels):
            raise ValueError(f"n={n} cannot cover the {len(levels)} values of {stratify!r}; use a larger n or stratify=None")
        chosen = [int(rng.choice(np.asarray(members))) for members in levels.values()]
    rest = np.setdiff1d(geometries.index.to_numpy(), chosen)
    chosen += [int(g) for g in rng.choice(rest, size=n - len(chosen), replace=False)]
    return tuple(sorted(chosen))


def grouped_split(frame: pd.DataFrame, held_out=HELD_OUT_GEOMETRIES):
    """Hold out whole geometries. Returns ``(history, current, answers)``.

    ``held_out`` is any collection of geometry numbers; the default is
    ``HELD_OUT_GEOMETRIES``, the split the notebooks' text describes, and
    ``random_geometries`` draws another. ``history`` is every design of the other
    geometries, with both outcomes: the labelled context. ``current`` is the held-out
    designs with the design columns and the geometry label only: what a design team
    would have before simulating. ``answers`` is the held-out heating loads, used only
    to score predictions.
    """
    held_out = [int(g) for g in held_out]
    unknown = sorted(set(held_out) - set(frame["geometry"].unique()))
    if unknown:
        raise ValueError(f"no geometry numbered {unknown}; the geometries are numbered 0 to {frame['geometry'].max()}")
    test = frame["geometry"].isin(held_out)
    if test.all() or not test.any():
        raise ValueError("held_out must leave at least one geometry on each side of the split")
    history = frame.loc[~test].copy()
    current = frame.loc[test, DESIGN_COLUMNS + ["geometry"]].copy()
    answers = frame.loc[test, TARGET].copy()
    return history, current, answers


def random_split(frame: pd.DataFrame, n_test: int = 192, seed: int = SEED):
    """Hold out ``n_test`` designs drawn at random, ignoring geometry. Same return as
    ``grouped_split``; the contrast that shows what a row-level split hides."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(frame), dtype=bool)
    test[rng.choice(len(frame), size=n_test, replace=False)] = True
    history = frame.loc[~test].copy()
    current = frame.loc[test, DESIGN_COLUMNS + ["geometry"]].copy()
    answers = frame.loc[test, TARGET].copy()
    return history, current, answers


# ---------------------------------------------------------------- models to compare


def fit_predict_tabpfn(history: pd.DataFrame, current: pd.DataFrame, features: list[str], seed: int = SEED) -> dict:
    """TabPFN with its default constructor. Returns the predictive mean, nine quantiles
    (10% to 90%) and the fit-plus-predict time in seconds."""
    from tabpfn import TabPFNRegressor

    start = time.perf_counter()
    model = TabPFNRegressor(random_state=seed)
    model.fit(history[features], history[TARGET])
    out = model.predict(current[features], output_type="main")
    seconds = time.perf_counter() - start
    quantiles = np.asarray(out["quantiles"])
    return {
        "mean": np.asarray(out["mean"]),
        "median": np.asarray(out["median"]),
        "quantiles": quantiles,
        "low": quantiles[0],
        "high": quantiles[-1],
        "seconds": seconds,
    }


def fit_predict_xgboost(history: pd.DataFrame, current: pd.DataFrame, features: list[str], seed: int = SEED) -> dict:
    """XGBoost with its default settings, one thread: the untuned gradient-boosting
    reference. Point predictions only."""
    from xgboost import XGBRegressor

    start = time.perf_counter()
    model = XGBRegressor(random_state=seed, n_jobs=1)
    model.fit(history[features], history[TARGET])
    mean = model.predict(current[features]).astype(float)
    return {"mean": mean, "seconds": time.perf_counter() - start}


MIXED_FORMULA = (
    f"{TARGET} ~ relative_compactness + overall_height + glazing_area + glazing_area:overall_height"
    " + C(glazing_distribution) + C(orientation)"
)


def fit_predict_mixed(history: pd.DataFrame, current: pd.DataFrame, formula: str = MIXED_FORMULA) -> dict:
    """Linear mixed-effects model with a random intercept for each geometry.

    For design i of geometry g the model is

        heating_load[g, i] = x[g, i] . beta + u[g] + e[g, i],
        u[g] ~ Normal(0, tau^2),  e[g, i] ~ Normal(0, sigma^2),

    where x holds the fixed-effect columns of ``formula`` (compactness and height
    describe the geometry, the glazing terms and orientation vary inside it, and
    the glazing-by-height term lets windows matter more on taller buildings), u[g]
    is the geometry's own offset that the shape columns do not explain, and e is the
    design-level residual. For a geometry outside the context u[g] is unknown, so the
    prediction is the fixed part x . beta and the predictive variance is
    tau^2 + sigma^2. The 10% and 90% quantiles of that normal distribution give an
    80% band comparable with TabPFN's.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import norm

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(formula, history, groups=history["geometry"]).fit(reml=True)
    mean = np.asarray(fit.predict(current), dtype=float)
    tau2, sigma2 = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    spread = np.sqrt(tau2 + sigma2) * norm.ppf(0.9)
    return {
        "mean": mean,
        "low": mean - spread,
        "high": mean + spread,
        "tau": float(np.sqrt(tau2)),
        "sigma": float(np.sqrt(sigma2)),
        "seconds": time.perf_counter() - start,
        "fit": fit,
    }


def fit_predict_two_stage(history: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Predict a geometry's level and the within-geometry glazing effect separately.

    Stage 1 fits a Gaussian process to the context geometries' mean heating loads,
    one point per geometry, as a function of their five standardized shape columns.
    Stage 2 takes, for each combination of overall height, glazing area and glazing
    distribution, the average amount by which designs with that combination sit above
    or below their own geometry's mean in the context, averaged over orientations.
    A prediction is the stage-1 level plus the stage-2 shift. The 80% band combines
    the Gaussian process's standard deviation for the level with the spread of the
    stage-2 residuals.
    """
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    start = time.perf_counter()
    levels = history.groupby("geometry").agg(**{c: (c, "first") for c in SHAPE_COLUMNS}, level=(TARGET, "mean"))
    centre, scale = levels[SHAPE_COLUMNS].mean(), levels[SHAPE_COLUMNS].std(ddof=0).replace(0, 1)
    kernel = ConstantKernel() * RBF(length_scale=np.ones(len(SHAPE_COLUMNS))) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit((levels[SHAPE_COLUMNS] - centre) / scale, levels["level"])
    level, level_sd = gp.predict((current[SHAPE_COLUMNS] - centre) / scale, return_std=True)

    cell = ["overall_height", "glazing_area", "glazing_distribution"]
    deviation = history[TARGET] - history.groupby("geometry")[TARGET].transform("mean")
    shift_table = deviation.groupby([history[c] for c in cell]).mean().rename("shift")
    shift = current[cell].join(shift_table, on=cell)["shift"].fillna(0.0).to_numpy()
    residual_sd = float((deviation - history[cell].join(shift_table, on=cell)["shift"]).std(ddof=1))

    mean = level + shift
    spread = np.sqrt(level_sd**2 + residual_sd**2) * norm.ppf(0.9)
    return {"mean": mean, "low": mean - spread, "high": mean + spread, "seconds": time.perf_counter() - start}


def leave_one_group_out(frame: pd.DataFrame, fit_predict, **kwargs) -> pd.DataFrame:
    """Predict every design from a model whose context excludes its whole geometry.

    ``fit_predict(history, current, **kwargs)`` must return a dict with ``"mean"``
    and, optionally, ``"low"`` and ``"high"``. The result has one row per design with
    the prediction, the band if given, and the fold's fit-and-predict seconds.
    """
    rows = []
    for geometry in sorted(frame["geometry"].unique()):
        history, current, answers = grouped_split(frame, held_out=(geometry,))
        out = fit_predict(history, current, **kwargs)
        part = pd.DataFrame({"geometry": geometry, "actual": answers, "mean": out["mean"]}, index=current.index)
        for key in ("low", "high"):
            if key in out:
                part[key] = out[key]
        part["seconds"] = out["seconds"]
        rows.append(part)
    return pd.concat(rows).loc[frame.index]


# ----------------------------------------------------------------------- summaries


def scores(actual, predicted) -> dict:
    """Mean absolute error, R^2 and mean signed error (prediction minus truth)."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    residual = predicted - actual
    return {
        "mae": float(np.abs(residual).mean()),
        "r2": float(1 - (residual**2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "signed": float(residual.mean()),
    }


def errors_by_geometry(geometry, actual, predictions: dict) -> pd.DataFrame:
    """For each geometry and each model: MAE and mean signed error.

    When a geometry's MAE is close to the size of its mean signed error, the model
    misses most designs of that geometry by about the same amount in the same
    direction: the error is a shared offset, not scatter.
    """
    frame = pd.DataFrame({"geometry": np.asarray(geometry), "actual": np.asarray(actual, float)})
    columns = {}
    for name, predicted in predictions.items():
        residual = np.asarray(predicted, float) - frame["actual"].to_numpy()
        grouped = pd.Series(residual).groupby(frame["geometry"].to_numpy())
        columns[(name, "MAE")] = grouped.apply(lambda r: np.abs(r).mean())
        columns[(name, "signed")] = grouped.mean()
    table = pd.DataFrame(columns)
    table.index.name = "geometry"
    return table


def coverage_by_geometry(geometry, actual, low, high) -> pd.DataFrame:
    """Share of designs whose truth falls inside the band, and mean band width, per
    geometry, with the overall values in a final row."""
    frame = pd.DataFrame(
        {"geometry": np.asarray(geometry), "inside": (np.asarray(actual) >= low) & (np.asarray(actual) <= high),
         "width": np.asarray(high) - np.asarray(low)}
    )
    table = frame.groupby("geometry")[["inside", "width"]].mean()
    table.loc["all"] = frame[["inside", "width"]].mean()
    return table.rename(columns={"inside": "coverage"})


def variance_shares(frame: pd.DataFrame) -> dict:
    """How much of the heating-load variance the grouping explains.

    ``between``: the share explained by each geometry's mean alone.
    ``with_glazing``: the share explained by the geometry mean plus an average shift
    for each glazing setup (area and distribution), the same for every geometry.
    ``icc``: the intraclass correlation from the one-way random-effects decomposition,
    the fraction of variance that two designs of the same geometry share.
    ``effective_n``: the row count divided by the design effect 1 + (m - 1) icc, where
    m is the number of designs per geometry; roughly how many independent designs the
    table is worth for learning anything that varies between geometries.
    """
    y = frame[TARGET]
    total = ((y - y.mean()) ** 2).sum()
    level = frame.groupby("geometry")[TARGET].transform("mean")
    within = y - level
    glazing = within.groupby([frame["glazing_area"], frame["glazing_distribution"]]).transform("mean")

    groups = frame.groupby("geometry")[TARGET]
    k, m = groups.ngroups, len(frame) / groups.ngroups
    msb = (groups.size() * (groups.mean() - y.mean()) ** 2).sum() / (k - 1)
    msw = (within**2).sum() / (len(frame) - k)
    icc = (msb - msw) / (msb + (m - 1) * msw)
    return {
        "between": float(((level - y.mean()) ** 2).sum() / total),
        "with_glazing": float(1 - ((within - glazing) ** 2).sum() / total),
        "icc": float(icc),
        "designs_per_geometry": float(m),
        "effective_n": float(len(frame) / (1 + (m - 1) * icc)),
    }


def per_geometry_mae(logo: pd.DataFrame) -> pd.Series:
    """MAE of a leave-one-group-out prediction table, one value per geometry."""
    return (logo["mean"] - logo["actual"]).abs().groupby(logo["geometry"]).mean()


def group_calibrated_band(logo: pd.DataFrame, level: float = 0.8):
    """Widen or narrow a leave-one-group-out band using the other geometries' misses.

    This is conformalized quantile regression with groups as the unit of
    exchangeability. For each design i of the table, the conformity score is
    s_i = max(low_i - y_i, y_i - high_i): negative when the truth is inside the band,
    and the distance to the nearer edge when it is outside. For each geometry g, the
    adjustment q_g is the ``level`` quantile (rounded up to an observed score) of the
    scores of all designs not in g, and g's band becomes [low - q_g, high + q_g]. A
    band that was too wide shrinks (q_g < 0), one that was too narrow grows. Because
    each score came from a model whose context excluded that score's own geometry,
    the calibration sees the same kind of miss the band has to cover: a whole new
    geometry. Scores within one geometry are not independent, so the nominal level is
    an approximate target here, not a guarantee.
    """
    score = np.maximum(logo["low"] - logo["actual"], logo["actual"] - logo["high"])
    low, high = logo["low"].copy(), logo["high"].copy()
    for g in logo["geometry"].unique():
        inside = logo["geometry"] == g
        q = np.quantile(score[~inside], level, method="higher")
        low[inside] -= q
        high[inside] += q
    return low, high


# ------------------------------------------------------------------------- figures


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


def plot_geometry_levels(frame: pd.DataFrame, held_out=HELD_OUT_GEOMETRIES) -> None:
    """Heating load of every design, one column of points per geometry, ordered by
    compactness within each height; the ``held_out`` geometries marked in orange."""
    if not _PLOTS:
        return
    names = geometry_names(frame)
    order = frame.groupby("geometry")[["overall_height", "relative_compactness"]].first().sort_values(
        ["overall_height", "relative_compactness"]).index
    fig, ax = plt.subplots(figsize=(9, 4))
    for x, g in enumerate(order):
        values = frame.loc[frame["geometry"] == g, TARGET]
        held = g in held_out
        ax.scatter(np.full(len(values), x) + np.random.default_rng(g).uniform(-0.15, 0.15, len(values)), values,
                   s=8, alpha=0.6, color="tab:orange" if held else "tab:blue")
    ax.set_xticks(range(len(order)), [names[g] for g in order], rotation=45, ha="right")
    ax.set_xlabel("geometry (relative compactness / overall height); orange = held out")
    ax.set_ylabel("heating load")
    _finish(fig)


def plot_predicted_vs_actual(actual, predictions: dict, geometry, names: pd.Series) -> None:
    """One panel per model: predicted against actual heating load, coloured by geometry."""
    if not _PLOTS:
        return
    actual = np.asarray(actual, float)
    geometry = np.asarray(geometry)
    fig, axes = plt.subplots(1, len(predictions), figsize=(4.6 * len(predictions), 4.4), squeeze=False)
    lo, hi = actual.min() - 1, actual.max() + 1
    for ax, (name, predicted) in zip(axes[0], predictions.items()):
        for g in np.unique(geometry):
            k = geometry == g
            ax.scatter(actual[k], np.asarray(predicted)[k], s=12, alpha=0.7, label=names[g])
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
        ax.set_title(name)
        ax.set_xlabel("actual heating load")
        ax.set_ylabel("predicted heating load")
    axes[0][0].legend(title="geometry", fontsize=8)
    _finish(fig)


def plot_group_errors(table: pd.DataFrame, names: pd.Series, metric: str = "signed") -> None:
    """Bars of a per-geometry error (``"signed"`` or ``"MAE"``) for each model."""
    if not _PLOTS:
        return
    data = table.xs(metric, axis=1, level=1)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(data) + 3), 3.8))
    width = 0.8 / data.shape[1]
    for j, model in enumerate(data.columns):
        ax.bar(np.arange(len(data)) + j * width - 0.4 + width / 2, data[model], width, label=model)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(range(len(data)), [names[g] for g in data.index], rotation=45, ha="right")
    ax.set_ylabel("mean signed error" if metric == "signed" else "MAE")
    ax.legend(fontsize=8)
    _finish(fig)


def find_design(current: pd.DataFrame, geometry: int, orientation: int = 2,
                glazing_area: float = 0.0, glazing_distribution: int = 0) -> str:
    """The identifier of the design with the given geometry, orientation and glazing setup.

    Every combination appears exactly once in the table, so the match is unique when
    the geometry is among the rows of ``current``. The defaults describe a design with
    no windows facing orientation 2; glazing distribution 0 means no windows.
    """
    match = current.index[
        (current["geometry"] == geometry)
        & (current["orientation"] == orientation)
        & np.isclose(current["glazing_area"], glazing_area)
        & (current["glazing_distribution"] == glazing_distribution)
    ]
    if len(match) != 1:
        raise KeyError(f"no single design with geometry {geometry}, orientation {orientation}, "
                       f"glazing area {glazing_area}, distribution {glazing_distribution} among these rows")
    return match[0]


def design_label(design: pd.Series, names: pd.Series) -> str:
    """A one-line description of one design: its geometry, glazing setup and orientation.

    ``design`` is one row of the frame, such as ``current.loc["E0505"]``, and ``names``
    the geometry labels from ``geometry_names``. Glazing distribution 0 means no windows.
    """
    if design["glazing_area"] == 0:
        glazing = "no windows"
    else:
        glazing = f"{design['glazing_area']:.0%} glazing, distribution {int(design['glazing_distribution'])}"
    return f"{design.name}: geometry {names[int(design['geometry'])]}, {glazing}, orientation {int(design['orientation'])}"


def predictive_density(output: dict, row: int) -> tuple[np.ndarray, np.ndarray]:
    """TabPFN's predictive density for one row, as bin edges and one density value per bin.

    ``output`` is the dictionary that ``TabPFNRegressor.predict(..., output_type="full")``
    returns, and ``row`` the position of a design in the rows that were predicted.

    TabPFN does not assume a shape such as a normal distribution for the target. It
    cuts the target axis into a fixed grid of a few thousand narrow bins and outputs
    one logit per bin; a softmax turns the logits into the probability that the target
    falls in each bin. Within a bin the density is flat, so the density there is the
    bin's probability divided by its width, and the whole distribution is a histogram
    with very fine bars. ``output["criterion"]`` holds the grid, already mapped to
    heating-load units, and ``output["logits"]`` the logits, one row per design.

    The first and last bins are special: beyond the grid, TabPFN spreads their mass as
    half-normal tails, so that any value has some density. They are left out here,
    because their bins are hundreds of units wide and their mass is negligible for
    this table (below one in a hundred thousand for the held-out designs).
    """
    edges = output["criterion"].borders.detach().cpu().numpy()
    probabilities = output["logits"][row].detach().cpu().float().softmax(-1).numpy()
    density = probabilities / np.diff(edges)
    return edges[1:-1], density[1:-1]


def plot_predictive_distribution(
    output: dict, current: pd.DataFrame, answers: pd.Series, designs: list[str], names: pd.Series
) -> None:
    """TabPFN's predictive density for each listed design, one panel per design.

    ``output`` comes from ``predict(current[features], output_type="full")``, so its rows
    follow ``current``'s index. Each panel shades the central 80% band (the 10% and 90%
    quantiles TabPFN reports), marks the mean, which is the point estimate used in the
    notebooks, the median and the true value, and shows the range holding the central
    99% of the probability. The panels share one heating-load axis, so the widths of
    the distributions can be compared directly.
    """
    if not _PLOTS:
        return
    quantiles = np.asarray(output["quantiles"])
    fig, axes = plt.subplots(len(designs), 1, figsize=(7, 2.4 * len(designs)), squeeze=False, sharex=True)
    for ax, design in zip(axes[:, 0], designs):
        row = current.index.get_loc(design)
        edges, density = predictive_density(output, row)
        cumulative = np.cumsum(density * np.diff(edges))
        shown = slice(np.searchsorted(cumulative, 0.005), np.searchsorted(cumulative, 0.995) + 1)
        low, high = quantiles[0, row], quantiles[-1, row]
        ax.stairs(density[shown], edges[shown.start : shown.stop + 1], color="tab:blue", lw=1)
        middle = (edges[:-1] + edges[1:]) / 2
        band = (middle >= low) & (middle <= high)
        ax.fill_between(middle[band], density[band], step="mid", color="tab:blue", alpha=0.25, lw=0,
                        label=f"80% band, {low:.1f} to {high:.1f}")
        ax.axvline(output["mean"][row], color="tab:blue", ls="--", lw=1.5, label=f"mean {output['mean'][row]:.2f}")
        ax.axvline(output["median"][row], color="tab:blue", ls=":", lw=1.5, label=f"median {output['median'][row]:.2f}")
        ax.axvline(answers[design], color="black", lw=2, label=f"true value {answers[design]:.2f}")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("density")
        ax.set_title(design_label(current.loc[design], names), fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("heating load, kWh per square metre")
    _finish(fig)
