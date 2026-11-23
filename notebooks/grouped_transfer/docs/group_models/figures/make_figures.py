"""Draw the figures and compute the illustrative numbers for the group-models notes.

The three notes in ``docs/group_models/`` explain the two models that use the grouping
in ``02_grouped_structure.ipynb``: the linear mixed-effects model and the two-stage
model. Their headline results come from the results ledger. This script computes the
quantities the notes use to explain how each model works, which the ledger does not
record: the fitted coefficients and standard errors on the full table, the estimated
offset of each geometry, each leave-one-geometry-out fold's variance estimates and
predicted geometry level, the Gaussian process's fitted length scales, the within-
geometry shift table, and how fast the mixed model's error falls once a few designs of
the new geometry are observed.

Run from anywhere with the ``tabicllab`` environment::

    python notebooks/grouped_transfer/docs/group_models/figures/make_figures.py

It writes four SVG figures and ``numbers.json`` next to itself and takes a few
seconds; no TabPFN fit is involved.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import helper as h  # noqa: E402

warnings.filterwarnings("ignore")

# Reference categorical palette (light mode): slots 1 to 3, and text inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
Z80 = float(norm.ppf(0.9))
FEW_SHOT_K = (0, 1, 2, 4, 8)
FEW_SHOT_DRAWS = 200

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.hashsalt": "group-models",
        "legend.frameon": False,
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


def geometry_levels(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("geometry").agg(**{c: (c, "first") for c in h.SHAPE_COLUMNS}, level=(h.TARGET, "mean"))


def full_table(frame: pd.DataFrame, names: pd.Series) -> dict:
    """The mixed model on all twelve geometries, against complete and no pooling."""
    reml = smf.mixedlm(h.MIXED_FORMULA, frame, groups=frame["geometry"]).fit(reml=True)
    ml = smf.mixedlm(h.MIXED_FORMULA, frame, groups=frame["geometry"]).fit(reml=False)
    ols = smf.ols(h.MIXED_FORMULA, frame).fit()
    tau2, sigma2 = float(reml.cov_re.iloc[0, 0]), float(reml.scale)
    offsets = pd.Series({g: float(v.iloc[0]) for g, v in reml.random_effects.items()})
    fixed_level = pd.Series(np.asarray(reml.predict(frame)), index=frame.index).groupby(frame["geometry"]).mean()
    coefficients = pd.DataFrame(
        {"estimate": reml.fe_params, "se_mixed": reml.bse_fe, "estimate_ols": ols.params, "se_ols": ols.bse}
    )
    return {
        "tau": tau2**0.5,
        "sigma": sigma2**0.5,
        "tau_ml": float(ml.cov_re.iloc[0, 0]) ** 0.5,
        "conditional_icc": tau2 / (tau2 + sigma2),
        "shrinkage_factor_64": tau2 / (tau2 + sigma2 / 64),
        "ols_residual_sd": float(np.sqrt(ols.scale)),
        "coefficients": coefficients.round(3).to_dict(orient="index"),
        "offsets": {names[g]: round(float(u), 2) for g, u in offsets.items()},
        "offsets_over_tau": {names[g]: round(float(u) / tau2**0.5, 2) for g, u in offsets.items()},
        "_fixed_level": fixed_level,
        "_offsets": offsets,
    }


def leave_one_out(frame: pd.DataFrame, names: pd.Series, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Per held-out geometry: both models' predicted level, band and errors, and few-shot updates."""
    rows, few_shot = [], {k: [] for k in FEW_SHOT_K}
    for g in sorted(frame["geometry"].unique()):
        history, current = frame[frame["geometry"] != g], frame[frame["geometry"] == g]
        y = current[h.TARGET].to_numpy()
        mixed = h.fit_predict_mixed(history, current)
        two = h.fit_predict_two_stage(history, current)

        levels = geometry_levels(history)
        centre, scale = levels[h.SHAPE_COLUMNS].mean(), levels[h.SHAPE_COLUMNS].std(ddof=0).replace(0, 1)
        kernel = ConstantKernel() * RBF(length_scale=np.ones(len(h.SHAPE_COLUMNS))) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=h.SEED)
        gp.fit((levels[h.SHAPE_COLUMNS] - centre) / scale, levels["level"])
        gp_mean, gp_sd = gp.predict(((current[h.SHAPE_COLUMNS] - centre) / scale).iloc[:1], return_std=True)
        length_scales = gp.kernel_.k1.k2.length_scale
        # With normalize_y the GP works on levels divided by their standard deviation
        # (numpy's, ddof=0); scale the white-noise term back to heating-load units. The
        # predictive sd combines this noise with the uncertainty of the smooth function.
        noise_sd = float(np.sqrt(gp.kernel_.k2.noise_level) * levels["level"].std(ddof=0))
        function_sd = float(np.sqrt(max(gp_sd[0] ** 2 - noise_sd**2, 0.0)))

        # Standard error of the fixed part x . beta_hat at each design, from the estimated
        # covariance of beta_hat; the band in helper.fit_predict_mixed leaves it out.
        fit = mixed["fit"]
        design = dmatrix(h.MIXED_FORMULA.split("~")[1], current, return_type="dataframe")[fit.fe_params.index].to_numpy()
        beta_cov = fit.cov_params().loc[fit.fe_params.index, fit.fe_params.index].to_numpy()
        fixed_se = np.sqrt(np.einsum("ij,jk,ik->i", design, beta_cov, design))

        # Stage 2's residual spread, as in helper.fit_predict_two_stage.
        cell = ["overall_height", "glazing_area", "glazing_distribution"]
        deviation = history[h.TARGET] - history.groupby("geometry")[h.TARGET].transform("mean")
        shift = deviation.groupby([history[c] for c in cell]).transform("mean")
        stage2_residual_sd = float((deviation - shift).std(ddof=1))

        mixed_error, two_error = mixed["mean"] - y, two["mean"] - y
        rows.append(
            {
                "geometry": names[g],
                "actual_level": y.mean(),
                "mixed_level": mixed["mean"].mean(),
                "mixed_tau": mixed["tau"],
                "mixed_sigma": mixed["sigma"],
                "mixed_mae": np.abs(mixed_error).mean(),
                "mixed_within_mae": np.abs(mixed_error - mixed_error.mean()).mean(),
                "mixed_band_width": (mixed["high"] - mixed["low"]).mean(),
                "mixed_fixed_part_se": float(fixed_se.mean()),
                "two_stage_level": float(gp_mean[0]),
                "two_stage_level_sd": float(gp_sd[0]),
                "two_stage_level_noise_sd": noise_sd,
                "two_stage_level_function_sd": function_sd,
                "two_stage_residual_sd": stage2_residual_sd,
                "two_stage_mae": np.abs(two_error).mean(),
                "two_stage_within_mae": np.abs(two_error - two_error.mean()).mean(),
                "two_stage_band_width": (two["high"] - two["low"]).mean(),
                **{f"length_scale_{c}": float(s) for c, s in zip(h.SHAPE_COLUMNS, length_scales)},
            }
        )

        # Observe k designs of the new geometry; the random effect's best estimate is the
        # shrunken mean residual of those k designs; score the other 64 - k designs.
        residual = y - mixed["mean"]
        tau2, sigma2 = mixed["tau"] ** 2, mixed["sigma"] ** 2
        for k in FEW_SHOT_K:
            errors = []
            for _ in range(FEW_SHOT_DRAWS):
                order = rng.permutation(len(y))
                seen, unseen = order[:k], order[k:]
                offset = tau2 / (tau2 + sigma2 / k) * residual[seen].mean() if k else 0.0
                errors.append(np.abs(residual[unseen] - offset).mean())
            few_shot[k].append(float(np.mean(errors)))
    return pd.DataFrame(rows), few_shot


def shift_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 on the full table: mean deviation from the geometry mean per (height, glazing setup)."""
    deviation = frame[h.TARGET] - frame.groupby("geometry")[h.TARGET].transform("mean")
    cell = ["overall_height", "glazing_area", "glazing_distribution"]
    return deviation.groupby([frame[c] for c in cell]).mean().rename("shift").reset_index()


def figure_levels(levels: pd.DataFrame, fixed_level: pd.Series, offsets: pd.Series, names: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for height, colour, label in ((3.5, BLUE, "one storey, 3.5 m"), (7.0, ORANGE, "two storeys, 7.0 m")):
        part = levels[levels["overall_height"] == height]
        x = part["relative_compactness"]
        ax.vlines(x, fixed_level[part.index], part["level"], color=colour, linewidth=2, alpha=0.55)
        ax.plot(x, fixed_level[part.index], linestyle="none", marker="_", markersize=14, markeredgewidth=2, color=colour)
        ax.plot(x, part["level"], linestyle="none", marker="o", markersize=8, color=colour, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    for g in levels.index:
        if abs(offsets[g]) > 4:
            ax.annotate(f"{names[g]}\n$u_g$ = {offsets[g]:+.1f}", (levels.loc[g, "relative_compactness"], levels.loc[g, "level"]),
                        xytext=(10, -4 if offsets[g] > 0 else -18), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlabel("relative compactness of the geometry")
    ax.set_ylabel("heating load, mean over the geometry's 64 designs")
    ax.set_title("Each geometry's level (dot), the fixed part (dash) and its offset $u_g$ (line)", fontsize=10, color=INK, loc="left")
    ax.legend(loc="center right")
    save(fig, "fig1_geometry_offsets.svg")


def figure_few_shot(few_shot: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ks = list(few_shot)
    per_geometry = np.array([few_shot[k] for k in ks])
    for column in per_geometry.T:
        ax.plot(ks, column, color=GRID, linewidth=1.2, marker="o", markersize=3, zorder=1)
    ax.plot(ks, per_geometry.mean(axis=1), color=BLUE, linewidth=2, marker="o", markersize=8, zorder=2,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="mean over the 12 geometries")
    for k, value in zip(ks, per_geometry.mean(axis=1)):
        ax.annotate(f"{value:.2f}", (k, value), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.plot([], [], color=GRID, linewidth=1.2, label="one held-out geometry")
    ax.set_xticks(ks)
    ax.set_xlabel("designs of the new geometry observed, $k$")
    ax.set_ylabel("MAE on its other designs")
    ax.set_title("Mixed model: error on a new geometry after observing $k$ of its designs", fontsize=10, color=INK, loc="left")
    ax.legend()
    save(fig, "fig2_few_designs.svg")


def figure_shift_table(shifts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    setups = shifts[["glazing_area", "glazing_distribution"]].drop_duplicates().sort_values(["glazing_area", "glazing_distribution"])
    position = {tuple(r): i for i, r in enumerate(setups.itertuples(index=False))}
    for height, colour, label in ((3.5, BLUE, "one storey, 3.5 m"), (7.0, ORANGE, "two storeys, 7.0 m")):
        part = shifts[shifts["overall_height"] == height]
        x = [position[(a, d)] for a, d in zip(part["glazing_area"], part["glazing_distribution"])]
        ax.plot(x, part["shift"], linestyle="none", marker="o", markersize=7, color=colour, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    for boundary in (0.5, 5.5, 10.5):
        ax.axvline(boundary, color=GRID, linewidth=1.2)
    for centre, text in ((0, "no\nwindows"), (3, "10% glazing"), (8, "25% glazing"), (13, "40% glazing")):
        ax.text(centre, 1.0, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(setups)), [str(d) for d in setups["glazing_distribution"]])
    ax.set_xlabel("glazing distribution code within each glazing area")
    ax.set_ylabel("shift from the geometry's mean")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_shift_table.svg")


def figure_level_predictions(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(table))
    for offset, prefix, spread, colour, label in (
        (-0.17, "mixed", table["mixed_tau"], BLUE, "mixed effects: fixed part, $\\pm 1.28\\,\\tau$"),
        (0.17, "two_stage", table["two_stage_level_sd"], AQUA, "two-stage: Gaussian process, $\\pm 1.28$ sd"),
    ):
        centre = table[f"{prefix}_level"]
        ax.errorbar(x + offset, centre, yerr=Z80 * spread, linestyle="none", marker="o", markersize=6, color=colour,
                    elinewidth=2, capsize=0, label=label, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.plot(x, table["actual_level"], linestyle="none", marker="D", markersize=6, color=INK, label="actual level")
    ax.axvline(5.5, color=GRID, linewidth=1.2)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_xticks(x, [name.replace(" / ", "\n") for name in table["geometry"]], fontsize=8.5)
    ax.set_ylabel("heating load, geometry mean")
    ax.set_title("Each geometry held out in turn: predicted level with an 80% interval, and the actual level",
                 fontsize=10, color=INK, loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_level_predictions.svg")


def main() -> None:
    rng = np.random.default_rng(h.SEED)
    frame = h.load_energy_efficiency()
    names = h.geometry_names(frame)
    levels = geometry_levels(frame)

    full = full_table(frame, names)
    table, few_shot = leave_one_out(frame, names, rng)
    shifts = shift_table(frame)

    figure_levels(levels, full["_fixed_level"], full["_offsets"], names)
    figure_few_shot(few_shot)
    figure_shift_table(shifts)
    figure_level_predictions(table)

    numbers = {
        "geometries": levels.assign(name=names).set_index("name").round(2).to_dict(orient="index"),
        "full_table": {k: v for k, v in full.items() if not k.startswith("_")},
        "leave_one_geometry_out": table.round(4).set_index("geometry").to_dict(orient="index"),
        "leave_one_geometry_out_means": table.drop(columns="geometry").mean().round(3).to_dict(),
        "few_shot_mixed_mae": {str(k): round(float(np.mean(v)), 3) for k, v in few_shot.items()},
        "few_shot_mixed_mae_by_geometry": {str(k): [round(x, 2) for x in v] for k, v in few_shot.items()},
        "shift_table": shifts.round(3).to_dict(orient="records"),
    }
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print(json.dumps({k: numbers[k] for k in ("full_table", "leave_one_geometry_out_means", "few_shot_mixed_mae")}, indent=1, default=float))


if __name__ == "__main__":
    main()
