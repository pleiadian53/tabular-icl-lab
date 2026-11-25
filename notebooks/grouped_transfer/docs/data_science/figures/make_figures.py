"""Draw the figures and compute the numbers for the data-science notes.

The first note in ``docs/data_science/`` reads the shape of TabPFN's predictive
distributions in ``01_unseen_geometry_transfer.ipynb``: the peak, the median and the
mean of each distribution, the probability that lies below the true value, and whether
the true value lies above the peak. This script repeats the notebook's default run,
TabPFN-3.5 with ``random_state=0`` fitted on the 576 designs of nine geometries and
predicting the 192 designs of geometries 2, 6 and 10, then computes those quantities
for every held-out design and draws four figures:

1. three summaries of a symmetric and of a right-skewed distribution, on textbook
   shapes, before any data;
2. one design's predictive density and its cumulative distribution, read at the
   true value;
3. one design of each held-out geometry, with the same glazing setup and orientation;
4. all 64 designs of each held-out geometry, measured from each design's own peak.

Run from anywhere with the ``tabicllab`` environment and the TabPFN-3.5 license
accepted::

    python notebooks/grouped_transfer/docs/data_science/figures/make_figures.py

It writes four SVG figures and ``numbers.json`` next to itself. The TabPFN fit takes
under a minute on a laptop CPU.
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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import helper as h  # noqa: E402

warnings.filterwarnings("ignore")

# Reference categorical palette (light mode), slots 1 to 3, a pale step of slot 1 for
# areas, and text inks. Each marker also has its own line style or shape, so colour is
# never the only cue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CURVE, AREA = "#5598e7", "#cde2fb"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PEAK, MEDIAN, MEAN, TRUTH = BLUE, AQUA, ORANGE, INK
EXAMPLE = "E0505"   # the design notebook 01 follows; figures 2 and 3 use its glazing setup
SETUP = ["orientation", "glazing_area", "glazing_distribution"]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
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
        "svg.hashsalt": "data-science",
        "legend.frameon": False,
    }
)


def save(fig, name: str, rect=(0, 0, 1, 1)) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


def stairs(ax, edges: np.ndarray, density: np.ndarray, **kwargs) -> None:
    """A histogram outline drawn as one line, so matplotlib can simplify its path."""
    ax.plot(np.repeat(edges, 2)[1:-1], np.repeat(density, 2), drawstyle="default", **kwargs)


# ----------------------------------------------------------------- the TabPFN run


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict, pd.Series]:
    version = h.init_tabpfn("v3.5")
    from tabpfn import TabPFNRegressor

    frame = h.load_energy_efficiency()
    names = h.geometry_names(frame)
    history, current, answers = h.grouped_split(frame, h.HELD_OUT_GEOMETRIES)
    model = TabPFNRegressor(random_state=0)
    model.fit(history[h.DESIGN_COLUMNS], history[h.TARGET])
    batch = model.predict(current[h.DESIGN_COLUMNS], output_type="full")
    batch["version"] = version
    return frame, current, answers, batch, names


def describe(batch: dict, current: pd.DataFrame, answers: pd.Series) -> pd.DataFrame:
    """One row per held-out design: the summaries of its predictive distribution."""
    q = np.asarray(batch["quantiles"])
    rows = {}
    for design in current.index:
        row = current.index.get_loc(design)
        edges, density = h.predictive_density(batch, row)
        prob = density * np.diff(edges)
        truth = float(answers[design])
        below = float(prob[edges[1:] <= truth].sum())
        # The exact cumulative probability adds the part of the bin holding the truth;
        # the density is flat inside a bin, so that part is proportional to its length.
        k = np.searchsorted(edges, truth) - 1
        exact = below + (float(density[k] * (truth - edges[k])) if 0 <= k < len(density) else 0.0)
        peak = int(density.argmax())
        rows[design] = {
            "geometry": int(current.loc[design, "geometry"]),
            "peak": float(edges[peak]),
            "median": float(batch["median"][row]),
            "mean": float(batch["mean"][row]),
            "q10": float(q[0, row]),
            "q90": float(q[-1, row]),
            "truth": truth,
            "probability_below_truth": below,
            "probability_below_truth_exact": exact,
            "truth_above_peak": truth > float(edges[peak]),
            "inner_mass": float(prob.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ----------------------------------------------------------------- figures


def figure_summaries() -> dict:
    """Figure 1: peak, median and mean of a symmetric and of a right-skewed shape."""
    normal = stats.norm(loc=6, scale=1.6)
    skewed = stats.gamma(a=2.0, scale=2.0)   # peak 2, median about 3.36, mean 4
    x = np.linspace(0, 16, 1200)
    shapes = [
        ("Symmetric: the three summaries coincide", normal, x[normal.pdf(x) > 1e-4], 6.0),
        ("Right-skewed: peak, then median, then mean", skewed, x[x > 0], 2.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    out = {}
    for ax, (title, dist, xs, mode) in zip(axes, shapes):
        pdf = dist.pdf(xs)
        ax.plot(xs, pdf, color=CURVE, lw=2)
        median, mean = float(dist.median()), float(dist.mean())
        ax.fill_between(xs[xs >= median], pdf[xs >= median], color=AREA, lw=0)
        for label, value, colour, style in (
            ("peak", mode, PEAK, "-"),
            ("median", median, MEDIAN, "--"),
            ("mean", mean, MEAN, "-."),
        ):
            ax.axvline(value, color=colour, ls=style, lw=2, label=f"{label} {value:.2f}")
        ax.set_title(title, loc="left")
        ax.set_xlabel("value")
        ax.set_xlim(0, 14)
        ax.legend(loc="upper right")
        out[title.split(":")[0].lower()] = {"peak": mode, "median": round(median, 3), "mean": round(mean, 3)}
    axes[0].set_ylabel("density")
    axes[1].text(9.0, 0.105, "shaded: the half of the\nprobability above the median", color=INK_2, fontsize=9)
    save(fig, "fig1_three_summaries.svg")
    return out


def figure_reading_at_truth(batch: dict, current: pd.DataFrame, table: pd.DataFrame, design: str) -> None:
    """Figure 2: one design's density, and its cumulative distribution read at the truth."""
    row = current.index.get_loc(design)
    edges, density = h.predictive_density(batch, row)
    cumulative = np.concatenate([[0.0], np.cumsum(density * np.diff(edges))])
    t = table.loc[design]
    lo, hi = np.searchsorted(cumulative, 0.003), np.searchsorted(cumulative, 0.997) + 1
    e, d = edges[lo : hi + 1], density[lo:hi]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    below = e[1:] <= t["truth"]
    stairs(left, e, d, color=CURVE, lw=1)
    centres = (e[:-1] + e[1:]) / 2
    left.fill_between(centres[below], d[below], step="mid", color=AREA, lw=0)
    left.axvline(t["truth"], color=TRUTH, lw=2, label=f"truth {t['truth']:.2f}")
    left.fill_between([], [], color=AREA, label=f"shaded area {t['probability_below_truth']:.2f}")
    left.set_title(f"Predicted density of {design}", loc="left")
    left.set_xlabel("heating load (kWh/m²)")
    left.set_ylabel("density")
    left.legend(loc="upper right")

    right.plot(edges[lo : hi + 1], cumulative[lo : hi + 1], color=CURVE, lw=2)
    p = t["probability_below_truth"]
    right.plot([t["truth"], t["truth"]], [0, p], color=TRUTH, lw=2, label=f"truth {t['truth']:.2f}")
    right.plot([edges[lo], t["truth"]], [p, p], color=TRUTH, lw=1, ls=":")
    right.annotate(
        f"{p:.2f} of the probability\nlies below the truth", xy=(edges[lo], p), xytext=(edges[lo] + 0.5, p + 0.08),
        color=INK, fontsize=9,
    )
    right.set_ylim(0, 1.02)
    right.set_title("Cumulative probability: the shaded area as a height", loc="left")
    right.set_xlabel("heating load (kWh/m²)")
    right.set_ylabel("probability below this value")
    right.legend(loc="lower right")
    save(fig, "fig2_probability_below_truth.svg")


def summary_lines(ax, t: pd.Series) -> None:
    for label, key, colour, style in (
        ("peak", "peak", PEAK, "-"),
        ("median", "median", MEDIAN, "--"),
        ("mean", "mean", MEAN, "-."),
        ("truth", "truth", TRUTH, "-"),
    ):
        ax.axvline(t[key], color=colour, ls=style, lw=2, label=f"{label:<7}{t[key]:5.1f}")


def figure_one_design_each(batch, current, table, names, designs: list[str]) -> None:
    """Figure 3: one design per held-out geometry, the same glazing setup, equal x scale."""
    fig, axes = plt.subplots(1, len(designs), figsize=(15, 3.9), sharey=True)
    for ax, design in zip(axes, designs):
        row = current.index.get_loc(design)
        edges, density = h.predictive_density(batch, row)
        t = table.loc[design]
        window = (edges[:-1] >= t["median"] - 12) & (edges[1:] <= t["median"] + 20)
        idx = np.flatnonzero(window)
        e, d = edges[idx[0] : idx[-1] + 2], density[idx[0] : idx[-1] + 1]
        below = e[1:] <= t["truth"]
        ax.fill_between(((e[:-1] + e[1:]) / 2)[below], d[below], step="mid", color=AREA, lw=0)
        stairs(ax, e, d, color=CURVE, lw=1)
        summary_lines(ax, t)
        ax.plot([], [], color=AREA, lw=8, label=f"below truth {t['probability_below_truth']:.2f}")
        ax.set_xlim(t["median"] - 12, t["median"] + 20)
        ax.set_title(f"Geometry {names[int(t['geometry'])]}: {design}", loc="left")
        ax.set_xlabel("heating load (kWh/m²)")
        ax.legend(loc="upper right", prop={"family": "monospace", "size": 8.5})
    axes[0].set_ylabel("predicted density")
    save(fig, "fig3_one_design_per_geometry.svg")


def figure_all_designs(table: pd.DataFrame, names: pd.Series) -> dict:
    """Figure 4: every held-out design, measured from its own peak."""
    geometries = list(h.HELD_OUT_GEOMETRIES)
    fig, axes = plt.subplots(1, len(geometries), figsize=(15, 8.8), sharex=True)
    ranges = {}
    for ax, g in zip(axes, geometries):
        part = table[table["geometry"] == g]
        rel = part[["q10", "q90", "median", "mean", "truth"]].sub(part["peak"], axis=0).sort_values("truth")
        y = np.arange(len(rel))
        ax.hlines(y, rel["q10"], rel["q90"], color=AREA, lw=4, label="80% band")
        ax.axvline(0, color=PEAK, lw=2, label="peak (each design's own)")
        ax.scatter(rel["median"], y, s=22, marker="s", color=MEDIAN, label="median", zorder=3)
        ax.scatter(rel["mean"], y, s=26, marker="^", color=MEAN, label="mean", zorder=3)
        ax.scatter(rel["truth"], y, s=30, marker="o", color=TRUTH, label="truth", zorder=4)
        above = int(part["truth_above_peak"].sum())
        ax.set_title(
            f"{names[g]}: truth above the peak in {above} of {len(part)}\n"
            f"mean probability below truth {part['probability_below_truth'].mean():.2f}",
            loc="left",
        )
        ax.set_xlabel("distance from the design's peak (kWh/m²)")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ranges[names[g]] = {
            "truth_minus_peak": [round(float(rel["truth"].min()), 2), round(float(rel["truth"].max()), 2)],
            "q10_minus_peak_mean": round(float(rel["q10"].mean()), 2),
            "q90_minus_peak_mean": round(float(rel["q90"].mean()), 2),
        }
    axes[0].set_ylabel("one row per design, sorted by where the truth falls")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10)
    save(fig, "fig4_all_designs_from_peak.svg", rect=(0, 0.04, 1, 1))
    return ranges


# ----------------------------------------------------------------- numbers


def by_geometry(table: pd.DataFrame, names: pd.Series) -> dict:
    out = {}
    for g, part in table.groupby("geometry"):
        ordered = (part["peak"] <= part["median"]) & (part["median"] <= part["mean"])
        out[names[g]] = {
            "designs": len(part),
            "band_width_80": round(float((part["q90"] - part["q10"]).mean()), 2),
            "q10_to_median": round(float((part["median"] - part["q10"]).mean()), 2),
            "median_to_q90": round(float((part["q90"] - part["median"]).mean()), 2),
            "mean_minus_median": round(float((part["mean"] - part["median"]).mean()), 2),
            "probability_below_truth": round(float(part["probability_below_truth"].mean()), 3),
            "truth_above_peak": int(part["truth_above_peak"].sum()),
            "peak_le_median_le_mean": int(ordered.sum()),
            "truth_in_80_band": int(((part["truth"] >= part["q10"]) & (part["truth"] <= part["q90"])).sum()),
            "mae_of_mean": round(float((part["mean"] - part["truth"]).abs().mean()), 2),
            "mae_of_median": round(float((part["median"] - part["truth"]).abs().mean()), 2),
            "mae_of_peak": round(float((part["peak"] - part["truth"]).abs().mean()), 2),
            "signed_error_of_mean": round(float((part["mean"] - part["truth"]).mean()), 2),
        }
    return out


def main() -> None:
    frame, current, answers, batch, names = run()
    table = describe(batch, current, answers)
    edges = batch["criterion"].borders.detach().cpu().numpy()
    anchor = current.loc[EXAMPLE, SETUP]
    same_setup = [d for d in current.index if (current.loc[d, SETUP] == anchor).all()]
    skew_example = next(d for d in same_setup if names[int(current.loc[d, "geometry"])].startswith("0.64"))

    shapes = figure_summaries()
    figure_reading_at_truth(batch, current, table, skew_example)
    figure_one_design_each(batch, current, table, names, same_setup)
    ranges = figure_all_designs(table, names)

    widths = np.diff(edges)
    inner = widths[1:-1]
    gap = table["probability_below_truth_exact"] - table["probability_below_truth"]
    numbers = {
        "run": {
            "tabpfn_version": batch["version"],
            "random_state": 0,
            "held_out_geometries": list(h.HELD_OUT_GEOMETRIES),
            "context_designs": int((~frame["geometry"].isin(h.HELD_OUT_GEOMETRIES)).sum()),
            "held_out_designs": len(table),
        },
        "grid": {
            "bins": int(len(widths)),
            "inner_bin_width_median": round(float(np.median(inner)), 4),
            "inner_bin_width_range": [round(float(inner.min()), 4), round(float(inner.max()), 4)],
            "tail_mass_max": float(1 - table["inner_mass"].min()),
            "whole_bin_shortfall_max": round(float(gap.max()), 5),
            "whole_bin_shortfall_mean": round(float(gap.mean()), 5),
        },
        "textbook_shapes": shapes,
        "example_designs": {
            d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in table.loc[d].items()} for d in same_setup
        },
        "by_geometry": by_geometry(table, names),
        "from_peak": ranges,
        "all_designs": {
            "probability_below_truth": round(float(table["probability_below_truth"].mean()), 3),
            "mae_of_mean": round(float((table["mean"] - table["truth"]).abs().mean()), 2),
            "mae_of_median": round(float((table["median"] - table["truth"]).abs().mean()), 2),
            "truth_in_80_band": int(((table["truth"] >= table["q10"]) & (table["truth"] <= table["q90"])).sum()),
        },
    }
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    print(json.dumps({k: numbers[k] for k in ("run", "grid", "by_geometry", "from_peak", "all_designs")}, indent=1))


if __name__ == "__main__":
    main()
