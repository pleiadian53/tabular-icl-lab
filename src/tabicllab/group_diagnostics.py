"""Find out whether a table's rows come in groups, and how much that matters.

Many tables are made of groups: designs that share one building geometry, cells from
one perturbation, patients from one hospital, readings from one machine. Rows inside a
group share most of what determines the target, so a model can be accurate on new rows
of a group it has seen and far off on a group it has not. Which kind of model to try
first, and which evaluation to trust, depends on whether such groups exist and how
strongly they shape the target. This module answers that in three steps.

1. **Propose grouping keys** (:func:`find_candidate_keys`). Any column whose values
   repeat, two or more rows per value on average, can define groups. Columns that split
   the rows into exactly the same groups are merged into one candidate, and for each
   candidate the other columns that are constant inside every group are listed: they
   are attributes of the group, not of the row. In the UCI building table, relative
   compactness and surface area split the 768 designs into the same 12 geometries, and
   wall area, roof area and height are constant within each one, which identifies the
   geometry as the entity even though no column names it. Candidates related to a
   higher-ranked one are labelled rather than treated as separate groupings: a
   *coarser* grouping merges whole groups of another candidate, as height merges the
   geometries into two; a *crossed* factor takes all its values inside every group of
   another, as orientation and glazing do within each geometry, so it varies by row,
   not by group. Only the unlabelled, primary candidates are checked by default.

2. **Measure how much the grouping explains** (:func:`variance_structure`). The share
   of the target's variance explained by the group means, the intraclass correlation
   (ICC) and the effective sample size. The ICC is the correlation between the targets
   of two rows drawn from the same group, equal to the between-group variance divided
   by the total; with ``m`` rows per group, ``n`` rows carry about as much information
   about anything that varies between groups as ``n / (1 + (m - 1) * ICC)`` independent
   rows would.

3. **Compare a random split with a grouped split** (:func:`compare_splits`). One fast
   model predicts every row twice out of fold: once with rows assigned to folds at
   random, so every test row's group is also in the training folds, and once with whole
   groups held out. The ratio of the two errors says how much harder a new group is
   than a new row of a known group. The grouped errors are then split into a part
   shared by all rows of a group, an offset, and the rest: a large offset share, or
   equivalently a high ICC of the errors, means a new group arrives with a shift its
   features do not predict.

:func:`diagnose_groups` runs all three and returns a :class:`GroupDiagnosis` whose
``report()`` states the findings in words with a suggested starting point. The
suggestion uses rough thresholds, :data:`THRESHOLDS`, and it cannot know the one thing
that decides which split is the honest one: whether the model will be used on new rows
of known groups or on new groups. The report says so.

Targets are treated as numbers. A binary target coded 0 and 1 works, with errors read
as differences in predicted probability; a multi-class target does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

#: Rough rules of thumb used by :meth:`GroupDiagnosis.report`. ``icc_negligible``: below
#: this ICC the grouping explains almost nothing. ``icc_weak``: below this ICC the
#: grouping is real but weak, rows of a group being only mildly alike. ``ratio_hard``: a grouped error at
#: least this many times the random-split error means new groups are distinctly harder.
#: ``offset_dominant``: when at least this share of the grouped squared error is a
#: per-group offset, the error on a new group is mainly a misplaced level.
#: ``few_groups``: below this many groups, anything that varies between groups is
#: poorly determined and model comparisons over groups are coarse.
THRESHOLDS = {"icc_negligible": 0.05, "icc_weak": 0.2, "ratio_hard": 1.5, "offset_dominant": 0.5, "few_groups": 20}

Key = str | Sequence[str]


def _group_codes(frame: pd.DataFrame, key: Key) -> np.ndarray:
    """Integer group code per row for a column name or a list of column names."""
    columns = [key] if isinstance(key, str) else list(key)
    return frame.groupby(columns, sort=False, dropna=False).ngroup().to_numpy()


def _key_name(key: Key) -> str:
    return key if isinstance(key, str) else " + ".join(key)


def variance_structure(y: Sequence[float], groups: Sequence) -> dict:
    """How much of the variance of ``y`` lies between the groups given by ``groups``.

    Returns the number of groups, their sizes, whether they are all the same size, the
    share of variance explained by the group means (``between_share``, the R^2 of
    predicting each row by its group's mean), the one-way ANOVA estimate of the
    intraclass correlation (``icc``, with the usual correction for unequal group sizes),
    the design effect ``1 + (m - 1) * icc`` with ``m`` the mean group size, and the
    effective sample size ``n / design_effect``. A negative ICC estimate, which happens
    when groups are less alike than chance, is reported as is and treated as 0 in the
    design effect.
    """
    y = np.asarray(y, dtype=float)
    codes = pd.factorize(pd.Series(groups), use_na_sentinel=False)[0]
    n, n_groups = y.size, int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=n_groups)
    means = np.bincount(codes, weights=y, minlength=n_groups) / sizes
    within = y - means[codes]
    total = float(((y - y.mean()) ** 2).sum())
    between_ss = float((sizes * (means - y.mean()) ** 2).sum())

    if n_groups < 2 or n_groups >= n:
        icc = float("nan")
    else:
        msb = between_ss / (n_groups - 1)
        msw = float((within**2).sum()) / (n - n_groups)
        n0 = (n - (sizes**2).sum() / n) / (n_groups - 1)
        denominator = msb + (n0 - 1) * msw
        icc = float((msb - msw) / denominator) if denominator > 0 else float("nan")
    mean_size = n / n_groups
    design_effect = 1 + (mean_size - 1) * max(icc, 0.0) if np.isfinite(icc) else float("nan")
    return {
        "n_groups": n_groups,
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
        "balanced": bool(sizes.min() == sizes.max()),
        "between_share": between_ss / total if total > 0 else float("nan"),
        "icc": icc,
        "design_effect": design_effect,
        "effective_n": n / design_effect if np.isfinite(design_effect) else float("nan"),
    }


def find_candidate_keys(
    frame: pd.DataFrame,
    target: str,
    exclude: Iterable[str] = (),
    min_rows_per_group: float = 2.0,
) -> pd.DataFrame:
    """Propose columns that could define groups, one row per distinct grouping.

    A column is a candidate when it takes at least two values and, on average, at least
    ``min_rows_per_group`` rows share each value. Columns that induce exactly the same
    partition of the rows are merged; ``columns`` lists them. ``attributes`` lists the
    other columns that are constant inside every group of the candidate: properties of
    the group rather than of the row, and a sign that the candidate is an entity such as
    a building geometry, a patient or a perturbation. The variance structure of the
    target under each grouping is attached. Rows are sorted by ICC, highest first.

    ``relation`` is empty for a primary candidate. It reads ``"coarser than K"`` when
    every group of candidate K lies inside one group of this candidate, so this
    candidate only merges K's groups, and ``"crossed with K"`` when K is ranked higher
    and every combination of the two candidates' values occurs, so this candidate
    varies inside each of K's groups like a design factor.
    """
    exclude = set(exclude) | {target}
    columns = [c for c in frame.columns if c not in exclude]
    n = len(frame)
    codes = {}
    for column in columns:
        values = frame[column]
        k = values.nunique(dropna=False)
        if k >= 2 and n / k >= min_rows_per_group:
            codes[column] = pd.factorize(values, use_na_sentinel=False)[0]

    # Merge columns that split the rows identically: same number of groups, and the
    # pair of columns has no more distinct combinations than either column alone.
    merged: list[list[str]] = []
    for column in codes:
        for group in merged:
            first = codes[group[0]]
            if first.max() == codes[column].max() and len(set(zip(first, codes[column]))) == first.max() + 1:
                group.append(column)
                break
        else:
            merged.append([column])

    rows = []
    y = frame[target].to_numpy(dtype=float)
    for group in merged:
        code = codes[group[0]]
        others = [c for c in frame.columns if c not in exclude and c not in group]
        constant = frame[others].groupby(code, sort=False).nunique(dropna=False).max() if others else pd.Series(dtype=int)
        attributes = [c for c in others if constant.get(c, 2) <= 1]
        rows.append(
            {
                "key": group[0],
                "columns": tuple(group),
                "attributes": tuple(attributes),
                **variance_structure(y, code),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("icc", ascending=False, na_position="last").reset_index(drop=True)

    relations = []
    for i, row in table.iterrows():
        relation = ""
        for j, other in table.iterrows():
            if j == i:
                continue
            if other["n_groups"] > row["n_groups"] and set(row["columns"]) <= set(other["attributes"]):
                relation = f"coarser than {other['key']}"
                break
        if not relation:
            mine = codes[row["key"]]
            for j in range(i):
                other = codes[table.at[j, "key"]]
                if len(set(zip(other, mine))) == (other.max() + 1) * (mine.max() + 1):
                    relation = f"crossed with {table.at[j, 'key']}"
                    break
        relations.append(relation)
    table.insert(3, "relation", relations)
    return table


def _default_model(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=seed)


def _prepare_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Text columns become pandas categoricals, which the default model handles natively."""
    x = frame[list(features)].copy()
    for column in x.columns:
        if x[column].dtype == object or pd.api.types.is_string_dtype(x[column]):
            x[column] = x[column].astype("category")
    return x


def compare_splits(
    frame: pd.DataFrame,
    target: str,
    key: Key,
    features: Sequence[str] | None = None,
    model: Callable[[], object] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    threads: int | None = 1,
) -> dict:
    """Out-of-fold errors of one model under a random split and under a grouped split.

    ``features`` defaults to every column except the target. Columns that only name a
    group, such as a hospital identifier, should be left out through ``features``: they
    help a random split by identifying the group and carry nothing to a new group.
    Columns that describe the group, such as a building's shape, belong in.

    ``model`` is a function returning a fresh scikit-learn-style regressor; the default
    is scikit-learn's ``HistGradientBoostingRegressor``, fast and without tuning.
    ``threads`` caps the threads native libraries may use during the fits; on small
    tables one thread is many times faster than the default, which oversubscribes
    the CPU. ``None`` leaves the libraries' own setting.

    Returns, for both splits, the mean absolute error and R^2; their ratio
    ``mae_ratio`` (grouped over random); and, for the grouped split, ``offset_share``,
    the fraction of the squared error that is a per-group offset (each group's mean
    error, weighted by group size), and ``residual_icc``, the ICC of the errors by
    group. Both are near 0 when errors are independent of the group and near 1 when a
    new group's rows are all off by the same amount.
    """
    from contextlib import nullcontext

    from sklearn.model_selection import GroupKFold, KFold
    from threadpoolctl import threadpool_limits

    features = [c for c in frame.columns if c != target] if features is None else list(features)
    make = model if model is not None else (lambda: _default_model(seed))
    x, y = _prepare_features(frame, features), frame[target].to_numpy(dtype=float)
    groups = _group_codes(frame, key)
    n_groups = int(groups.max()) + 1
    if n_groups < 2:
        raise ValueError(f"key {_key_name(key)!r} defines a single group")

    def out_of_fold(splitter, **split_kwargs) -> np.ndarray:
        prediction = np.empty_like(y)
        for train, test in splitter.split(x, y, **split_kwargs):
            fitted = make()
            fitted.fit(x.iloc[train], y[train])
            prediction[test] = fitted.predict(x.iloc[test])
        return prediction

    with threadpool_limits(threads) if threads is not None else nullcontext():
        random_prediction = out_of_fold(KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        grouped_prediction = out_of_fold(GroupKFold(n_splits=min(n_splits, n_groups)), groups=groups)

    def summary(prediction: np.ndarray) -> dict:
        error = prediction - y
        total = float(((y - y.mean()) ** 2).sum())
        return {"mae": float(np.abs(error).mean()), "r2": 1 - float((error**2).sum()) / total if total > 0 else float("nan")}

    error = grouped_prediction - y
    sizes = np.bincount(groups, minlength=n_groups)
    mean_error = np.bincount(groups, weights=error, minlength=n_groups) / sizes
    squared = float((error**2).sum())
    random_summary, grouped_summary = summary(random_prediction), summary(grouped_prediction)
    return {
        "key": _key_name(key),
        "n_groups": n_groups,
        "folds_grouped": min(n_splits, n_groups),
        "random": random_summary,
        "grouped": grouped_summary,
        "mae_ratio": grouped_summary["mae"] / random_summary["mae"] if random_summary["mae"] > 0 else float("inf"),
        "offset_share": float((sizes * mean_error**2).sum()) / squared if squared > 0 else float("nan"),
        "residual_icc": variance_structure(error, groups)["icc"],
        "predictions": {"random": random_prediction, "grouped": grouped_prediction},
    }


def _verdict(variance: dict, split: dict) -> tuple[str, str]:
    """A one-line reading of one key's numbers and a suggested starting point."""
    t = THRESHOLDS
    icc, ratio, offset = variance["icc"], split["mae_ratio"], split["offset_share"]
    if (not np.isfinite(icc) or icc < t["icc_negligible"]) and ratio < t["ratio_hard"]:
        return (
            "The grouping explains almost none of the target, and new groups are about as easy as new rows.",
            "Treat rows as independent: any strong tabular model, evaluated with an ordinary random split.",
        )
    if ratio < t["ratio_hard"] and icc < t["icc_weak"]:
        return (
            "The grouping is real but weak: rows of a group are only mildly alike, and a new group is not much harder than a new row.",
            "Model choice can treat rows as nearly independent; still evaluate with a grouped split if new groups will be met, "
            "since it costs little and the per-group offsets it reveals are the ones to watch.",
        )
    if ratio < t["ratio_hard"]:
        return (
            "Groups differ in the target, but the features predict those differences: a new group is not much harder than a new row.",
            "Standard tabular models that use the group-level features; still evaluate with a grouped split if new groups will be met.",
        )
    if offset >= t["offset_dominant"]:
        return (
            "New groups are distinctly harder, and most of their error is an offset shared by all rows of a group, which the features do not predict.",
            "For new groups: a mixed-effects model as the baseline, for its honest uncertainty and its separation of between- and within-group effects; "
            "report errors and coverage per group; plan to observe a few rows of each new group, which pins down its offset. "
            "For new rows of known groups: give the model the group, as a feature or a random effect.",
        )
    return (
        "New groups are distinctly harder, but the extra error is not mainly a shared offset: groups differ in how the features act, not only in level.",
        "Grouped evaluation, and models that let effects vary by group: random slopes in a mixed model, interactions with group-level features, or a model per group when groups are large.",
    )


@dataclass
class GroupDiagnosis:
    """The outcome of :func:`diagnose_groups`. ``candidates`` has one row per proposed
    grouping; ``checks`` maps each checked key to its :func:`compare_splits` result."""

    target: str
    n_rows: int
    candidates: pd.DataFrame
    checks: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One row per checked key: variance structure and split comparison together."""
        rows = []
        for name, split in self.checks.items():
            variance = split["variance"]
            rows.append(
                {
                    "key": name,
                    "n_groups": variance["n_groups"],
                    "icc": variance["icc"],
                    "between_share": variance["between_share"],
                    "effective_n": variance["effective_n"],
                    "mae_random": split["random"]["mae"],
                    "mae_grouped": split["grouped"]["mae"],
                    "mae_ratio": split["mae_ratio"],
                    "offset_share": split["offset_share"],
                    "residual_icc": split["residual_icc"],
                }
            )
        return pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()

    def report(self) -> str:
        """The findings in words, one paragraph per checked key."""
        lines = [f"Group diagnosis for target {self.target!r}, {self.n_rows} rows."]
        if self.candidates.empty:
            lines.append("No column repeats often enough to define groups; the rows look independent.")
            return "\n".join(lines)
        lines.append(f"{len(self.candidates)} candidate groupings, highest intraclass correlation first:")
        for row in self.candidates.itertuples(index=False):
            same = f" (same groups as {', '.join(row.columns[1:])})" if len(row.columns) > 1 else ""
            attrs = f"; constant within each group: {', '.join(row.attributes)}" if row.attributes else ""
            relation = f"; {row.relation}" if row.relation else ""
            lines.append(
                f"  - {row.key}{same}: {row.n_groups} groups, ICC {row.icc:.3f}, effective n {row.effective_n:.1f}{attrs}{relation}"
            )
        for name, split in self.checks.items():
            variance = split["variance"]
            reading, suggestion = _verdict(variance, split)
            lines += [
                "",
                f"Key {name!r}: {variance['n_groups']} groups of {variance['size_min']} to {variance['size_max']} rows.",
                f"  Group means explain {variance['between_share']:.0%} of the target's variance; ICC {variance['icc']:.3f}; "
                f"the {self.n_rows} rows are worth about {variance['effective_n']:.1f} independent rows for anything that varies between groups.",
                f"  Out-of-fold MAE {split['random']['mae']:.4g} with random folds, {split['grouped']['mae']:.4g} with whole groups held out "
                f"({split['mae_ratio']:.1f} times); {split['offset_share']:.0%} of the grouped squared error is a per-group offset "
                f"(ICC of the errors {split['residual_icc']:.3f}).",
                f"  Reading: {reading}",
                f"  Starting point: {suggestion}",
            ]
            if variance["n_groups"] < THRESHOLDS["few_groups"]:
                lines.append(
                    f"  Caution: with {variance['n_groups']} groups, effects that vary between groups are poorly determined; "
                    "report results per group and use a bootstrap over groups, not rows."
                )
        lines += [
            "",
            "Which split is the honest one depends on use: if the model will predict new rows of groups it has seen, the random split; "
            "if it will meet new groups, the grouped split.",
        ]
        return "\n".join(lines)


def diagnose_groups(
    frame: pd.DataFrame,
    target: str,
    keys: Sequence[Key] | None = None,
    exclude: Iterable[str] = (),
    features: Sequence[str] | None = None,
    n_check: int = 3,
    model: Callable[[], object] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    threads: int | None = 1,
) -> GroupDiagnosis:
    """Propose grouping keys, measure them, and compare random and grouped splits.

    ``keys`` names the groupings to check, each a column or a list of columns; by
    default the ``n_check`` primary candidates, those that are neither coarser than nor
    crossed with another candidate, with the highest ICC are checked. ``exclude``
    removes columns from the search for keys and from the features, for instance row
    identifiers or other outcomes measured alongside the target. ``features``,
    ``model``, ``n_splits``, ``seed`` and ``threads`` are passed to :func:`compare_splits`; by
    default the features are every column except the target and ``exclude``.
    """
    exclude = list(exclude)
    candidates = find_candidate_keys(frame, target, exclude=exclude)
    if keys is None:
        primary = candidates[candidates["relation"] == ""] if not candidates.empty else candidates
        keys = list(primary["key"].head(n_check)) if not primary.empty else []
    if features is None:
        features = [c for c in frame.columns if c != target and c not in exclude]
    diagnosis = GroupDiagnosis(target=target, n_rows=len(frame), candidates=candidates)
    y = frame[target].to_numpy(dtype=float)
    for key in keys:
        split = compare_splits(
            frame, target, key, features=features, model=model, n_splits=n_splits, seed=seed, threads=threads
        )
        split["variance"] = variance_structure(y, _group_codes(frame, key))
        diagnosis.checks[_key_name(key)] = split
    return diagnosis
