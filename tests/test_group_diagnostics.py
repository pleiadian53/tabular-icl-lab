"""Unit tests for the group diagnostics: key discovery, variance structure, split comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tabicllab.group_diagnostics import (
    compare_splits,
    diagnose_groups,
    find_candidate_keys,
    variance_structure,
)


def grouped_table(n_groups: int = 30, per_group: int = 16, offset_sd: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """Groups with a descriptor, a label and an offset the descriptor does not predict,
    crossed with a row-level factor and a continuous row feature."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), per_group)
    size = rng.uniform(1, 2, n_groups)[group]              # group attribute, unique per group
    kind = (np.arange(n_groups) % 3)[group]                # coarser: merges groups into 3
    setting = np.tile(np.arange(4), n_groups * per_group // 4)   # crossed with the groups
    x = rng.normal(size=group.size)
    offset = rng.normal(scale=offset_sd, size=n_groups)[group]
    y = 2 * x + setting + offset + rng.normal(scale=0.3, size=group.size)
    return pd.DataFrame({"label": [f"g{g}" for g in group], "size": size, "kind": kind, "setting": setting, "x": x, "y": y})


def test_variance_structure_on_a_hand_example() -> None:
    v = variance_structure([1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"])
    assert v["n_groups"] == 2 and v["balanced"]
    assert np.isclose(v["between_share"], 1.0) and np.isclose(v["icc"], 1.0)
    assert np.isclose(v["effective_n"], 2.0)                # two groups of identical copies


def test_variance_structure_matches_the_anova_formula_for_balanced_groups() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(8), 5)
    y = rng.normal(size=8)[groups] + rng.normal(size=40)
    means = pd.Series(y).groupby(groups).transform("mean").to_numpy()
    msb = (5 * (pd.Series(y).groupby(groups).mean() - y.mean()) ** 2).sum() / 7
    msw = ((y - means) ** 2).sum() / 32
    v = variance_structure(y, groups)
    assert np.isclose(v["icc"], (msb - msw) / (msb + 4 * msw))
    assert np.isclose(v["effective_n"], 40 / (1 + 4 * max(v["icc"], 0)))


def test_candidates_merge_identical_partitions_and_label_relations() -> None:
    table = find_candidate_keys(grouped_table(), "y")
    top = table.iloc[0]
    assert set(top["columns"]) == {"label", "size"}         # same 30 groups, merged
    assert top["relation"] == "" and top["n_groups"] == 30
    assert "kind" in top["attributes"]                       # constant within each group
    relation = dict(zip(table["key"], table["relation"]))
    assert relation["kind"].startswith("coarser than")
    assert relation["setting"].startswith("crossed with")
    assert "x" not in relation                               # continuous, no repeats


def test_an_unpredictable_offset_is_flagged_as_offset_dominated() -> None:
    frame = grouped_table()
    diagnosis = diagnose_groups(frame, "y", exclude=["label"])
    assert list(diagnosis.checks) == ["size"]                # only the primary candidate
    row = diagnosis.summary().loc["size"]
    assert row["icc"] > 0.5
    assert row["mae_ratio"] > 1.5 and row["offset_share"] > 0.5
    assert "offset shared by all rows" in diagnosis.report()


def test_independent_rows_are_read_as_independent() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"batch": np.repeat(np.arange(20), 10), "x": rng.normal(size=200)})
    frame["y"] = 2 * frame["x"] + rng.normal(scale=0.5, size=200)
    diagnosis = diagnose_groups(frame, "y", keys=["batch"], features=["x"])
    row = diagnosis.summary().loc["batch"]
    assert abs(row["icc"]) < 0.1 and row["mae_ratio"] < 1.3
    assert "Treat rows as independent" in diagnosis.report()


def test_group_differences_explained_by_a_feature_do_not_make_new_groups_harder() -> None:
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(40), 15)
    z = np.linspace(0, 4, 40)[groups]
    frame = pd.DataFrame({"z": z, "x": rng.normal(size=groups.size)})
    frame["y"] = 3 * frame["z"] + frame["x"] + rng.normal(scale=0.3, size=groups.size)
    split = compare_splits(frame, "y", "z", model=LinearRegression)
    assert variance_structure(frame["y"], groups)["icc"] > 0.8
    assert split["mae_ratio"] < 1.2 and split["offset_share"] < 0.2
