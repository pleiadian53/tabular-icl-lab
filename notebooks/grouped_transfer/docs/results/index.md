# The results ledger: transfer to unseen groups

*An append-only record of how TabPFN and its references behave when the rows to predict belong to a group that is absent from the context. This page holds what the characterizations add up to, the rules they follow, the settings they share and what they suggest next. Each characterization has its own page, which says what was asked, how it was run and what was measured, with a link from every number to the script or notebook and the report that produced it.*

> **What this ledger covers.** One public table with an exact group structure, the UCI Energy Efficiency data: 768 simulated buildings, each one of 12 geometries combined with one of 4 orientations and one of 16 glazing setups, so each geometry is a group of 64 designs. The numbers are a reference for grouped evaluation in the lab's applications, such as [perturbation response](../../../../examples/perturbation_response/docs/results/index.md), where a held-out perturbation plays the role of a held-out geometry.

---

## The characterizations

| # | Question | Answer | Page |
|---|---|---|---|
| 01 | How well does untuned TabPFN estimate heating load for three geometries it has never seen, against untuned gradient boosting, and where is its error? | Well on these three: MAE 0.95 against 1.90 for XGBoost, averaged over three seeds, on heating loads from 6 to 33. The error is almost entirely an offset shared by all designs of a geometry, largest on a two-storey geometry inside the context's range. TabPFN-2 does worse than XGBoost (2.60). Removing two redundant columns does not help. | [three unseen geometries](01-unseen-geometry-transfer.md) |
| 02 | Over all twelve geometries held out in turn, how do TabPFN, gradient boosting and models built around the grouping compare, what does a random split hide, and are 80% bands calibrated for a new geometry? | The three are among the easier geometries. Over all twelve, TabPFN-3.5 (2.92) is level with a mixed-effects model (2.68) and a two-stage model (2.80) and ahead of XGBoost (4.97) in point value, with no difference resolved by twelve groups. A random split scores the models nine to twenty-three times better and ranks XGBoost first. Overall band coverage is near nominal for every model while two geometries go almost entirely uncovered. | [the group structure](02-grouped-structure.md) |

---

## What they add up to

**On grouped data, the evaluation decides the answer.** The same models on the same table score an MAE around 0.3 when designs are held out at random, around 1 to 2 on three particular unseen geometries, and around 3 to 5 over all twelve. The ranking changes with the split as well: gradient boosting is best on random designs and worst on unseen geometries. Only the grouped evaluation answers the question a design team or an experimenter actually has, which is how good an estimate for a new group will be.

**The error on a new group is an offset, and its size is set by how surprising the group is.** A model that has not seen a geometry has to place the geometry's level, and every design inherits that placement. Where levels change smoothly with shape, even a step outside the context's range is cheap; where two neighbouring geometries differ by 13 units, every model misses whichever one is held out, in the direction of the other. TabPFN does not escape this, and two small models built around the grouping do as well as it does on average.

**Averages over rows hide failures over groups.** Bands that cover 76% to 83% of all designs, close to the nominal 80%, cover the hardest geometry not at all. For grouped data, errors and coverage need to be reported per group and intervals drawn over groups; with twelve groups, few differences between good models can be resolved.

---

## The rules every characterization follows

- **The model is used as shipped.** Default constructors, no tuning, the version and checkpoint recorded. TabPFN-3.5 is the main model; TabPFN-2 is reported beside it.
- **The split is by group.** Rows of a held-out geometry never appear in the context. A random split is reported only as the contrast that shows what it hides.
- **Every number comes from a saved report.** A report script whose docstring fixes the protocol before the full run. Each script first checks that it reproduces the number its notebook prints.
- **The group is the unit of replication.** Errors are reported per group as well as overall, intervals come from a bootstrap over groups, never over rows, and differences count only when a max-t simultaneous 95% interval lies wholly on one side of zero.
- **Several seeds where the model is seeded**, three by default.

---

## The settings every characterization shares

- **Code:** [`helper.py`](../../helper.py) holds the loader, the splits, the reference models and the plots; the bootstrap is the lab's shared `tabicllab.bootstrap`.
- **Data:** UCI Energy Efficiency (Tsanas and Xifara, 2012, DOI 10.24432/C51307, CC BY 4.0), downloaded once into the shared data lake under `energy/uci_energy_efficiency` and checked against a fixed SHA-256.
- **Software:** TabPFN 9.2.0 used as a library, checkpoints `tabpfn-v3.5-20260909` and `tabpfn-v2-regressor`; XGBoost 3.4, statsmodels 0.15, scikit-learn 1.9, PyTorch 2.14, Python 3.13.
- **Hardware:** a laptop CPU.

---

## What comes next

Three follow-ups are open, none yet committed to:

- **A few observations of the new group.** Add $k$ = 1, 2, 4 and 8 designs of the held-out geometry to the context and measure how fast the offset disappears, for TabPFN as a longer context and for the mixed model as an estimated random effect. This is the setting a design team or an experimenter faces after the first few simulations or assays of a new group, and it leads to choosing which ones to run first.
- **Group-level calibration.** Bands whose target is a stated share of groups rather than of rows, and a check on how many groups that needs.
- **The same evaluation on held-out perturbations**, where a perturbation is the group: per-group errors, per-group coverage and a bootstrap over groups in the lab's perturbation-response ledger.
