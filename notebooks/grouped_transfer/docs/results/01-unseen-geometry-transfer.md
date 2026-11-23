# 01 · Transfer to three unseen building geometries

*Part of [the results ledger](index.md), which holds what the characterizations add up to, the rules they follow and the settings they share. Notebook: [`01_unseen_geometry_transfer.ipynb`](../../01_unseen_geometry_transfer.ipynb). Report: [`01-unseen-geometry-transfer.json`](reports/01-unseen-geometry-transfer.json), from [`01_unseen_geometry_transfer_report.py`](../../01_unseen_geometry_transfer_report.py).*

**Question.** Given the simulated heating loads of 576 building designs on nine geometries, how well does TabPFN, untuned, estimate the heating load of 192 designs on three geometries it has never seen, compared with untuned gradient boosting; where is its error; and does removing two redundant input columns change it?

**Answer.** TabPFN-3.5 has a mean absolute error of 0.95 averaged over three seeds, against 1.90 for XGBoost, on heating loads that range from 6 to 33; it is better on every seed and on two of the three geometries. Its error is almost entirely a shift shared by all designs of a geometry, and it is largest on a two-storey geometry that lies inside the range of the context but among neighbours whose heating levels jump, not on the geometry that lies outside the context's range. TabPFN-2 does much worse on this split, 2.60, worse than XGBoost. Removing two columns that add no information makes TabPFN-3.5 worse on average, 1.18, but not on every seed, and its own seed-to-seed range is about as wide as that difference. **TabPFN transfers to unseen geometries far better than gradient boosting here, and its remaining error is a per-geometry offset.**

## What was measured

For a held-out design $i$ with true heating load $y_i$ and predicted mean $\hat y_i$:

- **MAE**, the mean of $|\hat y_i - y_i|$ over the 192 designs, the primary score, and $R^2$.
- **Mean signed error**, the mean of $\hat y_i - y_i$: negative when a model predicts too low. Per geometry, an MAE close to the size of the mean signed error means the error is a shared offset rather than scatter.
- **Band coverage and width** for TabPFN: the share of designs whose truth falls between the 10% and 90% quantiles of the predictive distribution, nominally 80%, and the mean distance between those quantiles.
- **Fit-and-predict time** on a laptop CPU.

The split is fixed: geometries 2, 6 and 10, numbered in the order they first appear in the source workbook, are held out. With three held-out geometries no bootstrap over geometries is meaningful, so no interval is attached to a difference between arms here; [02](02-grouped-structure.md) repeats the comparison over all twelve geometries, where it can be.

## What was compared

- **TabPFN-3.5**, the lab's main model, default constructor, seeds 0, 1 and 2.
- **TabPFN-2**, the same way, as the robustness arm.
- **XGBoost** with its default settings, one thread. Its fit is deterministic, so one seed suffices.
- Two **input sets**: all eight design columns, and six columns without `surface_area`, which equals wall area plus twice the roof area exactly, and `orientation`, which shows no consistent effect in the context. Cooling load, a second outcome of the same simulation, and the design identifier are never inputs.

## Results

### Headline, 192 held-out designs

| Arm | Inputs | MAE, mean over seeds (range) | $R^2$ | Mean signed error | 80% band coverage | Band width | Seconds |
|---|---|---|---|---|---|---|---|
| TabPFN-3.5 | eight | **0.95** (0.86 to 1.11) | 0.975 | −0.32 | 1.00 | 11.0 | 19.7 |
| TabPFN-3.5 | six | 1.18 (1.03 to 1.45) | 0.965 | −0.03 | 1.00 | 11.3 | 19.6 |
| TabPFN-2 | eight | 2.60 (2.38 to 2.78) | 0.845 | −1.87 | 0.62 | 7.1 | 9.5 |
| TabPFN-2 | six | 2.82 (2.77 to 2.85) | 0.805 | −2.21 | 0.67 | 8.0 | 7.6 |
| XGBoost | eight | 1.90 | 0.908 | −1.72 | | | 0.02 |
| XGBoost | six | 1.91 | 0.908 | −1.73 | | | 0.01 |

### By held-out geometry, MAE (mean signed error)

Geometries are named by relative compactness and overall height.

| Geometry | TabPFN-3.5, eight | TabPFN-3.5, six | TabPFN-2, eight | XGBoost, eight |
|---|---|---|---|---|
| 0.86 / 7.0 m, inside the context's range | 1.43 (−1.38) | 1.74 (−1.73) | 4.13 (−4.13) | 3.00 (−3.00) |
| 0.74 / 3.5 m, wall area outside the range | 0.68 (+0.67) | 0.93 (+0.93) | 1.19 (+1.02) | **0.38** (+0.15) |
| 0.64 / 3.5 m, inside the range | 0.75 (−0.26) | 0.87 (+0.72) | 2.49 (−2.49) | 2.33 (−2.33) |

TabPFN values are averaged over the three seeds.

## What this says

**The error is an offset per geometry.** For every arm and geometry the MAE is close to the size of the mean signed error. A model that has not seen a geometry has to place that geometry's overall level, and whatever it gets wrong there, all 64 designs inherit. Two designs on the same geometry share that offset, so ranking them is more reliable than the MAE suggests; the notebook shows a glazing difference of 7.5 units that XGBoost, despite its larger overall error, estimates more closely than TabPFN for exactly this reason.

**These three geometries are among the easier ones.** [02](02-grouped-structure.md) holds out each of the twelve geometries in turn; there TabPFN-3.5's MAE averages 2.92, about three times the 0.95 here, and the hardest geometries are two two-storey ones not held out on this page. The comparisons below describe these three geometries, not transfer in general.

**Difficulty follows smoothness, not range.** The out-of-range one-storey geometry is the easiest for three of the four arms, and there XGBoost beats TabPFN. One-storey heating levels in the context lie in a narrow band that changes smoothly with shape, so a small step beyond the range is safe. The hardest geometry, two storeys at compactness 0.86, lies between two context geometries whose levels differ by about six units and are themselves out of order with compactness; every arm places it too low.

**TabPFN-3.5's bands are wide rather than miscalibrated in the usual direction.** They cover every held-out design under every seed, with a mean width of about 11 units, roughly ten times the typical error. The model signals that a new geometry is uncertain, but too generously to be useful as an 80% band. TabPFN-2's bands go the other way, covering 62%, the under-coverage under shift also seen in the [out-of-the-box ledger](../../../tabpfn_capabilities/docs/results/index.md). [02](02-grouped-structure.md) asks whether a calibration using the other geometries' misses repairs both.

**Redundant columns are not invisible to TabPFN.** XGBoost is unchanged when the two columns are removed, as a tree ensemble that had ignored them should be. TabPFN-3.5 changes, worse on average and on two of three seeds, by about as much as its own seed-to-seed spread. The finding is that the input description matters to an in-context learner, not that six columns are reliably worse.

**TabPFN-2 does not transfer here.** Its error is nearly three times TabPFN-3.5's and larger than XGBoost's, with large negative offsets on the 0.86 and 0.64 geometries.

## Settings and reproducibility

- **Split:** geometries 2, 6 and 10 held out, 192 designs, with the other nine as context, 576 designs; the rule lives in `helper.grouped_split`, and geometries are numbered by `helper.load_energy_efficiency` in the order they first appear in the workbook.
- **Parity:** seed 0 of TabPFN-3.5 with eight columns reproduces the notebook's MAE, 0.873092, to six decimals.
- **Software:** TabPFN 9.2.0 as a library, checkpoints `tabpfn-v3.5-20260909` and `tabpfn-v2-regressor`; XGBoost 3.4.2, scikit-learn 1.9.1, PyTorch 2.14.1, Python 3.13.
- **Hardware:** laptop CPU (Darwin arm64); the full report runs in about three minutes.
