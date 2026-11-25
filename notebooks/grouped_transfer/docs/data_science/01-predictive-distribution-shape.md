# 01 · Reading the shape of a predictive distribution

*Part of [data-science notes](README.md). The distributions are the ones TabPFN-3.5 predicts in section 5 of [`01_unseen_geometry_transfer.ipynb`](../../01_unseen_geometry_transfer.ipynb), for the 192 designs of the three held-out building geometries; the notebook's summary table of their shapes is the starting point. Every number below, and every figure, comes from [`figures/make_figures.py`](figures/make_figures.py), which repeats the notebook's default run and saves the numbers in [`figures/numbers.json`](figures/numbers.json).*

A model that returns a whole distribution for each prediction gives far more than a single estimate, but only to a reader who knows what to look at. This note reads the distributions TabPFN predicts for new building designs and asks three questions of each one. Which single number best summarizes it? Is it symmetric or skewed, and in which direction? And where, inside it, did the true value land? The answers use ideas that hold for any predictive distribution, from a weather forecast to a delivery-time estimate: the peak, the median and the mean, skewness, the cumulative distribution, and calibration. The building table makes them concrete, because the true value of every prediction is known.

---

## 1. Why a distribution and not a number

A point estimate says "the heating load of this design is about 15.6". A predictive distribution says how plausible every possible value is: perhaps values between 12 and 15 are very plausible, values near 20 are possible, and values near 40 are conceivable but rare. Two designs can share the same point estimate and differ completely in how far the truth might be from it, and only the distribution tells them apart.

The notebook's task makes this matter. A design team wants heating-load estimates for buildings on geometries that were never simulated, and the model has learned from other geometries only. How sure it is depends on how far the new geometry lies from the ones it knows, and the distribution is where it says so.

## 2. How TabPFN represents a distribution

TabPFN does not assume the target follows a normal distribution or any other named family. It cuts the heating-load axis into $K = 5000$ bins with fixed edges $b_0 < b_1 < \dots < b_K$, and for each design it outputs one probability per bin, $\pi_k$ for bin $k$, with $\sum_{k=1}^{K} \pi_k = 1$. Inside bin $k$, which runs from $b_{k-1}$ to $b_k$, the predicted density is flat:

$$p(y) = \frac{\pi_k}{b_k - b_{k-1}} \quad \text{for } b_{k-1} \le y < b_k .$$

Here $y$ is a possible heating load and $p(y)$ the predicted density at that value: the bin's probability divided by its width, so that the area of each bar equals its probability. The result is a histogram with very fine bars. In this run the inner bins are 0.031 kWh/m² wide; the first and last bins stretch over more than two hundred units to cover the tails, but they hold less than one hundred-thousandth of the probability for any held-out design, so the figures leave them out. The notebook describes the grid, and why the curves look ragged at the scale of a few bins, in more detail.

Everything in this note is computed from the $\pi_k$ and the edges, which the notebook obtains with `predict(..., output_type="full")` and turns into densities with `helper.predictive_density`.

## 3. Three summaries of one distribution, and what skew does to them

### The peak, the median and the mean

A distribution has several natural "centres", and each answers a different question.

- **The peak**, which statisticians call the **mode**, is the single most likely value: the place where the density $p(y)$ is highest. For TabPFN's histogram it is the tallest bar. This note writes it $y_{\mathrm{peak}}$.
- **The median** $m$ splits the probability in half: the predicted chance that the heating load is below $m$ is 0.5, and the chance that it is above is also 0.5.
- **The mean** $\mu$ is the probability-weighted average of all values. For the histogram it is $\mu = \sum_{k=1}^{K} \pi_k c_k$, where $c_k$ is the centre of bin $k$. The mean is the point estimate the notebook reports for every design.

For a **symmetric** distribution, such as the normal bell curve, the three coincide. The left panel of figure 1 shows one: peak, median and mean all sit at 6.

![Two textbook distributions. Left, a symmetric bell curve with peak, median and mean all at 6. Right, a right-skewed distribution with its peak at 2, its median at 3.36 and its mean at 4; the half of the probability above the median is shaded and reaches far to the right.](figures/fig1_three_summaries.svg)

*Figure 1. Left: a symmetric distribution, where the three summaries coincide. Right: a right-skewed distribution (a gamma distribution with shape 2 and scale 2), where they separate in a fixed order: peak 2.00, median 3.36, mean 4.00. The shading marks the half of the probability above the median.*

### Right skew and left skew

A distribution is **right-skewed**, or positively skewed, when it has a long tail stretching to the right, toward large values, and a short one on the left. It is **left-skewed**, or negatively skewed, when the long tail stretches to the left. The direction is named after the tail, not after the hump: a right-skewed distribution has its hump on the left and its tail on the right, as in the right panel of figure 1. Throughout this note, "the right tail" means the long thin part of a right-skewed distribution.

The classic example is household income. Most households earn something near the typical income, and a few earn many times more. The distribution has a hump around ordinary incomes and a long tail of very high ones, so it is right-skewed, and this is why the mean income of a country is higher than its median income.

### Why skew pulls the three apart in a fixed order

In a right-skewed distribution the three summaries line up as peak, then median, then mean. The reason is the way each one responds to values far out in the tail.

- The peak ignores the tail entirely; it only asks where the density is highest, which is in the hump.
- The median counts values but does not weigh them. A value in the far tail counts as "one value above the median", exactly like a value just above it. The tail still pushes the median right, because the tail holds probability that must be balanced by probability on the left, but only by the amount the count requires.
- The mean weighs every value by its distance. A value far out in the tail pulls the mean toward itself in proportion to how far out it is.

A small example shows the difference. Five households earn 30, 40, 50, 60 and 1000 thousand a year. The median is 50, the middle value. The mean is $(30 + 40 + 50 + 60 + 1000)/5 = 236$. Replace 1000 by 100 and the median stays 50 while the mean falls to 56. The single far value moved the mean by 180 and the median not at all.

For a left-skewed distribution the order reverses: mean, then median, then peak, reading from left to right.

### A number for skew

Comparing distances between quantiles gives a simple measure of skew that needs no assumptions about the shape. Write $q_{0.1}$ and $q_{0.9}$ for the 10% and 90% quantiles, the values below which the predicted probability is 0.1 and 0.9. For a symmetric distribution the median lies halfway between them, so the distance from $q_{0.1}$ up to $m$ equals the distance from $m$ up to $q_{0.9}$. For a right-skewed distribution the upper distance is longer. **Kelly's skewness** turns this into one number between −1 and 1:

$$\text{Kelly skewness} = \frac{(q_{0.9} - m) - (m - q_{0.1})}{q_{0.9} - q_{0.1}} .$$

It is 0 for a symmetric distribution, positive for right skew and negative for left skew. Section 6 uses it on the three held-out geometries.

### Which summary to report depends on the cost of an error

Each summary is the best single guess under a particular way of scoring errors, if the distribution is right.

- If an error costs its **absolute size**, $|y - a|$ for a guess $a$ and a true value $y$, the median is the best guess: it gives the smallest expected absolute error. The reason fits in one sentence. Moving the guess a little to the right makes it closer to every outcome above it and further from every outcome below it, so the move pays off as long as more than half the probability lies above the guess, and stops paying off exactly at the median.
- If an error costs its **square**, $(y - a)^2$, as in least squares, the mean is the best guess.
- If only an **exact hit** counts, the peak is the best guess, since it is the most likely single value.

The notebook scores its estimates by mean absolute error, MAE. On that score, a distribution that described the truth correctly would make the median the better of the two. Section 7 comes back to this, because on one geometry the opposite happens, and that turns out to be informative.

## 4. Where did the truth land? The probability below the truth

### The cumulative distribution

The **cumulative distribution function** $F$ of a predicted distribution gives, for every value $y$, the predicted probability that the outcome is at most $y$:

$$F(y) = \int_{-\infty}^{y} p(u) \mathrm{d}u .$$

Here $u$ is just the variable of integration, running over all values up to $y$. In words, $F(y)$ is the area under the density curve to the left of $y$. It rises from 0 on the far left to 1 on the far right, and it is steep wherever the density is high. The median is the value where $F$ reaches 0.5, and the 10% and 90% quantiles are where it reaches 0.1 and 0.9.

Evaluating $F$ at the true heating load $y_{\mathrm{true}}$ answers the question "how much of its probability did the model put below what actually happened?" This number,

$$u = F(y_{\mathrm{true}}) ,$$

is called the **probability integral transform**, or PIT value, of the prediction. The notebook calls it the *probability below truth*.

![Left: the predicted density of design E0521, a sharp peak near 13.7 with a long right tail; the area to the left of the true value 16.73 is shaded and equals 0.75. Right: the cumulative probability of the same distribution, an S-shaped curve; at the true value 16.73 it reaches 0.75.](figures/fig2_probability_below_truth.svg)

*Figure 2. The same quantity seen two ways, for design E0521 on the 0.64 geometry. Left: the shaded area under the predicted density, to the left of the true value 16.73, is 0.75. Right: the cumulative distribution turns that area into a height, read off where the vertical line at the truth meets the curve.*

### How to read it

- **About 0.5**: the truth landed near the predicted median. The prediction was centred on what happened.
- **Close to 1**: almost all the probability lay below the truth. The model expected a lower value; the truth sits in its right tail.
- **Close to 0**: the model expected a higher value; the truth sits in its left tail.

One value of $u$ says little on its own, since even a perfect model is sometimes surprised. Its power comes from many predictions together, in section 7.

### How the notebook computes it

The notebook's function is:

```python
def probability_below_truth(design):
    edges, density = predictive_density(batch, current.index.get_loc(design))
    return (density * np.diff(edges))[edges[1:] <= answers[design]].sum()
```

`current.index.get_loc(design)` finds the row of the design in TabPFN's output. `density * np.diff(edges)` multiplies each bar's height by its width, which gives back the bin probabilities $\pi_k$. `edges[1:]` is the upper edge of every bin, so `edges[1:] <= answers[design]` selects the bins that lie entirely below the truth, and `.sum()` adds their probabilities.

A four-bin version makes the steps visible. With edges 10, 11, 12, 13, 14 and densities 0.1, 0.4, 0.3, 0.2, every bin is 1 wide, so the probabilities are also 0.1, 0.4, 0.3, 0.2. For a true value of 12.5, the upper edges 11, 12, 13, 14 give the selection True, True, False, False, and the sum is $0.1 + 0.4 = 0.5$.

The example also shows what the shortcut leaves out. The truth 12.5 lies inside the third bin, and half of that bin's 0.3 also lies below it, so the exact value of $F(12.5)$ is 0.65. The function counts only whole bins. Because the density is flat inside a bin, the exact value adds the fraction of the bin that lies below the truth:

$$F(y_{\mathrm{true}}) = \sum_{k: b_k \le y_{\mathrm{true}}} \pi_k + \pi_{j} \frac{y_{\mathrm{true}} - b_{j-1}}{b_j - b_{j-1}} ,$$

where $j$ is the bin that contains $y_{\mathrm{true}}$. With TabPFN's narrow bins the missing piece is small: over the 192 held-out designs it averages 0.002 and is at most 0.013. It is largest when the truth falls right at the peak, where each bin carries the most probability; for design E0505 the shortcut gives 0.553 and the exact value is 0.559. None of the conclusions below depends on it.

## 5. Is the truth above the peak?

The notebook's second shape column asks a yes-or-no question:

```python
def truth_above_peak(design):
    edges, density = predictive_density(batch, current.index.get_loc(design))
    return answers[design] > edges[density.argmax()]
```

`density.argmax()` is the position of the tallest bar, and `edges[...]` its lower edge, which stands in for the peak $y_{\mathrm{peak}}$. The function returns True when the truth lies above it.

Averaged over the 64 designs of a geometry, a True/False column becomes a fraction, because True counts as 1 and False as 0. A value of 1.00 means the truth was above the peak for every design.

The column complements the probability below the truth. The PIT value says how far into the distribution the truth landed; this column says on which side of the most likely value it landed. For a right-skewed distribution, "truth above the peak, again and again" means the truth keeps landing in the right tail.

One caution comes from the ragged texture of the histogram. The tallest bar is the tallest spike, which can sit a few bins away from where a smoothed curve would peak. For a distribution with a sharp hump this hardly matters. For a wide, flat-topped distribution the tallest spike can sit almost anywhere on the plateau, and the peak becomes a poor summary; section 6 meets one.

## 6. The three held-out geometries

The notebook holds out three geometries and predicts all 64 designs of each. They are named by relative compactness and height: **0.86 / 7.0 m**, a two-storey geometry inside the range of the training geometries; **0.74 / 3.5 m**, a one-storey geometry just outside that range; and **0.64 / 3.5 m**, a one-storey geometry inside it.

### The numbers

Each row averages over the 64 designs of one geometry. Distances are in kWh/m².

| Geometry | 80% band width, $q_{0.9} - q_{0.1}$ | $m - q_{0.1}$ | $q_{0.9} - m$ | $\mu - m$ | Probability below truth | Truth above peak | Peak ≤ median ≤ mean |
|---|---|---|---|---|---|---|---|
| 0.86 / 7.0 m | 14.74 | 7.41 | 7.33 | 0.06 | 0.61 | 44 of 64 | 9 of 64 |
| 0.74 / 3.5 m | 6.69 | 2.63 | 4.06 | 0.50 | 0.49 | 39 of 64 | 39 of 64 |
| 0.64 / 3.5 m | 9.41 | 2.53 | 6.87 | 1.29 | 0.71 | 64 of 64 | 58 of 64 |

Kelly's skewness, computed from the averaged distances in the table, is about 0.00 for the 0.86 geometry, 0.21 for the 0.74 geometry and 0.46 for the 0.64 geometry. So the two-storey geometry gets symmetric distributions, and both one-storey geometries get right-skewed ones, the 0.64 geometry much more strongly.

### One design of each geometry

Figure 3 shows one design of each geometry, all three with the same glazing setup and orientation as E0505, the design the notebook follows: 25% glazing in distribution 5, facing orientation 2. Each panel spans the same 32 units, so the widths can be compared directly.

![Three predicted densities. Left, design E0489 on the 0.86 geometry: a wide, low, ragged plateau from about 18 to 35, with median 27.3, mean 27.4, peak 28.0 and truth 29.7. Middle, design E0505 on the 0.74 geometry: a narrow hump near 12 with a right tail; peak, median, mean and truth all within 0.4 of each other. Right, design E0521 on the 0.64 geometry: a sharp hump near 13.7 with a long right tail; peak 13.7, median 14.2, mean 15.6, truth 16.7.](figures/fig3_one_design_per_geometry.svg)

*Figure 3. One design per held-out geometry, the same glazing setup. The shaded area is the probability below the truth. Each panel spans 32 kWh/m², so the widths are comparable.*

- **E0521, 0.64 geometry, on the right.** A sharp hump with a long right tail: right-skewed. The four lines are in the textbook order for right skew, peak 13.7, median 14.2, mean 15.6, and then the truth, 16.7, beyond all three. The truth sits in the right tail, with 0.75 of the probability below it.
- **E0505, 0.74 geometry, in the middle.** Also a hump with a right tail, but the truth, 12.1, landed at the hump, with peak, median and mean all within 0.4 of it. The probability below the truth is 0.55.
- **E0489, 0.86 geometry, on the left.** A wide, ragged plateau with no clear hump: the 80% band runs from 19.8 to 34.7. The median and the mean nearly coincide, 27.3 and 27.4, as they should for a symmetric shape, and the tallest spike, 28.0, sits on the plateau a little to their right. The truth, 29.7, is above all three, with 0.69 of the probability below it.

### All 64 designs of each geometry

Figure 4 repeats the comparison for every design. Each row is one design, measured from its own peak, so the vertical line at 0 is "the peak" for every row and anything to its right is higher than the most likely value. The pale bar is the design's 80% band, from $q_{0.1}$ to $q_{0.9}$. Rows are sorted by where the truth falls.

![Three panels of 64 rows each. In each row, a pale horizontal bar marks the 80% band, a square the median, a triangle the mean and a black dot the truth, all measured from the design's own peak at zero. Left, 0.86 geometry: bands from about minus 8 to plus 7; truths spread from minus 2 to plus 3, on both sides of the peak. Middle, 0.74 geometry: bands from about minus 2.5 to plus 4; truths in a tight column near zero. Right, 0.64 geometry: bands from about minus 2 to plus 7.5; every truth between plus 1.8 and plus 3.6, to the right of the peak.](figures/fig4_all_designs_from_peak.svg)

*Figure 4. Every held-out design, measured from its own peak (the blue line at 0). Pale bar: the 80% band. Square: median. Triangle: mean. Dot: the truth. All three panels share one horizontal scale.*

**0.64 geometry, right panel.** Two things hold for every row. The band reaches much further right of the peak than left, about 7.5 units against 1.9 on average, which is the right skew. And every truth lies to the right of the peak, between 1.8 and 3.6 units above it; none falls on the left. This is the 64 of 64 in the table. Within almost every row the order is the same: square nearest the peak, then triangle, then dot. The mean is the summary closest to the truth on every design, because the right tail pulls it toward where the truth is.

**0.74 geometry, middle panel.** The bands are just as right-skewed, but the truths form a tight column around the peak, from 0.4 below it to 1.35 above. They fall on both sides, 39 above and 25 below, and the median sits on top of them. Here the right tail is probability the model spent on outcomes that did not happen.

**0.86 geometry, left panel.** The bands are wide and centred: about 8 units below the peak and 7 above. The truths spread from 2 below the peak to 3 above. The squares and triangles mostly sit together, since the shape is symmetric. In the upper rows they sit on the peak, but in the lower third of the panel they sit 1 to 2 units to its left, and the bands there run from about 10 below the peak to 6 above it. In those designs the tallest spike landed right of the middle of the plateau. That is the unstable peak described in section 5: on a flat-topped distribution the tallest spike lands at different places from design to design, so "above the peak" means little here, and the 44 of 64 should not be read as a lean. What does matter on this geometry is the level. The mean is too low by 1.34 on average while its MAE is 1.36, so almost every design is missed by about the same amount in the same direction: the model has not placed the geometry's overall heating level, and the wide distribution is how it says so.

## 7. What the shape says, and what it does not

### Skew is not bias

The 0.64 and 0.74 geometries get distributions of the same kind, a hump with a long right tail. Looking at the shapes alone, nothing separates them. Only the truth does: on the 0.74 geometry it lands at the hump, on the 0.64 geometry in the tail, every time. A skewed predictive distribution is a statement about what the model thinks might happen, and it can be honest or not; whether the truth keeps landing on one side is a separate question, answered by the PIT values and the above-the-peak counts, never by the shape.

### The mean beat the median on absolute error, and that is a symptom

Section 3 showed that, if the predicted distribution were correct, the median would be the best guess under absolute error. The notebook scores by MAE, and the outcome differs by geometry:

| Geometry | MAE of the mean | MAE of the median | MAE of the peak | Mean signed error of the mean |
|---|---|---|---|---|
| 0.86 / 7.0 m | 1.36 | 1.47 | 1.54 | −1.34 |
| 0.74 / 3.5 m | 0.51 | 0.18 | 0.24 | +0.48 |
| 0.64 / 3.5 m | 0.75 | 2.00 | 2.64 | −0.71 |
| all 192 designs | 0.87 | 1.22 | | |

On the 0.74 geometry the median wins clearly, 0.18 against 0.51, as it should when the truth sits in the middle of the distribution. The mean pays for the right tail it was pulled toward: it is too high by 0.48 on average.

On the 0.64 geometry the median loses badly, 2.00 against 0.75. That cannot happen often under a correct distribution, so it is evidence that the distribution is off in a specific direction: the truth lands in the right tail far more often than the distribution says it should. The mean wins only because the tail drags it toward where the truth keeps landing. It is right for the wrong reason, and it is still too low by 0.71 on average.

Over all 192 designs the mean wins, 0.87 against 1.22, mostly because of the 0.64 geometry. The general point: when the summary that the scoring rule should favour loses, look at where the truths fall inside the distributions before choosing the other summary.

### Calibration: what the PIT values should look like

A predictive distribution is **calibrated** when its probabilities match how often things happen: outcomes it gives a 10% chance of falling below some value should fall below it about 10% of the time, and so on for every level. Calibration has a clean test through the PIT values. If the outcome really follows the predicted distribution, then $u = F(y_{\mathrm{true}})$ is uniformly distributed between 0 and 1: about 10% of the predictions have $u$ below 0.1, about half have $u$ below 0.5, about 10% have $u$ above 0.9. A histogram of PIT values over many predictions should therefore be flat. Common departures have recognizable shapes:

- most values near 0.5, with few near 0 or 1: the distributions are **too wide**, and the model is less certain than it needs to be;
- many values near 0 and near 1: the distributions are **too narrow**, and the model is overconfident;
- values piling up toward 1, or toward 0: the distributions are **shifted**, and the model predicts too low, or too high.

The 192 designs show two of these at once. Every one of the 192 truths lies inside its 80% band, so every PIT value is between 0.1 and 0.9, where a calibrated model would have put about 38 outside: the bands are too wide here. And on the 0.64 geometry the PIT values pile toward the upper end, averaging 0.71: the distributions are also shifted. A distribution can be too wide and off-centre at the same time.

Three geometries cannot settle how calibrated TabPFN is for new geometries in general. The 64 designs of one geometry are not 64 independent checks, since they share one geometry and their errors share one offset, so the evidence is closer to three checks than to 192. Notebook 02 measures the coverage of the 80% bands with each of the twelve geometries held out in turn, and the [results ledger](../results/index.md) records it.

## 8. Reading any predictive distribution

The same questions apply to any model that returns distributions, whatever the domain.

1. **Look at the shape before choosing a summary.** For a symmetric distribution the peak, the median and the mean agree. For a skewed one they separate, in the order peak, median, mean for right skew and the reverse for left skew, and the distance between them tells how strong the skew is.
2. **Choose the summary to fit the cost of an error.** Absolute cost calls for the median, squared cost for the mean, all-or-nothing for the peak, provided the distribution is right.
3. **Treat the peak of a ragged or flat-topped histogram with care.** It can move a long way for reasons that have nothing to do with the data.
4. **Locate the truth inside the distribution, not only its distance from the point estimate.** The probability below the truth, $F(y_{\mathrm{true}})$, says where it landed; over many predictions, a flat histogram of these values means calibrated distributions, a hump means too wide, a U shape means too narrow, and a pile at one end means a shift.
5. **Count independent checks honestly.** Predictions that share a group share its errors, so a calibration check over many rows of a few groups is a check over a few groups.

## Reproducing the numbers and figures

The script repeats the notebook's default run, TabPFN-3.5 with `random_state=0` fitted on the 576 designs of nine geometries, predicts the 192 designs of geometries 2, 6 and 10, computes every quantity above and draws the four figures. It runs in under a minute on a laptop CPU with the `tabicllab` environment and the TabPFN-3.5 license accepted:

```bash
python notebooks/grouped_transfer/docs/data_science/figures/make_figures.py
```
