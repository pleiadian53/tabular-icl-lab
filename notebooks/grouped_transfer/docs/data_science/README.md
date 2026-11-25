# Data-science notes

The grouped-transfer notebooks report numbers whose meaning rests on statistical ideas that are worth more than a line in a notebook: what a predictive distribution's shape says, which single number summarizes it best, and how to tell whether its probabilities can be trusted. These notes explain those ideas on the notebooks' own outputs, where every prediction can be compared with a known true value. The ideas are general, and the notes say so wherever a point carries over to other models and other data.

| Note | What it covers |
|---|---|
| [01 · Reading the shape of a predictive distribution](01-predictive-distribution-shape.md) | How TabPFN represents a predictive distribution as a fine histogram; the peak, the median and the mean, and why right and left skew pull them apart in a fixed order; Kelly's quantile skewness; which summary suits which cost of error; the cumulative distribution and the probability below the truth, also called the PIT value, with the notebook's code worked by hand; whether the truth lies above the peak; the three held-out geometries of notebook 01, one design and then all 64 designs each; why the mean beating the median on absolute error is a symptom; and how PIT values reveal distributions that are too wide, too narrow or shifted. |

Each note defines its symbols where they first appear. They assume what a probability density is and what a mean and a median are; everything else is built up in the text.

## The numbers and figures

Each note's quantities and figures come from a script in [`figures/`](figures/) that writes the figures as SVG and the numbers to [`figures/numbers.json`](figures/numbers.json). Note 01 needs one TabPFN-3.5 fit, under a minute on a laptop CPU with the `tabicllab` environment:

```bash
python notebooks/grouped_transfer/docs/data_science/figures/make_figures.py
```
