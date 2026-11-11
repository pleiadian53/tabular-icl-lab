# tabular-icl-lab

A research lab for **tabular foundation models and in-context learning (ICL)**. The aim is to understand how these models work, find out where they beat the methods biologists and clinicians already use, and take them a step further by fitting such models to priors built for a scientific domain.

## The idea in one paragraph

A tabular foundation model such as [TabPFN](https://github.com/PriorLabs/TabPFN) is a transformer pretrained on millions of synthetic datasets drawn from a prior over data-generating processes. At prediction time it does no gradient training at all: the labeled rows are passed in as context, and a single forward pass returns a predictive distribution for the query rows. In effect the network has been fitted to approximate Bayesian inference under its prior, which is why such models are called prior-fitted networks (PFNs). Three properties follow, and they are what this lab is interested in:

- **Small data was the home turf, and the ceiling is rising fast.** The first versions were built for tables of up to about ten thousand rows, where deep models overfit and tuning a gradient-boosted ensemble is fragile. The current release, TabPFN-3.5, accepts up to a million rows and twenty thousand features, and Prior Labs has announced ten million rows for TabPFN-4. A larger context is a capability, not an advantage: whether an in-context learner beats a well-tuned conventional model at a given size is an empirical question, and this lab treats it as one.
- **Uncertainty comes for free.** Regression returns a full predictive distribution, not a point estimate, so calibration and active learning are first-class.
- **The prior is the model.** What the network "knows" is determined by the synthetic prior it was fitted to. Change the prior and you change the inductive bias, which opens a route to domain-specific models trained without any real data.

## Research directions

**1. Strong references for small-data biology.** Many questions in computational biology are small, structured and tabular once the heavy lifting is done: a few hundred measured perturbations, a few thousand labeled protein variants, a cohort of patients. A tuning-free learner with calibrated uncertainty is the natural reference any new method should beat, and sometimes it is the right tool outright.

- **Single-cell perturbation response** *(first application)*. Predicting the effect of combining two genetic perturbations. The simplest prediction, adding the two single-gene effects, has proven very hard to beat. The question here is whether an in-context learner can predict the part that is not additive, the genetic interaction, from the single-gene measurements.
- **Protein fitness from few labels.** Protein language model embeddings as features, with the predictive distribution steering which variants to measure next.
- **Clinical and EHR tables**, the classical home of tabular learning.

**2. In-context learners as evaluation probes.** Self-supervised representations are usually judged by a linear probe. An in-context learner is a stronger probe that still needs no tuning, which makes it a useful second opinion on what a frozen representation contains, especially when labels are scarce.

**3. Domain priors.** The longer-term direction: fit a PFN to a prior built from a scientific simulator, for example gene regulatory networks under intervention, so that the inductive bias encodes how the system works rather than generic tabular regularities.

## Why tabular-icl-lab exists

tabular-icl-lab is a spin-off of its sibling project [ssl-lab](https://github.com/pleiadian53/ssl-lab), a research lab for self-supervised learning whose flagship application is single-cell perturbation response. Work there showed that a training-free additive reference beats every learned model on predicting the mean effect of unseen gene combinations, which made a strong, tuning-free small-data learner the obvious next reference to bring in. Tabular ICL turned out to be a field worth its own lab, with applications across the sibling projects: [ssl-lab](https://github.com/pleiadian53/ssl-lab) (perturbation response, evaluation probes), [protein-ml-lab](https://github.com/pleiadian53/protein-ml-lab) (protein fitness), and [genai-lab](https://github.com/pleiadian53/genai-lab) (generative models for computational biology).

## Relationship to TabPFN

This lab builds on TabPFN as a library and does not fork it. TabPFN's code is released under the Apache License 2.0. Its model weights are licensed separately: the TabPFN-2 weights under an Apache-2.0-based license with an attribution requirement, and the newer TabPFN-2.5 to 3.5 weights under non-commercial licenses that require accepting the terms on first use. Everything in this repository uses TabPFN through its public interface, and results state which model version and weights produced them.

## Layout

| Folder | What it holds |
|---|---|
| `src/tabicllab/` | the lab's code: adapters, feature builders, evaluation |
| `examples/` | one folder per application, each with its own README and results |
| `docs/` | tutorials and write-ups |
| `tests/` | unit tests |

**How code matures here.** Work starts as practical, use-case-driven scripts in `examples/<topic>/`. These are not throwaway proofs of concept: they are meant to mature, and the pieces that prove reusable are lifted into `src/tabicllab/`. A group of related scripts that settles into a coherent tool graduates into an application, and an application that stabilizes, with a fixed feature set, end-to-end integration and documentation that needs no insider context, becomes a product. The path is `examples/<topic>` → application → product, and one project can carry several applications at different stages.

## Getting started

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate tabicllab
pytest
```

The environment installs TabPFN from a local clone next to this repository (`../TabPFN`), so upstream changes can be followed with `git pull`. To install the released package instead, replace that line in `environment.yml` with `tabpfn`. On first use TabPFN opens a browser to accept the weights' license; on a headless machine, set `TABPFN_TOKEN` instead (see TabPFN's README).

## Status

Scaffold. The first experiment, an in-context learner for genetic interactions in perturbation response, is being designed.
