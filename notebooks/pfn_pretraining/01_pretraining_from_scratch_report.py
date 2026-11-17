"""Report for notebook 01: what do the prior and the model's scale each change?

Context. ``01_pretraining_from_scratch.ipynb`` pretrains a small prior-fitted network three
times: a small model on a prior of straight class boundaries, the same small model on a
richer prior that adds a random sine wave to each boundary, and a larger model trained
longer on the richer prior. One pretraining run is one draw from a noisy process, so a
single run cannot say whether a difference between two of them is real. This script runs
the notebook once per seed and compares the three models, and two classical learners, on
the same fixed evaluation draws.

Notation. An evaluation problem is a two-feature binary classification problem with a
known boundary (line, sine, or sine plus a fast wave; see ``helper.py``). Each problem has
20 fixed draws of 40 labeled points and 200 query points. For a model m, a problem and a
draw d, acc(m, d) is the fraction of the draw's query points classified correctly.

PROTOCOL (fixed in this docstring before the full run)
-------------------------------------------------------
Runs. The notebook is executed with SEED = 0, 1 and 2. Each of its three pretraining runs
starts from torch.manual_seed(SEED); the notebook's last cell writes
``docs/results/reports/01-pretraining-from-scratch-seed{SEED}.json``. The seed-0 execution
is saved into the notebook file, so the notebook shows seed 0.

Arms.
  small_line    embedding 48, 2 layers, 750 steps, line prior
  small_wave    embedding 48, 2 layers, 750 steps, wave prior
  large_wave    embedding 96, 3 layers, 2,500 steps, wave prior
  logistic      logistic regression fitted on each draw's 40 labeled points
  forest        random forest, 100 trees, random_state 0, fitted the same way
  ceiling       the classifier that knows the true boundary; it errs only where label
                noise flipped a point, so it is the best achievable accuracy (reported,
                not contrasted; added before the aggregation first ran)
All pretraining uses batch 32, AdamW at learning rate 1e-3 with a cosine decay to 0.

Aggregation. For a pretrained arm, acc(m, d) is averaged over the three seeds first;
the classical learners are deterministic. An arm's score on a problem is its mean over
the 20 draws. The spread over seeds is reported as the per-seed means.

Primary family, twelve contrasts, paired over draws within each problem:
  prior          small_wave - small_line    on line, sine, sine + fast wave
  scale          large_wave - small_wave    on each problem
  vs forest      large_wave - forest        on each problem
  vs logistic    large_wave - logistic      on each problem
Joint paired bootstrap over the 20 draws (10,000 resamples, generator seed 0, one resample
shared by all twelve contrasts), max-t simultaneous 95% intervals (``tabicllab.bootstrap``).
A difference counts only when its interval lies wholly on one side of zero. The draws are
the unit of resampling, so the intervals describe uncertainty over evaluation problems of
this kind, not over pretraining seeds; the seed spread is reported beside them.

Expectations stated before the run, to be checked rather than assumed: the wave prior
helps on the sine problem and not on the line; scale helps most where the prior already
covers the problem; on the problem outside both priors, nothing is predicted.

Run: from this directory, ``python 01_pretraining_from_scratch_report.py --run`` (executes
the notebook three times on the CPU, then aggregates) or without ``--run`` to aggregate
existing per-seed reports. ``--seeds 0`` aggregates a subset of the seeds; the report then
records ``"complete": false`` and every conclusion drawn from it is provisional. ``--smoke``
executes the notebook once with 20 steps per pretraining run and checks its report; the GPU
pod runs it before the full run (``run_r4_3_pod.sh``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01_pretraining_from_scratch.ipynb"
REPORTS = HERE / "docs" / "results" / "reports"
OUT = REPORTS / "01-pretraining-from-scratch.json"
SEEDS = [0, 1, 2]
ARMS = {
    "small_line": "small, line prior",
    "small_wave": "small, wave prior",
    "large_wave": "large, wave prior",
}
BASELINE_ARMS = {"logistic": "logistic regression", "forest": "random forest"}
CONTRASTS = [
    ("prior", "small_wave", "small_line"),
    ("scale", "large_wave", "small_wave"),
    ("vs forest", "large_wave", "forest"),
    ("vs logistic", "large_wave", "logistic"),
]


SMOKE_SEED = 99


def run_notebook(seed: int, save: bool, steps: int | None = None) -> float:
    """Execute the notebook with SEED replaced; save it in place only when ``save``.

    ``steps`` replaces every pretraining run's number of steps, for a smoke run that checks
    the notebook executes end to end without spending the full budget.
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    setup = next(c for c in nb.cells if c.cell_type == "code" and "SEED = 0" in c.source)
    setup.source = setup.source.replace("SEED = 0", f"SEED = {seed}")
    if steps is not None:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("num_steps=750", f"num_steps={steps}").replace(
                    "num_steps=2500", f"num_steps={steps}")
    start = time.time()
    NotebookClient(nb, timeout=7200, kernel_name="tabicllab", resources={"metadata": {"path": str(HERE)}}).execute()
    if save:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = [o for o in cell.outputs if not (o.output_type == "stream" and o.name == "stderr")]
        nbformat.write(nb, NOTEBOOK)
    return time.time() - start


def aggregate(seeds: list[int]) -> dict:
    from tabicllab.bootstrap import maxt_family, resample_indices

    runs = [json.loads((REPORTS / f"01-pretraining-from-scratch-seed{s}.json").read_text()) for s in seeds]
    problems = list(runs[0]["baselines"]["random forest"]["accuracy_per_draw"])

    per_draw: dict[str, dict[str, np.ndarray]] = {}
    arms: dict[str, dict] = {}
    for key, name in ARMS.items():
        by_seed = np.array([[r["models"][name]["accuracy_per_draw"][p] for p in problems] for r in runs])  # seed, problem, draw
        per_draw[key] = {p: by_seed[:, i].mean(axis=0) for i, p in enumerate(problems)}
        arms[key] = {
            "label": name,
            "parameters": runs[0]["models"][name]["parameters"],
            "train_seconds_by_seed": [r["models"][name]["train_seconds"] for r in runs],
            "accuracy": {p: float(per_draw[key][p].mean()) for p in problems},
            "accuracy_by_seed": {p: [float(x) for x in by_seed[:, i].mean(axis=1)] for i, p in enumerate(problems)},
        }
    for key, name in BASELINE_ARMS.items():
        draws = runs[0]["baselines"][name]["accuracy_per_draw"]
        per_draw[key] = {p: np.array(draws[p]) for p in problems}
        arms[key] = {"label": name, "accuracy": {p: float(np.mean(draws[p])) for p in problems}}
    from helper import ceiling_accuracy_per_draw

    ceiling = ceiling_accuracy_per_draw()
    arms["ceiling"] = {"label": "true boundary", "accuracy": {p: float(np.mean(ceiling[p])) for p in problems}}
    for a in arms.values():
        a["accuracy"]["mean"] = float(np.mean([a["accuracy"][p] for p in problems]))

    n_draws = len(per_draw["forest"][problems[0]])
    idx = resample_indices(n_draws, n_boot=10_000, seed=0)
    diffs = {
        f"{label}: {a} - {b} | {p}": per_draw[a][p] - per_draw[b][p]
        for label, a, b in CONTRASTS
        for p in problems
    }
    return {
        "protocol": "see the docstring of 01_pretraining_from_scratch_report.py",
        "seeds": seeds,
        "complete": seeds == SEEDS,
        "problems": problems,
        "evaluation": runs[0]["evaluation"],
        "environment": runs[0]["environment"],
        "arms": arms,
        "contrasts": maxt_family(diffs, idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=int, nargs="*", metavar="SEED", help="execute the notebook for these seeds first (default: all)")
    parser.add_argument("--no-aggregate", action="store_true", help="only run, do not aggregate")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, metavar="SEED",
                        help="aggregate only these seeds; the report then records itself as incomplete")
    parser.add_argument("--smoke", action="store_true",
                        help=f"execute the notebook with seed {SMOKE_SEED} and 20 steps per run, check its report, delete it")
    args = parser.parse_args()
    if args.smoke:
        path = REPORTS / f"01-pretraining-from-scratch-seed{SMOKE_SEED}.json"
        seconds = run_notebook(SMOKE_SEED, save=False, steps=20)
        report = json.loads(path.read_text())
        path.unlink()
        assert set(report["models"]) == set(ARMS.values()), report["models"].keys()
        print(f"smoke: notebook ran in {seconds:.1f} s on {report['environment']['device']}; report checked and removed")
        return
    runtimes = {}
    if args.run is not None:
        for seed in args.run or SEEDS:
            runtimes[seed] = round(run_notebook(seed, save=(seed == 0)), 1)
            print(f"seed {seed}: notebook ran in {runtimes[seed]} s", flush=True)
    if args.no_aggregate:
        return
    report = aggregate(args.seeds)
    if runtimes:
        report["notebook_runtime_s"] = runtimes
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT.relative_to(HERE)}")
    for key, a in report["arms"].items():
        print(f"{key:12s}", {p: round(v, 3) for p, v in a["accuracy"].items()})
    print("critical value", round(report["contrasts"]["critical_value"], 3))
    for name, c in report["contrasts"]["contrasts"].items():
        lo, hi = c["ci_simultaneous"]
        print(f"  {name:45s} {c['diff']:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    sys.exit(main())
