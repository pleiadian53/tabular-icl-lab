# ops: running this lab's experiments on a GPU pod

Heavy runs go to a RunPod pod; the laptop does data preparation, tests and smoke runs. [SkyPilot](https://skypilot.readthedocs.io) orchestrates and RunPod is the provider, the same arrangement as the sibling lab [ssl-lab](https://github.com/pleiadian53/ssl-lab).

## What a task file holds

Each experiment has a SkyPilot task in `ops/skypilot/` and a pod-side wrapper next to its script in `examples/`. The task names an ordered list of GPU types (SkyPilot falls back to the next when one is unavailable), the NGC PyTorch image, and the shared RunPod network volume mounted at `/runpod-volume`. Inputs are read from the volume; outputs go to this lab's own namespace, `/runpod-volume/tabular-icl-lab/`. The wrapper checks its inputs (including the data fingerprints the results depend on), runs a smoke version, and only then the full run.

## Run one

From the repo root, with the `sky` CLI installed (`pip install "skypilot[runpod]"`) and RunPod credentials configured:

```bash
sky launch --dryrun -c ticl-r1 ops/skypilot/r1_1.yaml                          # preview, no cloud calls
sky launch -c ticl-r1 ops/skypilot/r1_1.yaml --idle-minutes-to-autostop 30 --down -y --detach-run
sky logs ticl-r1                                                               # follow the job
rsync -Pavz ticl-r1:/runpod-volume/tabular-icl-lab/output/r1_1_residual_ceiling/ output/r1_1_residual_ceiling/
sky down ticl-r1 -y                                                            # release the pod
```

`--idle-minutes-to-autostop 30 --down` deletes the pod 30 minutes after the job ends, as a safety net if it is not released by hand. Results stay on the network volume either way.

## Experiments that need TabPFN

TabPFN-3.5's weights are downloaded from Hugging Face after a license check against the user's Prior Labs account, which needs `TABPFN_TOKEN`. The pod downloads them once into `/runpod-volume/tabular-icl-lab/models/tabpfn/` (the wrapper sets `TABPFN_MODEL_CACHE_DIR` there), and later runs read them from the volume without a token. Pass the token as a SkyPilot secret, which SkyPilot redacts from its logs and YAML output, never as a plain `--env`:

```bash
TABPFN_TOKEN="$(cat ~/.cache/tabpfn/auth_token)" \
  sky launch -c ticl-r1 ops/skypilot/r1_3.yaml --secret TABPFN_TOKEN \
  --idle-minutes-to-autostop 30 --down -y --detach-run     # first run: downloads the weights
sky launch -c ticl-r1 ops/skypilot/r1_3.yaml --secret TABPFN_TOKEN="" \
  --idle-minutes-to-autostop 30 --down -y --detach-run     # later runs: weights already on the volume
```

The NGC image's CUDA build of torch must not be replaced by a PyPI build, so the task installs `tabpfn` without dependencies, pinned to the upstream commit the lab develops against, and adds its other dependencies explicitly.

| Task | Experiment | Needs TabPFN |
|---|---|---|
| `skypilot/r1_1.yaml` | R1.1, the residual ceiling (`examples/perturbation_response/01_residual_ceiling.py`) | no |
| `skypilot/r1_3.yaml` | R1.3, round 1: TabPFN-3.5 against additivity, its no-interaction variants and tuned learners (`examples/perturbation_response/02_round1_residual_learners.py`) | yes |
