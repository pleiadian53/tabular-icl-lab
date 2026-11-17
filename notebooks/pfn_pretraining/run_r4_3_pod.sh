#!/usr/bin/env bash
# R4.3, pretraining a tabular in-context learner from scratch, on the GPU pod: preflight, a
# smoke run, then the notebook for the given pretraining seeds under the protocol fixed in
# 01_pretraining_from_scratch_report.py. Any preflight or smoke failure stops the job
# before the full run.
#
# Nothing is read from the volume: the notebook generates its own synthetic data. Each seed
# writes docs/results/reports/01-pretraining-from-scratch-seed{SEED}.json in the pod's copy
# of the repository; the reports are copied to this lab's namespace on the volume, to be
# fetched and aggregated on the laptop. The notebook file itself is not saved here: the
# committed notebook shows seed 0.
#
# Usage (on the pod, from the repo root):
#   SEEDS="1 2" bash notebooks/pfn_pretraining/run_r4_3_pod.sh
set -uo pipefail
VOL="${VOL:-/runpod-volume}"
OUT="${OUT:-$VOL/tabular-icl-lab/output/r4_3_pfn_pretraining}"
SEEDS="${SEEDS:-1 2}"
HERE=notebooks/pfn_pretraining
mkdir -p "$OUT"

echo "==== preflight ===="
python - <<'EOF' || { echo "PREFLIGHT FAILED: imports"; exit 1; }
import torch, sklearn, matplotlib, pandas, nbclient, nbformat, ipykernel
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("sklearn", sklearn.__version__, "nbclient", nbclient.__version__)
assert torch.cuda.is_available(), "no GPU visible"
EOF
jupyter kernelspec list | grep -q tabicllab || { echo "PREFLIGHT FAILED: no tabicllab kernel"; exit 1; }

cd "$HERE" || exit 1
echo "==== smoke (seed 99, 20 steps per pretraining run) ===="
python 01_pretraining_from_scratch_report.py --smoke > "$OUT/smoke.log" 2>&1 \
  || { echo "SMOKE FAILED"; tail -30 "$OUT/smoke.log"; exit 1; }
tail -1 "$OUT/smoke.log"

echo "==== full run, seeds $SEEDS ===="
# shellcheck disable=SC2086
python -u 01_pretraining_from_scratch_report.py --run $SEEDS --no-aggregate 2>&1 | tee "$OUT/run.log"
status=${PIPESTATUS[0]}
cp docs/results/reports/01-pretraining-from-scratch-seed*.json "$OUT/"
ls -la "$OUT"
echo "R4.3 exit status $status. Fetch: $OUT/"
exit $status
