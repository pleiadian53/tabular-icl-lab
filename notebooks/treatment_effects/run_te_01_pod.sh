#!/usr/bin/env bash
# Treatment effects, notebook 01, on the GPU pod: preflight, a smoke execution (TE_SMOKE=1,
# 300 members, a light TabPFN), then the full execution. Any preflight or smoke failure stops
# the job before the full run.
#
# Nothing is read from the volume except the TabPFN-3.5 weights, which live in this lab's
# namespace. The executed notebook and the logs are copied to $OUT, to be fetched to the
# laptop, where the executed notebook replaces the unexecuted one.
#
# Usage (on the pod, from the repo root):
#   bash notebooks/treatment_effects/run_te_01_pod.sh
set -uo pipefail
VOL="${VOL:-/runpod-volume}"
OUT="${OUT:-$VOL/tabular-icl-lab/output/te_01}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$VOL/tabular-icl-lab/models/tabpfn}"
export TABPFN_NO_BROWSER=1
HERE=notebooks/treatment_effects
NOTEBOOK=01_effects_with_a_known_answer.ipynb
WEIGHTS="$TABPFN_MODEL_CACHE_DIR/tabpfn-v3.5-20260909.safetensors"
mkdir -p "$OUT"

echo "==== preflight ===="
[ -f "$WEIGHTS" ] || { echo "PREFLIGHT FAILED: no TabPFN-3.5 weights at $WEIGHTS"; exit 1; }
python - <<'EOF' || { echo "PREFLIGHT FAILED: imports"; exit 1; }
import torch, tabpfn, xgboost, doubleml, sklearn, nbclient, tabicllab.treatment_effects
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("tabpfn", tabpfn.__version__, "xgboost", xgboost.__version__, "doubleml", doubleml.__version__,
      "sklearn", sklearn.__version__)
assert torch.cuda.is_available(), "no GPU visible"
EOF
jupyter kernelspec list | grep -q tabicllab || { echo "PREFLIGHT FAILED: no tabicllab kernel"; exit 1; }
nproc; free -g | head -2

execute() {   # $1: output path; runs the notebook in place of a copy, from its own folder
  python - "$1" <<'EOF'
import sys
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("01_effects_with_a_known_answer.ipynb", 4)
try:
    NotebookClient(nb, timeout=7200, kernel_name="tabicllab", resources={"metadata": {"path": "."}}).execute()
finally:
    nbformat.write(nb, sys.argv[1])
EOF
}

cd "$HERE" || exit 1
echo "==== smoke (TE_SMOKE=1) ===="
start=$(date +%s)
TE_SMOKE=1 execute "$OUT/smoke.ipynb" > "$OUT/smoke.log" 2>&1 || { echo "SMOKE FAILED"; tail -40 "$OUT/smoke.log"; exit 1; }
echo "smoke done in $(( $(date +%s) - start )) s"

echo "==== full run ===="
start=$(date +%s)
execute "$OUT/$NOTEBOOK" > "$OUT/run.log" 2>&1
status=$?
echo "full run exit status $status after $(( $(date +%s) - start )) s"
[ $status -eq 0 ] || tail -40 "$OUT/run.log"
ls -la "$OUT"
echo "Fetch: $OUT/"
exit $status
