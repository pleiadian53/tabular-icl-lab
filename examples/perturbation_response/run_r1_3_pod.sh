#!/usr/bin/env bash
# R1.3, round 1, on the pod: preflight, a smoke run, then the full run under the protocol
# fixed in 02_round1_residual_learners.py. Any preflight or smoke failure stops the job
# before the full run.
#
# Inputs are read, never written: the Norman cache staged on the shared volume. Outputs,
# and the TabPFN-3.5 weights, live in this lab's own namespace on the volume. The weights
# are downloaded once (that needs TABPFN_TOKEN, passed as a SkyPilot secret, for the
# license check); later runs find them on the volume and need no token.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_r1_3_pod.sh
set -uo pipefail
VOL="${VOL:-/runpod-volume}"
DATA="${DATA:-$VOL/ssl-lab/data/norman2019}"
OUT="${OUT:-$VOL/tabular-icl-lab/output}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$VOL/tabular-icl-lab/models/tabpfn}"
export TABPFN_NO_BROWSER=1
EP=examples/perturbation_response
DE_MD5=9844c2c25d2903872e1c0a92c722f105       # the scored-gene list every ssl-lab result used
SPLITS_MD5=82f1f43952731c64551629701c5dc041
WEIGHTS="$TABPFN_MODEL_CACHE_DIR/tabpfn-v3.5-20260909.safetensors"

echo "==== preflight ===="
for f in tokens_meta.npz de_genes.json splits.json manifest.json; do
  [ -f "$DATA/$f" ] || { echo "PREFLIGHT FAILED: missing $DATA/$f"; exit 1; }
done
[ "$(md5sum "$DATA/de_genes.json" | cut -d' ' -f1)" = "$DE_MD5" ] || { echo "PREFLIGHT FAILED: de_genes.json md5"; exit 1; }
[ "$(md5sum "$DATA/splits.json" | cut -d' ' -f1)" = "$SPLITS_MD5" ] || { echo "PREFLIGHT FAILED: splits.json md5"; exit 1; }
mkdir -p "$TABPFN_MODEL_CACHE_DIR" "$OUT/r1_3_round1" "$OUT/r1_3_smoke"
if [ ! -f "$WEIGHTS" ] && [ -z "${TABPFN_TOKEN:-}" ]; then
  echo "PREFLIGHT FAILED: no TabPFN-3.5 weights at $WEIGHTS and no TABPFN_TOKEN to download them"; exit 1
fi
python - <<'EOF' || { echo "PREFLIGHT FAILED: imports"; exit 1; }
import torch, tabpfn, lightgbm, sklearn, numpy, scipy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("tabpfn", tabpfn.__version__, "lightgbm", lightgbm.__version__, "sklearn", sklearn.__version__)
EOF
nproc; free -g | head -2

echo "==== smoke (training and validation pairs only; downloads the weights on first use) ===="
python $EP/02_round1_residual_learners.py --data-dir "$DATA" --smoke \
  --out "$OUT/r1_3_smoke/report.json" > "$OUT/r1_3_smoke/run.log" 2>&1 \
  || { echo "SMOKE FAILED"; tail -30 "$OUT/r1_3_smoke/run.log"; exit 1; }
tail -2 "$OUT/r1_3_smoke/run.log"
ls -la "$WEIGHTS" || { echo "weights missing after smoke"; exit 1; }
unset TABPFN_TOKEN    # the full run reads the weights from the volume

echo "==== full run ===="
python $EP/02_round1_residual_learners.py --data-dir "$DATA" \
  --out "$OUT/r1_3_round1/report.json" 2>&1 | tee "$OUT/r1_3_round1/run.log"
status=${PIPESTATUS[0]}
echo "R1.3 exit status $status. Fetch: $OUT/r1_3_round1/"
exit $status
