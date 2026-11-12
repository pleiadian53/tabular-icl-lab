#!/usr/bin/env bash
# R1.1 on the pod: preflight, a smoke run, then the full run under the protocol fixed
# in 01_residual_ceiling.py. Any preflight or smoke failure stops the job before the
# full run.
#
# Inputs are read, never written: the Norman cache staged on the shared volume.
# Outputs go to this lab's own namespace on the volume.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_r1_1_pod.sh
set -uo pipefail
VOL="${VOL:-/runpod-volume}"
DATA="${DATA:-$VOL/ssl-lab/data/norman2019}"
OUT="${OUT:-$VOL/tabular-icl-lab/output}"
EP=examples/perturbation_response
DE_MD5=9844c2c25d2903872e1c0a92c722f105       # the scored-gene list every ssl-lab result used
SPLITS_MD5=82f1f43952731c64551629701c5dc041

echo "==== preflight ===="
for f in tokens_meta.npz de_genes.json splits.json manifest.json; do
  [ -f "$DATA/$f" ] || { echo "PREFLIGHT FAILED: missing $DATA/$f"; exit 1; }
done
[ "$(md5sum "$DATA/de_genes.json" | cut -d' ' -f1)" = "$DE_MD5" ] || { echo "PREFLIGHT FAILED: de_genes.json md5"; exit 1; }
[ "$(md5sum "$DATA/splits.json" | cut -d' ' -f1)" = "$SPLITS_MD5" ] || { echo "PREFLIGHT FAILED: splits.json md5"; exit 1; }
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" || exit 1
nproc; free -g | head -2
mkdir -p "$OUT/r1_1_residual_ceiling" "$OUT/r1_1_smoke"

echo "==== smoke ===="
python $EP/01_residual_ceiling.py --data-dir "$DATA" --limit 2 --reps 2 \
  --out "$OUT/r1_1_smoke/report.json" > "$OUT/r1_1_smoke/run.log" 2>&1 \
  || { echo "SMOKE FAILED"; tail -20 "$OUT/r1_1_smoke/run.log"; exit 1; }
tail -2 "$OUT/r1_1_smoke/run.log"

echo "==== full run ===="
python $EP/01_residual_ceiling.py --data-dir "$DATA" \
  --out "$OUT/r1_1_residual_ceiling/report.json" 2>&1 | tee "$OUT/r1_1_residual_ceiling/run.log"
status=${PIPESTATUS[0]}
echo "R1.1 exit status $status. Fetch: $OUT/r1_1_residual_ceiling/"
exit $status
