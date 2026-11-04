#!/usr/bin/env bash
# Sweep one config.py parameter with the standalone (NetSquid-only) driver.
# Does not edit config.py. Works on macOS and Linux.
#
# Usage:   ./sweep_standalone.sh PARAM START STEP STOP [extra run_standalone.py args]
# Example: ./sweep_standalone.sh TEMPORAL_JITTER_STDEV_PS 0 0.5 20 --runs 10000 --workers 8
#
# Writes results/<PARAM>/<PARAM>_<value>.json (configuration_name = value), the
# same JSON layout the QSI driver produced, so the paper_data/plot_*.py scripts
# can read the folder.
set -euo pipefail
if [ $# -lt 4 ]; then sed -n '2,10p' "$0"; exit 1; fi
PARAM=$1; START=$2; STEP=$3; STOP=$4; shift 4
OUT="results/${PARAM}"; mkdir -p "$OUT"
for v in $(seq "$START" "$STEP" "$STOP"); do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')  ${PARAM}=${v} ==="
  python run_standalone.py --set "${PARAM}=${v}" --label "${v}" --out "${OUT}/${PARAM}_${v}.json" "$@" | tail -n 1
done
echo "=== $(date '+%Y-%m-%d %H:%M:%S')  sweep complete: ${OUT} ==="
