#!/bin/bash
# Usage: ./bfcl_capacity_sweep.sh [--dry-run]
#
# Replays the BFCL trace at each QPS point, twice: baseline, then Sutradhara.
# Env: BFCL_TRACE, BFCL_GPU, BFCL_NUMA.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
EXPERIMENT_SCRIPT="$DIR/run_experiments.py"

BFCL_TRACE="${BFCL_TRACE:-$REPO_ROOT/experiment_traces/bfcl_trace.json}"
GPU="${BFCL_GPU:-2}"
NUMA="${BFCL_NUMA:-0}"

CHUNK=256
REQUESTS=56
MEM=0.65        # induces KV pressure
SEED=42         # Poisson arrivals only

COMMON=(--trace "$BFCL_TRACE" --bfcl-trace
        --chunk-sizes "$CHUNK" --requests "$REQUESTS"
        --gpu-memory-utilization "$MEM"
        --seed "$SEED" --shuffle-seed -1
        --gpus "$GPU" --numa-nodes "$NUMA")
SD_FLAGS=(--decode-stream --workload-aware-cache)

QPS_LIST=(0.5 1.0 1.5 2.0 3.0 4.0)

DRY_RUN=false
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN=true; done

if [[ ! -f "$BFCL_TRACE" ]]; then
  if $DRY_RUN; then
    echo "[dry-run] WARNING: trace not found: $BFCL_TRACE"
  else
    echo "ERROR: trace not found: $BFCL_TRACE" >&2
    exit 1
  fi
fi

BATCH_LOG="$REPO_ROOT/logs/bfcl_capacity_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$BATCH_LOG")"
trap 'kill 0 2>/dev/null; exit 130' INT TERM

log() { echo "$@" | tee -a "$BATCH_LOG"; }

# run_exp ARGS... — one run at a time on one GPU.
run_exp() {
  local port=$((RANDOM % 5000 + 10000))
  local cmd=(python "$EXPERIMENT_SCRIPT" --base-port "$port" "$@")
  if $DRY_RUN; then
    log "[dry-run] ${cmd[*]}"
  else
    log "${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee -a "$BATCH_LOG"
  fi
}

log "BFCL capacity sweep | QPS: ${QPS_LIST[*]}"
log "Trace:  $BFCL_TRACE"
log "Fixed:  chunk=$CHUNK requests=$REQUESTS mem=$MEM seed=$SEED shuffle-seed=-1"
log "GPU:    $GPU (NUMA $NUMA) | $(date)"
log "============================================"

for QPS in "${QPS_LIST[@]}"; do
  run_exp "${COMMON[@]}" --qps "$QPS"
  run_exp "${COMMON[@]}" "${SD_FLAGS[@]}" --qps "$QPS"
  log "--- QPS=$QPS done (baseline, ds_kv) ---"
done

log "Done. Results under experiments/$(basename "$BFCL_TRACE" .json)/"
log "Render the table with analysis/bfcl_paper_experiments.ipynb"
