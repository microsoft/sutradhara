#!/bin/bash
# Usage: ./bfcl_disaggregated.sh [--dry-run]
#
# Replays the BFCL trace at QPS 1.0 using disaggregated prefill/decode:
# baseline first, then Sutradhara.
#
# Env:
#   BFCL_TRACE       Trace path.
#   BFCL_GPUS        Space-separated GPU pairs (default: "0 1").
#   BFCL_NUMA_NODES  Space-separated NUMA nodes, one per pair (default: "0").

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
EXPERIMENT_SCRIPT="$DIR/run_experiments.py"

BFCL_TRACE="${BFCL_TRACE:-$REPO_ROOT/experiment_traces/bfcl_trace.json}"
read -r -a GPUS <<< "${BFCL_GPUS:-0 1}"
read -r -a NUMA_NODES <<< "${BFCL_NUMA_NODES:-0}"

QPS=1.0
CHUNK=256
REQUESTS=56
MEM=0.65
SEED=42

if (( ${#GPUS[@]} == 0 || ${#GPUS[@]} % 2 != 0 )); then
  echo "ERROR: BFCL_GPUS must contain an even number of GPU IDs" >&2
  exit 1
fi

PAIR_COUNT=$(( ${#GPUS[@]} / 2 ))
if (( ${#NUMA_NODES[@]} != PAIR_COUNT )); then
  echo "ERROR: BFCL_NUMA_NODES must contain one node per GPU pair ($PAIR_COUNT expected)" >&2
  exit 1
fi

TRACES=()
QPS_VALUES=()
CHUNK_SIZES=()
for ((i = 0; i < PAIR_COUNT; i++)); do
  TRACES+=("$BFCL_TRACE")
  QPS_VALUES+=("$QPS")
  CHUNK_SIZES+=("$CHUNK")
done

COMMON=(--disaggregated
        --trace "${TRACES[@]}" --bfcl-trace
        --qps "${QPS_VALUES[@]}"
        --chunk-sizes "${CHUNK_SIZES[@]}"
        --requests "$REQUESTS"
        --gpu-memory-utilization "$MEM"
        --seed "$SEED" --shuffle-seed -1
        --gpus "${GPUS[@]}" --numa-nodes "${NUMA_NODES[@]}")
SD_FLAGS=(--decode-stream --workload-aware-cache)

DRY_RUN=false
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=true
  else
    echo "ERROR: unknown argument: $arg" >&2
    exit 1
  fi
done

if [[ ! -f "$EXPERIMENT_SCRIPT" ]]; then
  echo "ERROR: experiment runner not found: $EXPERIMENT_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$BFCL_TRACE" ]]; then
  if $DRY_RUN; then
    echo "[dry-run] WARNING: trace not found: $BFCL_TRACE"
  else
    echo "ERROR: trace not found: $BFCL_TRACE" >&2
    exit 1
  fi
fi

BATCH_LOG="$REPO_ROOT/logs/bfcl_disaggregated_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$BATCH_LOG")"

log() { echo "$@" | tee -a "$BATCH_LOG"; }

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

log "BFCL disaggregated runs | QPS: $QPS"
log "Trace:  $BFCL_TRACE"
log "Fixed:  chunk=$CHUNK requests=$REQUESTS mem=$MEM seed=$SEED shuffle-seed=-1"
log "GPUs:   ${GPUS[*]} (prefill/decode pairs)"
log "NUMA:   ${NUMA_NODES[*]} (one per pair) | $(date)"
log "============================================"

run_exp "${COMMON[@]}"
run_exp "${COMMON[@]}" "${SD_FLAGS[@]}"

log "--- QPS=$QPS done (disaggregated baseline, disaggregated ds_kv) ---"
log "Done. Results under experiments/$(basename "$BFCL_TRACE" .json)/"
log "Render the table with analysis/bfcl_paper_experiments.ipynb"
