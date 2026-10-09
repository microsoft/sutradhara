#!/bin/bash
# Usage: ./swe_capacity_sweep.sh [--dry-run]
#
# Replays the SWE-bench trace at each QPS point, twice: baseline, then
# Sutradhara decode-stream. Env: SWE_TRACE, SWE_GPUS, SWE_NUMA, SWE_MODEL.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
EXPERIMENT_SCRIPT="$DIR/run_experiments.py"

SWE_TRACE="${SWE_TRACE:-$REPO_ROOT/experiment_traces/swe_bench_trace.json}"
# TP=2: pass two contiguous GPU IDs (space-separated).
GPUS="${SWE_GPUS:-0 1}"
NUMA="${SWE_NUMA:-0}"
MODEL="${SWE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

CHUNK=256
REQUESTS=0      # all requests in the trace
MEM=0.80
MAX_LEN=32768
TP=2
SEED=42         # Poisson arrivals only

# shellcheck disable=SC2206
GPU_ARR=($GPUS)

COMMON=(--trace "$SWE_TRACE" --swe-trace
        --chunk-sizes "$CHUNK" --requests "$REQUESTS"
        --gpu-memory-utilization "$MEM"
        --seed "$SEED" --shuffle-seed -1
        --gpus "${GPU_ARR[@]}" --numa-nodes "$NUMA"
        --model "$MODEL"
        --tensor-parallel-size "$TP"
        --max-model-len "$MAX_LEN")
SD_FLAGS=(--decode-stream)

QPS_LIST=(0.02 0.04 0.08 0.16 0.32)

DRY_RUN=false
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN=true; done

if [[ ! -f "$SWE_TRACE" ]]; then
  if $DRY_RUN; then
    echo "[dry-run] WARNING: trace not found: $SWE_TRACE"
  else
    echo "ERROR: trace not found: $SWE_TRACE" >&2
    exit 1
  fi
fi

BATCH_LOG="$REPO_ROOT/logs/swe_capacity_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$BATCH_LOG")"
trap 'kill 0 2>/dev/null; exit 130' INT TERM

log() { echo "$@" | tee -a "$BATCH_LOG"; }

# run_exp ARGS... — one run at a time (TP group).
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

log "SWE-bench capacity sweep | QPS: ${QPS_LIST[*]}"
log "Trace:  $SWE_TRACE"
log "Model:  $MODEL (TP=$TP)"
log "Fixed:  chunk=$CHUNK requests=$REQUESTS mem=$MEM maxlen=$MAX_LEN seed=$SEED shuffle-seed=-1"
log "GPUs:   ${GPU_ARR[*]} (NUMA $NUMA) | $(date)"
log "============================================"

for QPS in "${QPS_LIST[@]}"; do
  run_exp "${COMMON[@]}" --qps "$QPS"
  run_exp "${COMMON[@]}" "${SD_FLAGS[@]}" --qps "$QPS"
  log "--- QPS=$QPS done (baseline, ds) ---"
done

log "Done. Results under experiments/$(basename "$SWE_TRACE" .json)/"
