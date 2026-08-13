# Experiment runner

`run_experiments.py` drives an end-to-end experiment: it starts a patched vLLM
server, replays an agentic trace against it, and writes metrics into a
structured directory.

```bash
python run_experiments.py \
  --trace path/to/trace.json \
  --qps 0.01 \
  --chunk-sizes 256 \
  --gpus 0 \
  --numa-nodes 0 \
  --requests 60 \
  --prefill-split
```

## Arguments

| Arg | Description | Default |
|-----|-------------|---------|
| `--trace` | Trace file path(s) — one per GPU | required |
| `--qps` | Poisson arrival rate(s) — one per `--trace` | required |
| `--chunk-sizes` | Max batched tokens per GPU | required |
| `--gpus` | GPU IDs | required |
| `--numa-nodes` | NUMA node IDs (defaults to `--gpus`) | `--gpus` |
| `--requests` | Requests to replay (`0` = all) | `0` |
| `--base-port` | Starting port for vLLM server(s) | `8000` |
| `--gpu-memory-utilization` | GPU memory fraction | `0.90` |
| `--seed` | Seed for the Poisson arrival schedule; also seeds request ordering unless `--shuffle-seed` is given | `42` |
| `--shuffle-seed` | Seed for request ordering only, independent of arrivals (`-1` replays in trace order, unshuffled) | `--seed` |
| `--model` | HuggingFace model name | `Qwen/Qwen3-14B` |

The list args (`--trace`, `--qps`, `--chunk-sizes`, `--gpus`, `--numa-nodes`)
must have equal length: one experiment per GPU, run in parallel.

## Optimization flags

| Flag | Tag | Description |
|------|-----|-------------|
| `--prefill-split` | `ps` | Split prefill to overlap with tool execution |
| `--decode-stream` | `ds` | Launch tools mid-decode at trigger indices |
| `--workload-aware-cache` | `kv` | Priority-based KV-cache eviction |
| `--track-eviction-types` | `et` | Record per-block eviction counts |

Flags compose into the output tag: `ps_ds`, `ps_ds_kv`, etc. No flags →
`baseline`.

## Trace formats

The default trace format is a JSON array of chat-message dicts. BFCL v4 traces
use a different on-disk layout (JSONL) and a different prompt format (ChatML),
so they need an explicit flag:

| Flag | Description |
|------|-------------|
| `--prod-trace` | Parse the trace as JSON message arrays — the default; equivalent to passing neither flag |
| `--bfcl-trace` | Parse the trace as BFCL v4 JSONL and use the ChatML prompt parsers |

The two are mutually exclusive; passing both is an error. `--prod-trace` exists
to state the format explicitly on the command line and record it in
`metadata.json` — it does not change behavior relative to omitting it.

`--bfcl-trace` switches three things at once: the trace loader, the KV-hint
builder, and the prefill splitter. The trace file must be valid JSONL — one
complete JSON object per line. A truncated final record is a malformed trace and
will fail loudly rather than being silently skipped.

**`--prefill-split` does nothing on BFCL traces.** The split-point logic only
exists for the JSON message-array format; on ChatML it degrades to baseline and
logs a warning. BFCL's full optimization stack is therefore `--decode-stream
--workload-aware-cache` (tag `ds_kv`), not `ps_ds_kv`.

## BFCL capacity sweep

`bfcl_capacity_sweep.sh` replays the BFCL trace at each QPS point, twice per
point — baseline, then Sutradhara (`--decode-stream --workload-aware-cache`, tag
`ds_kv`). It is self-contained; `--dry-run` prints the commands without launching
anything.

```bash
./scripts/bfcl_capacity_sweep.sh [--dry-run]
```

Runs on a **single GPU**, sequentially: 6 QPS points × 2 arms = 12 runs. GPU and
NUMA node come from `BFCL_GPU` (default 2) and `BFCL_NUMA` (default 0).

Fixed for every run: `--chunk-sizes 256 --requests 56
--gpu-memory-utilization 0.65 --seed 42 --shuffle-seed -1`. Requests are never
shuffled and the Poisson arrival seed is always 42, so both arms replay an
identical arrival schedule.

**`--prefill-split` is absent.** It is a no-op on BFCL's ChatML prompts, so the
full BFCL stack is DS+KV, never PS+DS+KV.

Results are read by `analysis/bfcl_paper_experiments.ipynb`, which prints the
median FTR/E2E table directly from the CSVs under `experiments/`.

Both arms replay the **same** trace. Select it with the `BFCL_TRACE` environment
variable:

```bash
BFCL_TRACE=experiment_traces/<file>.json ./bfcl_capacity_sweep.sh
```

The default trace, `experiment_traces/bfcl_trace.json`, holds 56 `web_search`
requests with their recorded per-call tool latencies (1.094 s mean, 1.703 s
stdev, 3.23 tool steps per request, tool fan-out 2.04).

Model generalizability across serving models is not covered: `run_experiments.py`
passes `--model` to the server but never passes a matching `--tokenizer` to
`trace_replay`, so serving a non-Qwen model would compute KV hints and
decode-stream trigger indices with the wrong tokenizer.

## Disaggregated serving

`--disaggregated` runs prefill and decode on separate GPUs, connected by a proxy
that routes requests and transfers the KV cache (via vLLM's `NixlConnector` over
NVLink).

- GPUs are **paired**: first = prefill, second = decode (even count required).
- `--trace`, `--qps`, `--chunk-sizes` take **one value per pair**.
- Ports are auto-assigned, 3 per pair (prefill, decode, proxy), from `--base-port`.
- Composes with all optimization flags; tags gain a `disagg` prefix
  (`disagg`, `disagg_ps`, …).

```bash
# Two disaggregated experiments on 4 GPUs
python run_experiments.py \
  --disaggregated \
  --trace path/to/t1.json path/to/t2.json \
  --qps 0.01 0.01 \
  --chunk-sizes 256 256 \
  --gpus 0 1 2 3 \
  --numa-nodes 0 1 2 3 \
  --requests 10
# Pair 0: GPU0 prefill:8000 + GPU1 decode:8001 + proxy:8002
# Pair 1: GPU2 prefill:8003 + GPU3 decode:8004 + proxy:8005
```

## Output layout

```
experiments/{trace}/chunk{N}_qps{X}_mem{YYY}/{tag}/seed{S}/{timestamp}/
├── metadata.json
├── logs/
│   ├── vllm_server.log          # single-instance mode
│   ├── vllm_prefill.log         # disaggregated mode
│   ├── vllm_decode.log          # disaggregated mode
│   ├── proxy.log                # disaggregated mode
│   └── trace_replay.log
└── metrics/
    ├── request_metrics.csv
    ├── iteration_metrics.csv
    ├── batch_metrics.csv
    ├── batch_creation_times.csv
    ├── overlap_metrics.csv
    └── eviction_counts.json      # only with --track-eviction-types
```
