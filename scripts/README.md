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
| `--trace` | Trace file path(s) — one per experiment | required |
| `--qps` | Poisson arrival rate(s) — one per `--trace` | required |
| `--chunk-sizes` | Max batched tokens — one per `--trace` | required |
| `--gpus` | GPU IDs — length must be `len(--trace) * --tensor-parallel-size` | required |
| `--numa-nodes` | NUMA node IDs — one per `--trace` (defaults to the first GPU of each TP group) | first GPU / group |
| `--requests` | Requests to replay (`0` = all) | `0` |
| `--base-port` | Starting port for vLLM server(s) | `8000` |
| `--gpu-memory-utilization` | GPU memory fraction | `0.90` |
| `--seed` | Seed for the Poisson arrival schedule; also seeds request ordering unless `--shuffle-seed` is given | `42` |
| `--shuffle-seed` | Seed for request ordering only, independent of arrivals (`-1` replays in trace order, unshuffled) | `--seed` |
| `--model` | HuggingFace model name (also passed as `--tokenizer` to `trace_replay`) | `Qwen/Qwen3-14B` |
| `--tensor-parallel-size` | vLLM TP size; pass `TP` contiguous GPUs per experiment in `--gpus` | `1` |
| `--max-model-len` | vLLM `--max-model-len` | `65536` |
| `--sequential-tools` | Serialize in-iteration tool sleeps (auto-enabled with `--swe-trace`) | off |

`--trace`, `--qps`, and `--chunk-sizes` must have equal length (one value per
experiment). Experiments run in parallel. With `--tensor-parallel-size t`,
`--gpus` is partitioned into contiguous groups of `t`.

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

| Flag | Description |
|------|-------------|
| `--prod-trace` | JSON message arrays — the default; equivalent to passing neither flag |
| `--bfcl-trace` | BFCL v4 JSONL + ChatML prompt parsers |
| `--swe-trace` | SWE-agent traces (terminus `<tool_call>` text or mini-swe structured) |

These flags are mutually exclusive. `--prod-trace` only records the format in
`metadata.json`; behavior matches omitting it.

`--bfcl-trace` switches the trace loader, KV-hint builder, and prefill splitter.
The file must be valid JSONL (one complete JSON object per line). A truncated
final record fails loudly rather than being skipped.

`--swe-trace` uses `swe_trace_loader` (dict-of-instances JSON, or a directory of
`.json` files). Default experiment file:
`experiment_traces/swe_bench_trace.json`. `--sequential-tools` is enabled
automatically (bash tools share shell state, so in-iteration sleeps must sum).

**`--prefill-split` does nothing on BFCL traces.** Split-point logic only exists
for the JSON message-array format; on ChatML it degrades to baseline and logs a
warning. BFCL's full stack is therefore `--decode-stream --workload-aware-cache`
(tag `ds_kv`), not `ps_ds_kv`.

## BFCL capacity sweep

`bfcl_capacity_sweep.sh` replays the BFCL trace at each QPS point, twice —
baseline, then Sutradhara (`--decode-stream --workload-aware-cache`, tag
`ds_kv`). `--dry-run` prints commands without launching.

```bash
./scripts/bfcl_capacity_sweep.sh [--dry-run]
```

Single GPU, sequential: 6 QPS points × 2 arms = 12 runs. Override with
`BFCL_TRACE`, `BFCL_GPU` (default `2`), `BFCL_NUMA` (default `0`).

Fixed: `--chunk-sizes 256 --requests 56 --gpu-memory-utilization 0.65
--seed 42 --shuffle-seed -1`. Prefill-split is omitted (no-op on ChatML).

Default trace: `experiment_traces/bfcl_trace.json` (56 `web_search` requests).
Results: `analysis/bfcl_paper_experiments.ipynb`.

## SWE-bench capacity sweep

`swe_capacity_sweep.sh` replays the SWE-bench trace at each QPS point, twice —
baseline, then Sutradhara (`--decode-stream`, tag `ds`). `--dry-run` prints
commands without launching.

```bash
./scripts/swe_capacity_sweep.sh [--dry-run]
```

Tensor parallel 2, sequential: 5 QPS points × 2 arms = 10 runs. Override with
`SWE_TRACE`, `SWE_GPUS` (default `0 1`), `SWE_NUMA` (default `0`), `SWE_MODEL`
(default `Qwen/Qwen3-30B-A3B-Instruct-2507`).

Fixed: `--chunk-sizes 256 --requests 0 --gpu-memory-utilization 0.80
--max-model-len 32768 --seed 42 --shuffle-seed -1 --tensor-parallel-size 2`.

Default trace: `experiment_traces/swe_bench_trace.json` (16 terminus trajectories;
tool latency from per-call `duration` wait budgets).

## Disaggregated serving

`--disaggregated` runs prefill and decode on separate GPUs, connected by a proxy
that routes requests and transfers the KV cache (via vLLM's `NixlConnector` over
NVLink). `--tensor-parallel-size > 1` is not supported in this mode.

- GPUs are **paired**: first = prefill, second = decode (even count required).
- `--trace`, `--qps`, `--chunk-sizes`, `--numa-nodes`: one value **per pair**.
- Ports are auto-assigned, 3 per pair (prefill, decode, proxy), from `--base-port`.
- Composes with optimization flags; tags gain a `disagg` prefix
  (`disagg`, `disagg_ps`, …).

```bash
# Two disaggregated experiments on 4 GPUs
python run_experiments.py \
  --disaggregated \
  --trace path/to/t1.json path/to/t2.json \
  --qps 0.01 0.01 \
  --chunk-sizes 256 256 \
  --gpus 0 1 2 3 \
  --numa-nodes 0 1 \
  --requests 10
# Pair 0: GPU0 prefill + GPU1 decode + proxy
# Pair 1: GPU2 prefill + GPU3 decode + proxy
```

`bfcl_disaggregated.sh` is the BFCL QPS sweep in this mode (`BFCL_TRACE`,
`BFCL_GPUS`, `BFCL_NUMA_NODES`).

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
