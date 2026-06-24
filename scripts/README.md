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
| `--seed` | Seed for the Poisson arrival schedule | `42` |
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
