# Sutradhara

Sutradhara is a co-designed agentic inference system that couples the
orchestrator with the LLM serving engine. A thin orchestrator↔engine API enables
three cross-layer optimizations:

- **Tool-aware prompt splitting** — overlap tool execution with the prefill of
  later prompt segments.
- **Streaming tool execution** — dispatch tool calls during decode instead of
  waiting for the full output.
- **Orchestrator-aware cache management** — semantic hints drive KV-cache
  eviction to lift hit rates.

It is implemented on [vLLM](https://github.com/vllm-project/vllm) as a small,
pure-Python patch. Full design and evaluation are in the paper:

> **Sutradhara: Co-designing Orchestration and Serving for Agentic LLM Inference**
> [arXiv:2601.12967](https://arxiv.org/abs/2601.12967)

## Hardware

- NVIDIA A100 80GB GPUs. Four GPUs reproduce the paper's parallel configurations;
  a single GPU is enough to run one experiment.
- One Qwen3-14B instance per GPU (tensor-parallel = 1).

## Install

Install [pixi](https://pixi.sh) if you don't have it:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Clone and build:

```bash
pixi install
pixi run install-serving-layer
```

`pixi install` creates an isolated environment (Python 3.12, CUDA 12.9,
torch 2.8.0+cu129, and all dependencies) — no system-wide CUDA changes needed.
`pixi run install-serving-layer` clones official vLLM `v0.11.0`, installs it
editable using vLLM's official precompiled binaries.

Verify:

```bash
pixi run python -c "import sutradhara.orchestrator; print('orchestrator OK')"
pixi run python -c "from vllm.v1.core.kv_block_types import KVBlockType; print('vLLM patch OK')"
```

Then `pixi shell` to enter the environment, or prefix commands with `pixi run`.

## Running experiments

`scripts/run_experiments.py` starts a patched vLLM server, replays an agentic
trace against it, and writes metrics into a structured directory. See
[`scripts/README.md`](scripts/README.md) for all options and the optimization
flags.
