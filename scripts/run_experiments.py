"""Run E2E experiments: start vLLM servers, replay traces, collect metrics.

Output layout:  experiments/{trace}/chunk{N}_qps{X}/{opt_tag}/{timestamp}/
Optimization tag is auto-derived from flags:  baseline | ps | ds | kv | ps_ds | …
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def wait_for_server(port: str, timeout: int = 300) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_vllm_server(chunk_size, gpu, numa_node, port, log_file, metrics_dir,
                     async_scheduling=True, gpu_memory_utilization=0.90,
                     track_eviction_types=False, workload_aware_cache=False,
                     kv_transfer_config=None, disable_chunked_prefill=False,
                     extra_env="", model="Qwen/Qwen3-14B"):
    async_flag = "--async-scheduling " if async_scheduling else ""
    eviction_env = "VLLM_TRACK_EVICTION_TYPES=1 " if track_eviction_types else ""
    # Force LRU when tracking evictions without priority eviction.
    disable_priority = ("VLLM_DISABLE_PRIORITY_EVICTION=1 "
                        if track_eviction_types and not workload_aware_cache
                        else "")
    kv_flag = (f"--kv-transfer-config '{json.dumps(kv_transfer_config)}' "
               if kv_transfer_config else "")
    eager_flag = "--enforce-eager " if kv_transfer_config else ""
    # For disaggregated prefill: process entire prompt in one step
    batched_tokens = 60000 if disable_chunked_prefill else chunk_size
    chunked_flag = "--no-enable-chunked-prefill " if disable_chunked_prefill else ""
    # Qwen models need rope-scaling for extended context
    rope_flag = (
        """--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' """
        if "qwen" in model.lower() else ""
    )
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} VLLM_METRICS_DIR={metrics_dir} "
        f"{eviction_env}{disable_priority}{extra_env}"
        f"numactl --cpunodebind {numa_node} --membind {numa_node} "
        f"vllm serve {model} "
        f"--tensor-parallel-size 1 "
        f"{rope_flag}"
        f"--max-model-len 60000 "
        f"--max-num-batched-tokens {batched_tokens} "
        f"--dtype bfloat16 "
        f"{async_flag}"
        f"--scheduling-policy priority "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"{eager_flag}"
        f"{chunked_flag}"
        f"{kv_flag}"
        f"--port {port} > {log_file} 2>&1"
    )
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    print(f"  vLLM server  PID={proc.pid}  port={port}")
    return proc


def kill_process(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            # Force kill if SIGTERM didn't work (vLLM spawns sub-workers)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=3)
            except Exception:
                pass


def start_disagg_proxy(prefill_port, decode_port, proxy_port, log_file,
                      prefill_kv_addr="", decode_kv_addr=""):
    proxy_dir = os.path.join(REPO_ROOT, "serving_layer", "vllm",
                             "benchmarks", "disagg_benchmarks")
    kv_args = ""
    if prefill_kv_addr and decode_kv_addr:
        kv_args = (f"--prefill-kv-addr {prefill_kv_addr} "
                   f"--decode-kv-addr {decode_kv_addr} ")
    cmd = (
        f"cd {proxy_dir} && python disagg_prefill_proxy_server.py "
        f"--port {proxy_port} "
        f"--prefill-url http://localhost:{prefill_port}/v1/completions "
        f"--decode-url http://localhost:{decode_port}/v1/completions "
        f"{kv_args}"
        f"> {log_file} 2>&1"
    )
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    print(f"  proxy server PID={proc.pid}  port={proxy_port}")
    return proc


def wait_for_port(port, timeout=30):
    """Wait for a TCP port to accept connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", int(port)), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def start_trace_replay(trace_file, num_requests, port, numa_node,
                       qps, metrics_dir, log_file, prefill_split,
                       decode_stream=False, workload_aware_cache=False,
                       track_eviction_types=False, seed=42,
                       tool_time_multiplier=3.1):
    ps_flag = "--prefill-split " if prefill_split else ""
    ds_flag = "--decode-tool-streaming " if decode_stream else ""
    wac_flag = "--workload-aware-cache " if workload_aware_cache else ""
    tet_flag = "--track-eviction-types " if track_eviction_types else ""
    cmd = (
        f"numactl --cpunodebind {numa_node} --membind {numa_node} "
        f"python -m sutradhara.orchestrator.trace_replay "
        f"--trace {trace_file} "
        f"--vllm-url http://localhost:{port} "
        f"--qps {qps} "
        f"--num-requests {num_requests} "
        f"--metrics-dir {metrics_dir} "
        f"--seed {seed} "
        f"--tool-time-multiplier {tool_time_multiplier} "
        f"{ps_flag}"
        f"{ds_flag}"
        f"{wac_flag}"
        f"{tet_flag}"
        f">> {log_file} 2>&1"
    )
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    print(f"  trace replay PID={proc.pid}")
    return proc


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

def _opt_tag(prefill_split: bool, decode_stream: bool,
             workload_aware_cache: bool = False,
             track_eviction_types: bool = False,
             disaggregated: bool = False) -> str:
    """baseline | ps | ds | kv | ps_ds | disagg | disagg_ps | … """
    parts = []
    if prefill_split:
        parts.append("ps")
    if decode_stream:
        parts.append("ds")
    if workload_aware_cache:
        parts.append("kv")
    if track_eviction_types:
        parts.append("et")
    opt = "_".join(parts) if parts else "baseline"
    if disaggregated:
        return f"disagg_{opt}" if parts else "disagg"
    return opt


def make_experiment_dir(base, trace_file, chunk_size, qps, prefill_split,
                        decode_stream, workload_aware_cache=False,
                        gpu_memory_utilization=0.90,
                        track_eviction_types=False,
                        disaggregated=False,
                        seed=42):
    """Create and return (exp_dir, run_id)."""
    trace = Path(trace_file).stem
    mem_tag = f"mem{int(gpu_memory_utilization * 100):03d}"
    config = f"chunk{chunk_size}_qps{qps}_{mem_tag}"
    tag = _opt_tag(prefill_split, decode_stream, workload_aware_cache,
                   track_eviction_types, disaggregated)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join(base, trace, config, tag, f"seed{seed}", ts)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)

    run_id = f"{trace}_{config}_{tag}_seed{seed}_{ts}"
    return exp_dir, run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    n = len(args.trace)
    ports = [int(args.base_port) + i for i in range(n)]
    base = os.path.join(REPO_ROOT, "experiments")

    # --- set up directories & metadata ---
    experiments = []
    for i in range(n):
        exp_dir, run_id = make_experiment_dir(
            base, args.trace[i], args.chunk_sizes[i], args.qps[i],
            args.prefill_split, args.decode_stream, args.workload_aware_cache,
            args.gpu_memory_utilization, args.track_eviction_types,
            seed=args.seed,
        )
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "trace_file": args.trace[i],
            "num_requests": args.requests or "all",
            "chunk_size": args.chunk_sizes[i],
            "qps": args.qps[i],
            "seed": args.seed,
            "model": args.model,
            "prefill_split": args.prefill_split,
            "decode_stream": args.decode_stream,
            "workload_aware_cache": args.workload_aware_cache,
            "track_eviction_types": args.track_eviction_types,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tool_time_multiplier": args.tool_time_multiplier,
            "disaggregated": False,
            "gpu": args.gpus[i],
            "numa_node": args.numa_nodes[i],
            "port": ports[i],
        }
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        experiments.append({"dir": exp_dir, "meta": metadata})
        print(f"[{i+1}/{n}] {run_id}")

    # --- start vLLM servers ---
    vllm_procs = []
    try:
        print(f"\nStarting {n} vLLM server(s)...")
        for i in range(n):
            vllm_procs.append(start_vllm_server(
                chunk_size=args.chunk_sizes[i],
                gpu=args.gpus[i],
                numa_node=args.numa_nodes[i],
                port=str(ports[i]),
                log_file=os.path.join(experiments[i]["dir"], "logs", "vllm_server.log"),
                metrics_dir=os.path.join(experiments[i]["dir"], "metrics"),
                gpu_memory_utilization=args.gpu_memory_utilization,
                track_eviction_types=args.track_eviction_types,
                workload_aware_cache=args.workload_aware_cache,
                model=args.model,
            ))

        # health-check
        print("Waiting for servers...")
        for i, port in enumerate(ports):
            if wait_for_server(str(port)):
                print(f"  ✓ port {port} ready")
            else:
                print(f"  ✗ port {port} failed to start — aborting")
                return

        # --- start trace replays ---
        print(f"\nStarting {n} trace replay(s)...")
        replay_procs = []
        for i in range(n):
            replay_procs.append(start_trace_replay(
                trace_file=os.path.join(REPO_ROOT, args.trace[i]),
                num_requests=args.requests,
                port=str(ports[i]),
                numa_node=args.numa_nodes[i],
                qps=args.qps[i],
                metrics_dir=os.path.join(experiments[i]["dir"], "metrics"),
                log_file=os.path.join(experiments[i]["dir"], "logs", "trace_replay.log"),
                prefill_split=args.prefill_split,
                decode_stream=args.decode_stream,
                workload_aware_cache=args.workload_aware_cache,
                track_eviction_types=args.track_eviction_types,
                seed=args.seed,
                tool_time_multiplier=args.tool_time_multiplier,
            ))

        # wait for completion
        for i, proc in enumerate(replay_procs):
            proc.wait()
            print(f"  ✓ experiment {i+1}/{n} done → {experiments[i]['dir']}")

        print(f"\nAll done. Results under: {base}/")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for proc in vllm_procs:
            kill_process(proc)


def run_disaggregated(args):
    n_pairs = len(args.trace)
    base_port = int(args.base_port)
    base = os.path.join(REPO_ROOT, "experiments")

    # --- set up directories & metadata ---
    experiments = []
    for i in range(n_pairs):
        prefill_gpu = args.gpus[i * 2]
        decode_gpu = args.gpus[i * 2 + 1]
        prefill_port = base_port + i * 3
        decode_port = base_port + i * 3 + 1
        proxy_port = base_port + i * 3 + 2

        exp_dir, run_id = make_experiment_dir(
            base, args.trace[i], args.chunk_sizes[i], args.qps[i],
            args.prefill_split, args.decode_stream, args.workload_aware_cache,
            args.gpu_memory_utilization, args.track_eviction_types,
            disaggregated=True,
            seed=args.seed,
        )
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "trace_file": args.trace[i],
            "num_requests": args.requests or "all",
            "chunk_size": args.chunk_sizes[i],
            "qps": args.qps[i],
            "seed": args.seed,
            "model": args.model,
            "prefill_split": args.prefill_split,
            "decode_stream": args.decode_stream,
            "workload_aware_cache": args.workload_aware_cache,
            "track_eviction_types": args.track_eviction_types,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tool_time_multiplier": args.tool_time_multiplier,
            "disaggregated": True,
            "prefill_gpu": prefill_gpu,
            "decode_gpu": decode_gpu,
            "numa_node": args.numa_nodes[i],
            "prefill_port": prefill_port,
            "decode_port": decode_port,
            "proxy_port": proxy_port,
        }
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        experiments.append({"dir": exp_dir, "meta": metadata})
        print(f"[{i+1}/{n_pairs}] {run_id}")

    # --- start vLLM prefill + decode pairs ---
    vllm_procs = []
    proxy_procs = []
    try:
        print(f"\nStarting {n_pairs} disaggregated pair(s)...")
        for i in range(n_pairs):
            m = experiments[i]["meta"]
            log_dir = os.path.join(experiments[i]["dir"], "logs")
            metrics_dir = os.path.join(experiments[i]["dir"], "metrics")
            numa = args.numa_nodes[i]
            # NixlConnector: async KV transfer via NIXL (UCX cuda_ipc over
            # NVLink on same host).  Each instance needs a unique side-channel
            # port for the NIXL handshake.
            nixl_base_port = base_port + 200 + i * 2
            kv_cfg = {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            }
            ucx_env = ("UCX_TLS=cuda_ipc,cuda_copy,tcp "
                       "UCX_NET_DEVICES=all "
                       "UCX_LOG_LEVEL=info ")

            # Prefill instance — no chunked prefill
            vllm_procs.append(start_vllm_server(
                chunk_size=args.chunk_sizes[i],
                gpu=m["prefill_gpu"],
                numa_node=numa,
                port=str(m["prefill_port"]),
                log_file=os.path.join(log_dir, "vllm_prefill.log"),
                metrics_dir=metrics_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                track_eviction_types=args.track_eviction_types,
                workload_aware_cache=args.workload_aware_cache,
                disable_chunked_prefill=True,
                kv_transfer_config=kv_cfg,
                extra_env=(
                    f"{ucx_env}"
                    f"VLLM_NIXL_SIDE_CHANNEL_PORT={nixl_base_port} "
                ),
                model=args.model,
            ))

            # Decode instance — no chunked prefill
            vllm_procs.append(start_vllm_server(
                chunk_size=args.chunk_sizes[i],
                gpu=m["decode_gpu"],
                numa_node=numa,
                port=str(m["decode_port"]),
                log_file=os.path.join(log_dir, "vllm_decode.log"),
                metrics_dir=metrics_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                track_eviction_types=args.track_eviction_types,
                workload_aware_cache=args.workload_aware_cache,
                disable_chunked_prefill=True,
                kv_transfer_config=kv_cfg,
                extra_env=(
                    f"{ucx_env}"
                    f"VLLM_NIXL_SIDE_CHANNEL_PORT={nixl_base_port + 1} "
                ),
                model=args.model,
            ))

        # Health-check all vLLM instances
        print("Waiting for vLLM servers...")
        for i in range(n_pairs):
            m = experiments[i]["meta"]
            for label, port in [("prefill", m["prefill_port"]),
                                ("decode", m["decode_port"])]:
                if wait_for_server(str(port)):
                    print(f"  ✓ pair {i} {label} port {port} ready")
                else:
                    print(f"  ✗ pair {i} {label} port {port} failed — aborting")
                    return

        # --- start proxy servers ---
        print(f"\nStarting {n_pairs} proxy server(s)...")
        for i in range(n_pairs):
            m = experiments[i]["meta"]
            proxy_procs.append(start_disagg_proxy(
                prefill_port=m["prefill_port"],
                decode_port=m["decode_port"],
                proxy_port=m["proxy_port"],
                log_file=os.path.join(experiments[i]["dir"], "logs", "proxy.log"),
            ))

        print("Waiting for proxy servers...")
        for i in range(n_pairs):
            m = experiments[i]["meta"]
            if wait_for_server(str(m["proxy_port"])):
                print(f"  ✓ pair {i} proxy port {m['proxy_port']} ready")
            else:
                print(f"  ✗ pair {i} proxy port {m['proxy_port']} failed — aborting")
                return

        # --- start trace replays (pointed at proxy) ---
        print(f"\nStarting {n_pairs} trace replay(s)...")
        replay_procs = []
        for i in range(n_pairs):
            m = experiments[i]["meta"]
            replay_procs.append(start_trace_replay(
                trace_file=os.path.join(REPO_ROOT, args.trace[i]),
                num_requests=args.requests,
                port=str(m["proxy_port"]),
                numa_node=args.numa_nodes[i],
                qps=args.qps[i],
                metrics_dir=os.path.join(experiments[i]["dir"], "metrics"),
                log_file=os.path.join(experiments[i]["dir"], "logs",
                                      "trace_replay.log"),
                prefill_split=args.prefill_split,
                decode_stream=args.decode_stream,
                workload_aware_cache=args.workload_aware_cache,
                track_eviction_types=args.track_eviction_types,
                seed=args.seed,
                tool_time_multiplier=args.tool_time_multiplier,
            ))

        for i, proc in enumerate(replay_procs):
            proc.wait()
            print(f"  ✓ experiment {i+1}/{n_pairs} done → {experiments[i]['dir']}")

        print(f"\nAll done. Results under: {base}/")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for proc in proxy_procs:
            kill_process(proc)
        for proc in vllm_procs:
            kill_process(proc)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--trace", nargs="+", required=True)
    p.add_argument("--qps", nargs="+", type=float, required=True)
    p.add_argument("--chunk-sizes", nargs="+", type=int, required=True)
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--numa-nodes", nargs="+", type=int, default=None,
                   help="NUMA node IDs (defaults to --gpus)")
    p.add_argument("--requests", type=int, default=0, help="0 = all (default)")
    p.add_argument("--base-port", default="8000")
    p.add_argument("--prefill-split", action="store_true", default=False)
    p.add_argument("--decode-stream", action="store_true", default=False)
    p.add_argument("--workload-aware-cache", action="store_true", default=False)
    p.add_argument("--track-eviction-types", action="store_true", default=False)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for request ordering (default: 42)")
    p.add_argument("--tool-time-multiplier", type=float, default=3.1,
                   help="Multiplier for simulated tool latency (default: 3.1)")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-14B",
                   help="HuggingFace model name (default: Qwen/Qwen3-14B)")
    p.add_argument("--disaggregated", action="store_true", default=False,
                   help="Disaggregated prefill/decode: GPUs are paired "
                        "(first=prefill, second=decode). --trace/--qps/"
                        "--chunk-sizes provide one value per GPU pair.")

    args = p.parse_args()

    if args.disaggregated:
        if len(args.gpus) % 2 != 0:
            p.error("--gpus must have even length in disaggregated mode")
        n_pairs = len(args.gpus) // 2
        if args.numa_nodes is None:
            p.error("--numa-nodes is required in disaggregated mode "
                    "(one per GPU pair)")
        lists = [args.trace, args.qps, args.chunk_sizes, args.numa_nodes]
        if any(len(l) != n_pairs for l in lists):
            p.error("--trace, --qps, --chunk-sizes, --numa-nodes must have "
                    f"length == len(--gpus)/2 ({n_pairs}) in disaggregated "
                    "mode (one value per GPU pair)")
        run_disaggregated(args)
    else:
        if args.numa_nodes is None:
            args.numa_nodes = list(args.gpus)
        lists = [args.trace, args.qps, args.chunk_sizes, args.gpus,
                 args.numa_nodes]
        if len(set(len(l) for l in lists)) != 1:
            p.error("--trace, --qps, --chunk-sizes, --gpus, --numa-nodes "
                    "must have equal length")
        run(args)


if __name__ == "__main__":
    main()
