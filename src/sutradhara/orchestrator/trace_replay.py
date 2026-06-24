import argparse
import asyncio
from pathlib import Path

from sutradhara.orchestrator.common.poisson_arrival_distribution import (
    sample_poisson_inter_arrivals,
    get_request_order,
)
from sutradhara.orchestrator.common.request_iter_extractor import load_agentic_requests
from sutradhara.orchestrator.llm_client.vllm_client import VLLMClient
from sutradhara.orchestrator.logger.logger import setup_logging
from sutradhara.orchestrator.policies.latency_optimal import LatencyOptimalPolicy
from sutradhara.orchestrator.simulator.orchestrator_simulator import (
    OrchestratorSimulator,
)
from sutradhara.orchestrator.simulator.tool_simulator import ToolSimulator

logger = setup_logging()


async def async_main(args):
    vllm_client = VLLMClient(args.vllm_url, tokenizer_name=args.tokenizer)

    try:
        ok = await vllm_client.health_check()
        if not ok:
            logger.error("vLLM server not available at %s", args.vllm_url)
            return

        if args.qps <= 0:
            logger.error("QPS must be greater than zero. Received %s", args.qps)
            return

        requests_list = load_agentic_requests(args.trace)
        if args.num_requests and args.num_requests > 0:
            requests_list = requests_list[: args.num_requests]
        if not requests_list:
            logger.error("No requests scheduled for replay from trace %s", args.trace)
            return

        # Create policy and orchestrator
        policy = LatencyOptimalPolicy()
        orchestrator = OrchestratorSimulator(
            vllm_client=vllm_client,
            tool_simulator=ToolSimulator(),
            policy=policy,
            prefill_split=args.prefill_split,
            decode_tool_streaming=args.decode_tool_streaming,
            workload_aware_caching=args.workload_aware_cache
            or args.track_eviction_types,
            tool_time_multiplier=args.tool_time_multiplier,
            track_max_kv_hits=args.track_max_kv_hits,
        )

        logger.info("Running with seed=%d", args.seed)

        # Simulate Poisson arrivals — sleep then submit
        inter_arrivals = sample_poisson_inter_arrivals(
            len(requests_list), args.qps, seed=args.seed
        )
        requests_list = get_request_order(requests=requests_list, seed=args.seed)
        loop = asyncio.get_running_loop()
        base_time = loop.time()
        cumulative = 0.0

        for request, arrival_time in zip(requests_list, inter_arrivals):
            cumulative += arrival_time
            sleep_duration = (base_time + cumulative) - loop.time()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            orchestrator.submit(request, arrival_s=cumulative)

        logger.info("Submitted %d requests", len(requests_list))
        await orchestrator.wait_for_all_requests_to_complete(Path(args.metrics_dir))

    finally:
        await vllm_client.aclose()


def main():
    parser = argparse.ArgumentParser(
        description="Replay multiple agentic requests using Poisson arrivals",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to a single trace file to replay end-to-end",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL",
    )
    parser.add_argument(
        "--qps",
        type=float,
        required=True,
        help="Target requests per second for the Poisson arrival process",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=0,
        help="Number of requests to replay from the trace (0 means all)",
    )
    parser.add_argument(
        "--metrics-dir",
        required=True,
        help="Directory where metrics CSV files will be stored",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-14B",
        help="HuggingFace tokenizer name/path (must match the served model)",
    )
    parser.add_argument(
        "--prefill-split",
        action="store_true",
        default=False,
        help="Enable prefill-split optimization (overlap tool execution with Split A prefill)",
    )
    parser.add_argument(
        "--decode-tool-streaming",
        action="store_true",
        default=False,
        help="Enable decode tool streaming optimization (dispatch tools early)",
    )
    parser.add_argument(
        "--workload-aware-cache",
        action="store_true",
        default=False,
        help="Enable priority based kv cache eviction",
    )
    parser.add_argument(
        "--track-eviction-types",
        action="store_true",
        default=False,
        help="Pass orchestrator hints for eviction tracking (observe-only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for request ordering (default: 42)",
    )
    parser.add_argument(
        "--tool-time-multiplier",
        type=float,
        default=3.1,
        help="Multiplier for simulated tool latency (default: 3.1)",
    )
    parser.add_argument(
        "--track-max-kv-hits",
        action="store_true",
        default=False,
        help="Compute the theoretical max KV prefix-reuse ratio per iteration "
        "(off by default; tokenizes the full prompt — for offline analysis "
        "only)",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
