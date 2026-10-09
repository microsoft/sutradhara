import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from sutradhara.orchestrator.common.max_kv_hit_rate import g_kv_hit_rate_tracker
from sutradhara.orchestrator.optimizations.prefill_split import (
    PrefillSplitter,
    ProductionPrefillSplitter,
)
from sutradhara.orchestrator.optimizations.decode_tool_streaming import (
    DeferredToolExec,
    StreamingToolWatcher,
    ToolTriggerDetector,
    TraceTriggerDetector,
    has_trigger_indices,
)
from sutradhara.orchestrator.llm_client.vllm_client import VLLMClient
from sutradhara.orchestrator.logger.logger import setup_logging
from sutradhara.orchestrator.metrics.collector import MetricsCollector
from sutradhara.orchestrator.metrics.metrics_csv import save_metrics_csv
from sutradhara.orchestrator.metrics.request_metrics import RequestMetrics
from sutradhara.orchestrator.policies.base import SchedulingPolicy
from sutradhara.orchestrator.common.request_classes import (
    AgenticRequest,
    RequestReplayOutcome,
)
from sutradhara.orchestrator.simulator.tool_simulator import ToolSimulator

logger = setup_logging()


@dataclass
class _IterState:
    """Mutable state carried between iterations of replay_request."""

    tool_tasks: List[asyncio.Task] = field(default_factory=list)
    tool_meta: Dict = field(default_factory=dict)
    split_a_task: Optional[asyncio.Task] = None
    split_a_start: Optional[float] = None

    def clear_tools(self) -> None:
        self.tool_tasks = []
        self.tool_meta = {}

    def consume_split_a(self):
        task, start = self.split_a_task, self.split_a_start
        self.split_a_task = None
        self.split_a_start = None
        return task, start


@dataclass
class _ReplayContext:
    """Immutable request-level context for an entire replay."""

    request_id: str
    request_priority: int
    trace_iterations: List[Dict[str, Any]]
    collector: MetricsCollector
    relative_arrival_time_s: float


@dataclass
class _IterConfig:
    """Per-iteration configuration (read-only within iteration handlers)."""

    iteration_id: int
    trace_iter: Dict[str, Any]
    prompt: str
    max_tokens: int
    detector: Any  # TraceTriggerDetector | DeferredToolExec
    raw_tool_info: List
    prompt_split: int
    next_iter_is_split: bool
    next_split_point: int
    iter_time_start: float


class OrchestratorSimulator:
    def __init__(
        self,
        vllm_client: VLLMClient,
        tool_simulator: ToolSimulator,
        policy: SchedulingPolicy,
        prefill_split: bool = False,
        decode_tool_streaming: bool = False,
        workload_aware_caching: bool = False,
        track_max_kv_hits: bool = False,
        prefill_splitter: Optional[PrefillSplitter] = None,
        sequential_tools: bool = False,
    ):
        self.vllm_client = vllm_client
        self.tool_simulator = tool_simulator
        self.policy = policy
        self.prefill_split = prefill_split
        self.decode_tool_streaming = decode_tool_streaming
        self.workload_aware_caching = workload_aware_caching

        self._tasks: List[asyncio.Task] = []
        self._base_time: Optional[float] = None
        self._request_replayer = AgenticRequestReplayer(
            vllm_client=self.vllm_client,
            tool_simulator=self.tool_simulator,
            prefill_split=self.prefill_split,
            decode_tool_streaming=self.decode_tool_streaming,
            workload_aware_caching=self.workload_aware_caching,
            track_max_kv_hits=track_max_kv_hits,
            prefill_splitter=prefill_splitter,
            sequential_tools=sequential_tools,
        )

    def submit(self, agentic_request: AgenticRequest, arrival_s: float) -> None:
        if self._base_time is None:
            loop = asyncio.get_running_loop()
            self._base_time = loop.time()

        # TODO: Global FIFO Priority; Ideally you'll do something more interesting here
        # This is the place to change if you want to introduce routing in multi replica
        priority = self.policy.schedule(agentic_request)

        task = asyncio.create_task(
            self._run_request(agentic_request, arrival_s, priority)
        )
        self._tasks.append(task)

    async def _run_request(
        self,
        request: AgenticRequest,
        arrival_s: float,
        priority: int,
    ) -> RequestReplayOutcome:
        assert self._base_time is not None
        loop = asyncio.get_running_loop()
        start = loop.time()

        metrics = await self._request_replayer.replay_request(
            request_id=request.request_id,
            trace_iterations=request.iterations,
            request_priority=priority,
            relative_arrival_time_s=arrival_s,
        )

        orchestrator_queue_delay_ms = (
            max((start - self._base_time) - arrival_s, 0.0) * 1000.0
        )
        metrics.orchestrator_queue_delay_ms = orchestrator_queue_delay_ms
        metrics.e2e_time_ms += orchestrator_queue_delay_ms

        end = loop.time()
        return RequestReplayOutcome(
            request_id=request.request_id,
            arrival_s=arrival_s,
            start_s=start - self._base_time,
            end_s=end - self._base_time,
            metrics=metrics,
        )

    async def wait_for_all_requests_to_complete(self, metrics_dir: Path) -> None:
        """Await all dispatched requests, summarize, and save metrics."""
        assert self._base_time is not None
        outcomes: List[RequestReplayOutcome] = await asyncio.gather(*self._tasks)
        save_metrics_csv(metrics_dir, outcomes)


class SequentialToolCall:
    """
    Optionally serialize tool sleeps within one decode drain.
    """

    def __init__(self, sequential: bool) -> None:
        self._sequential = sequential
        self._tail: Optional[asyncio.Task] = None

    def launch(self, coro) -> asyncio.Task:
        if not self._sequential:
            return asyncio.create_task(coro)

        prev = self._tail

        async def _run():
            if prev is not None:
                await prev
            return await coro

        task = asyncio.create_task(_run())
        self._tail = task
        return task


class AgenticRequestReplayer:
    def __init__(
        self,
        vllm_client: VLLMClient,
        tool_simulator: ToolSimulator,
        prefill_split: bool = False,
        decode_tool_streaming: bool = False,
        workload_aware_caching: bool = False,
        track_max_kv_hits: bool = False,
        prefill_splitter: Optional[PrefillSplitter] = None,
        sequential_tools: bool = False,
    ):
        self.vllm_client = vllm_client
        self.tool_simulator = tool_simulator
        self.prefill_split = prefill_split
        self.decode_tool_streaming = decode_tool_streaming
        self.workload_aware_caching = workload_aware_caching
        self.track_max_kv_hits = track_max_kv_hits
        self.prefill_splitter = prefill_splitter or ProductionPrefillSplitter()
        self.sequential_tools = sequential_tools

    def _max_kv_hits(self, prompt: str) -> float:
        """Theoretical max KV prefix-reuse ratio for *prompt*."""
        if not self.track_max_kv_hits:
            return -1.0
        return g_kv_hit_rate_tracker.compute_max_kv_hits(prompt)

    @staticmethod
    def _build_llm_response(
        text: str,
        metrics: Dict[str, Any],
        usage: Dict[str, Any],
        ttft_fallback_ms: float,
        total_time_fallback_ms: float,
        **overrides: Any,
    ) -> Dict[str, Any]:
        response = {
            "text": text,
            "tokens": [text],
            "ttft_ms": metrics.get("ttft_ms", ttft_fallback_ms),
            "total_time_ms": metrics.get("e2e_latency_ms", total_time_fallback_ms),
            "waiting_time_ms": metrics.get("waiting_time", 0.0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "kv_cache_hit_percentage": metrics.get(
                "prefill_kv_cache_hit_percentage",
                metrics.get("kv_cache_hit_percentage", 0.0),
            ),
            "preempted": metrics.get("preempted", False),
            "time_spent_in_preemption_ms": metrics.get(
                "time_spent_in_preemption_ms", 0.0
            ),
            "batch_metrics": metrics.get("batch_metrics") or [],
            "tool_results": [],
            "pending_tasks": [],
            "launched_tool_count": 0,
        }
        response.update(overrides)
        return response

    def _tool_coro(self, tool_info: Dict):
        """Coroutine that simulates a single tool call."""
        latency = tool_info.get("latency", 0)
        return self.tool_simulator.execute_tool(
            tool_name="simulated_tool",
            tool_args={"trace_info": tool_info},
            latency_ms=latency,
        )

    async def _drive_split_a(
        self,
        split_a_prompt: str,
        priority: int,
        request_id: str,
        iteration_id: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """Send the split-A prefix as a non-streaming request (1 token)."""
        iter_req_id = f"{request_id}_iter{iteration_id}_split_a"
        gen = self.vllm_client.submit_partial_prefill(
            split_a_prompt, priority, iter_req_id
        )
        # Non-streaming generator yields exactly one chunk
        result = await gen.__anext__()
        await gen.aclose()
        end_time = time.perf_counter()
        wall_ms = (end_time - start_time) * 1000
        return self._build_llm_response(
            text=result.get("text", ""),
            metrics=result.get("metrics") or {},
            usage=result.get("usage") or {},
            ttft_fallback_ms=wall_ms,
            total_time_fallback_ms=wall_ms,
        )

    async def _drain_stream_and_launch_tools(
        self,
        generator: AsyncGenerator[Dict[str, Any], None],
        start_time: float,
        request_id_tag: str,
        iteration_id: int,
        tool_trigger: ToolTriggerDetector,
        wait_for_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        Drains the vLLM SSE stream, fires tool calls via the tool_trigger,
        and returns aggregated metrics. If wait_for_tools is False, returns
        running tool tasks in 'pending_tasks' for cross-iteration overlap.
        """
        full_text_parts: list[str] = []
        first_token_time = None
        first_tool_dispatch_time = None

        final_usage = {}
        final_metrics = {}

        token_idx = 0
        launched_tool_tasks = []
        sequential_tool_call = SequentialToolCall(self.sequential_tools)

        async for chunk in generator:
            token = chunk.get("text", "")

            if token:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                full_text_parts.append(token)

                # Ask the tool_trigger which tools (if any) to fire at this token
                for tool_info in tool_trigger.on_token(token_idx, token):
                    if first_tool_dispatch_time is None:
                        first_tool_dispatch_time = time.perf_counter()
                    logger.info(
                        "Triggering tool at token %d for %s", token_idx, request_id_tag
                    )
                    launched_tool_tasks.append(
                        sequential_tool_call.launch(self._tool_coro(tool_info))
                    )
                token_idx += 1

            if chunk.get("usage"):
                final_usage = chunk["usage"]

            if chunk.get("metrics") is not None:
                final_metrics = chunk["metrics"]

        # --- Deferred / end-of-stream tools ---
        for tool_info in tool_trigger.on_stream_end():
            if first_tool_dispatch_time is None:
                first_tool_dispatch_time = time.perf_counter()
            logger.info("Triggering deferred tool for %s", request_id_tag)
            launched_tool_tasks.append(
                sequential_tool_call.launch(self._tool_coro(tool_info))
            )

        tool_results = []
        pending_tasks = []
        tool_call_time_ms = 0.0

        # CRITICAL CHANGE: Decision to await or defer
        if wait_for_tools:
            if launched_tool_tasks:
                tool_results = await asyncio.gather(*launched_tool_tasks)
                if first_tool_dispatch_time is not None:
                    tool_call_time_ms = (
                        time.perf_counter() - first_tool_dispatch_time
                    ) * 1000
        else:
            # Pass them back to the caller to overlap with next iteration
            pending_tasks = launched_tool_tasks

        end_time = time.perf_counter()

        full_text = "".join(full_text_parts)
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else 0
        total_time_ms = (end_time - start_time) * 1000

        return self._build_llm_response(
            text=full_text,
            metrics=final_metrics,
            usage=final_usage,
            ttft_fallback_ms=ttft_ms,
            total_time_fallback_ms=total_time_ms,
            tool_results=tool_results,
            pending_tasks=pending_tasks,
            launched_tool_count=len(launched_tool_tasks),
            first_tool_dispatch_time=first_tool_dispatch_time,
            tool_call_time_ms=tool_call_time_ms,
        )

    # --- Small helpers for cross-iteration state management ---

    def _save_pending_tools(
        self,
        carry: _IterState,
        llm_response: Dict,
        raw_tool_info: List,
    ) -> None:
        """Stash tool tasks + metadata for overlap with the next split-A."""
        carry.tool_tasks = llm_response["pending_tasks"]
        carry.tool_meta = {
            "count": llm_response["launched_tool_count"],
            "trace_info": raw_tool_info,
            "first_tool_dispatch_time": llm_response.get("first_tool_dispatch_time"),
        }

    def _pre_launch_next_split_a(
        self,
        carry: _IterState,
        ctx: _ReplayContext,
        iteration_id: int,
        next_split_point: int,
    ) -> None:
        """Fire the next iteration's split-A immediately (overlap optimisation)."""
        next_prompt = ctx.trace_iterations[iteration_id + 1]["prompt_json"]
        carry.split_a_start = time.perf_counter()
        carry.split_a_task = asyncio.create_task(
            self._drive_split_a(
                next_prompt[:next_split_point],
                ctx.request_priority,
                ctx.request_id,
                iteration_id + 1,
                carry.split_a_start,
            )
        )

    def _finalize_tools(
        self,
        cfg: "_IterConfig",
        ctx: "_ReplayContext",
        carry: "_IterState",
        llm_response: Dict,
    ) -> None:
        if cfg.next_iter_is_split:
            self._save_pending_tools(carry, llm_response, cfg.raw_tool_info)
            self._pre_launch_next_split_a(
                carry,
                ctx,
                cfg.iteration_id,
                cfg.next_split_point,
            )
        else:
            carry.clear_tools()

    def _build_iter_config(
        self,
        iteration_id: int,
        trace_iterations: List[Dict[str, Any]],
        split_cache: Dict[int, int],
    ) -> _IterConfig:
        """Compute read-only per-iteration configuration."""
        trace_iter = trace_iterations[iteration_id]
        prompt = trace_iter["prompt_json"]

        def split_point(idx: int) -> int:
            # get_split_a_end_token is a pure function of the iteration's
            # prompt_json; cache it so a look-ahead and the later current-iter
            # lookup don't both re-parse the same prompt.
            if idx not in split_cache:
                split_cache[idx] = self.prefill_splitter.get_split_a_end_token(
                    trace_iterations[idx].get("prompt_json", "")
                )
            return split_cache[idx]

        prompt_split = split_point(iteration_id) if self.prefill_split else -1
        max_tokens = trace_iter.get("response_length", 512)
        raw_tool_info = trace_iter.get("tool_info", [])

        if self.decode_tool_streaming:
            if has_trigger_indices(raw_tool_info):
                # Trace already carries pre-recorded trigger positions — replay them.
                detector = TraceTriggerDetector(raw_tool_info, max_tokens)
            else:
                # No baked-in positions — watch the live decode stream and
                # dispatch each tool as its JSON object completes.
                detector = StreamingToolWatcher(raw_tool_info, max_tokens)
        else:
            detector = DeferredToolExec(raw_tool_info)

        next_iter_is_split = False
        next_split_point = -1
        if self.prefill_split and iteration_id + 1 < len(trace_iterations):
            next_split_point = split_point(iteration_id + 1)
            next_iter_is_split = next_split_point > 0

        return _IterConfig(
            iteration_id=iteration_id,
            trace_iter=trace_iter,
            prompt=prompt,
            max_tokens=max_tokens,
            detector=detector,
            raw_tool_info=raw_tool_info,
            prompt_split=prompt_split,
            next_iter_is_split=next_iter_is_split,
            next_split_point=next_split_point,
            iter_time_start=time.perf_counter(),
        )

    # --- Iteration handlers ---

    async def _handle_standard_iteration(
        self,
        cfg: _IterConfig,
        ctx: _ReplayContext,
        carry: _IterState,
    ) -> None:
        """Process one non-split iteration (decode → tools → optional pre-launch)."""
        req_start = time.perf_counter()
        iter_request_id = f"{ctx.request_id}_iter{cfg.iteration_id}"

        hints = (
            self.vllm_client.tag_kv_blocks(cfg.prompt)
            if self.workload_aware_caching
            else None
        )
        response_generator = self.vllm_client.extend_prefill(
            cfg.prompt,
            cfg.max_tokens,
            ctx.request_priority,
            iter_request_id,
            orchestrator_hints=hints,
        )

        should_wait = not cfg.next_iter_is_split

        llm_response = await self._drain_stream_and_launch_tools(
            response_generator,
            req_start,
            request_id_tag=iter_request_id,
            iteration_id=cfg.iteration_id,
            tool_trigger=cfg.detector,
            wait_for_tools=should_wait,
        )

        self._finalize_tools(cfg, ctx, carry, llm_response)

        iter_time_end = time.perf_counter()
        # When this iteration's tools are deferred (next iter is a split), they
        # are awaited and reported under the next split-A, so tool_results is
        # empty here (n_tool_calls=0 while triggered_count may be > 0).
        tool_results = llm_response.get("tool_results", [])

        ctx.collector.add_iteration(
            str(cfg.iteration_id),
            llm_response,
            cfg.trace_iter,
            self._max_kv_hits(cfg.prompt),
            iter_e2e_time=(iter_time_end - cfg.iter_time_start) * 1000,
            tool_execution_info={
                "tool_results": tool_results,
                "triggered_count": len(cfg.raw_tool_info),
                "n_tool_calls": len(tool_results),
                "tool_call_time_ms": llm_response.get("tool_call_time_ms", 0.0),
            },
            relative_arrival_time_s=ctx.relative_arrival_time_s,
        )

    async def _handle_split_iteration(
        self,
        cfg: _IterConfig,
        ctx: _ReplayContext,
        carry: _IterState,
    ) -> None:
        """Process one prefill-split iteration (split-A ‖ tools → split-B → optional pre-launch)."""
        # Use pre-launched split-A or create on-the-fly as fallback
        if carry.split_a_task is not None:
            split_a_task, split_a_start = carry.consume_split_a()
        else:
            split_a_start = time.perf_counter()
            split_a_task = asyncio.create_task(
                self._drive_split_a(
                    cfg.prompt[: cfg.prompt_split],
                    ctx.request_priority,
                    ctx.request_id,
                    cfg.iteration_id,
                    split_a_start,
                )
            )

        logger.info(
            "Overlapping Split A with %d pending tools",
            len(carry.tool_tasks),
        )

        # Gather split-A and carry-over tools concurrently
        tool_end_time = None
        if carry.tool_tasks:

            async def _gather_tools(tasks):
                results = await asyncio.gather(*tasks)
                return results, time.perf_counter()

            tool_meta_task = asyncio.create_task(_gather_tools(carry.tool_tasks))
            results = await asyncio.gather(split_a_task, tool_meta_task)
            split_a_result = results[0]
            prev_tool_results, tool_end_time = results[1]
        else:
            split_a_result = await split_a_task
            prev_tool_results = []

        split_a_end = time.perf_counter()

        first_dispatch_a = carry.tool_meta.get("first_tool_dispatch_time")
        tool_call_time_ms_a = (
            (tool_end_time - first_dispatch_a) * 1000
            if first_dispatch_a and tool_end_time
            else 0.0
        )

        ctx.collector.add_iteration(
            f"{cfg.iteration_id}_split_a",
            split_a_result,
            cfg.trace_iter,
            self._max_kv_hits(cfg.prompt[: cfg.prompt_split]),
            iter_e2e_time=(split_a_end - split_a_start) * 1000,
            tool_execution_info={
                "tool_results": prev_tool_results,
                "triggered_count": carry.tool_meta.get("count", 0),
                "note": "Tools overlapped with Split A",
                "n_tool_calls": len(prev_tool_results),
                "tool_call_time_ms": tool_call_time_ms_a,
            },
            relative_arrival_time_s=ctx.relative_arrival_time_s,
        )

        carry.clear_tools()

        # Launch Split B (full context generation)
        split_b_start = time.perf_counter()
        split_b_request_id = f"{ctx.request_id}_iter{cfg.iteration_id}_split_b"
        split_b_hints = (
            self.vllm_client.tag_kv_blocks(cfg.prompt)
            if self.workload_aware_caching
            else None
        )
        response_generator_b = self.vllm_client.extend_prefill(
            cfg.prompt,
            cfg.max_tokens,
            ctx.request_priority,
            split_b_request_id,
            orchestrator_hints=split_b_hints,
        )

        should_wait_b = not cfg.next_iter_is_split

        llm_response_b = await self._drain_stream_and_launch_tools(
            response_generator_b,
            split_b_start,
            request_id_tag=split_b_request_id,
            iteration_id=cfg.iteration_id,
            tool_trigger=cfg.detector,
            wait_for_tools=should_wait_b,
        )

        self._finalize_tools(cfg, ctx, carry, llm_response_b)

        split_b_end = time.perf_counter()

        ctx.collector.add_iteration(
            f"{cfg.iteration_id}_split_b",
            llm_response_b,
            cfg.trace_iter,
            self._max_kv_hits(cfg.prompt),
            iter_e2e_time=(split_b_end - split_b_start) * 1000,
            tool_execution_info={
                "tool_results": llm_response_b.get("tool_results", []),
                "n_tool_calls": len(llm_response_b.get("tool_results", [])),
                "tool_call_time_ms": llm_response_b.get("tool_call_time_ms", 0.0),
            },
            relative_arrival_time_s=ctx.relative_arrival_time_s,
        )

    async def replay_request(
        self,
        request_id: str,
        trace_iterations: List[Dict[str, Any]],
        request_priority: int,
        relative_arrival_time_s: float,
    ) -> RequestMetrics:
        if not trace_iterations:
            raise ValueError("Trace iterations are required.")

        ctx = _ReplayContext(
            request_id=request_id,
            request_priority=request_priority,
            trace_iterations=trace_iterations,
            collector=MetricsCollector(request_id),
            relative_arrival_time_s=relative_arrival_time_s,
        )
        carry = _IterState()
        split_cache: Dict[int, int] = {}

        logger.info("Starting replay of Agentic Request %s", request_id)

        for iteration_id in range(len(trace_iterations)):
            cfg = self._build_iter_config(iteration_id, trace_iterations, split_cache)

            if cfg.prompt_split <= 0:
                await self._handle_standard_iteration(cfg, ctx, carry)
            else:
                await self._handle_split_iteration(cfg, ctx, carry)

        return ctx.collector.build_request_metrics()
