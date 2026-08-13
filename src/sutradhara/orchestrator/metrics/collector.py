"""Metrics collector for orchestrating metrics collection across the orchestrator."""

import time
from typing import Any, Dict, List, Optional

from sutradhara.orchestrator.logger.logger import setup_logging
from sutradhara.orchestrator.metrics.batch_metrics import BatchMetrics
from sutradhara.orchestrator.metrics.iteration_metrics import IterationMetrics
from sutradhara.orchestrator.metrics.overlap_metrics import OverlapMetrics
from sutradhara.orchestrator.metrics.request_metrics import RequestMetrics

logger = setup_logging()


class MetricsCollector:
    """Collects and aggregates metrics for agentic requests."""

    def __init__(self, request_id: str, enable_logging: bool = True):
        self.request_id = request_id
        self.start_time = time.time()
        self.iterations: List[IterationMetrics] = []
        self.overlap_metrics: List[OverlapMetrics] = []
        self.enable_logging = enable_logging

    def add_iteration(
        self,
        iteration_id: str,
        llm_response: Dict[str, Any],
        trace_iter: Dict[str, Any],
        max_kv_hit_rate: float,
        iter_e2e_time: float = 0.0,
        tool_execution_info: Optional[Dict[str, Any]] = None,
        relative_arrival_time_s: float = 0.0,
    ) -> IterationMetrics:
        metrics = self._extract_llm_metrics(llm_response)
        batch_metrics = self._extract_batch_metrics(
            llm_response.get("batch_metrics", [])
        )

        prefill_ms = self._calculate_prefill_time(batch_metrics)
        decode_ms = self._calculate_decode_time(batch_metrics)

        # Tool call counts from execution info
        # tool_call_time_ms is a timer-based measurement: first tool dispatch → next phase start
        n_tool_calls = 0
        tool_call_time_ms = 0.0
        if tool_execution_info:
            n_tool_calls = tool_execution_info.get("n_tool_calls", 0)
            tool_call_time_ms = tool_execution_info.get("tool_call_time_ms", 0.0)

        iter_e2e_time = (
            iter_e2e_time if iter_e2e_time != 0.0 else metrics["total_time_ms"]
        )
        iter_type = trace_iter.get("iter_type", "TOOL_DECODE")

        # Create iteration metrics
        iter_metrics = IterationMetrics(
            iteration_id=iteration_id,
            iter_type=iter_type,
            prompt_length=llm_response.get("prompt_tokens", 0),
            response_length=llm_response.get("completion_tokens", 0),
            kv_cache_hit_percentage=metrics.get("kv_cache_hit_percentage", 0.0) * 100.0,
            max_kv_cache_hit_percentage=max_kv_hit_rate * 100.0,
            waiting_time_ms=metrics["waiting_time_ms"],
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            batch_metrics=batch_metrics,
            n_tool_calls=n_tool_calls,
            tool_call_time_ms=tool_call_time_ms,
            relative_arrival_time_s=relative_arrival_time_s,
            ttft_ms=metrics["ttft_ms"],
            e2e_time_ms=iter_e2e_time,
            preempted=metrics.get("preempted", False), # type: ignore
            time_spent_in_preemption=metrics.get("time_spent_in_preemption_ms", 0.0),
        )

        self.iterations.append(iter_metrics)

        if self.enable_logging:
            self._log_iteration_metrics(metrics, llm_response, trace_iter)

        if "_split_a" in iteration_id:
            overlap = self._create_overlap_metrics(iter_metrics)
            self.overlap_metrics.append(overlap)

        return iter_metrics

    def _create_overlap_metrics(self, split_a: IterationMetrics) -> OverlapMetrics:
        return OverlapMetrics(
            request_id=self.request_id,
            iteration_id=split_a.iteration_id,
            waiting_ms=split_a.waiting_time_ms,
            prefill_ms=split_a.prefill_ms,
            tool_time_ms=split_a.tool_call_time_ms,
            e2e_time_ms=split_a.e2e_time_ms,
        )

    def build_request_metrics(
        self, orchestrator_queue_delay_ms: float = 0.0
    ) -> RequestMetrics:
        """Build final RequestMetrics from collected iterations.

        All totals are automatically computed from iterations via properties.
        """
        total_e2e_time_ms = sum(
            iter_metrics.e2e_time_ms for iter_metrics in self.iterations
        )
        request_e2e_time_ms = total_e2e_time_ms

        return RequestMetrics(
            request_id=self.request_id,
            e2e_time_ms=request_e2e_time_ms,
            orchestrator_queue_delay_ms=orchestrator_queue_delay_ms,
            iterations=self.iterations,
            relative_arrival_time_s=(
                self.iterations[0].relative_arrival_time_s if self.iterations else 0.0
            ),
            overlap_metrics=self.overlap_metrics,
        )

    @staticmethod
    def _extract_llm_metrics(llm_response: Dict[str, Any]) -> Dict[str, float]:
        return {
            "ttft_ms": llm_response.get("ttft_ms", 0.0),
            "total_time_ms": llm_response.get("total_time_ms", 0.0),
            "waiting_time_ms": llm_response.get("waiting_time_ms", 0.0),
            "kv_cache_hit_percentage": llm_response.get("kv_cache_hit_percentage", 0.0),
            "preempted": llm_response.get("preempted", False),
            "time_spent_in_preemption_ms": llm_response.get(
                "time_spent_in_preemption_ms", 0.0
            ),
        }

    @staticmethod
    def _extract_batch_metrics(
        batch_metrics_payload: List[Dict[str, Any]],
    ) -> List[BatchMetrics]:
        return [
            BatchMetrics(
                batch_id=entry.get("batch_id", 0),
                batch_total_tokens=entry.get("batch_total_tokens", 0),
                num_prefills=entry.get("num_prefills", 0),
                num_decodes=entry.get("num_decodes", 0),
                latency_ms=entry.get("latency_ms", 0.0),
            )
            for entry in batch_metrics_payload
        ]

    @staticmethod
    def _calculate_prefill_time(batch_metrics: List[BatchMetrics]) -> float:
        return sum(
            metrics.latency_ms for metrics in batch_metrics if metrics.num_prefills > 0
        )

    @staticmethod
    def _calculate_decode_time(batch_metrics: List[BatchMetrics]) -> float:
        return sum(
            metrics.latency_ms for metrics in batch_metrics if metrics.num_decodes > 0
        )

    @staticmethod
    def _log_iteration_metrics(
        metrics: Dict[str, float],
        llm_response: Dict[str, Any],
        trace_iter: Dict[str, Any],
    ) -> None:
        """Log metrics for a single iteration using already-extracted metrics."""
        logger.info(f"LLM response time: {metrics['total_time_ms']:.2f}ms")
        logger.info(f"Generated tokens: {llm_response.get('completion_tokens', 0)}")
        logger.info(f"TTFT: {metrics['ttft_ms']:.2f}ms")
        if metrics["waiting_time_ms"]:
            logger.info(f"Waiting time: {metrics['waiting_time_ms']:.2f}ms")
        logger.info(f"KV Cache Hit %: {metrics['kv_cache_hit_percentage']:.2%}")
        if metrics.get("preempted"):
            logger.info(
                f"Preempted: yes ({metrics.get('time_spent_in_preemption_ms', 0.0):.2f} ms)"
            )

        n_tool_calls = trace_iter.get("n_tool_calls", 0)
        if n_tool_calls > 0:
            logger.info(f"Trace had {n_tool_calls} tool call(s):")
            for i, tool_info in enumerate(
                trace_iter.get("tool_info", [])[:n_tool_calls]
            ):
                tool_name = (
                    tool_info.get("tool_name")
                    or tool_info.get("function")
                    or tool_info.get("name")
                    or f"tool_{i}"
                )
                tool_latency = tool_info.get("latency", 0)
                logger.info(f"  - {tool_name}: {tool_latency:.2f}ms")
