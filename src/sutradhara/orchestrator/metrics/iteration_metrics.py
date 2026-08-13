"""Iteration-level metrics (one LLM call): scheduling delay, prefill, TTFT
(prefill + delay), decode time, prefix-cache hit, and preemption recompute."""

from dataclasses import dataclass, field
from typing import List, Optional

from sutradhara.orchestrator.metrics.batch_metrics import BatchMetrics


@dataclass
class IterationMetrics:
    iteration_id: str
    iter_type: str  # TOOL_DECODE or RESPONSE_DECODE
    prompt_length: int
    response_length: int

    kv_cache_hit_percentage: float
    max_kv_cache_hit_percentage: float

    waiting_time_ms: float
    prefill_ms: float
    decode_ms: float

    batch_metrics: List[BatchMetrics] = field(default_factory=list)

    n_tool_calls: int = 0
    tool_call_time_ms: float = 0.0

    ttft_ms: float = 0.0
    e2e_time_ms: float = 0.0

    preempted: Optional[bool] = False
    time_spent_in_preemption: Optional[float] = 0.0
    relative_arrival_time_s: float = 0.0

    @property
    def n_batches(self) -> int:
        return len(self.batch_metrics)

    def to_csv_row(self, request_id: str = "") -> dict:
        """Return a dictionary suitable for CSV export (excludes nested batch_metrics)."""
        row = {
            "request_id": request_id,
            "iteration_id": self.iteration_id,
            "iter_type": self.iter_type,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "kv_cache_hit_percentage": self.kv_cache_hit_percentage,
            "max_kv_cache_hit_percentage": self.max_kv_cache_hit_percentage,
            "n_tool_calls": self.n_tool_calls,
            "tool_call_time_ms": self.tool_call_time_ms,
            "n_batches": self.n_batches,
            "waiting_time_ms": self.waiting_time_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "ttft_ms": self.ttft_ms,
            "e2e_time_ms": self.e2e_time_ms,
            "preempted": self.preempted,
            "time_spent_in_preemption_ms": self.time_spent_in_preemption,
        }
        return row
