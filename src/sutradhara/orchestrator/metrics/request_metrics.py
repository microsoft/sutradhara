"""Request-level metrics for a whole agentic request.

FTR (First Token Rendered) = time of the first token of the request's last
iteration, measured from request arrival.
e2e request time = sum of per-iteration e2e times = prefill + decode + tool +
scheduling-delay totals (sanity check).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .iteration_metrics import IterationMetrics
from .overlap_metrics import OverlapMetrics


@dataclass
class RequestMetrics:
    request_id: str
    e2e_time_ms: float
    orchestrator_queue_delay_ms: float = 0.0
    iterations: List[IterationMetrics] = field(default_factory=list)
    overlap_metrics: List[OverlapMetrics] = field(default_factory=list)
    relative_arrival_time_s: float = 0.0

    # Computed properties - these are calculated from iterations
    @property
    def total_iterations(self) -> int:
        """Total number of iterations."""
        return len(self.iterations)

    @property
    def total_llm_wait_time_ms(self) -> float:
        """Total waiting time accumulated inside vLLM across iterations."""
        return sum(m.waiting_time_ms for m in self.iterations)

    @property
    def total_scheduling_delay_ms(self) -> float:
        """Total scheduling delay = LLM wait + orchestrator queue delay."""
        return self.total_llm_wait_time_ms + self.orchestrator_queue_delay_ms

    @property
    def total_prefill_time_ms(self) -> float:
        """Total prefill time across all iterations."""
        return sum(m.prefill_ms for m in self.iterations)

    @property
    def total_decode_time_ms(self) -> float:
        """Total decode time across all iterations."""
        return sum(m.decode_ms for m in self.iterations)

    @property
    def total_tool_time_ms(self) -> float:
        """Total tool time on the critical path across all iterations.

        For split_a iterations, uses overlap critical_path to avoid
        double-counting time already in prefill/scheduling totals.
        """
        overlap_cp = {o.iteration_id: o.critical_path_time_ms
                      for o in self.overlap_metrics}
        return sum(
            overlap_cp.get(m.iteration_id, m.tool_call_time_ms)
            for m in self.iterations
        )

    @property
    def first_token_render_ms(self) -> float:
        """Time to first token using multi-iteration definition."""
        if not self.iterations:
            return 0.0
        if len(self.iterations) == 1:
            return self.iterations[0].ttft_ms or 0.0
        total_e2e_time = sum(m.e2e_time_ms for m in self.iterations)
        return total_e2e_time - (self.iterations[-1].decode_ms or 0.0)

    @property
    def total_preemption_time_ms(self) -> float:
        return sum(
            (iter_metrics.time_spent_in_preemption or 0.0)
            for iter_metrics in self.iterations
        )

    @property
    def n_preempted_iterations(self) -> int:
        return sum(1 for iter_metrics in self.iterations if iter_metrics.preempted)

    def to_csv_row(
        self,
        orchestrator_queue_delay_ms: Optional[float] = None,
        request_id_override: Optional[str] = None,
    ) -> dict:
        """Return a dictionary suitable for CSV export (excludes nested iterations).
        Args:
            orchestrator_queue_delay_ms: Optional override for queue delay that
                will also be used to recompute total scheduling delay. Defaults
                to the value stored on the object.
            request_id_override: Optional override for request_id value.
        """
        queue_delay_ms = (
            self.orchestrator_queue_delay_ms
            if orchestrator_queue_delay_ms is None
            else orchestrator_queue_delay_ms
        )
        total_scheduling_delay_ms = self.total_llm_wait_time_ms + queue_delay_ms
        request_id_value = request_id_override or self.request_id
        return {
            "request_id": request_id_value,
            "relative_arrival_time_s": self.relative_arrival_time_s,
            "total_iterations": self.total_iterations,
            "total_llm_wait_time_ms": self.total_llm_wait_time_ms,
            "orchestrator_queue_delay_ms": queue_delay_ms,
            "total_scheduling_delay_ms": total_scheduling_delay_ms,
            "total_prefill_time_ms": self.total_prefill_time_ms,
            "total_decode_time_ms": self.total_decode_time_ms,
            "total_tool_time_ms": self.total_tool_time_ms,
            "first_token_render_ms": self.first_token_render_ms,
            "n_preempted_iterations": self.n_preempted_iterations,
            "total_preemption_time_ms": self.total_preemption_time_ms,
            "e2e_time_ms": self.e2e_time_ms,
        }

    def to_csv_payload(
        self,
        orchestrator_queue_delay_ms: Optional[float] = None,
    ) -> dict:
        """Return CSV-aligned rows for request/iteration/batch/tool metrics."""
        request_row = self.to_csv_row(
            orchestrator_queue_delay_ms=orchestrator_queue_delay_ms,
            request_id_override=self.request_id,
        )

        iteration_rows = []
        batch_rows = []
        overlap_rows = []
        for iteration in self.iterations:
            iteration_rows.append(iteration.to_csv_row(request_id=self.request_id))
            for batch_metric in iteration.batch_metrics:
                batch_rows.append(
                    batch_metric.to_csv_row(
                        request_id=self.request_id,
                        iteration_id=iteration.iteration_id,
                    )
                )

        for overlap_metric in self.overlap_metrics:
            overlap_rows.append(overlap_metric.to_csv_row())

        return {
            "request": request_row,
            "iterations": iteration_rows,
            "batches": batch_rows,
            "overlaps": overlap_rows,
        }
