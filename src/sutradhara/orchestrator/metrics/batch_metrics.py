"""Batch-level metrics (one vLLM scheduler batch): latency and composition."""

from dataclasses import dataclass


@dataclass
class BatchMetrics:
    batch_id: int
    batch_total_tokens: int
    num_prefills: int
    num_decodes: int
    latency_ms: float

    def to_csv_row(self, request_id: str = "", iteration_id: str = "") -> dict:
        """Return a dictionary suitable for CSV export."""
        latency = round(self.latency_ms, 2)
        return {
            "request_id": request_id,
            "iteration_id": iteration_id,
            "batch_id": self.batch_id,
            "batch_total_tokens": self.batch_total_tokens,
            "num_prefills": self.num_prefills,
            "num_decodes": self.num_decodes,
            "latency_ms": latency,
        }
