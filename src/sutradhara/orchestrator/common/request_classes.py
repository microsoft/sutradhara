from dataclasses import dataclass
from typing import Any, Dict, List

from sutradhara.orchestrator.metrics.request_metrics import RequestMetrics


@dataclass
class AgenticRequest:
    request_id: str
    iterations: List[Dict[str, Any]]


@dataclass
class RequestReplayOutcome:
    request_id: str
    arrival_s: float
    start_s: float
    end_s: float
    metrics: RequestMetrics

    def to_dict(self) -> Dict[str, Any]:
        csv_payload = self.metrics.to_csv_payload()
        return {
            "request_id": self.request_id,
            "arrival_s": self.arrival_s,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "latency_ms": (self.end_s - self.arrival_s) * 1000.0,
            "queue_delay_ms": (self.start_s - self.arrival_s) * 1000.0,
            "metrics": csv_payload,
        }
