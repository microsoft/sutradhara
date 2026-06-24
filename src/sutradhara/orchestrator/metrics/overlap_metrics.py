from dataclasses import dataclass


@dataclass
class OverlapMetrics:
    request_id: str
    iteration_id: str

    waiting_ms: float       # split_a scheduling delay
    prefill_ms: float       # split_a prefill time
    tool_time_ms: float     # pipeline timer: first dispatch → tool completion
    e2e_time_ms: float      # overlap phase wall-clock: split_a_start → split_a_end

    @property
    def llm_time_ms(self) -> float:
        return self.waiting_ms + self.prefill_ms

    @property
    def critical_path_time_ms(self) -> float:
        """Extra time tools added beyond LLM in the overlap phase."""
        return max(0.0, self.e2e_time_ms - self.llm_time_ms)

    @property
    def is_tool_bottleneck(self) -> bool:
        return self.critical_path_time_ms > 0

    @property
    def saved_ms(self) -> float:
        """Tool pipeline time hidden by overlapping with LLM."""
        return self.tool_time_ms - self.critical_path_time_ms

    def to_csv_row(self) -> dict:
        return {
            "request_id": self.request_id,
            "iteration_id": self.iteration_id,
            "waiting_ms": self.waiting_ms,
            "prefill_ms": self.prefill_ms,
            "llm_time_ms": self.llm_time_ms,
            "tool_time_ms": self.tool_time_ms,
            "e2e_time_ms": self.e2e_time_ms,
            "critical_path_time_ms": self.critical_path_time_ms,
            "saved_ms": self.saved_ms,
            "is_tool_bottleneck": self.is_tool_bottleneck,
        }
