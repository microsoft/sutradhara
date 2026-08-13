"""Latency-optimal scheduling policy.

FIFO priority assignment — request 0 gets priority 0, etc.
Groundwork for more sophisticated policies that may reorder, batch, or
re-prioritise requests based on runtime signals.
"""

from sutradhara.orchestrator.policies.base import SchedulingPolicy
from sutradhara.orchestrator.common.request_classes import AgenticRequest


class LatencyOptimalPolicy(SchedulingPolicy):
    def __init__(self):
        self._next_priority = 0

    def schedule(self, request: AgenticRequest) -> int:
        priority = self._next_priority
        self._next_priority += 1
        return priority
