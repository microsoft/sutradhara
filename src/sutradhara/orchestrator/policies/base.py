"""Abstract base for scheduling policies."""

from abc import ABC, abstractmethod

from sutradhara.orchestrator.common.request_classes import AgenticRequest


class SchedulingPolicy(ABC):
    """Decides the priority for each incoming request.

    Subclasses implement `schedule` which is called by the orchestrator
    every time a new request arrives.  Future policies may also inspect
    cluster state; for now the interface is intentionally minimal.
    """

    @abstractmethod
    def schedule(self, request: AgenticRequest) -> int:
        """Return a priority for *config* (lower value = higher priority)."""
        ...
