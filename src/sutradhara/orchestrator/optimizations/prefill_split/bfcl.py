"""Prefill split for BFCL v4 traces: no split-point logic for ChatML, so a no-op."""

from sutradhara.orchestrator.logger.logger import setup_logging
from sutradhara.orchestrator.optimizations.prefill_split.base import PrefillSplitter

logger = setup_logging()


class BFCLPrefillSplitter(PrefillSplitter):
    """No-ops to baseline; only the JSON message-array format has a splitter."""

    def __init__(self):
        logger.warning(
            "Prefill-split has no ChatML implementation for BFCL traces — "
            "degrading to baseline (no split); this will not overlap tool "
            "execution with prefill and may hurt latency vs. a working split."
        )

    def get_split_a_end_token(self, prompt: str) -> int:
        return -1
