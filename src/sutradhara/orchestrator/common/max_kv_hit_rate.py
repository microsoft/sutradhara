from typing import Dict, List

import tiktoken


class KVHitRateTracker:
    """Tracks the theoretical max KV-cache prefix-reuse ratio of each prompt
    against all previously-seen prompts.
    """

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        # Trie of token ids; each node maps token_id -> child node.
        self._root: Dict[int, dict] = {}

    def _longest_prefix_match(self, token_ids: List[int]) -> int:
        """Length of the longest prefix of ``token_ids`` that is also a prefix
        of some previously inserted prompt."""
        node = self._root
        depth = 0
        for tok in token_ids:
            nxt = node.get(tok)
            if nxt is None:
                break
            node = nxt
            depth += 1
        return depth

    def _insert(self, token_ids: List[int]) -> None:
        node = self._root
        for tok in token_ids:
            node = node.setdefault(tok, {})

    def compute_max_kv_hits(self, input_prompt: str) -> float:
        token_ids = self.encoding.encode(input_prompt)
        if not token_ids:
            return 0.0
        hits = self._longest_prefix_match(token_ids)
        self._insert(token_ids)
        return hits / len(token_ids)


# TODO: For prototyping sake let's use a global object
g_kv_hit_rate_tracker = KVHitRateTracker()
