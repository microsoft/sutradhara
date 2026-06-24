import random
from typing import List, Optional


def sample_poisson_inter_arrivals(
    count: int, qps: float, seed: Optional[int] = None
) -> List[float]:
    if count <= 0:
        return []
    if qps <= 0:
        raise ValueError(f"QPS must be positive, received {qps}")

    rng = random.Random(seed)
    return [rng.expovariate(qps) for _ in range(count)]

def get_request_order(requests, seed: Optional[int] = None):
    indices = list(range(len(requests)))
    random.Random(seed).shuffle(indices)
    return [requests[i] for i in indices]
