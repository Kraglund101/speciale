"""Reusable pilot subset generation for fast ablation runs.

Works with any dataset that has:
  - .samples: List[Dict] with at least "anomaly_type" key (and optionally "product")
  - .type_to_samples: Dict[str, List[int]]  (anomaly_type -> sample indices)
"""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def generate_pilot_subset(
    samples: List[Dict],
    type_to_samples: Dict[str, List[int]],
    target_size: int = 10000,
    min_per_type: int = 5,
    seed: int = 42,
) -> List[int]:
    """Generate a balanced pilot subset across anomaly types.

    Strategy: proportional allocation across types, with a floor of
    ``min_per_type`` per type (if the type has enough samples).

    Args:
        samples: The full sample list (used for length only).
        type_to_samples: Mapping from anomaly_type to list of sample indices.
        target_size: Desired subset size.
        min_per_type: Minimum samples per type (capped by availability).
        seed: Random seed for reproducibility.

    Returns:
        List of sample indices (shuffled).
    """
    rng = random.Random(seed)

    if target_size >= len(samples):
        return list(range(len(samples)))

    types = sorted(type_to_samples.keys())
    total = sum(len(type_to_samples[t]) for t in types)

    # Proportional allocation with floor
    allocations: Dict[str, int] = {}
    for t in types:
        n_available = len(type_to_samples[t])
        proportional = max(1, round(target_size * n_available / total))
        allocated = max(min(min_per_type, n_available), proportional)
        allocations[t] = min(allocated, n_available)

    # If over budget, trim largest types first
    while sum(allocations.values()) > target_size:
        biggest = max(allocations, key=lambda t: allocations[t])
        allocations[biggest] -= 1

    # If under budget, add to types with remaining capacity
    while sum(allocations.values()) < target_size:
        candidates = [t for t in types if allocations[t] < len(type_to_samples[t])]
        if not candidates:
            break
        # Prefer types that are most under-represented
        t = min(candidates, key=lambda t: allocations[t] / max(len(type_to_samples[t]), 1))
        allocations[t] += 1

    # Sample indices
    indices: List[int] = []
    for t in types:
        pool = type_to_samples[t]
        n = allocations[t]
        indices.extend(rng.sample(pool, n))

    rng.shuffle(indices)

    n_types_used = sum(1 for t in types if allocations[t] > 0)
    print(f"Pilot subset: {len(indices)} samples from {n_types_used}/{len(types)} types")
    return indices


def apply_pilot_subset(dataset, indices: List[int]) -> None:
    """Restrict a dataset to the given sample indices (in-place).

    Rebuilds ``dataset.samples`` and ``dataset.type_to_samples``.
    """
    orig_n = len(dataset.samples)
    new_samples = [dataset.samples[i] for i in indices]
    dataset.samples = new_samples

    # Rebuild type_to_samples
    dataset.type_to_samples = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        dataset.type_to_samples[s["anomaly_type"]].append(i)

    print(f"  Applied pilot subset: {orig_n} -> {len(dataset.samples)} samples")


def save_pilot_subset(indices: List[int], path: Path) -> None:
    """Save pilot subset indices to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(indices, f)
    print(f"  Saved pilot subset ({len(indices)} indices) to {path}")


def load_pilot_subset(path: Path) -> List[int]:
    """Load pilot subset indices from JSON."""
    with open(path) as f:
        indices = json.load(f)
    print(f"  Loaded pilot subset: {len(indices)} indices from {path}")
    return indices
