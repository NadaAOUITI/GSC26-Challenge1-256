"""Local-only robust aggregators for attack/defense diagnostics.

These helpers are for eval scripts. Portal defense submissions must stay
self-contained in defense_submission*.py (module-scope rules).
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from utilities.checks import EXPECTED_STATE_LAYOUT, fedavg, validate_state_dict


def _flatten_state(state: dict) -> torch.Tensor:
    pieces = []
    for key in EXPECTED_STATE_LAYOUT:
        pieces.append(
            state[key].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        )
    return torch.cat(pieces, dim=0)


def coordinate_median(models: list[dict]) -> dict:
    """Coordinate-wise median across client state dicts."""
    if not models:
        raise ValueError("Need at least one model.")
    parameter_names = list(models[0].keys())
    aggregated = OrderedDict()
    for name in parameter_names:
        reference = models[0][name]
        stacked = torch.stack(
            [
                model[name].detach().to(device="cpu", dtype=torch.float64)
                for model in models
            ],
            dim=0,
        )
        aggregated[name] = (
            torch.median(stacked, dim=0)
            .values
            .to(dtype=reference.dtype)
            .contiguous()
        )
    return validate_state_dict(aggregated, "coordinate_median output")


def multi_krum(models: list[dict], f: int | None = None) -> dict:
    """Multi-Krum: keep N-f lowest-scoring clients, then FedAvg."""
    n = len(models)
    if f is None:
        f = n // 3
    f = max(0, min(f, n - 1))

    vectors = torch.stack([_flatten_state(model) for model in models], dim=0)
    pairwise = torch.cdist(vectors, vectors, p=2)

    neighbor_count = n - f - 2
    if neighbor_count < 1:
        neighbor_count = max(1, n - 1)

    scores = []
    for index in range(n):
        dists = pairwise[index].clone()
        dists[index] = float("inf")
        nearest, _ = torch.topk(
            dists,
            k=min(neighbor_count, n - 1),
            largest=False,
        )
        scores.append(float(nearest.sum().item()))

    keep_count = max(1, n - f)
    ranked = sorted(range(n), key=lambda i: scores[i])
    survivors = [models[index] for index in ranked[:keep_count]]
    return fedavg(survivors)


def trimmed_mean_hybrid(models: list[dict], f: int | None = None) -> dict:
    """
    Drop the f clients farthest from the coordinate-wise median, then FedAvg.

    Uses full-vector L2 distance to the median model as the outlier score.
    """
    n = len(models)
    if f is None:
        f = n // 3
    f = max(0, min(f, n - 1))
    if f == 0 or n <= 1:
        return fedavg(models)

    median_state = coordinate_median(models)
    median_vec = _flatten_state(median_state)
    distances = [
        float(torch.norm(_flatten_state(model) - median_vec, p=2).item())
        for model in models
    ]
    keep_count = max(1, n - f)
    ranked = sorted(range(n), key=lambda i: distances[i])
    survivors = [models[index] for index in ranked[:keep_count]]
    return fedavg(survivors)


def bulyan_lite(models: list[dict], f: int | None = None) -> dict:
    """
    Bulyan-style: Multi-Krum keep-set, then coordinate median of survivors.

    Simplified one-shot variant suitable for this challenge's single round.
    """
    n = len(models)
    if f is None:
        f = n // 3
    f = max(0, min(f, n - 1))

    vectors = torch.stack([_flatten_state(model) for model in models], dim=0)
    pairwise = torch.cdist(vectors, vectors, p=2)
    neighbor_count = n - f - 2
    if neighbor_count < 1:
        neighbor_count = max(1, n - 1)

    scores = []
    for index in range(n):
        dists = pairwise[index].clone()
        dists[index] = float("inf")
        nearest, _ = torch.topk(
            dists,
            k=min(neighbor_count, n - 1),
            largest=False,
        )
        scores.append(float(nearest.sum().item()))

    # Bulyan keeps n - 2f clients in classic form; use n - f for conservatism.
    keep_count = max(1, n - f)
    ranked = sorted(range(n), key=lambda i: scores[i])
    survivors = [models[index] for index in ranked[:keep_count]]
    return coordinate_median(survivors)


AGGREGATORS = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "multikrum": multi_krum,
    "trimmed_mean": trimmed_mean_hybrid,
    "bulyan_lite": bulyan_lite,
}


def get_aggregator(name: str):
    key = name.strip().lower()
    if key not in AGGREGATORS:
        raise ValueError(
            f"Unknown aggregator {name!r}. Choose from: "
            + ", ".join(sorted(AGGREGATORS))
        )
    return AGGREGATORS[key]
