from collections import OrderedDict

import torch


def robust_aggregation(num_models, models):
    """
    Multi-Krum aggregation (Blanchard et al.).

    Score each client by sum of distances to its nearest neighbors,
    keep the N - f lowest-scoring models (f = N // 3), and FedAvg them.

    Portal result (2026-07-20): ASR 0.905, CleanAcc 0.848, score ~0.547
    (worse than FedAvg baseline). Kept here for reference only.
    """
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")

    if len(models) != num_models:
        raise ValueError(
            f"num_models={num_models}, "
            f"but received {len(models)} models."
        )

    parameter_names = list(models[0].keys())
    n = num_models
    f = n // 3

    vectors = []
    for model in models:
        pieces = []
        for key in parameter_names:
            pieces.append(
                model[key]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .reshape(-1)
            )
        vectors.append(torch.cat(pieces, dim=0))
    stacked = torch.stack(vectors, dim=0)

    pairwise = torch.cdist(stacked, stacked, p=2)

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

    keep_count = n - f
    if keep_count < 1:
        keep_count = n
    ranked = sorted(range(n), key=lambda i: scores[i])
    keep_indices = ranked[:keep_count]
    survivors = [models[index] for index in keep_indices]
    survivor_count = len(survivors)

    aggregated = OrderedDict()
    for name in parameter_names:
        reference_param = survivors[0][name]
        accumulator = torch.zeros_like(
            reference_param,
            dtype=torch.float64,
            device="cpu",
        )
        for model in survivors:
            value = model[name]
            if value.shape != reference_param.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name!r}."
                )
            accumulator.add_(
                value.detach().to(device="cpu", dtype=torch.float64)
            )
        aggregated[name] = (
            accumulator
            .div(float(survivor_count))
            .to(dtype=reference_param.dtype)
            .contiguous()
        )

    return aggregated
