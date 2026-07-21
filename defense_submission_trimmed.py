from collections import OrderedDict

import torch


def robust_aggregation(num_models, models):
    """
    Trimmed-mean hybrid defense.

    1. Build a coordinate-wise median reference.
    2. Drop the f = N // 3 clients farthest (L2) from that reference.
    3. FedAvg the survivors.

    Prepared for tomorrow's portal slot after Multi-Krum and plain median
    both left ASR ~0.9 on the official eval.
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

    # Coordinate-wise median reference.
    median_state = OrderedDict()
    for name in parameter_names:
        reference_param = models[0][name]
        stacked = torch.stack(
            [
                model[name]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                for model in models
            ],
            dim=0,
        )
        median_state[name] = torch.median(stacked, dim=0).values

    median_pieces = []
    for name in parameter_names:
        median_pieces.append(median_state[name].reshape(-1))
    median_vector = torch.cat(median_pieces, dim=0)

    distances = []
    for model in models:
        pieces = []
        for name in parameter_names:
            pieces.append(
                model[name]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .reshape(-1)
            )
        vector = torch.cat(pieces, dim=0)
        distances.append(float(torch.norm(vector - median_vector, p=2).item()))

    keep_count = n - f
    if keep_count < 1:
        keep_count = n
    ranked = sorted(range(n), key=lambda i: distances[i])
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
