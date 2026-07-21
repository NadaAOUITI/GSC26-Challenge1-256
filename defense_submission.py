from collections import OrderedDict

import torch


def robust_aggregation(num_models, models):
    """
    Coordinate-wise median aggregation.

    For each parameter element, take the median across clients.
    Active portal submission (defense slot #2, 2026-07-20).
    """
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")

    if len(models) != num_models:
        raise ValueError(
            f"num_models={num_models}, "
            f"but received {len(models)} models."
        )

    parameter_names = list(models[0].keys())
    aggregated = OrderedDict()

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
        if stacked[0].shape != reference_param.shape:
            raise ValueError(
                f"Shape mismatch for parameter {name!r}."
            )
        aggregated[name] = (
            torch.median(stacked, dim=0)
            .values
            .to(dtype=reference_param.dtype)
            .contiguous()
        )

    return aggregated
