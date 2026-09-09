import torch


def test_bound_lpips_inputs_limits_long_edge_and_preserves_gradient() -> None:
    from texture.lpips_memory import bound_lpips_inputs

    prediction = torch.rand(2, 3, 128, 96, requires_grad=True)
    target = torch.rand(2, 3, 128, 96)

    bounded_prediction, bounded_target = bound_lpips_inputs(
        prediction,
        target,
        max_size=64,
    )

    assert bounded_prediction.shape == (2, 3, 64, 48)
    assert bounded_target.shape == (2, 3, 64, 48)

    bounded_prediction.sum().backward()
    assert prediction.grad is not None
