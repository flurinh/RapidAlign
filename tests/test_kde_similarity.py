import math

import pytest

try:
    import torch
except ModuleNotFoundError:
    pytestmark = pytest.mark.skip(reason="requires torch")

from rapidalign import kde_mmd_loss


def test_identical_zero_loss():
    x = torch.randn(32, 3)
    loss, Kxx, Kyy, Kxy = kde_mmd_loss(x, x, sigma=0.3, center=True)
    assert loss.item() < 1e-6
    # Kxx == Kyy, Kxy == Kxx
    assert torch.allclose(Kxx, Kxy, atol=1e-6)


def test_translation_invariance():
    x = torch.randn(40, 3)
    y = x + torch.tensor([0.5, -0.2, 0.1])
    loss_centered, *_ = kde_mmd_loss(x, y, sigma=0.2, center=True)
    loss_uncentered, *_ = kde_mmd_loss(x, y, sigma=0.2, center=False)
    assert loss_centered.item() < 1e-6
    assert loss_uncentered.item() > loss_centered.item()


def test_mismatched_cardinalities():
    x = torch.randn(25, 3)
    y = x[:18]  # subset
    loss, *_ = kde_mmd_loss(x, y, sigma=0.25, center=True)
    assert math.isfinite(loss.item())


def test_cosine_loss_identical_zero():
    x = torch.randn(16, 3)
    loss, *_ = kde_mmd_loss(x, x, sigma=0.3, center=True, cosine=True)
    assert abs(loss.item()) < 1e-6

