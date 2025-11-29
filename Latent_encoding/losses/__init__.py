"""Loss functions for slot-latent experiments."""

from .kernel_correlation import (
    kernel_correlation_loss_pyg,
    global_distance_kernel_loss_pyg,
    local_distance_kernel_loss_pyg,
)

__all__ = [
    "kernel_correlation_loss_pyg",
    "global_distance_kernel_loss_pyg",
    "local_distance_kernel_loss_pyg",
]
