"""Autograd-enabled segmented KDE loss."""
from __future__ import annotations

from typing import Tuple

import torch

from .kde_gradients import kde_mmd_backward_ragged_cpu, kde_mmd_forward_cpu

try:  # pragma: no cover
    from . import _cuda
except ImportError:  # pragma: no cover
    _cuda = None


class SegmentedKDELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        w_x: torch.Tensor,
        w_y: torch.Tensor,
        ptr_x: torch.Tensor,
        ptr_y: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if sigma.dim() != 0:
            raise ValueError("sigma must be a scalar tensor")

        ctx.sigma = sigma
        ctx.device = x.device
        ctx.dtypes = (x.dtype, y.dtype, w_x.dtype, w_y.dtype, sigma.dtype)

        if x.is_cuda:
            if _cuda is None:
                raise RuntimeError("CUDA extension not available")

            x32 = x.contiguous().to(torch.float32)
            y32 = y.contiguous().to(torch.float32)
            w_x32 = w_x.contiguous().to(torch.float32)
            w_y32 = w_y.contiguous().to(torch.float32)

            Kxx, Kyy, Kxy = _cuda.kde_mmd_forward_segmented(
                x32,
                y32,
                w_x32,
                w_y32,
                ptr_x.contiguous(),
                ptr_y.contiguous(),
                float(sigma.item()),
            )
            losses = Kxx + Kyy - 2.0 * Kxy
            ctx.save_for_backward(x, y, w_x, w_y, ptr_x, ptr_y)
            return losses

        sigma_val = sigma.item()
        num_pairs = ptr_x.numel() - 1
        losses = torch.zeros(num_pairs, dtype=torch.float64)
        for b in range(num_pairs):
            xs = slice(ptr_x[b].item(), ptr_x[b + 1].item())
            ys = slice(ptr_y[b].item(), ptr_y[b + 1].item())
            fwd = kde_mmd_forward_cpu(x[xs], y[ys], w_x[xs], w_y[ys], sigma_val)
            losses[b] = fwd.loss

        ctx.save_for_backward(x, y, w_x, w_y, ptr_x, ptr_y)
        return losses

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        x, y, w_x, w_y, ptr_x, ptr_y = ctx.saved_tensors
        sigma = ctx.sigma

        if x.is_cuda:
            if _cuda is None:
                raise RuntimeError("CUDA extension not available")

            grad_loss_d = grad_loss.to(torch.float64)
            x32 = x.contiguous().to(torch.float32)
            y32 = y.contiguous().to(torch.float32)
            w_x32 = w_x.contiguous().to(torch.float32)
            w_y32 = w_y.contiguous().to(torch.float32)

            grad_x32, grad_y32, grad_wx64, grad_wy64, grad_sigma_pairs = _cuda.kde_mmd_backward_segmented(
                x32,
                y32,
                w_x32,
                w_y32,
                ptr_x.contiguous(),
                ptr_y.contiguous(),
                grad_loss_d.contiguous(),
                float(sigma.item()),
            )

            dx_dtype, dy_dtype, dwx_dtype, dwy_dtype, sigma_dtype = ctx.dtypes

            grad_x = grad_x32.to(dx_dtype)
            grad_y = grad_y32.to(dy_dtype)
            grad_wx = grad_wx64.to(dwx_dtype)
            grad_wy = grad_wy64.to(dwy_dtype)
            grad_sigma = grad_sigma_pairs.sum().to(sigma_dtype).unsqueeze(0)
            return grad_x, grad_y, grad_wx, grad_wy, None, None, grad_sigma

        sigma_val = sigma.item()
        grads = kde_mmd_backward_ragged_cpu(x, y, w_x, w_y, ptr_x, ptr_y, sigma_val)

        grad_loss = grad_loss.to(torch.float64)
        repeat_x = torch.repeat_interleave(grad_loss, ptr_x.diff())
        repeat_y = torch.repeat_interleave(grad_loss, ptr_y.diff())

        grad_x = grads["x"] * repeat_x.unsqueeze(1)
        grad_y = grads["y"] * repeat_y.unsqueeze(1)
        grad_wx = grads["w_x"] * repeat_x
        grad_wy = grads["w_y"] * repeat_y
        grad_sigma = (grad_loss * grads["sigma_per_pair"]).sum().unsqueeze(0)

        return grad_x, grad_y, grad_wx, grad_wy, None, None, grad_sigma


def segmented_kde_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: torch.Tensor,
    w_y: torch.Tensor,
    ptr_x: torch.Tensor,
    ptr_y: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return SegmentedKDELossFunction.apply(x, y, w_x, w_y, ptr_x, ptr_y, sigma)


__all__ = ["segmented_kde_loss", "SegmentedKDELossFunction"]
