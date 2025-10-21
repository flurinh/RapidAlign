import torch
import pytest

from rapidalign.kde_gradients import (
    kde_mmd_forward_cpu,
    kde_mmd_backward_cpu,
    kde_mmd_backward_ragged_cpu,
)
from rapidalign.autograd_kde import segmented_kde_loss


def finite_difference_grad_x(x, y, w_x, w_y, sigma, eps=1e-6):
    grad = torch.zeros_like(x)
    for i in range(x.size(0)):
        for d in range(x.size(1)):
            x_pos = x.clone()
            x_neg = x.clone()
            x_pos[i, d] += eps
            x_neg[i, d] -= eps
            loss_pos = kde_mmd_forward_cpu(x_pos, y, w_x, w_y, sigma).loss
            loss_neg = kde_mmd_forward_cpu(x_neg, y, w_x, w_y, sigma).loss
            grad[i, d] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad


def finite_difference_grad_y(x, y, w_x, w_y, sigma, eps=1e-6):
    grad = torch.zeros_like(y)
    for j in range(y.size(0)):
        for d in range(y.size(1)):
            y_pos = y.clone()
            y_neg = y.clone()
            y_pos[j, d] += eps
            y_neg[j, d] -= eps
            loss_pos = kde_mmd_forward_cpu(x, y_pos, w_x, w_y, sigma).loss
            loss_neg = kde_mmd_forward_cpu(x, y_neg, w_x, w_y, sigma).loss
            grad[j, d] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad


def finite_difference_grad_w(x, y, w_x, w_y, sigma, which="x", eps=1e-6):
    w = w_x if which == "x" else w_y
    grad = torch.zeros_like(w)
    for i in range(w.size(0)):
        w_pos = w.clone()
        w_neg = w.clone()
        w_pos[i] += eps
        w_neg[i] -= eps
        if which == "x":
            loss_pos = kde_mmd_forward_cpu(x, y, w_pos, w_y, sigma).loss
            loss_neg = kde_mmd_forward_cpu(x, y, w_neg, w_y, sigma).loss
        else:
            loss_pos = kde_mmd_forward_cpu(x, y, w_x, w_pos, sigma).loss
            loss_neg = kde_mmd_forward_cpu(x, y, w_x, w_neg, sigma).loss
        grad[i] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad


def finite_difference_grad_sigma(x, y, w_x, w_y, sigma, eps=1e-6):
    loss_pos = kde_mmd_forward_cpu(x, y, w_x, w_y, sigma + eps).loss
    loss_neg = kde_mmd_forward_cpu(x, y, w_x, w_y, sigma - eps).loss
    return (loss_pos - loss_neg) / (2.0 * eps)


def test_identical_pair_gradients_zero():
    torch.manual_seed(0)
    dtype = torch.float64
    x = torch.randn(4, 3, dtype=dtype)
    y = x.clone()
    w = torch.full((x.size(0),), 1.0 / x.size(0), dtype=dtype)
    sigma = 0.35

    grads = kde_mmd_backward_cpu(x, y, w, w, sigma)

    assert torch.allclose(grads["x"], torch.zeros_like(x), atol=1e-10)
    assert torch.allclose(grads["y"], torch.zeros_like(y), atol=1e-10)
    assert torch.allclose(grads["w_x"], torch.zeros_like(w), atol=1e-10)
    assert torch.allclose(grads["w_y"], torch.zeros_like(w), atol=1e-10)
    assert torch.isclose(grads["sigma"], torch.tensor(0.0, dtype=dtype), atol=1e-10)


def test_identical_pair_gradients_match_finite_difference():
    torch.manual_seed(42)
    dtype = torch.float64
    x = torch.randn(3, 2, dtype=dtype)
    y = x.clone()
    w = torch.full((x.size(0),), 1.0 / x.size(0), dtype=dtype)
    sigma = 0.25

    grads = kde_mmd_backward_cpu(x, y, w, w, sigma)
    fd_grad = finite_difference_grad_x(x, y, w, w, sigma, eps=1e-7)

    assert torch.allclose(grads["x"], fd_grad, atol=1e-8)


def test_noisy_pair_gradients_match_finite_difference():
    torch.manual_seed(123)
    dtype = torch.float64
    x = torch.randn(4, 3, dtype=dtype)
    y = x + 0.05 * torch.randn_like(x)
    w = torch.full((x.size(0),), 1.0 / x.size(0), dtype=dtype)
    sigma = 0.3

    grads = kde_mmd_backward_cpu(x, y, w, w, sigma)

    fd_x = finite_difference_grad_x(x, y, w, w, sigma, eps=1e-7)
    fd_y = finite_difference_grad_y(x, y, w, w, sigma, eps=1e-7)
    fd_wx = finite_difference_grad_w(x, y, w, w, sigma, which="x", eps=1e-7)
    fd_wy = finite_difference_grad_w(x, y, w, w, sigma, which="y", eps=1e-7)
    fd_sigma = finite_difference_grad_sigma(x, y, w, w, sigma, eps=1e-7)

    assert torch.allclose(grads["x"], fd_x, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads["y"], fd_y, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads["w_x"], fd_wx, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads["w_y"], fd_wy, atol=1e-7, rtol=1e-6)
    assert torch.isclose(grads["sigma"], fd_sigma, atol=1e-7, rtol=1e-6)


def test_variable_size_gradients_match_finite_difference():
    torch.manual_seed(7)
    dtype = torch.float64
    x = torch.randn(3, 3, dtype=dtype)
    y = torch.randn(5, 3, dtype=dtype)
    w_x = torch.rand(3, dtype=dtype)
    w_y = torch.rand(5, dtype=dtype)
    w_x = w_x / w_x.sum()
    w_y = w_y / w_y.sum()
    sigma = 0.4

    grads = kde_mmd_backward_cpu(x, y, w_x, w_y, sigma)

    fd_x = finite_difference_grad_x(x, y, w_x, w_y, sigma, eps=1e-7)
    fd_y = finite_difference_grad_y(x, y, w_x, w_y, sigma, eps=1e-7)
    fd_wx = finite_difference_grad_w(x, y, w_x, w_y, sigma, which="x", eps=1e-7)
    fd_wy = finite_difference_grad_w(x, y, w_x, w_y, sigma, which="y", eps=1e-7)
    fd_sigma = finite_difference_grad_sigma(x, y, w_x, w_y, sigma, eps=1e-7)

    assert torch.allclose(grads["x"], fd_x, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads["y"], fd_y, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads["w_x"], fd_wx, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads["w_y"], fd_wy, atol=1e-7, rtol=1e-6)
    assert torch.isclose(grads["sigma"], fd_sigma, atol=1e-7, rtol=1e-6)


def test_ragged_batch_gradients_matches_per_pair():
    torch.manual_seed(99)
    dtype = torch.float64
    ptr_x = torch.tensor([0, 3, 7], dtype=torch.long)
    ptr_y = torch.tensor([0, 4, 6], dtype=torch.long)

    x = torch.randn(ptr_x[-1], 2, dtype=dtype)
    y = torch.randn(ptr_y[-1], 2, dtype=dtype)
    w_x = torch.rand(ptr_x[-1], dtype=dtype)
    w_y = torch.rand(ptr_y[-1], dtype=dtype)
    for b in range(ptr_x.numel() - 1):
        s = slice(ptr_x[b].item(), ptr_x[b + 1].item())
        t = slice(ptr_y[b].item(), ptr_y[b + 1].item())
        w_x[s] /= w_x[s].sum()
        w_y[t] /= w_y[t].sum()

    sigma = 0.35

    ragged = kde_mmd_backward_ragged_cpu(x, y, w_x, w_y, ptr_x, ptr_y, sigma)

    for b in range(ptr_x.numel() - 1):
        xs = slice(ptr_x[b].item(), ptr_x[b + 1].item())
        ys = slice(ptr_y[b].item(), ptr_y[b + 1].item())
        indiv = kde_mmd_backward_cpu(x[xs], y[ys], w_x[xs], w_y[ys], sigma)

        assert torch.allclose(ragged["x"][xs], indiv["x"])
        assert torch.allclose(ragged["y"][ys], indiv["y"])
        assert torch.allclose(ragged["w_x"][xs], indiv["w_x"])
        assert torch.allclose(ragged["w_y"][ys], indiv["w_y"])

    expected_sigma = []
    for b in range(ptr_x.numel() - 1):
        xs = slice(ptr_x[b].item(), ptr_x[b + 1].item())
        ys = slice(ptr_y[b].item(), ptr_y[b + 1].item())
        expected_sigma.append(
            kde_mmd_backward_cpu(x[xs], y[ys], w_x[xs], w_y[ys], sigma)["sigma"]
        )
    expected_sigma = torch.stack(expected_sigma)

    assert torch.allclose(ragged["sigma_per_pair"], expected_sigma)
    assert torch.isclose(ragged["sigma_total"], expected_sigma.sum())


def test_gradcheck_cpu_single_pair():
    torch.manual_seed(5)
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    w_x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    w_y = torch.randn(3, dtype=torch.float64, requires_grad=True)
    ptr = torch.tensor([0, 3], dtype=torch.long)
    sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def func(x_, y_, w_x_, w_y_, sigma_):
        return segmented_kde_loss(x_, y_, w_x_, w_y_, ptr, ptr, sigma_).sum()

    assert torch.autograd.gradcheck(func, (x, y, w_x, w_y, sigma), eps=1e-6, atol=1e-5)


def test_gradcheck_cpu_ragged():
    torch.manual_seed(6)
    ptr_x = torch.tensor([0, 2, 5], dtype=torch.long)
    ptr_y = torch.tensor([0, 3, 4], dtype=torch.long)
    x = torch.randn(ptr_x[-1], 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(ptr_y[-1], 3, dtype=torch.float64, requires_grad=True)
    w_x = torch.randn(ptr_x[-1], dtype=torch.float64, requires_grad=True)
    w_y = torch.randn(ptr_y[-1], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    def func(x_, y_, w_x_, w_y_, sigma_):
        return segmented_kde_loss(x_, y_, w_x_, w_y_, ptr_x, ptr_y, sigma_).sum()

    assert torch.autograd.gradcheck(func, (x, y, w_x, w_y, sigma), eps=1e-6, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_backward_matches_cpu():
    torch.manual_seed(42)
    ptr_x = torch.tensor([0, 3, 7], dtype=torch.long)
    ptr_y = torch.tensor([0, 2, 5], dtype=torch.long)

    x_cpu = torch.randn(ptr_x[-1], 3, dtype=torch.float64, requires_grad=True)
    y_cpu = torch.randn(ptr_y[-1], 3, dtype=torch.float64, requires_grad=True)
    w_x_cpu = torch.randn(ptr_x[-1], dtype=torch.float64, requires_grad=True)
    w_y_cpu = torch.randn(ptr_y[-1], dtype=torch.float64, requires_grad=True)
    sigma_cpu = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)

    loss_cpu = segmented_kde_loss(x_cpu, y_cpu, w_x_cpu, w_y_cpu, ptr_x, ptr_y, sigma_cpu).sum()
    loss_cpu.backward()

    x_gpu = x_cpu.detach().to(torch.float32).cuda().requires_grad_(True)
    y_gpu = y_cpu.detach().to(torch.float32).cuda().requires_grad_(True)
    w_x_gpu = w_x_cpu.detach().to(torch.float64).cuda().requires_grad_(True)
    w_y_gpu = w_y_cpu.detach().to(torch.float64).cuda().requires_grad_(True)
    sigma_gpu = sigma_cpu.detach().to(torch.float64).cuda().requires_grad_(True)
    ptr_x_gpu = ptr_x.cuda()
    ptr_y_gpu = ptr_y.cuda()

    loss_gpu = segmented_kde_loss(x_gpu, y_gpu, w_x_gpu, w_y_gpu, ptr_x_gpu, ptr_y_gpu, sigma_gpu).sum()
    loss_gpu.backward()

    atol = 1e-4
    rtol = 1e-3

    assert torch.allclose(x_gpu.grad.detach().cpu().double(), x_cpu.grad, atol=atol, rtol=rtol)
    assert torch.allclose(y_gpu.grad.detach().cpu().double(), y_cpu.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w_x_gpu.grad.detach().cpu(), w_x_cpu.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w_y_gpu.grad.detach().cpu(), w_y_cpu.grad, atol=atol, rtol=rtol)
    assert torch.isclose(sigma_gpu.grad.detach().cpu(), sigma_cpu.grad, atol=atol, rtol=rtol)
