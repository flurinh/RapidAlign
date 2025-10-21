"""Benchmark forward/backward runtime of segmented KDE (CPU vs CUDA)."""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import torch

from rapidalign.autograd_kde import segmented_kde_loss


@dataclass
class Case:
    batch: int
    min_nodes: int
    max_nodes: int
    dim: int = 3


def build_random_case(case: Case, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    sizes_x = []
    sizes_y = []
    for _ in range(case.batch):
        nx = torch.randint(case.min_nodes, case.max_nodes + 1, (1,), generator=g).item()
        ny = torch.randint(case.min_nodes, case.max_nodes + 1, (1,), generator=g).item()
        sizes_x.append(nx)
        sizes_y.append(ny)

    ptr_x = torch.tensor([0] + list(itertools.accumulate(sizes_x)), dtype=torch.long)
    ptr_y = torch.tensor([0] + list(itertools.accumulate(sizes_y)), dtype=torch.long)

    total_x = ptr_x[-1].item()
    total_y = ptr_y[-1].item()

    x = torch.randn(total_x, case.dim, dtype=torch.float64, generator=g)
    y = torch.randn(total_y, case.dim, dtype=torch.float64, generator=g)

    w_x = torch.rand(total_x, dtype=torch.float64, generator=g)
    w_y = torch.rand(total_y, dtype=torch.float64, generator=g)

    for b in range(case.batch):
        xs = slice(ptr_x[b].item(), ptr_x[b + 1].item())
        ys = slice(ptr_y[b].item(), ptr_y[b + 1].item())
        w_x[xs] /= w_x[xs].sum().clamp_min(1e-9)
        w_y[ys] /= w_y[ys].sum().clamp_min(1e-9)

    sigma = torch.tensor(0.4, dtype=torch.float64)

    return x, y, w_x, w_y, ptr_x, ptr_y, sigma


def time_forward_backward(device: torch.device, repeats: int, warmup: int, tensors):
    x, y, w_x, w_y, ptr_x, ptr_y, sigma = tensors

    if device.type == "cuda":
        x = x.to(device=device, dtype=torch.float32).requires_grad_(True)
        y = y.to(device=device, dtype=torch.float32).requires_grad_(True)
        w_x = w_x.to(device=device, dtype=torch.float64).requires_grad_(True)
        w_y = w_y.to(device=device, dtype=torch.float64).requires_grad_(True)
        ptr_x = ptr_x.to(device)
        ptr_y = ptr_y.to(device)
        sigma = sigma.to(device=device, dtype=torch.float64).requires_grad_(True)
        sync = torch.cuda.synchronize
    else:
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        w_x = w_x.clone().requires_grad_(True)
        w_y = w_y.clone().requires_grad_(True)
        ptr_x = ptr_x.clone()
        ptr_y = ptr_y.clone()
        sigma = sigma.clone().requires_grad_(True)

        def sync():
            return None

    # warmup
    for _ in range(warmup):
        loss = segmented_kde_loss(x, y, w_x, w_y, ptr_x, ptr_y, sigma).sum()
        loss.backward()
        for tensor in (x, y, w_x, w_y, sigma):
            if tensor.grad is not None:
                tensor.grad.zero_()
        if device.type == "cuda":
            sync()

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            sync()
        t0 = time.perf_counter()
        loss = segmented_kde_loss(x, y, w_x, w_y, ptr_x, ptr_y, sigma).sum()
        loss.backward()
        if device.type == "cuda":
            sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        for tensor in (x, y, w_x, w_y, sigma):
            if tensor.grad is not None:
                tensor.grad.zero_()

    avg = sum(times) / len(times)
    return avg, loss.detach().cpu().item()


def main():
    parser = argparse.ArgumentParser(description="Segmented KDE forward/backward benchmark")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cases", nargs="*", default=["4x48-96", "8x96-192", "12x128-256"])
    args = parser.parse_args()

    parsed_cases = []
    for spec in args.cases:
        # format: "BxMin-Max"
        batch_part, size_part = spec.split("x")
        min_part, max_part = size_part.split("-")
        parsed_cases.append(Case(batch=int(batch_part), min_nodes=int(min_part), max_nodes=int(max_part)))

    has_cuda = torch.cuda.is_available() and (_cuda_available())

    dev_info = "CUDA available" if has_cuda else "CUDA unavailable"
    print(f"Environment: torch {torch.__version__}, device=cpu, {dev_info}")
    if has_cuda:
        cap = torch.cuda.get_device_properties(0)
        print(f"GPU: {cap.name} (cc {cap.major}.{cap.minor})\n")

    header = ["Case", "Device", "Avg time (ms)", "Loss"]
    print(" | ".join(header))
    print("-" * 72)

    for case in parsed_cases:
        tensors = build_random_case(case, seed=1234)

        avg_cpu, loss_cpu = time_forward_backward(torch.device("cpu"), args.repeats, args.warmup, tensors)
        print(f"{case.batch}x[{case.min_nodes},{case.max_nodes}] | CPU  | {avg_cpu*1000:.3f} | {loss_cpu:.4f}")

        if has_cuda:
            avg_gpu, loss_gpu = time_forward_backward(torch.device("cuda"), args.repeats, args.warmup, tensors)
            print(f"{case.batch}x[{case.min_nodes},{case.max_nodes}] | CUDA | {avg_gpu*1000:.3f} | {loss_gpu:.4f}")
        else:
            print(f"{case.batch}x[{case.min_nodes},{case.max_nodes}] | CUDA | n/a (device unavailable)")


def _cuda_available() -> bool:
    try:
        from rapidalign import _cuda  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
