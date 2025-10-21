"""Benchmark baseline vs learned KDE/MMD similarity speed on synthetic batches."""
import time
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from rapidalign import kde_mmd_loss_dense
from rapidalign.learned_kernel import LearnedKernel


def make_batch(batch_size: int, num_nodes: int, seed: int = 0):
    torch.manual_seed(seed)
    graphs_q = []
    graphs_t = []
    for i in range(batch_size):
        base = torch.randn(num_nodes, 2)
        noise = torch.randn_like(base) * 0.1
        perturbed = base + noise
        graphs_q.append(Data(pos=base))
        graphs_t.append(Data(pos=perturbed))
    batch_q = Batch.from_data_list(graphs_q)
    batch_t = Batch.from_data_list(graphs_t)
    X, mask_x = to_dense_batch(batch_q.pos, batch_q.batch)
    Y, mask_y = to_dense_batch(batch_t.pos, batch_t.batch)
    return X, Y, mask_x, mask_y


def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Benchmarking on {device}')

    batch_sizes = [4, 8, 16, 32]
    num_nodes = 256
    sigmas = (0.05, 0.1, 0.2)

    learned_kernel = LearnedKernel(sigmas=sigmas).to(device)

    for B in batch_sizes:
        X, Y, mask_x, mask_y = make_batch(B, num_nodes)
        X = X.to(device)
        Y = Y.to(device)
        mask_x = mask_x.to(device)
        mask_y = mask_y.to(device)
        weights_x = mask_x.to(torch.float32)
        weights_y = mask_y.to(torch.float32)

        # Warm-up
        _ = kde_mmd_loss_dense(X, Y, mask_x, mask_y, x_w=weights_x, y_w=weights_y,
                               sigma=0.1, center=True)
        _ = kde_mmd_loss_dense(X, Y, mask_x, mask_y, x_w=weights_x, y_w=weights_y,
                               sigma=0.15, center=True)
        _ = learned_kernel(X[0], Y[0])

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(10):
            _ = kde_mmd_loss_dense(X, Y, mask_x, mask_y, x_w=weights_x, y_w=weights_y,
                                   sigma=0.1, center=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        baseline_time = (time.time() - start) / 10

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(10):
            loss = 0.0
            for sigma in sigmas:
                loss += kde_mmd_loss_dense(X, Y, mask_x, mask_y, x_w=weights_x, y_w=weights_y,
                                           sigma=sigma, center=True)[0].mean()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        multi_sigma_time = (time.time() - start) / 10

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(10):
            total = 0.0
            for b in range(B):
                total = total + learned_kernel(X[b, mask_x[b]], Y[b, mask_y[b]])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        learned_time = (time.time() - start) / 10

        print(f"Batch {B:2d} | baseline single σ: {baseline_time*1e3:.2f} ms | "+
              f"3×σ sum: {multi_sigma_time*1e3:.2f} ms | learned kernel: {learned_time*1e3:.2f} ms")


if __name__ == '__main__':
    benchmark()
