import math
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("requires torch", allow_module_level=True)

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_batch
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("requires torch_geometric", allow_module_level=True)

from rapidalign import kde_mmd_loss, kde_mmd_loss_dense, pyg_kde_mmd_loss


def make_graph(n_nodes: int, noise: float = 0.0, drop: int = 0, seed: int = 0) -> Data:
    torch.manual_seed(seed)
    pos = torch.randn(n_nodes, 3)
    if noise > 0:
        pos = pos + torch.randn_like(pos) * noise
    if drop > 0 and drop < n_nodes:
        perm = torch.randperm(n_nodes)
        pos = pos[perm[:-drop]]
    return Data(pos=pos)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kde_cuda_matches_cpu():
    device = torch.device("cuda")

    graphs_q = [make_graph(12, seed=1), make_graph(8, seed=2)]
    graphs_t = [make_graph(12, noise=0.05, seed=3), make_graph(10, noise=0.1, seed=4)]

    batch_q = Batch.from_data_list(graphs_q)
    batch_t = Batch.from_data_list(graphs_t)

    X_cpu, mask_x_cpu = to_dense_batch(batch_q.pos, batch_q.batch)
    Y_cpu, mask_y_cpu = to_dense_batch(batch_t.pos, batch_t.batch)

    weights_x = mask_x_cpu.to(torch.float32)
    weights_y = mask_y_cpu.to(torch.float32)

    loss_gpu, Kxx_gpu, Kyy_gpu, Kxy_gpu = kde_mmd_loss_dense(
        X_cpu.to(device), Y_cpu.to(device),
        mask_x_cpu.to(device), mask_y_cpu.to(device),
        x_w=weights_x.to(device), y_w=weights_y.to(device),
        sigma=0.25, center=False)

    losses_cpu = []
    Kxx_cpu = []
    Kyy_cpu = []
    Kxy_cpu = []
    for gq, gt in zip(graphs_q, graphs_t):
        loss, kxx, kyy, kxy = kde_mmd_loss(
            gq.pos, gt.pos,
            sigma=0.25,
            center=False)
        losses_cpu.append(loss)
        Kxx_cpu.append(kxx)
        Kyy_cpu.append(kyy)
        Kxy_cpu.append(kxy)

    losses_cpu = torch.stack(losses_cpu).to(device=device, dtype=torch.float64)
    Kxx_cpu = torch.stack(Kxx_cpu).to(device=device, dtype=torch.float64)
    Kyy_cpu = torch.stack(Kyy_cpu).to(device=device, dtype=torch.float64)
    Kxy_cpu = torch.stack(Kxy_cpu).to(device=device, dtype=torch.float64)

    loss_gpu = loss_gpu.to(losses_cpu.dtype)
    Kxx_gpu = Kxx_gpu.to(Kxx_cpu.dtype)
    Kyy_gpu = Kyy_gpu.to(Kyy_cpu.dtype)
    Kxy_gpu = Kxy_gpu.to(Kxy_cpu.dtype)

    assert torch.allclose(loss_gpu, losses_cpu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(Kxx_gpu, Kxx_cpu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(Kyy_gpu, Kyy_cpu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(Kxy_gpu, Kxy_cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pyg_kde_matches_dense_cuda():
    device = torch.device("cuda")

    graphs_src = [make_graph(16, seed=7), make_graph(11, seed=8), make_graph(5, seed=9)]
    graphs_tgt = [make_graph(14, noise=0.02, seed=10), make_graph(13, noise=0.05, seed=11), make_graph(8, noise=0.01, seed=12)]

    batch_src = Batch.from_data_list(graphs_src)
    batch_tgt = Batch.from_data_list(graphs_tgt)

    X, mask_x = to_dense_batch(batch_src.pos, batch_src.batch)
    Y, mask_y = to_dense_batch(batch_tgt.pos, batch_tgt.batch)
    weights_x = mask_x.to(torch.float32)
    weights_y = mask_y.to(torch.float32)

    loss_dense, Kxx_dense, Kyy_dense, Kxy_dense = kde_mmd_loss_dense(
        X.to(device), Y.to(device),
        mask_x.to(device), mask_y.to(device),
        x_w=weights_x.to(device), y_w=weights_y.to(device),
        sigma=0.3, center=True)

    loss_pyg, Kxx_pyg, Kyy_pyg, Kxy_pyg = pyg_kde_mmd_loss(
        batch_src.pos.to(device),
        batch_tgt.pos.to(device),
        src_batch=batch_src.batch.to(device),
        tgt_batch=batch_tgt.batch.to(device),
        sigma=0.3,
        center=True)

    assert torch.allclose(loss_pyg, loss_dense, atol=1e-4, rtol=1e-4)
    assert torch.allclose(Kxx_pyg, Kxx_dense, atol=1e-4, rtol=1e-4)
    assert torch.allclose(Kyy_pyg, Kyy_dense, atol=1e-4, rtol=1e-4)
    assert torch.allclose(Kxy_pyg, Kxy_dense, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kde_cuda_translation_invariance():
    device = torch.device("cuda")

    graph = make_graph(20, seed=10)
    shift = torch.tensor([0.5, -0.2, 0.3])
    shifted = Data(pos=graph.pos + shift)

    batch_q = Batch.from_data_list([graph])
    batch_t = Batch.from_data_list([shifted])

    X, mask_x = to_dense_batch(batch_q.pos, batch_q.batch)
    Y, mask_y = to_dense_batch(batch_t.pos, batch_t.batch)

    weights_x = mask_x.to(torch.float32)
    weights_y = mask_y.to(torch.float32)

    loss_centered, *_ = kde_mmd_loss_dense(
        X.to(device), Y.to(device),
        mask_x.to(device), mask_y.to(device),
        x_w=weights_x.to(device), y_w=weights_y.to(device),
        sigma=0.3, center=True)

    loss_uncentered, *_ = kde_mmd_loss_dense(
        X.to(device), Y.to(device),
        mask_x.to(device), mask_y.to(device),
        x_w=weights_x.to(device), y_w=weights_y.to(device),
        sigma=0.3, center=False)

    assert loss_centered.item() < 1e-6
    assert loss_uncentered.item() > loss_centered.item()

    # Rotation invariance: rotate both clouds by same random rotation
    torch.manual_seed(2024)
    axis = torch.randn(3)
    axis = axis / axis.norm()
    angle = torch.rand(1).item() * 2 * math.pi
    K = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    R = torch.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    rotated_graph = Data(pos=(graph.pos @ R.T))
    rotated_shifted = Data(pos=(shifted.pos @ R.T))

    batch_rot_q = Batch.from_data_list([rotated_graph])
    batch_rot_t = Batch.from_data_list([rotated_shifted])
    Xr, mask_xr = to_dense_batch(batch_rot_q.pos, batch_rot_q.batch)
    Yr, mask_yr = to_dense_batch(batch_rot_t.pos, batch_rot_t.batch)

    weights_xr = mask_xr.to(torch.float32)
    weights_yr = mask_yr.to(torch.float32)

    loss_rot, *_ = kde_mmd_loss_dense(
        Xr.to(device), Yr.to(device),
        mask_xr.to(device), mask_yr.to(device),
        x_w=weights_xr.to(device), y_w=weights_yr.to(device),
        sigma=0.3, center=True)

    assert loss_rot.item() < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_noise_and_node_removal_increase_loss():
    device = torch.device("cuda")
    base = make_graph(64, seed=50)

    noise_levels = [0.0, 0.05, 0.1, 0.2]
    torch.manual_seed(99)
    noise_direction = torch.randn_like(base.pos)
    graphs_noisy = [Data(pos=base.pos + n * noise_direction) for n in noise_levels]

    # Stack base with noisy variants
    batch_base = Batch.from_data_list([base])
    X, mask_x = to_dense_batch(batch_base.pos, batch_base.batch)
    weights_x = mask_x.to(torch.float32)

    losses = []
    for i, g in enumerate(graphs_noisy):
        batch_g = Batch.from_data_list([g])
        Y, mask_y = to_dense_batch(batch_g.pos, batch_g.batch)
        weights_y = mask_y.to(torch.float32)

        loss, *_ = kde_mmd_loss_dense(
            X.to(device), Y.to(device),
            mask_x.to(device), mask_y.to(device),
            x_w=weights_x.to(device), y_w=weights_y.to(device),
            sigma=0.2, center=True)
        losses.append(loss.item())

    # Ensure loss grows with noise (monotonic non-decreasing)
    assert losses == sorted(losses)

    # Node removal should also increase loss
    torch.manual_seed(123)
    perm = torch.randperm(base.pos.size(0))
    trimmed = base.pos[perm[:-20]].clone()
    graph_drop = Data(pos=trimmed)
    batch_drop = Batch.from_data_list([graph_drop])
    Y_drop, mask_y_drop = to_dense_batch(batch_drop.pos, batch_drop.batch)
    weights_y_drop = mask_y_drop.to(torch.float32)

    loss_drop, *_ = kde_mmd_loss_dense(
        X.to(device), Y_drop.to(device),
        mask_x.to(device), mask_y_drop.to(device),
        x_w=weights_x.to(device), y_w=weights_y_drop.to(device),
        sigma=0.2, center=True)

    assert loss_drop.item() >= losses[0]


def test_pyg_kde_cpu_matches_per_graph():
    graphs_src = [make_graph(6, seed=21), make_graph(4, seed=22)]
    graphs_tgt = [make_graph(6, noise=0.05, seed=23), make_graph(7, noise=0.02, seed=24)]

    batch_src = Batch.from_data_list(graphs_src)
    batch_tgt = Batch.from_data_list(graphs_tgt)

    loss_pyg, Kxx_pyg, Kyy_pyg, Kxy_pyg = pyg_kde_mmd_loss(
        batch_src.pos,
        batch_tgt.pos,
        src_batch=batch_src.batch,
        tgt_batch=batch_tgt.batch,
        sigma=0.25,
        center=True)

    per_graph = []
    per_kxx = []
    per_kyy = []
    per_kxy = []
    for g_src, g_tgt in zip(graphs_src, graphs_tgt):
        loss, kxx, kyy, kxy = kde_mmd_loss(g_src.pos, g_tgt.pos, sigma=0.25, center=True)
        per_graph.append(loss)
        per_kxx.append(kxx)
        per_kyy.append(kyy)
        per_kxy.append(kxy)

    assert torch.allclose(loss_pyg.cpu(), torch.stack(per_graph), atol=1e-6, rtol=1e-6)
    assert torch.allclose(Kxx_pyg.cpu(), torch.stack(per_kxx), atol=1e-6, rtol=1e-6)
    assert torch.allclose(Kyy_pyg.cpu(), torch.stack(per_kyy), atol=1e-6, rtol=1e-6)
    assert torch.allclose(Kxy_pyg.cpu(), torch.stack(per_kxy), atol=1e-6, rtol=1e-6)
