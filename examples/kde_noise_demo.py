"""Demo: KDE/MMD similarity vs. noise and node removal for PyG graphs."""
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from rapidalign import kde_mmd_loss_dense


def make_base_graph(num_nodes: int, seed: int = 0) -> Data:
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3)
    return Data(pos=pos)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base = make_base_graph(128, seed=123)

    batch_base = Batch.from_data_list([base])
    X, mask_x = to_dense_batch(batch_base.pos, batch_base.batch)
    weights_x = mask_x.to(torch.float32)

    X = X.to(device)
    mask_x = mask_x.to(device)
    weights_x = weights_x.to(device)

    # Pre-sample a deterministic noise direction
    torch.manual_seed(999)
    noise_direction = torch.randn_like(base.pos)

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    print("\nNoise sweep (translation invariant):")
    prev_loss = None
    for level in noise_levels:
        noisy_graph = Data(pos=base.pos + level * noise_direction)
        batch_noisy = Batch.from_data_list([noisy_graph])
        Y, mask_y = to_dense_batch(batch_noisy.pos, batch_noisy.batch)
        weights_y = mask_y.to(torch.float32)

        loss, *_ = kde_mmd_loss_dense(
            X, Y.to(device),
            mask_x, mask_y.to(device),
            x_w=weights_x, y_w=weights_y.to(device),
            sigma=0.2, center=True)

        print(f"  noise={level:.2f} -> loss={loss.item():.6f}")
        if prev_loss is not None:
            assert loss.item() >= prev_loss - 1e-6
        prev_loss = loss.item()

    # Node removal experiment
    torch.manual_seed(1234)
    perm = torch.randperm(base.pos.size(0))
    trimmed = base.pos[perm[:-40]].clone()
    drop_graph = Data(pos=trimmed)

    batch_drop = Batch.from_data_list([drop_graph])
    Y_drop, mask_y_drop = to_dense_batch(batch_drop.pos, batch_drop.batch)
    weights_y_drop = mask_y_drop.to(torch.float32)

    loss_drop, *_ = kde_mmd_loss_dense(
        X, Y_drop.to(device),
        mask_x, mask_y_drop.to(device),
        x_w=weights_x, y_w=weights_y_drop.to(device),
        sigma=0.2, center=True)

    print(f"\nNode removal (40 removed out of {base.pos.size(0)}): loss={loss_drop.item():.6f}")


if __name__ == "__main__":
    main()
