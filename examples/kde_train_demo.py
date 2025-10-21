"""Train a learnable KDE/MMD similarity to correlate with synthetic noise/masking.

We generate random 2D graphs, perturb them with noise or node additions/removals,
then learn the Gaussian bandwidth sigma so that the MMD loss tracks the
perturbation magnitude.
"""
import random
import math
import torch
import matplotlib.pyplot as plt

from torch import nn
from rapidalign import kde_mmd_loss
from rapidalign.learned_kernel import StructuredKernel


def node_features(pos: torch.Tensor) -> torch.Tensor:
    centroid = pos.mean(dim=0, keepdim=True)
    centered = pos - centroid
    radial = centered.norm(dim=1, keepdim=True)
    if pos.size(0) > 1:
        dist = torch.cdist(pos, pos)
        k = min(5, pos.size(0) - 1)
        knn = dist.topk(k + 1, largest=False).values[:, 1:]
        knn_mean = knn.mean(dim=1, keepdim=True)
        knn_std = knn.std(dim=1, keepdim=True)
    else:
        knn_mean = torch.zeros(1, 1, dtype=pos.dtype, device=pos.device)
        knn_std = torch.zeros(1, 1, dtype=pos.dtype, device=pos.device)
    return torch.cat([centered, radial, knn_mean, knn_std], dim=1)


def graph_descriptor(pos: torch.Tensor) -> torch.Tensor:
    centroid = pos.mean(dim=0)
    centered = pos - centroid
    radius = centered.norm(dim=1).mean()
    if pos.size(0) > 1:
        dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
        mean_dist = dist.mean()
        var_dist = dist.var()
    else:
        mean_dist = torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        var_dist = torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
    bbox = (pos.max(dim=0).values - pos.min(dim=0).values)
    area = (bbox[0] * bbox[1])
    return torch.stack([mean_dist, var_dist, radius, area])


def to_device(graph: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in graph.items()}


class SyntheticGraphDataset:
    def __init__(self, base_nodes=128, noise_max=0.2, drop_max=20, add_max=20, seed=0):
        self.base_nodes = base_nodes
        self.noise_max = noise_max
        self.drop_max = drop_max
        self.add_max = add_max
        self.rng = random.Random(seed)

    def sample(self):
        torch.manual_seed(self.rng.randint(0, 10_000))
        base_pos = torch.randn(self.base_nodes, 2)
        direction = torch.randn_like(base_pos)

        noise_level = self.rng.random() * self.noise_max
        target_pos = base_pos + noise_level * direction

        base = {
            'pos': base_pos,
            'feat': node_features(base_pos),
            'desc': graph_descriptor(base_pos)
        }
        target = {
            'pos': target_pos,
            'feat': node_features(target_pos),
            'desc': graph_descriptor(target_pos)
        }

        label = torch.tensor(noise_level, dtype=torch.float32)
        return base, target, label



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    dataset = SyntheticGraphDataset(drop_max=0, add_max=0)
    model = StructuredKernel(init_sigma=0.1, learn_scale=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    batch_size = 16
    num_steps = 2000
    for step in range(1, num_steps + 1):
        graphs_base = []
        graphs_target = []
        labels = []
        for _ in range(batch_size):
            base_graph, target_graph, label = dataset.sample()
            graphs_base.append(to_device(base_graph, device))
            graphs_target.append(to_device(target_graph, device))
            labels.append(label.to(device))

        preds = torch.stack([model(b, t) for b, t in zip(graphs_base, graphs_target)])
        labels_t = torch.stack(labels)
        loss = ((preds - labels_t) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if step % 200 == 0:
            sigma_val = torch.exp(model.log_sigma).detach().cpu().item()
            print(f'Step {step:4d} | train MSE={loss.item():.6f} | sigma={sigma_val:.4f}')

    # Evaluate correlation on new samples
    preds = []
    labels = []
    with torch.no_grad():
        for _ in range(200):
            base_graph, target_graph, label = dataset.sample()
            base_graph = to_device(base_graph, device)
            target_graph = to_device(target_graph, device)
            pred = model(base_graph, target_graph)
            preds.append(pred.item())
            labels.append(label.item())

    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels)
    corr = torch.corrcoef(torch.stack([preds_t, labels_t]))[0, 1].item()
    print(f'Correlation between predicted loss and synthetic label: {corr:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history)
    axes[0].set_title('Training MSE')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')

    axes[1].scatter(labels, preds, alpha=0.6)
    axes[1].set_title('Target vs predicted loss')
    axes[1].set_xlabel('Synthetic label')
    axes[1].set_ylabel('Predicted loss')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
