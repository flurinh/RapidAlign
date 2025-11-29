"""Synthetic dataset for slot-latent graph experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


@dataclass
class SyntheticPointCloudConfig:
    """Configuration for the synthetic dataset."""

    num_graphs: int = 10_000
    num_nodes: int = 8
    min_num_nodes: int | None = None
    max_num_nodes: int | None = None
    num_node_features: int = 0
    seed: int = 0
    feature_mode: str = "ones"
    avg_edge_length: float = 1.0
    min_degree: int = 1
    max_degree: int = 6


class SyntheticPointCloudDataset(Dataset):
    """Simple synthetic dataset of variable-size sparse point clouds."""

    def __init__(
        self,
        num_graphs: int = 10_000,
        num_nodes: int = 8,
        num_node_features: int = 0,
        seed: int = 0,
        feature_mode: str = "ones",
        min_num_nodes: int | None = None,
        max_num_nodes: int | None = None,
        avg_edge_length: float = 1.0,
        min_degree: int = 1,
        max_degree: int = 6,
    ) -> None:
        super().__init__()
        self.config = SyntheticPointCloudConfig(
            num_graphs=num_graphs,
            num_nodes=num_nodes,
            min_num_nodes=min_num_nodes,
            max_num_nodes=max_num_nodes,
            num_node_features=num_node_features,
            seed=seed,
            feature_mode=feature_mode,
            avg_edge_length=avg_edge_length,
            min_degree=min_degree,
            max_degree=max_degree,
        )
        self._rng = random.Random(seed)

    @property
    def num_graphs(self) -> int:
        return self.config.num_graphs

    @property
    def num_node_features(self) -> int:
        return self.config.num_node_features if self.config.num_node_features > 0 else 1

    def current_max_nodes(self, epoch: int | None = None) -> int:
        return self.config.max_num_nodes or self.config.num_nodes

    def sample_graph(self, epoch: int | None = None) -> Data:
        min_nodes = self.config.min_num_nodes or self.config.num_nodes
        max_nodes = self.current_max_nodes(epoch)
        max_nodes = max(max_nodes, min_nodes)
        n = self._rng.randint(min_nodes, max_nodes)
        pos = torch.randn(n, 3)
        pos = pos - pos.mean(dim=0, keepdim=True)
        dists = torch.cdist(pos, pos)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        mean_dist = dists[mask].mean().clamp(min=1e-6)
        pos = pos * (self.config.avg_edge_length / mean_dist)
        if self.config.feature_mode == "gaussian":
            dim = self.config.num_node_features or 1
            x = torch.randn(n, dim)
        else:
            x = torch.ones(n, 1)
        neighbors = []
        for i in range(n):
            order = torch.argsort(torch.cdist(pos[i : i + 1], pos).squeeze(0))
            max_deg = min(self.config.max_degree, n - 1)
            k = self._rng.randint(self.config.min_degree, max_deg if max_deg >= self.config.min_degree else self.config.min_degree)
            for idx in order[1 : k + 1]:
                neighbors.append((i, int(idx)))
        if neighbors:
            src, dst = zip(*neighbors)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        return Data(x=x, pos=pos, edge_index=edge_index)

    def __getitem__(self, idx: int) -> Data:
        del idx
        return self.sample_graph()

    def __len__(self) -> int:
        return self.config.num_graphs
