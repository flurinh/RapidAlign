"""PyG QM9 dataset wrapper that returns graph point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9


_VALID_SPLITS = ("train", "val", "test")


class QM9PointCloudDataset(Dataset):
    """Thin wrapper around PyG's QM9 dataset with deterministic splits."""

    def __init__(
        self,
        root: Path | str | None = None,
        split: str = "train",
        limit: int | None = None,
        split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        max_nodes: int | None = None,
        center: bool = True,
    ) -> None:
        super().__init__()
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {_VALID_SPLITS}.")
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "qm9"
        self.root.mkdir(parents=True, exist_ok=True)
        self.center = center
        self.max_nodes = max_nodes
        self.split = split
        self.split_seed = split_seed
        self.split_fractions = split_fractions

        self._dataset = QM9(str(self.root))
        self._node_feature_dim = self._infer_node_dim()
        self.indices = self._build_split_indices()
        if self.max_nodes is not None:
            self.indices = [idx for idx in self.indices if self._dataset[idx].num_nodes <= self.max_nodes]
        if limit is not None:
            self.indices = self.indices[: max(limit, 0)]
        if not self.indices:
            raise ValueError(
                "No QM9 samples available after applying the requested split/filters."
            )

    def _infer_node_dim(self) -> int:
        sample = self._dataset[0]
        if sample.x is not None:
            return int(sample.x.size(-1))
        z = sample.z.long()
        return int(z.max().item()) + 1

    def _build_split_indices(self) -> List[int]:
        perm = torch.randperm(len(self._dataset), generator=torch.Generator().manual_seed(self.split_seed))
        n_train = int(self.split_fractions[0] * len(self._dataset))
        n_val = int(self.split_fractions[1] * len(self._dataset))
        splits: Dict[str, torch.Tensor] = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        return splits[self.split].tolist()

    @property
    def num_node_features(self) -> int:
        return self._node_feature_dim

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        data = self._dataset[self.indices[idx]].clone()
        if data.x is None:
            data.x = F.one_hot(data.z.long(), num_classes=self._node_feature_dim).float()
        if self.center:
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        if self.max_nodes is not None and data.num_nodes > self.max_nodes:
            raise ValueError(
                f"Graph with {data.num_nodes} nodes exceeds configured max_nodes={self.max_nodes}."
            )
        return data


__all__ = ["QM9PointCloudDataset"]
