"""Datasets for slot-latent experiments."""

from .qm9 import QM9PointCloudDataset
from .synthetic import SyntheticPointCloudDataset

__all__ = ["SyntheticPointCloudDataset", "QM9PointCloudDataset"]
