#!/usr/bin/env python3
"""Generate a synthetic alignment demo HTML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from Latent_encoding.utils import apply_random_se3, kabsch_align  # type: ignore
else:
    from .utils import apply_random_se3, kabsch_align


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = repo_root / "Latent_encoding" / "alignment_demo.html"

    rng = np.random.default_rng(0)
    points = rng.standard_normal((32, 3))
    points_t = torch.tensor(points, dtype=torch.float64)
    transformed, rot, trans = apply_random_se3(points_t)
    aligned = kabsch_align(transformed, points_t)
    rel_error = torch.norm(aligned - points_t) / torch.norm(points_t)
    print(f"Relative alignment error: {rel_error.item():.3e}")

    transformed_np = transformed.cpu().numpy()
    aligned_np = aligned.cpu().numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            name="Original",
            marker=dict(size=5, color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=transformed_np[:, 0],
            y=transformed_np[:, 1],
            z=transformed_np[:, 2],
            mode="markers",
            name="Rotated+Translated",
            marker=dict(size=4, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=aligned_np[:, 0],
            y=aligned_np[:, 1],
            z=aligned_np[:, 2],
            mode="markers",
            name="Aligned",
            marker=dict(size=3, color="green"),
        )
    )
    fig.update_layout(title="Synthetic Graph Alignment Example", scene=dict(aspectmode="data"))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    print(f"Visualization saved to {output}")


if __name__ == "__main__":
    main()
