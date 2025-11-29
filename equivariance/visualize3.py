"""
Graph <-> Field Transformations using Irreducible Representations (Irreps)

This script demonstrates bidirectional transformations between:
1. GRAPH SPACE: Discrete node positions with irrep features
2. FIELD SPACE: Continuous functions over R³ with irrep values at every point

Key concepts:
- Graph → Field: "Lifting" - spreading discrete features into continuous space
- Field → Graph: "Sampling/Discretization" - evaluating the field at specific points
"""

import numpy as np
from scipy.special import sph_harm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IrrepFeatures:
    """
    Container for irrep features at a single point or node.

    Features are organized by angular momentum L:
    - L=0: scalars, shape (n_channels, 1)
    - L=1: vectors, shape (n_channels, 3)
    - L=2: tensors, shape (n_channels, 5)
    - L=l: shape (n_channels, 2*l+1)
    """
    features: Dict[int, np.ndarray]  # {L: features array}

    @property
    def l_max(self) -> int:
        return max(self.features.keys()) if self.features else -1

    def get_flat(self) -> np.ndarray:
        """Flatten all features into a single vector."""
        arrays = []
        for l in sorted(self.features.keys()):
            arrays.append(self.features[l].flatten())
        return np.concatenate(arrays) if arrays else np.array([])

    @classmethod
    def from_flat(cls, flat: np.ndarray, channels_per_l: Dict[int, int]) -> 'IrrepFeatures':
        """Reconstruct from flattened vector."""
        features = {}
        idx = 0
        for l in sorted(channels_per_l.keys()):
            n_channels = channels_per_l[l]
            dim = 2 * l + 1
            size = n_channels * dim
            features[l] = flat[idx:idx + size].reshape(n_channels, dim)
            idx += size
        return cls(features)

    def __add__(self, other: 'IrrepFeatures') -> 'IrrepFeatures':
        """Add two irrep features (must have same structure)."""
        result = {}
        for l in self.features:
            result[l] = self.features[l] + other.features[l]
        return IrrepFeatures(result)

    def __mul__(self, scalar: float) -> 'IrrepFeatures':
        """Scalar multiplication."""
        return IrrepFeatures({l: f * scalar for l, f in self.features.items()})

    def __rmul__(self, scalar: float) -> 'IrrepFeatures':
        return self.__mul__(scalar)


@dataclass
class Graph:
    """
    A graph with 3D node positions and irrep features.
    """
    positions: np.ndarray  # (N, 3)
    features: List[IrrepFeatures]  # List of N IrrepFeatures
    edges: Optional[np.ndarray] = None  # (E, 2) edge indices

    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    def get_feature_matrix(self, l: int) -> np.ndarray:
        """Get features for a specific L as (N, n_channels, 2l+1) array."""
        return np.stack([f.features[l] for f in self.features])


# =============================================================================
# RADIAL BASIS FUNCTIONS
# =============================================================================

class RadialBasis(ABC):
    """Abstract base class for radial basis functions."""

    @abstractmethod
    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate basis at distances r."""
        pass

    @abstractmethod
    def derivative(self, r: np.ndarray) -> np.ndarray:
        """Derivative with respect to r."""
        pass


class GaussianRBF(RadialBasis):
    """Gaussian radial basis: exp(-alpha * r²)"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-self.alpha * r ** 2)

    def derivative(self, r: np.ndarray) -> np.ndarray:
        return -2 * self.alpha * r * self(r)


class ExponentialRBF(RadialBasis):
    """Exponential radial basis: exp(-alpha * r)"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-self.alpha * r)

    def derivative(self, r: np.ndarray) -> np.ndarray:
        return -self.alpha * self(r)


class BesselRBF(RadialBasis):
    """
    Bessel radial basis (used in DimeNet, GemNet):
    sin(n * pi * r / cutoff) / r
    """

    def __init__(self, n: int = 1, cutoff: float = 5.0):
        self.n = n
        self.cutoff = cutoff

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r_safe = np.where(r < 1e-8, 1e-8, r)
        mask = r < self.cutoff
        result = np.zeros_like(r)
        result[mask] = np.sin(self.n * np.pi * r_safe[mask] / self.cutoff) / r_safe[mask]
        return result

    def derivative(self, r: np.ndarray) -> np.ndarray:
        r_safe = np.where(r < 1e-8, 1e-8, r)
        mask = r < self.cutoff
        result = np.zeros_like(r)
        arg = self.n * np.pi * r_safe[mask] / self.cutoff
        result[mask] = (self.n * np.pi / self.cutoff * np.cos(arg) - np.sin(arg) / r_safe[mask]) / r_safe[mask]
        return result


class SmoothCutoff:
    """Smooth cutoff function for locality."""

    def __init__(self, cutoff: float, skin: float = 0.5):
        self.cutoff = cutoff
        self.skin = skin
        self.inner = cutoff - skin

    def __call__(self, r: np.ndarray) -> np.ndarray:
        result = np.ones_like(r)
        mask_transition = (r > self.inner) & (r < self.cutoff)
        mask_outside = r >= self.cutoff

        x = (r[mask_transition] - self.inner) / self.skin
        result[mask_transition] = 0.5 * (1 + np.cos(np.pi * x))
        result[mask_outside] = 0.0

        return result


# =============================================================================
# SPHERICAL HARMONICS UTILITIES
# =============================================================================

def cart_to_spherical(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical (r, theta, phi).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    r_safe = np.where(r < 1e-10, 1e-10, r)

    theta = np.arccos(np.clip(z / r_safe, -1, 1))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return r, theta, phi


def real_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute real spherical harmonic Y_l^m.
    """
    if m == 0:
        return sph_harm(0, l, phi, theta).real
    elif m > 0:
        Y_pos = sph_harm(m, l, phi, theta)
        return np.sqrt(2) * ((-1) ** m * Y_pos.real)
    else:
        Y_pos = sph_harm(-m, l, phi, theta)
        return np.sqrt(2) * ((-1) ** (-m + 1) * Y_pos.imag)


def compute_spherical_harmonics(l_max: int, directions: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute all spherical harmonics up to l_max for given directions.
    """
    _, theta, phi = cart_to_spherical(directions)

    result = {}
    for l in range(l_max + 1):
        Y_l = np.zeros((len(directions), 2 * l + 1))
        for i, m in enumerate(range(-l, l + 1)):
            Y_l[:, i] = real_spherical_harmonic(l, m, theta, phi)
        result[l] = Y_l

    return result


# =============================================================================
# GRAPH -> FIELD TRANSFORMATION
# =============================================================================

class GraphToField:
    """
    Transform a graph with irrep features into a continuous field.

    The field at point x is defined as:

    f(x) = Σ_i R(|x - x_i|) * Σ_l Σ_m h_i^{l,m} * Y_l^m(x - x_i)

    where:
    - x_i are node positions
    - R is a radial basis function
    - h_i^{l,m} are the irrep features at node i
    - Y_l^m are real spherical harmonics
    """

    def __init__(
            self,
            radial_basis: RadialBasis = None,
            cutoff: float = 5.0,
            normalize: bool = True
    ):
        self.radial_basis = radial_basis or GaussianRBF(alpha=0.5)
        self.cutoff = SmoothCutoff(cutoff)
        self.normalize = normalize

    def __call__(self, graph: Graph, query_points: np.ndarray) -> List[IrrepFeatures]:
        """
        Evaluate the field at query points.
        """
        M = len(query_points)
        l_max = graph.features[0].l_max

        channels_per_l = {l: f.shape[0] for l, f in graph.features[0].features.items()}

        results = []
        for _ in range(M):
            features = {l: np.zeros((channels_per_l[l], 2 * l + 1)) for l in channels_per_l}
            results.append(IrrepFeatures(features))

        weights = np.zeros(M)

        for i, (pos, node_feat) in enumerate(zip(graph.positions, graph.features)):
            diff = query_points - pos
            distances = np.linalg.norm(diff, axis=1)

            radial = self.radial_basis(distances) * self.cutoff(distances)

            if np.all(radial < 1e-10):
                continue

            mask = distances > 1e-10
            directions = np.zeros_like(diff)
            directions[mask] = diff[mask] / distances[mask, np.newaxis]

            Y = compute_spherical_harmonics(l_max, directions)

            for j in range(M):
                if radial[j] < 1e-10:
                    continue

                for l in channels_per_l:
                    contribution = node_feat.features[l] * radial[j]
                    if l > 0:
                        contribution = contribution * Y[l][j]

                    results[j].features[l] += contribution

            weights += radial

        if self.normalize:
            for j in range(M):
                if weights[j] > 1e-10:
                    for l in results[j].features:
                        results[j].features[l] /= weights[j]

        return results

    def evaluate_scalar_field(self, graph: Graph, query_points: np.ndarray) -> np.ndarray:
        """Convenience method to evaluate just the L=0 (scalar) component."""
        field_values = self(graph, query_points)
        return np.stack([f.features[0] for f in field_values])


# =============================================================================
# FIELD -> GRAPH TRANSFORMATION
# =============================================================================

class FieldToGraph:
    """
    Sample a continuous field to create a graph with irrep features.
    """

    def __init__(self, field_fn: Callable[[np.ndarray], List[IrrepFeatures]]):
        self.field_fn = field_fn

    def sample_at_positions(
            self,
            positions: np.ndarray,
            connectivity_radius: float = 2.0
    ) -> Graph:
        """Sample the field at specified positions."""
        features = self.field_fn(positions)

        tree = cKDTree(positions)
        pairs = tree.query_pairs(connectivity_radius, output_type='ndarray')
        edges = pairs if len(pairs) > 0 else None

        return Graph(positions=positions, features=features, edges=edges)

    def sample_on_grid(
            self,
            bounds: Tuple[np.ndarray, np.ndarray],
            resolution: int = 10,
            threshold: float = 0.1,
            connectivity_radius: float = 2.0
    ) -> Graph:
        """Sample on a regular grid and keep points above threshold."""
        min_corner, max_corner = bounds

        x = np.linspace(min_corner[0], max_corner[0], resolution)
        y = np.linspace(min_corner[1], max_corner[1], resolution)
        z = np.linspace(min_corner[2], max_corner[2], resolution)

        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=1)

        field_values = self.field_fn(grid_points)

        magnitudes = np.array([np.linalg.norm(f.features[0]) for f in field_values])
        mask = magnitudes > threshold

        if not np.any(mask):
            return Graph(positions=np.zeros((0, 3)), features=[], edges=None)

        positions = grid_points[mask]
        features = [f for f, m in zip(field_values, mask) if m]

        if len(positions) > 1:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(connectivity_radius, output_type='ndarray')
            edges = pairs if len(pairs) > 0 else None
        else:
            edges = None

        return Graph(positions=positions, features=features, edges=edges)

    def sample_local_maxima(
            self,
            bounds: Tuple[np.ndarray, np.ndarray],
            resolution: int = 20,
            connectivity_radius: float = 2.0
    ) -> Graph:
        """Find local maxima of the scalar field and sample there."""
        min_corner, max_corner = bounds

        x = np.linspace(min_corner[0], max_corner[0], resolution)
        y = np.linspace(min_corner[1], max_corner[1], resolution)
        z = np.linspace(min_corner[2], max_corner[2], resolution)

        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=1)

        field_values = self.field_fn(grid_points)
        scalar_field = np.array([np.sum(f.features[0] ** 2) for f in field_values])
        scalar_grid = scalar_field.reshape(resolution, resolution, resolution)

        maxima_positions = []
        maxima_indices = []

        for i in range(1, resolution - 1):
            for j in range(1, resolution - 1):
                for k in range(1, resolution - 1):
                    val = scalar_grid[i, j, k]
                    neighborhood = scalar_grid[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]

                    if val == np.max(neighborhood) and val > 0.01:
                        pos = np.array([x[i], y[j], z[k]])
                        maxima_positions.append(pos)
                        maxima_indices.append(i * resolution * resolution + j * resolution + k)

        if not maxima_positions:
            return Graph(positions=np.zeros((0, 3)), features=[], edges=None)

        positions = np.array(maxima_positions)
        features = [field_values[idx] for idx in maxima_indices]

        if len(positions) > 1:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(connectivity_radius, output_type='ndarray')
            edges = pairs if len(pairs) > 0 else None
        else:
            edges = None

        return Graph(positions=positions, features=features, edges=edges)


# =============================================================================
# VISUALIZATION (matplotlib version)
# =============================================================================

class FieldVisualizer:
    """Visualize fields and graphs with irrep features using matplotlib."""

    def plot_graph_3d(self, graph: Graph, title: str = "Graph", ax=None):
        """Plot a 3D graph with its irrep features."""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Plot edges
        if graph.edges is not None:
            for edge in graph.edges:
                i, j = edge
                ax.plot(
                    [graph.positions[i, 0], graph.positions[j, 0]],
                    [graph.positions[i, 1], graph.positions[j, 1]],
                    [graph.positions[i, 2], graph.positions[j, 2]],
                    'gray', linewidth=1, alpha=0.5
                )

        # Get scalar magnitudes for coloring
        if graph.features and 0 in graph.features[0].features:
            scalar_mags = np.array([np.linalg.norm(f.features[0]) for f in graph.features])
        else:
            scalar_mags = np.ones(graph.n_nodes)

        # Plot nodes
        scatter = ax.scatter(
            graph.positions[:, 0],
            graph.positions[:, 1],
            graph.positions[:, 2],
            c=scalar_mags,
            s=100 + 200 * scalar_mags / (np.max(scalar_mags) + 1e-8),
            cmap='viridis',
            edgecolors='black'
        )

        # Plot vector features (L=1) if present
        if graph.features and 1 in graph.features[0].features:
            for i, (pos, feat) in enumerate(zip(graph.positions, graph.features)):
                vec = feat.features[1][0]
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8) * 0.4
                ax.quiver(
                    pos[0], pos[1], pos[2],
                    vec_norm[0], vec_norm[1], vec_norm[2],
                    color='red', arrow_length_ratio=0.3
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        return ax

    def plot_scalar_field_slice(
            self,
            field_fn: Callable,
            bounds: Tuple[np.ndarray, np.ndarray],
            z_slice: float = 0.0,
            resolution: int = 50,
            ax=None
    ):
        """Plot a 2D slice of the scalar field."""
        min_corner, max_corner = bounds

        x = np.linspace(min_corner[0], max_corner[0], resolution)
        y = np.linspace(min_corner[1], max_corner[1], resolution)
        xg, yg = np.meshgrid(x, y, indexing='ij')

        points = np.stack([xg.flatten(), yg.flatten(),
                           np.full(resolution ** 2, z_slice)], axis=1)

        field_values = field_fn(points)
        scalar_field = np.array([np.sum(f.features[0]) for f in field_values])
        scalar_grid = scalar_field.reshape(resolution, resolution)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(
            scalar_grid.T, origin='lower',
            extent=[min_corner[0], max_corner[0], min_corner[1], max_corner[1]],
            cmap='viridis', aspect='equal'
        )
        plt.colorbar(im, ax=ax, label='Field Value')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Scalar Field Slice (z={z_slice})')

        return ax

    def create_comparison_figure(
            self,
            original_graph: Graph,
            field_fn: Callable,
            reconstructed_graph: Graph,
            bounds: Tuple[np.ndarray, np.ndarray]
    ):
        """Create a comparison visualization."""
        fig = plt.figure(figsize=(18, 5))

        # Original graph
        ax1 = fig.add_subplot(131, projection='3d')
        self.plot_graph_3d(original_graph, "Original Graph", ax1)

        # Field slice
        ax2 = fig.add_subplot(132)
        self.plot_scalar_field_slice(field_fn, bounds, z_slice=0.0, ax=ax2)
        # Add original node positions
        ax2.scatter(
            original_graph.positions[:, 0],
            original_graph.positions[:, 1],
            c='red', s=50, marker='x', label='Original nodes'
        )
        ax2.legend()

        # Reconstructed graph
        ax3 = fig.add_subplot(133, projection='3d')
        self.plot_graph_3d(reconstructed_graph, "Reconstructed Graph", ax3)

        plt.tight_layout()
        return fig


# =============================================================================
# DEMONSTRATION
# =============================================================================

def create_example_graph() -> Graph:
    """Create an example molecular-like graph."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, -0.5],
        [-0.5, 0.87, -0.5],
        [-0.5, -0.87, -0.5],
        [0.0, 0.0, 1.0],
    ])

    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 1], [1, 4], [2, 4], [3, 4]
    ])

    features = []
    for i, pos in enumerate(positions):
        feat = {
            0: np.array([[1.0 if i == 0 else 0.5],
                         [0.2 * i]]).reshape(2, 1),
            1: np.array([pos / (np.linalg.norm(pos) + 0.1)]).reshape(1, 3),
            2: np.random.randn(1, 5) * 0.1
        }
        features.append(IrrepFeatures(feat))

    return Graph(positions=positions, features=features, edges=edges)


def run_demonstration():
    """Run the full demonstration pipeline."""
    print("\n" + "=" * 80)
    print("GRAPH <-> FIELD TRANSFORMATION DEMONSTRATION")
    print("=" * 80)

    os.makedirs('outputs', exist_ok=True)

    # Step 1: Create example graph
    print("\n[Step 1] Creating example graph...")
    graph = create_example_graph()
    print(f"  - {graph.n_nodes} nodes")
    print(f"  - {len(graph.edges)} edges")
    print(f"  - Irrep structure: L=0 (2 ch), L=1 (1 ch), L=2 (1 ch)")

    # Step 2: Graph -> Field
    print("\n[Step 2] Graph → Field transformation...")
    g2f = GraphToField(
        radial_basis=GaussianRBF(alpha=0.5),
        cutoff=3.0,
        normalize=True
    )

    def field_fn(points):
        return g2f(graph, points)

    print("  - Using Gaussian RBF with α=0.5")
    print("  - Cutoff radius: 3.0")

    # Step 3: Visualize
    print("\n[Step 3] Visualizing...")
    viz = FieldVisualizer()
    bounds = (np.array([-2, -2, -2]), np.array([2, 2, 2]))

    # Original graph
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    viz.plot_graph_3d(graph, "Original Graph with Irrep Features", ax1)
    fig1.savefig('outputs/graph_original.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  - Saved: graph_original.png")

    # Field slice
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    viz.plot_scalar_field_slice(field_fn, bounds, z_slice=0.0, ax=ax2)
    ax2.scatter(graph.positions[:, 0], graph.positions[:, 1],
                c='red', s=80, marker='x', linewidths=2, label='Nodes')
    ax2.legend()
    fig2.savefig('outputs/field_slice.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  - Saved: field_slice.png")

    # Multiple slices
    fig_slices, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, z_val in zip(axes, [-0.5, 0.0, 0.5]):
        viz.plot_scalar_field_slice(field_fn, bounds, z_slice=z_val, ax=ax)
        ax.set_title(f'z = {z_val}')
    fig_slices.suptitle('Scalar Field at Different Z-slices', fontsize=14)
    plt.tight_layout()
    fig_slices.savefig('outputs/field_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig_slices)
    print("  - Saved: field_slices.png")

    # Step 4: Field -> Graph
    print("\n[Step 4] Field → Graph transformation...")
    f2g = FieldToGraph(field_fn)

    # Method 1: Sample at original positions
    print("\n  Method 1: Sample at original positions")
    graph_sampled = f2g.sample_at_positions(graph.positions, connectivity_radius=2.0)
    print(f"    - Recovered {graph_sampled.n_nodes} nodes")

    orig_feats = np.array([f.features[0].flatten() for f in graph.features])
    sampled_feats = np.array([f.features[0].flatten() for f in graph_sampled.features])
    mse = np.mean((orig_feats - sampled_feats) ** 2)
    print(f"    - Feature MSE: {mse:.6f}")

    # Method 2: Sample on grid
    print("\n  Method 2: Sample on regular grid")
    graph_grid = f2g.sample_on_grid(bounds, resolution=10, threshold=0.3, connectivity_radius=1.5)
    print(f"    - Found {graph_grid.n_nodes} nodes above threshold")

    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    viz.plot_graph_3d(graph_grid, "Sampled on Grid (threshold=0.3)", ax3)
    fig3.savefig('outputs/graph_sampled_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("    - Saved: graph_sampled_grid.png")

    # Method 3: Find local maxima
    print("\n  Method 3: Find local maxima")
    graph_maxima = f2g.sample_local_maxima(bounds, resolution=25, connectivity_radius=2.0)
    print(f"    - Found {graph_maxima.n_nodes} local maxima")

    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    viz.plot_graph_3d(graph_maxima, "Sampled at Local Maxima", ax4)
    fig4.savefig('outputs/graph_sampled_maxima.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("    - Saved: graph_sampled_maxima.png")

    # Step 5: Comparison
    print("\n[Step 5] Creating comparison visualization...")
    fig_compare = viz.create_comparison_figure(graph, field_fn, graph_maxima, bounds)
    fig_compare.savefig('outputs/comparison_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig_compare)
    print("  - Saved: comparison_pipeline.png")

    # Step 6: Different radial basis functions comparison
    print("\n[Step 6] Comparing different radial basis functions...")
    rbfs = [
        ('Gaussian α=0.5', GaussianRBF(alpha=0.5)),
        ('Gaussian α=2.0', GaussianRBF(alpha=2.0)),
        ('Exponential α=1.0', ExponentialRBF(alpha=1.0)),
    ]

    fig_rbf, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, rbf) in zip(axes, rbfs):
        g2f_tmp = GraphToField(radial_basis=rbf, cutoff=3.0, normalize=True)
        field_fn_tmp = lambda pts, g2f=g2f_tmp: g2f(graph, pts)
        viz.plot_scalar_field_slice(field_fn_tmp, bounds, z_slice=0.0, ax=ax)
        ax.scatter(graph.positions[:, 0], graph.positions[:, 1],
                   c='red', s=50, marker='x', linewidths=2)
        ax.set_title(name)

    fig_rbf.suptitle('Effect of Different Radial Basis Functions', fontsize=14)
    plt.tight_layout()
    fig_rbf.savefig('outputs/rbf_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig_rbf)
    print("  - Saved: rbf_comparison.png")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The pipeline demonstrates:

1. GRAPH → FIELD (Lifting):
   - Each node's irrep features are spread into space using:
     * Radial basis functions (Gaussian, exponential, Bessel, etc.)
     * Spherical harmonics for directional encoding (L > 0)
   - The field at any point is a weighted sum of node contributions

2. FIELD → GRAPH (Discretization):
   - Method A: Sample at known positions (exact if positions match)
   - Method B: Sample on a grid with thresholding
   - Method C: Find local maxima (like finding atoms from density)

Key concepts:
- L=0 (scalars): Encoded directly, spread by radial basis
- L=1 (vectors): Directionally encoded via Y_1^m spherical harmonics
- L=2 (tensors): Quadrupolar patterns via Y_2^m

The mathematical formulation:
  f(x) = Σ_i R(|x - x_i|) * Σ_l Σ_m h_i^{l,m} * Y_l^m((x - x_i)/|x - x_i|)

Where:
  - f(x) is the field value at point x
  - R(r) is the radial basis function
  - h_i^{l,m} are the irrep coefficients at node i
  - Y_l^m are real spherical harmonics
    """)

    print("\nOutput files created:")
    print("  - graph_original.png       : Original graph visualization")
    print("  - field_slice.png          : 2D slice of the scalar field")
    print("  - field_slices.png         : Multiple z-slices")
    print("  - graph_sampled_grid.png   : Graph from grid sampling")
    print("  - graph_sampled_maxima.png : Graph from local maxima")
    print("  - comparison_pipeline.png  : Full pipeline comparison")
    print("  - rbf_comparison.png       : Different RBF effects")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
    print("\n✓ Graph <-> Field transformation demonstration complete!")