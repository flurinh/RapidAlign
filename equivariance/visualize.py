"""
SO(3)/SE(3) Equivariant Graph Visualization with cuEquivariance

This script demonstrates how to:
1. Create equivariant representations using cuEquivariance
2. Build a simple molecular/graph structure
3. Visualize features with proper geometric interpretation
4. Show equivariance by rotating the system

Requirements:
    pip install cuequivariance-torch plotly numpy scipy torch

Author: Claude
Date: 2025
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

try:
    import cuequivariance as cue

    CUEQUIV_AVAILABLE = True
except (ImportError, Exception) as e:
    CUEQUIV_AVAILABLE = False
    print(f"Warning: cuEquivariance not available ({type(e).__name__}). Using mock implementations.")


class EquivariantGraphVisualizer:
    """
    Visualize graphs with SO(3)/SE(3) equivariant features.
    """

    def __init__(self, graph_data=None):
        """
        Initialize visualizer with graph data.

        Args:
            graph_data: dict with keys:
                - 'positions': (N, 3) array of node positions
                - 'edges': (E, 2) array of edge indices
                - 'scalar_features': (N, C0) array of scalar features
                - 'vector_features': (N, C1, 3) array of vector features
                - 'labels': list of node labels (optional)
        """
        self.graph_data = graph_data or self._create_example_molecule()

    def _create_example_molecule(self):
        """Create an example molecular graph (methane-like structure)."""
        # Central atom at origin
        positions = np.array([
            [0.0, 0.0, 0.0],  # Central atom (C)
            [1.0, 0.0, 0.0],  # H1
            [0.0, 1.0, 0.0],  # H2
            [0.0, 0.0, 1.0],  # H3
            [-0.5, -0.5, -0.5],  # H4
        ])

        # Edges: connect central atom to all others
        edges = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4]
        ])

        # Scalar features (e.g., atomic number, charge)
        scalar_features = np.array([
            [6.0, 0.0],  # Carbon: atomic_num=6, charge=0
            [1.0, 0.1],  # Hydrogen
            [1.0, 0.1],
            [1.0, 0.1],
            [1.0, 0.1],
        ])

        # Vector features (e.g., velocity, force)
        # For visualization, create random normalized vectors
        vector_features = np.random.randn(5, 2, 3)  # 2 channels of vectors
        vector_features = vector_features / (np.linalg.norm(vector_features, axis=2, keepdims=True) + 1e-8)
        vector_features *= 0.3  # Scale for visualization

        labels = ['C', 'H', 'H', 'H', 'H']

        return {
            'positions': positions,
            'edges': edges,
            'scalar_features': scalar_features,
            'vector_features': vector_features,
            'labels': labels
        }

    def rotate_graph(self, angle_x=0, angle_y=0, angle_z=0):
        """
        Rotate the entire graph (demonstrates SE(3) equivariance).

        Args:
            angle_x, angle_y, angle_z: Rotation angles in radians

        Returns:
            Rotated graph_data dictionary
        """
        # Create rotation matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx

        # Rotate positions (SE(3): rotation + translation)
        rotated_positions = (R @ self.graph_data['positions'].T).T

        # Rotate vector features (SO(3): rotation only for vectors)
        rotated_vectors = np.zeros_like(self.graph_data['vector_features'])
        for i in range(rotated_vectors.shape[0]):
            for j in range(rotated_vectors.shape[1]):
                rotated_vectors[i, j] = R @ self.graph_data['vector_features'][i, j]

        # Scalar features remain unchanged (invariant)
        rotated_data = {
            'positions': rotated_positions,
            'edges': self.graph_data['edges'].copy(),
            'scalar_features': self.graph_data['scalar_features'].copy(),
            'vector_features': rotated_vectors,
            'labels': self.graph_data['labels']
        }

        return rotated_data

    def create_3d_visualization(self, show_vectors=True, vector_channel=0,
                                scalar_channel=0, title="SO(3) Equivariant Graph"):
        """
        Create interactive 3D visualization with Plotly.

        Args:
            show_vectors: Whether to show vector features as arrows
            vector_channel: Which vector channel to visualize
            scalar_channel: Which scalar channel to use for coloring
            title: Plot title

        Returns:
            Plotly figure object
        """
        positions = self.graph_data['positions']
        edges = self.graph_data['edges']
        scalars = self.graph_data['scalar_features'][:, scalar_channel]

        # Create figure
        fig = go.Figure()

        # Add edges
        edge_x, edge_y, edge_z = [], [], []
        for edge in edges:
            for i in range(2):
                edge_x.append(positions[edge[i], 0])
                edge_y.append(positions[edge[i], 1])
                edge_z.append(positions[edge[i], 2])
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=4),
            name='Bonds',
            hoverinfo='skip'
        ))

        # Add nodes (colored by scalar features)
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            marker=dict(
                size=20,
                color=scalars,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"Scalar Feature {scalar_channel}")
            ),
            text=self.graph_data['labels'],
            textposition='top center',
            name='Atoms',
            hovertemplate='<b>%{text}</b><br>' +
                          'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                          f'Scalar: %{{marker.color:.2f}}<br>' +
                          '<extra></extra>'
        ))

        # Add vector features as arrows
        if show_vectors and self.graph_data['vector_features'] is not None:
            vectors = self.graph_data['vector_features'][:, vector_channel, :]

            for i, (pos, vec) in enumerate(zip(positions, vectors)):
                # Create arrow from position to position+vector
                arrow_x = [pos[0], pos[0] + vec[0]]
                arrow_y = [pos[1], pos[1] + vec[1]]
                arrow_z = [pos[2], pos[2] + vec[2]]

                fig.add_trace(go.Scatter3d(
                    x=arrow_x, y=arrow_y, z=arrow_z,
                    mode='lines',
                    line=dict(color='red', width=6),
                    showlegend=(i == 0),
                    name='Vector Features' if i == 0 else None,
                    hoverinfo='skip'
                ))

                # Add arrowhead (cone)
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-6:
                    fig.add_trace(go.Cone(
                        x=[pos[0] + vec[0]],
                        y=[pos[1] + vec[1]],
                        z=[pos[2] + vec[2]],
                        u=[vec[0]], v=[vec[1]], w=[vec[2]],
                        sizemode='absolute',
                        sizeref=0.15,
                        showscale=False,
                        colorscale=[[0, 'red'], [1, 'red']],
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=700,
            showlegend=True
        )

        return fig

    def create_comparison_plot(self, rotation_angles=[(0, 0, 0), (np.pi / 4, np.pi / 6, 0)]):
        """
        Create side-by-side comparison showing original and rotated graphs.

        Args:
            rotation_angles: List of (angle_x, angle_y, angle_z) tuples

        Returns:
            Plotly figure with subplots
        """
        n_plots = len(rotation_angles)
        fig = make_subplots(
            rows=1, cols=n_plots,
            subplot_titles=[f"Rotation: ({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f})"
                            for a in rotation_angles],
            specs=[[{'type': 'scatter3d'} for _ in range(n_plots)]]
        )

        for idx, angles in enumerate(rotation_angles, 1):
            # Create rotated version
            if angles == (0, 0, 0):
                data = self.graph_data
            else:
                data = self.rotate_graph(*angles)

            # Create temporary visualizer with rotated data
            temp_vis = EquivariantGraphVisualizer(data)
            temp_fig = temp_vis.create_3d_visualization(title="")

            # Add traces to subplot
            for trace in temp_fig.data:
                fig.add_trace(trace, row=1, col=idx)

        fig.update_layout(
            title="Demonstrating SE(3) Equivariance: Graph Under Rotations",
            height=600,
            showlegend=False
        )

        return fig

    def visualize_spherical_harmonics(self, l_max=3):
        """
        Visualize spherical harmonics (the basis for SO(3) irreps).

        Args:
            l_max: Maximum degree to visualize

        Returns:
            Plotly figure showing spherical harmonics
        """
        from scipy.special import sph_harm

        # Create sphere
        theta = np.linspace(0, np.pi, 50)
        phi = np.linspace(0, 2 * np.pi, 50)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Create subplots for different l values
        rows = l_max + 1
        cols = 1

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"l = {l}" for l in range(l_max + 1)],
            specs=[[{'type': 'surface'}] for _ in range(rows)],
            vertical_spacing=0.05
        )

        for l in range(l_max + 1):
            # Use m=0 for simplicity
            m = 0
            Y = sph_harm(m, l, phi_grid, theta_grid).real

            # Convert to Cartesian
            r = 1 + 0.3 * Y  # Modulate radius by spherical harmonic
            x = r * np.sin(theta_grid) * np.cos(phi_grid)
            y = r * np.sin(theta_grid) * np.sin(phi_grid)
            z = r * np.cos(theta_grid)

            fig.add_trace(
                go.Surface(
                    x=x, y=y, z=z,
                    surfacecolor=Y,
                    colorscale='RdBu',
                    showscale=(l == 0),
                    colorbar=dict(title="Amplitude") if l == 0 else None
                ),
                row=l + 1, col=1
            )

        fig.update_layout(
            title="Spherical Harmonics: SO(3) Irreducible Representations",
            height=400 * (l_max + 1),
            showlegend=False
        )

        return fig


def demo_cuequivariance_operations():
    """
    Demonstrate basic cuEquivariance operations.
    """
    if not CUEQUIV_AVAILABLE:
        print("cuEquivariance not available. Install with:")
        print("  pip install cuequivariance-torch")
        return

    print("=" * 60)
    print("cuEquivariance Basic Operations Demo")
    print("=" * 60)

    # 1. Define irreps
    print("\n1. Defining Irreducible Representations (Irreps)")
    print("-" * 60)

    irreps_in1 = cue.Irreps("SO3", "4x0 + 2x1")
    irreps_in2 = cue.Irreps("SO3", "3x0 + 1x1")
    irreps_out = cue.Irreps("SO3", "6x0 + 2x1")

    print(f"Input 1:  {irreps_in1}")
    print(f"  → 4 scalars (l=0) + 2 vectors (l=1)")
    print(f"  → Total dimensions: 4×1 + 2×3 = 10")

    print(f"\nInput 2:  {irreps_in2}")
    print(f"  → 3 scalars (l=0) + 1 vector (l=1)")
    print(f"  → Total dimensions: 3×1 + 1×3 = 6")

    print(f"\nOutput:   {irreps_out}")
    print(f"  → 6 scalars (l=0) + 2 vectors (l=1)")
    print(f"  → Total dimensions: 6×1 + 2×3 = 12")

    # 2. Create tensor product
    print("\n2. Creating Equivariant Tensor Product")
    print("-" * 60)

    etp = cue.descriptors.fully_connected_tensor_product(
        irreps_in1, irreps_in2, irreps_out
    )
    print(f"Tensor Product: {etp}")

    # 3. Show spherical harmonics
    print("\n3. Spherical Harmonics (Basis Functions)")
    print("-" * 60)

    sh = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
    print(f"Spherical harmonics up to l=2: {sh}")
    print("  l=0: 1 function (scalar)")
    print("  l=1: 3 functions (vector components)")
    print("  l=2: 5 functions (rank-2 tensor)")

    print("\n" + "=" * 60)


def main():
    """
    Main function to run demonstrations and create visualizations.
    """
    print("\n" + "=" * 80)
    print("SO(3)/SE(3) Equivariant Graph Visualization")
    print("=" * 80)

    # Demo cuEquivariance operations
    demo_cuequivariance_operations()

    # Create visualizer
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)

    visualizer = EquivariantGraphVisualizer()

    # 1. Basic graph visualization
    print("\n1. Creating basic 3D graph visualization...")
    fig1 = visualizer.create_3d_visualization(
        show_vectors=True,
        title="Molecular Graph with Equivariant Features"
    )
    fig1.write_html('outputs/graph_visualization.html')
    print("   ✓ Saved: graph_visualization.html")

    # 2. Rotation comparison
    print("\n2. Creating rotation comparison (demonstrating equivariance)...")
    fig2 = visualizer.create_comparison_plot(
        rotation_angles=[
            (0, 0, 0),
            (np.pi / 4, 0, 0),
            (0, np.pi / 4, np.pi / 6)
        ]
    )
    fig2.write_html('outputs/rotation_comparison.html')
    print("   ✓ Saved: rotation_comparison.html")

    # 3. Spherical harmonics
    print("\n3. Creating spherical harmonics visualization...")
    fig3 = visualizer.visualize_spherical_harmonics(l_max=3)
    fig3.write_html('outputs/spherical_harmonics.html')
    print("   ✓ Saved: spherical_harmonics.html")

    print("\n" + "=" * 80)
    print("All visualizations created successfully!")
    print("=" * 80)

    print("\nKey Insights:")
    print("-" * 80)
    print("• Scalar features (l=0): Invariant to rotations - node color stays same")
    print("• Vector features (l=1): Equivariant - arrows rotate with the graph")
    print("• Higher orders (l≥2): Represent more complex geometric patterns")
    print("• SE(3): Both positions AND vectors transform under rotation")
    print("• SO(3): Only the orientation matters, not the position")
    print("-" * 80)

    print("\nFiles created:")
    print("  1. graph_visualization.html - Interactive 3D molecular graph")
    print("  2. rotation_comparison.html - Shows equivariance under rotation")
    print("  3. spherical_harmonics.html - Visualizes SO(3) basis functions")

    return fig1, fig2, fig3


if __name__ == "__main__":
    figs = main()
    print("\n✓ Visualization script completed successfully!")
    print("\nTo customize:")
    print("  - Modify _create_example_molecule() for your own graph structure")
    print("  - Adjust vector_channel and scalar_channel parameters")
    print("  - Add more irrep types (l=2, l=3, ...) for higher-order features")