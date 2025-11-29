"""
Understanding Irreps: From Node Features to Fields

This script explores:
1. What irreps are as features at each node
2. How irreps encode neighborhood information
3. Representing entire graphs with irreps
4. When irreps constitute a field

Author: Claude
Date: 2025
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import sph_harm
import torch

try:
    import cuequivariance as cue

    CUEQUIV_AVAILABLE = True
except:
    CUEQUIV_AVAILABLE = False
    print("Note: cuEquivariance not available. Using conceptual demonstrations.")


class IrrepExplorer:
    """
    Explore irreducible representations as node features.
    """

    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    def create_simple_graph(self):
        """Create a simple graph for demonstration."""
        # Central node with 4 neighbors in different directions
        positions = np.array([
            [0.0, 0.0, 0.0],  # Central node
            [1.0, 0.0, 0.0],  # Right
            [0.0, 1.0, 0.0],  # Up
            [-1.0, 0.0, 0.0],  # Left
            [0.0, -1.0, 0.0],  # Down
            [0.0, 0.0, 1.0],  # Forward
        ])

        edges = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]
        ])

        return positions, edges

    def visualize_l0_scalars(self):
        """
        L=0 (Scalars): Rotation-invariant features

        These represent quantities that don't change under rotation:
        - Atomic number
        - Total charge
        - Node degree
        - Distance to neighbors
        """
        positions, edges = self.create_simple_graph()

        fig = go.Figure()

        # Scalar values (just some example values)
        scalar_values = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add nodes with scalar coloring
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=scalar_values * 10,
                color=scalar_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Scalar Value (l=0)")
            ),
            text=[f"Node {i}<br>Scalar: {v:.1f}" for i, v in enumerate(scalar_values)],
            hoverinfo='text'
        ))

        fig.update_layout(
            title="L=0 Features (Scalars): Rotation Invariant<br>" +
                  "<sub>Size and color show scalar magnitude - same under any rotation</sub>",
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            width=800, height=600
        )

        return fig

    def visualize_l1_vectors(self):
        """
        L=1 (Vectors): Directional features

        These represent directional quantities:
        - Dipole moment
        - Mean neighbor direction
        - Velocity
        - Force
        """
        positions, edges = self.create_simple_graph()

        fig = go.Figure()

        # Vector features: point from central node toward neighbors
        vectors = np.zeros((6, 3))
        vectors[0] = np.array([0.0, 0.0, 0.5])  # Central node: average direction
        for i in range(1, 6):
            direction = positions[i] - positions[0]
            vectors[i] = 0.3 * direction / np.linalg.norm(direction)

        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=15, color='lightblue'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add vector arrows
        for i, (pos, vec) in enumerate(zip(positions, vectors)):
            # Arrow line
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + vec[0]],
                y=[pos[1], pos[1] + vec[1]],
                z=[pos[2], pos[2] + vec[2]],
                mode='lines',
                line=dict(color='red', width=8),
                showlegend=(i == 0),
                name='Vector Feature (l=1)' if i == 0 else None,
                hoverinfo='skip'
            ))

            # Arrowhead
            if np.linalg.norm(vec) > 1e-6:
                fig.add_trace(go.Cone(
                    x=[pos[0] + vec[0]],
                    y=[pos[1] + vec[1]],
                    z=[pos[2] + vec[2]],
                    u=[vec[0]], v=[vec[1]], w=[vec[2]],
                    sizemode='absolute',
                    sizeref=0.2,
                    showscale=False,
                    colorscale=[[0, 'red'], [1, 'red']],
                    showlegend=False,
                    hoverinfo='skip'
                ))

        fig.update_layout(
            title="L=1 Features (Vectors): Equivariant Direction<br>" +
                  "<sub>Red arrows rotate with the graph - encode directional info</sub>",
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            width=800, height=600
        )

        return fig

    def visualize_l2_tensors(self):
        """
        L=2 (Tensors): Quadrupole-like features

        These represent:
        - Quadrupole moments
        - Stress/strain tensors
        - Second-order neighborhood patterns
        - Ellipsoidal shapes
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "Spherical (isotropic)",
                "Prolate (elongated)",
                "Oblate (flattened)"
            ],
            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
        )

        # Create sphere coordinates
        theta = np.linspace(0, np.pi, 30)
        phi = np.linspace(0, 2 * np.pi, 30)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Different l=2, m=0 spherical harmonics (quadrupole patterns)
        for col, (a, b, c, title) in enumerate([
            (1.0, 1.0, 1.0, "Spherical"),
            (1.0, 1.0, 1.5, "Prolate"),
            (1.5, 1.5, 1.0, "Oblate")
        ], 1):
            # Modulate radius with spherical harmonic Y_2^0
            Y20 = sph_harm(0, 2, phi_grid, theta_grid).real
            r = 1 + 0.3 * Y20

            # Apply anisotropy
            x = a * r * np.sin(theta_grid) * np.cos(phi_grid)
            y = b * r * np.sin(theta_grid) * np.sin(phi_grid)
            z = c * r * np.cos(theta_grid)

            fig.add_trace(
                go.Surface(
                    x=x, y=y, z=z,
                    surfacecolor=Y20,
                    colorscale='RdBu',
                    showscale=(col == 1),
                    colorbar=dict(title="Y₂⁰") if col == 1 else None
                ),
                row=1, col=col
            )

        fig.update_layout(
            title="L=2 Features (Rank-2 Tensors): Quadrupole Patterns<br>" +
                  "<sub>Encode ellipsoidal/quadrupolar neighborhood patterns</sub>",
            height=500,
            showlegend=False
        )

        return fig

    def demonstrate_neighborhood_encoding(self):
        """
        Show how irreps at a node encode information about its neighborhood.
        """
        # Create a more interesting local structure
        positions = np.array([
            [0.0, 0.0, 0.0],  # Central node
            [1.0, 0.0, 0.0],  # 1 neighbor to the right
            [1.0, 0.0, 0.0],  # (duplicate for emphasis)
            [0.5, 0.87, 0.0],  # 1 neighbor upper-right
            [-0.5, -0.5, 0.0],  # 1 neighbor lower-left
        ])

        edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])

        # Compute spherical harmonic features from neighborhood
        relative_positions = positions[1:] - positions[0]
        distances = np.linalg.norm(relative_positions, axis=1)

        # Normalize to unit sphere
        directions = relative_positions / (distances[:, np.newaxis] + 1e-8)

        # Convert to spherical coordinates
        r = distances
        theta = np.arccos(np.clip(directions[:, 2], -1, 1))
        phi = np.arctan2(directions[:, 1], directions[:, 0])

        # Compute spherical harmonics for each neighbor direction
        l_max = 2
        features = {}

        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, phi, theta)
                # Weight by distance (closer neighbors contribute more)
                weighted = Y_lm * np.exp(-distances)
                features[f"Y_{l}^{m}"] = np.sum(weighted)

        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                "Local Neighborhood Structure",
                "Encoded Features (Spherical Harmonics)"
            ],
            specs=[[{'type': 'scatter3d'}], [{'type': 'bar'}]],
            row_heights=[0.6, 0.4]
        )

        # Plot the graph
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=4),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

        # Central node
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            marker=dict(size=20, color='red'),
            name='Central Node',
            hoverinfo='text',
            text='Central Node'
        ), row=1, col=1)

        # Neighbor nodes
        fig.add_trace(go.Scatter3d(
            x=positions[1:, 0],
            y=positions[1:, 1],
            z=positions[1:, 2],
            mode='markers',
            marker=dict(size=15, color='lightblue'),
            name='Neighbors',
            hoverinfo='skip'
        ), row=1, col=1)

        # Plot the features as a bar chart
        feature_names = list(features.keys())
        feature_values = [np.abs(features[k]) for k in feature_names]

        colors = ['green' if 'Y_0' in name else 'blue' if 'Y_1' in name else 'red'
                  for name in feature_names]

        fig.add_trace(go.Bar(
            x=feature_names,
            y=feature_values,
            marker_color=colors,
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Magnitude: %{y:.3f}<extra></extra>'
        ), row=2, col=1)

        fig.update_layout(
            title="How Irreps Encode Neighborhood Information<br>" +
                  "<sub>Green: l=0 (scalar), Blue: l=1 (vector), Red: l=2 (tensor)</sub>",
            height=900,
            showlegend=True
        )

        return fig, features

    def visualize_graph_as_irrep_combination(self):
        """
        Show how an entire graph can be represented as a combination of irreps.

        KEY INSIGHT: Each node has irrep features, and the full graph is the
        collection of all node features + connectivity.
        """
        positions, edges = self.create_simple_graph()

        # Generate random irrep features for each node
        # Format: each node has [n_scalars, n_vectors, n_tensors]
        n_nodes = len(positions)

        # l=0: 2 scalar channels per node
        l0_features = np.random.randn(n_nodes, 2)

        # l=1: 3 vector channels per node (each is 3D)
        l1_features = np.random.randn(n_nodes, 3, 3)
        l1_features = l1_features / (np.linalg.norm(l1_features, axis=2, keepdims=True) + 1e-8)
        l1_features *= 0.3

        # l=2: 1 tensor channel per node (5D representation of quadrupole)
        l2_features = np.random.randn(n_nodes, 5)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"L=0 Features<br>{n_nodes} nodes × 2 scalars = {n_nodes * 2} values",
                f"L=1 Features<br>{n_nodes} nodes × 3 vectors × 3D = {n_nodes * 9} values",
                f"L=2 Features<br>{n_nodes} nodes × 5 components = {n_nodes * 5} values"
            ],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )

        # Column 1: Scalars
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=np.abs(l0_features[:, 0]) * 20,
                color=l0_features[:, 0],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Scalar", x=0.3)
            ),
            showlegend=False,
            hoverinfo='text',
            text=[f"Node {i}<br>s₀={l0_features[i, 0]:.2f}<br>s₁={l0_features[i, 1]:.2f}"
                  for i in range(n_nodes)]
        ), row=1, col=1)

        # Column 2: Vectors
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=10, color='lightblue'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)

        # Add vector arrows (just first channel for clarity)
        for i, (pos, vec) in enumerate(zip(positions, l1_features[:, 0, :])):
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + vec[0]],
                y=[pos[1], pos[1] + vec[1]],
                z=[pos[2], pos[2] + vec[2]],
                mode='lines',
                line=dict(color='red', width=6),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)

        # Column 3: Tensors (show as size)
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=3)

        tensor_magnitudes = np.linalg.norm(l2_features, axis=1)
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=tensor_magnitudes * 20,
                color=tensor_magnitudes,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Tensor Mag", x=1.0)
            ),
            showlegend=False,
            hoverinfo='text',
            text=[f"Node {i}<br>||T||={tensor_magnitudes[i]:.2f}"
                  for i in range(n_nodes)]
        ), row=1, col=3)

        total_params = n_nodes * 2 + n_nodes * 9 + n_nodes * 5

        fig.update_layout(
            title=f"Graph Representation as Irrep Features<br>" +
                  f"<sub>Total: {total_params} parameters across all nodes and channels</sub>",
            height=600,
            showlegend=False
        )

        return fig, {
            'l0': l0_features,
            'l1': l1_features,
            'l2': l2_features,
            'total_params': total_params
        }

    def explain_fields(self):
        """
        Explain when irrep features constitute a field.

        A FIELD is when irreps are defined at every point in space (or graph),
        not just at discrete nodes.
        """
        # Create a grid showing field vs. discrete features

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Discrete Node Features<br>(Not a Field)",
                "Continuous Field<br>(Interpolated)"
            ],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )

        positions, edges = self.create_simple_graph()

        # Left: Discrete features at nodes
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[edge[0], 0], positions[edge[1], 0]],
                y=[positions[edge[0], 1], positions[edge[1], 1]],
                z=[positions[edge[0], 2], positions[edge[1], 2]],
                mode='lines',
                line=dict(color='gray', width=4),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=20, color='red'),
            name='Discrete Features',
            hoverinfo='text',
            text=['Features defined<br>ONLY at nodes']
        ), row=1, col=1)

        # Right: Field (continuous)
        # Create a grid of points
        x = np.linspace(-1.5, 1.5, 15)
        y = np.linspace(-1.5, 1.5, 15)
        z = np.linspace(-1.5, 1.5, 15)
        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')

        # Flatten for plotting
        points = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=1)

        # Compute field values (distance to nearest node)
        field_values = np.zeros(len(points))
        for i, pt in enumerate(points):
            distances = np.linalg.norm(positions - pt, axis=1)
            field_values[i] = np.exp(-np.min(distances))

        # Sample for visualization
        sample_idx = np.random.choice(len(points), 500, replace=False)

        fig.add_trace(go.Scatter3d(
            x=points[sample_idx, 0],
            y=points[sample_idx, 1],
            z=points[sample_idx, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=field_values[sample_idx],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Field Value", x=1.0)
            ),
            name='Continuous Field',
            hoverinfo='skip'
        ), row=1, col=2)

        # Show original nodes too
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond'),
            name='Original Nodes',
            hoverinfo='skip'
        ), row=1, col=2)

        fig.update_layout(
            title="Node Features vs. Fields<br>" +
                  "<sub>Left: Features only at nodes | Right: Features everywhere in space</sub>",
            height=600,
            showlegend=True
        )

        return fig


def create_comprehensive_guide():
    """
    Create a comprehensive guide to understanding irreps.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE GUIDE: Understanding Irreps as Graph Features")
    print("=" * 80)

    explorer = IrrepExplorer()

    # Part 1: Basic irrep types
    print("\n" + "-" * 80)
    print("PART 1: What are Irreps?")
    print("-" * 80)
    print("""
Irreducible Representations (Irreps) are the "atomic" building blocks for 
representing geometric quantities that transform in well-defined ways under rotations.

Think of them as different "types" of features, classified by how they rotate:

L=0 (Scalars):     Don't change direction when you rotate
                   Examples: temperature, mass, charge, distance

L=1 (Vectors):     Rotate like arrows
                   Examples: velocity, force, electric field

L=2 (Tensors):     Rotate like quadrupoles/ellipsoids  
                   Examples: stress tensor, quadrupole moment

L=3, L=4, ...:     Higher-order patterns (octupoles, hexadecapoles, ...)

Each node in your graph can have features from ALL of these types simultaneously!
    """)

    fig1 = explorer.visualize_l0_scalars()
    fig1.write_html('outputs/irrep_l0_scalars.html')
    print("✓ Created: irrep_l0_scalars.html")

    fig2 = explorer.visualize_l1_vectors()
    fig2.write_html('outputs/irrep_l1_vectors.html')
    print("✓ Created: irrep_l1_vectors.html")

    fig3 = explorer.visualize_l2_tensors()
    fig3.write_html('outputs/irrep_l2_tensors.html')
    print("✓ Created: irrep_l2_tensors.html")

    # Part 2: Neighborhood encoding
    print("\n" + "-" * 80)
    print("PART 2: How Irreps Encode Neighborhood Information")
    print("-" * 80)
    print("""
When we use irreps as node features, we're encoding information about the 
node's local neighborhood in a rotation-equivariant way.

The key insight: Different irrep types capture different aspects of geometry:

- L=0: How MANY neighbors, total mass, average distance
       → Rotation-invariant aggregate info

- L=1: WHERE neighbors are (net direction)
       → "Center of mass" of neighborhood
       → Points toward asymmetry

- L=2: SHAPE of neighborhood distribution
       → Is it elongated in one direction?
       → Quadrupolar patterns

- L≥3: Increasingly fine-grained directional patterns

Example: A node with 3 neighbors to the right will have:
  - L=0: "I have 3 neighbors"
  - L=1: "They're mostly to my right" (vector pointing right)
  - L=2: "They form a line" (tensor showing elongation)
    """)

    fig4, features = explorer.demonstrate_neighborhood_encoding()
    fig4.write_html('outputs/irrep_neighborhood_encoding.html')
    print("✓ Created: irrep_neighborhood_encoding.html")

    print("\nComputed features for central node:")
    for name, value in list(features.items())[:8]:
        print(f"  {name}: {value:.4f}")

    # Part 3: Graph representation
    print("\n" + "-" * 80)
    print("PART 3: Representing Entire Graphs with Irreps")
    print("-" * 80)
    print("""
YES! An entire graph can be represented as a combination of irreps.

The full representation consists of:

1. CONNECTIVITY: The graph structure (edges)
   → This is usually fixed or changes slowly

2. NODE FEATURES: Irrep features at each node
   → Each node i has features: [h_i^(0), h_i^(1), h_i^(2), ...]
   → h_i^(l) is the L=l irrep feature at node i

3. GLOBAL FEATURES (optional): Graph-level irreps
   → Aggregate information about the entire graph

Mathematical notation:
  Node i features: h_i = ⊕[l=0 to L_max] h_i^(l)

Where:
  - h_i^(0) ∈ ℝ^{n_0}          (n_0 scalar channels)
  - h_i^(1) ∈ ℝ^{n_1 × 3}      (n_1 vector channels, each 3D)
  - h_i^(2) ∈ ℝ^{n_2 × 5}      (n_2 tensor channels, each 5D)
  - h_i^(l) ∈ ℝ^{n_l × (2l+1)} (n_l channels, each (2l+1)D)

The ⊕ symbol means "direct sum" - we just concatenate all these features!

Total parameters per node: n_0·1 + n_1·3 + n_2·5 + ... + n_L·(2L+1)
    """)

    fig5, feature_info = explorer.visualize_graph_as_irrep_combination()
    fig5.write_html('outputs/irrep_full_graph.html')
    print("✓ Created: irrep_full_graph.html")
    print(f"\nExample: 6 nodes with 2x0 + 3x1 + 1x2 irreps")
    print(f"  → Total parameters: {feature_info['total_params']}")

    # Part 4: Fields
    print("\n" + "-" * 80)
    print("PART 4: When are Irreps a Field?")
    print("-" * 80)
    print("""
A FIELD is a function that assigns irrep features to EVERY point in space,
not just at discrete graph nodes.

Mathematical definition:
  f: ℝ³ → ⊕[l] ℝ^{n_l × (2l+1)}

For each position x ∈ ℝ³, f(x) gives you irrep features.

Examples:

NOT A FIELD:
  - Discrete node features on a molecular graph
  - Features only at atom positions
  - Most GNNs operate here

IS A FIELD:  
  - Electron density in quantum mechanics: ρ(x) (scalar field)
  - Electromagnetic field: E(x), B(x) (vector fields)
  - Continuous neural fields (Neural Radiance Fields)
  - Fields on manifolds or meshes with interpolation

How to make node features into a field:
  1. Use radial basis functions centered at nodes
  2. Interpolate between node values
  3. Learn a continuous neural network f(x)

Key distinction:
  • Node features: f defined only at N points → N × d parameters
  • Field: f defined everywhere → ∞ points, but parameterized by 
    neural network or basis functions

In practice, many "field" methods still discretize space, but the key is
that the representation is designed to generalize to new spatial locations.
    """)

    fig6 = explorer.explain_fields()
    fig6.write_html('outputs/irrep_fields.html')
    print("✓ Created: irrep_fields.html")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Key Takeaways")
    print("=" * 80)
    print("""
1. IRREPS ARE FEATURE TYPES:
   Different L values encode different geometric aspects of the neighborhood

2. NODES HAVE IRREP FEATURES:
   Each node can have features from multiple irrep types simultaneously:
   node_features = [scalars, vectors, tensors, ...]

3. GRAPHS ARE COLLECTIONS OF IRREPS:
   Full graph = node features + connectivity
   Total representation size = (# nodes) × (# channels) × (irrep dimensions)

4. FIELDS ARE CONTINUOUS EXTENSIONS:
   Instead of features at N discrete points, fields define features 
   at ALL points in space (or on a manifold)

5. WHY THIS MATTERS:
   - Equivariance: Features transform correctly under rotations
   - Efficiency: Don't need to learn rotation-specific features
   - Interpretability: Each L has geometric meaning
   - Expressiveness: Can represent complex directional patterns

Files created:
  1. irrep_l0_scalars.html - Scalar features (rotation invariant)
  2. irrep_l1_vectors.html - Vector features (equivariant directions)
  3. irrep_l2_tensors.html - Tensor features (quadrupole patterns)
  4. irrep_neighborhood_encoding.html - How irreps encode local structure
  5. irrep_full_graph.html - Full graph representation
  6. irrep_fields.html - Node features vs. continuous fields
    """)

    print("=" * 80)
    print("\nNext steps to deepen understanding:")
    print("  1. Experiment with different L_max values")
    print("  2. Try different numbers of channels per irrep type")
    print("  3. Implement message passing that mixes irrep types")
    print("  4. Compare parameter count: all scalars vs. mixed irreps")
    print("=" * 80)


if __name__ == "__main__":
    create_comprehensive_guide()
    print("\n✓ Irrep exploration complete!")