import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.data import Batch
from pathlib import Path
import sys


# Import your modules
from data.synthetic import SyntheticPointCloudDataset
from data.noise import NoiseConfig, noisify_batch


def main():

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    print("Loading dataset...")
    # Get a single clean graph to serve as the base for all comparisons
    ds = SyntheticPointCloudDataset(num_graphs=1, min_num_nodes=30, max_num_nodes=50)
    clean_data = ds[0]
    batch = Batch.from_data_list([clean_data])
    clean_pos = clean_data.pos.numpy()

    # ==========================================================================
    # 1. Define Noise Scenarios (4 Levels per Type)
    # ==========================================================================
    print("Generating noise samples...")

    # Helper to make 4 configs of increasing severity for a given type
    def make_levels(base_type, param_name, values):
        configs = []
        for v in values:
            # Create a dict for the specific parameter
            kwargs = {
                "types": (base_type,),
                param_name: v,
                # Ensure rigid doesn't count towards loss unless specified
                "include_rigid_in_supervision": False
            }
            # Add defaults for other params if needed to isolate the effect
            configs.append(NoiseConfig(**kwargs))
        return configs

    # Define the dropdown categories
    categories = {
        "Rigid (L=0)": make_levels("rigid", "max_rotation_deg", [45, 90, 135, 180]),
        "Global Gaussian": make_levels("global_gaussian", "sigma_global", [0.05, 0.15, 0.3, 0.6]),
        "Local Gaussian": make_levels("local_gaussian", "sigma_local", [0.2, 0.5, 1.0, 2.0]),
        "Anisotropic Scale": make_levels("anisotropic_scale", "scale_std", [0.1, 0.3, 0.5, 0.8]),
        "Shear": make_levels("shear", "shear_max", [0.2, 0.5, 0.8, 1.2]),
        "Bend": make_levels("bend", "bend_max_deg", [20, 45, 90, 135]),
        "Drift": make_levels("drift", "drift_max", [0.5, 1.0, 1.5, 2.5]),
        "Overlap": make_levels("overlap", "overlap_fraction", [0.1, 0.3, 0.5, 0.8]),
    }

    # ==========================================================================
    # 2. Build Figure and Traces
    # ==========================================================================
    # 2x2 Grid
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        horizontal_spacing=0.05, vertical_spacing=0.1
    )

    # Storage for update menu logic
    all_buttons = []

    # We simply append traces sequentially.
    # Total Traces = Num_Categories * 4_Levels * 3_Traces_Per_Level
    # We must carefully manage the "visible" array for the dropdown.

    trace_start_index = 0

    for cat_name, configs in categories.items():
        # Store metadata for the 4 subplots of this category to update titles later
        subplot_titles = []

        # We need to know which traces belong to this category to toggle visibility
        # Indices in the global trace list
        cat_trace_indices = []

        for idx, cfg in enumerate(configs):
            row = (idx // 2) + 1
            col = (idx % 2) + 1

            # Apply Noise
            noisy_batch, _, metas = noisify_batch(batch, cfg)
            noisy_pos = noisy_batch.pos.numpy()
            meta = metas[0]

            # Format Title with L metrics
            title = (f"L={meta['L_true']:.3f} "
                     f"(N:{meta['L_node']:.2f}, E:{meta['L_edge']:.2f})")
            subplot_titles.append(title)

            # --- Trace 1: Clean (Blue) ---
            t1 = go.Scatter3d(
                x=clean_pos[:, 0], y=clean_pos[:, 1], z=clean_pos[:, 2],
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.3),
                name='Clean', showlegend=False,
                visible=(cat_name == list(categories.keys())[0])  # Visible if first cat
            )
            fig.add_trace(t1, row=row, col=col)
            cat_trace_indices.append(trace_start_index)
            trace_start_index += 1

            # --- Trace 2: Noisy (Red) ---
            t2 = go.Scatter3d(
                x=noisy_pos[:, 0], y=noisy_pos[:, 1], z=noisy_pos[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.8),
                name='Noisy', showlegend=False,
                visible=(cat_name == list(categories.keys())[0])
            )
            fig.add_trace(t2, row=row, col=col)
            cat_trace_indices.append(trace_start_index)
            trace_start_index += 1

            # --- Trace 3: Displacement Lines (Grey) ---
            xe, ye, ze = [], [], []
            for p_c, p_n in zip(clean_pos, noisy_pos):
                xe.extend([p_c[0], p_n[0], None])
                ye.extend([p_c[1], p_n[1], None])
                ze.extend([p_c[2], p_n[2], None])

            t3 = go.Scatter3d(
                x=xe, y=ye, z=ze,
                mode='lines',
                line=dict(color='grey', width=1),
                opacity=0.3,
                showlegend=False,
                visible=(cat_name == list(categories.keys())[0])
            )
            fig.add_trace(t3, row=row, col=col)
            cat_trace_indices.append(trace_start_index)
            trace_start_index += 1

        # Create Button for this Category
        # 1. Visibility Vector: True only for this category's traces
        total_traces = len(categories) * 4 * 3
        visible_mask = [False] * total_traces
        for i in cat_trace_indices:
            visible_mask[i] = True

        # 2. Annotations (Subplot Titles): Need to construct full list of 4 annotations
        # Position mapping for 2x2 grid titles
        # (This is approximate; Plotly requires manual x/y for layout.annotations)
        # However, updating 'layout.annotations' replaces ALL annotations.

        new_annotations = [
            dict(text=subplot_titles[0], x=0.225, y=1.0, showarrow=False, xref="paper", yref="paper",
                 font=dict(size=12)),
            dict(text=subplot_titles[1], x=0.775, y=1.0, showarrow=False, xref="paper", yref="paper",
                 font=dict(size=12)),
            dict(text=subplot_titles[2], x=0.225, y=0.45, showarrow=False, xref="paper", yref="paper",
                 font=dict(size=12)),
            dict(text=subplot_titles[3], x=0.775, y=0.45, showarrow=False, xref="paper", yref="paper",
                 font=dict(size=12))
        ]

        button = dict(
            label=cat_name,
            method="update",
            args=[
                {"visible": visible_mask},
                {"annotations": new_annotations, "title": f"Noise Type: {cat_name}"}
            ]
        )
        all_buttons.append(button)

    # ==========================================================================
    # 3. Final Layout
    # ==========================================================================
    # Set initial titles (for the first category)
    first_cat = list(categories.keys())[0]
    # We trigger the first button's logic implicitly by setting layout

    fig.update_layout(
        title=f"Noise Type: {first_cat}",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=all_buttons,
            x=0.1, y=1.15,
            showactive=True
        )],
        margin=dict(l=20, r=20, t=100, b=20),
        # Ensure aspect ratio is preserved
        scene1=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        scene3=dict(aspectmode='data'),
        scene4=dict(aspectmode='data'),
    )

    # Initialize annotations for the first view
    fig.update_layout(annotations=all_buttons[0]['args'][1]['annotations'])

    filename = "severity_grid.html"
    fig.write_html(filename)
    print(f"Visualization saved to {filename}")


if __name__ == "__main__":
    main()