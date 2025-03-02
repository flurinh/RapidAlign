"""
Benchmark Visualization Script for Point Cloud Alignment Optimizations

This script reads benchmark results from CSV files and creates visualizations
to compare the performance of different optimization techniques.

Requirements:
- pandas
- matplotlib
- numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 11})

def plot_grid_acceleration_benchmarks(csv_file):
    """Plot benchmarks for grid acceleration with different cell sizes"""
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    # Read the CSV file
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Execution time vs. grid cell size
    axs[0, 0].plot(df['grid_cell_size'], df['total_time'], 'o-', linewidth=2)
    axs[0, 0].set_xlabel('Grid Cell Size')
    axs[0, 0].set_ylabel('Execution Time (ms)')
    axs[0, 0].set_title('Execution Time vs. Grid Cell Size')
    axs[0, 0].grid(True)
    
    # Plot 2: Mean error vs. grid cell size
    axs[0, 1].plot(df['grid_cell_size'], df['mean_error'], 'o-', color='green', linewidth=2)
    axs[0, 1].set_xlabel('Grid Cell Size')
    axs[0, 1].set_ylabel('Mean Error')
    axs[0, 1].set_title('Alignment Error vs. Grid Cell Size')
    axs[0, 1].grid(True)
    
    # Create a second y-axis for max error if it exists
    if 'max_error' in df.columns:
        ax2 = axs[0, 1].twinx()
        ax2.plot(df['grid_cell_size'], df['max_error'], 'x--', color='red', alpha=0.7)
        ax2.set_ylabel('Max Error', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot 3: Plot the tradeoff between speed and error
    axs[1, 0].scatter(df['total_time'], df['mean_error'], s=80, c=df['grid_cell_size'], 
                      cmap='viridis', alpha=0.8)
    axs[1, 0].set_xlabel('Execution Time (ms)')
    axs[1, 0].set_ylabel('Mean Error')
    axs[1, 0].set_title('Speed-Accuracy Tradeoff for Grid Cell Sizes')
    
    # Add cell size annotations
    for i, row in df.iterrows():
        axs[1, 0].annotate(f"{row['grid_cell_size']:.3f}", 
                          (row['total_time'], row['mean_error']),
                          textcoords="offset points", 
                          xytext=(0, 5), 
                          ha='center')
    
    axs[1, 0].grid(True)
    
    # Plot 4: Nearest neighbor time vs grid cell size (if available)
    if 'nearest_neighbor_time' in df.columns and df['nearest_neighbor_time'].sum() > 0:
        axs[1, 1].plot(df['grid_cell_size'], df['nearest_neighbor_time'], 'o-', color='purple', linewidth=2)
        axs[1, 1].set_xlabel('Grid Cell Size')
        axs[1, 1].set_ylabel('Nearest Neighbor Search Time (ms)')
        axs[1, 1].set_title('Nearest Neighbor Search Time vs. Grid Cell Size')
    else:
        # Just duplicate the most important plot if NN time is not available
        axs[1, 1].plot(df['grid_cell_size'], df['total_time'], 'o-', color='blue', linewidth=2)
        axs[1, 1].set_xlabel('Grid Cell Size')
        axs[1, 1].set_ylabel('Execution Time (ms)')
        axs[1, 1].set_title('Execution Time vs. Grid Cell Size (Duplicate)')
    
    axs[1, 1].grid(True)
    
    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig('grid_acceleration_benchmark_plots.png', dpi=300)
    print("Saved grid acceleration benchmark plots to grid_acceleration_benchmark_plots.png")
    plt.close()

def plot_optimization_combinations(csv_file):
    """Plot benchmarks for combinations of optimizations"""
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    # Read the CSV file
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Create a new optimization_name column for better labeling
    optimization_names = {
        0: 'Baseline',
        1: 'Grid Accel',
        2: 'CUDA Streams',
        3: 'Grid+Streams'
    }
    df['optimization_name'] = df['optimization_flags'].map(optimization_names)
    
    # Get unique point counts and batch sizes
    point_counts = sorted(df['point_count'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create figure for total time
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Total time vs. point count by optimization
    # Group by point count and optimization, averaging over batch sizes
    time_by_point_opt = df.groupby(['point_count', 'optimization_name'])['total_time'].mean().reset_index()
    
    # Pivot for easier plotting
    pivot_time = time_by_point_opt.pivot(index='point_count', columns='optimization_name', values='total_time')
    
    # Plot each optimization
    colors = ['blue', 'green', 'red', 'purple']
    for i, opt in enumerate(pivot_time.columns):
        axs[0, 0].plot(pivot_time.index, pivot_time[opt], 'o-', 
                       color=colors[i % len(colors)], 
                       linewidth=2, 
                       label=opt)
    
    axs[0, 0].set_xlabel('Point Count')
    axs[0, 0].set_ylabel('Average Execution Time (ms)')
    axs[0, 0].set_title('Execution Time vs. Point Count by Optimization')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Use log scale for x and y axes if ranges are large
    if max(pivot_time.index) / min(pivot_time.index) > 10:
        axs[0, 0].set_xscale('log')
    
    if pivot_time.max().max() / pivot_time.min().min() > 10:
        axs[0, 0].set_yscale('log')
    
    # Plot 2: Total time vs. batch size by optimization
    # Group by batch size and optimization, averaging over point counts
    time_by_batch_opt = df.groupby(['batch_size', 'optimization_name'])['total_time'].mean().reset_index()
    
    # Pivot for easier plotting
    pivot_time_batch = time_by_batch_opt.pivot(index='batch_size', columns='optimization_name', values='total_time')
    
    # Plot each optimization
    for i, opt in enumerate(pivot_time_batch.columns):
        axs[0, 1].plot(pivot_time_batch.index, pivot_time_batch[opt], 'o-', 
                       color=colors[i % len(colors)], 
                       linewidth=2, 
                       label=opt)
    
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Average Execution Time (ms)')
    axs[0, 1].set_title('Execution Time vs. Batch Size by Optimization')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Use log scale for x and y axes if ranges are large
    if max(pivot_time_batch.index) / min(pivot_time_batch.index) > 10:
        axs[0, 1].set_xscale('log')
    
    if pivot_time_batch.max().max() / pivot_time_batch.min().min() > 10:
        axs[0, 1].set_yscale('log')
    
    # Plot 3: Speedup relative to baseline
    # Filter for baseline results
    baseline_df = df[df['optimization_flags'] == 0].copy()
    
    # Merge with original dataframe to compute speedup
    merged_df = df.merge(
        baseline_df[['point_count', 'batch_size', 'total_time']],
        on=['point_count', 'batch_size'],
        suffixes=('', '_baseline')
    )
    
    # Compute speedup
    merged_df['speedup'] = merged_df['total_time_baseline'] / merged_df['total_time']
    
    # Group by point count and optimization for speedup plot
    speedup_by_point = merged_df.groupby(['point_count', 'optimization_name'])['speedup'].mean().reset_index()
    pivot_speedup = speedup_by_point.pivot(index='point_count', columns='optimization_name', values='speedup')
    
    # Remove baseline from pivot (speedup = 1.0)
    if 'Baseline' in pivot_speedup.columns:
        pivot_speedup = pivot_speedup.drop('Baseline', axis=1)
    
    # Plot speedup vs point count
    for i, opt in enumerate(pivot_speedup.columns):
        axs[1, 0].plot(pivot_speedup.index, pivot_speedup[opt], 'o-', 
                       color=colors[(i+1) % len(colors)], 
                       linewidth=2, 
                       label=opt)
    
    axs[1, 0].set_xlabel('Point Count')
    axs[1, 0].set_ylabel('Speedup (vs. Baseline)')
    axs[1, 0].set_title('Speedup vs. Point Count by Optimization')
    axs[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.7)  # Add reference line at y=1.0
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Use log scale for x-axis if range is large
    if max(pivot_speedup.index) / min(pivot_speedup.index) > 10:
        axs[1, 0].set_xscale('log')
    
    # Plot 4: Error comparison
    # Group by optimization type and average over all tests
    error_by_opt = df.groupby('optimization_name')[['mean_error', 'max_error']].mean().reset_index()
    
    # Create bar chart for error
    bar_width = 0.35
    x = np.arange(len(error_by_opt))
    
    axs[1, 1].bar(x - bar_width/2, error_by_opt['mean_error'], bar_width, label='Mean Error')
    axs[1, 1].bar(x + bar_width/2, error_by_opt['max_error'], bar_width, label='Max Error')
    
    axs[1, 1].set_xlabel('Optimization Type')
    axs[1, 1].set_ylabel('Error')
    axs[1, 1].set_title('Error Comparison by Optimization')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(error_by_opt['optimization_name'])
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig('optimization_combinations_plots.png', dpi=300)
    print("Saved optimization combinations plots to optimization_combinations_plots.png")
    plt.close()

def plot_detailed_timings(csv_file):
    """Plot detailed timing breakdowns if available"""
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    # Read the CSV file
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Check if detailed timing columns exist and have data
    timing_columns = ['centroid_time', 'nearest_neighbor_time', 'covariance_time', 
                      'svd_time', 'transform_time']
    
    has_detailed_timing = all(col in df.columns for col in timing_columns) and df[timing_columns].sum().sum() > 0
    
    if not has_detailed_timing:
        print("Detailed timing information not available in the CSV file")
        return
    
    # Create a mapping for optimization flags
    optimization_names = {
        0: 'Baseline',
        1: 'Grid Accel',
        2: 'CUDA Streams',
        3: 'Grid+Streams'
    }
    df['optimization_name'] = df['optimization_flags'].map(optimization_names)
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Stacked bar chart of execution time breakdown by optimization
    # Group by optimization type and average over all tests
    time_breakdown = df.groupby('optimization_name')[timing_columns].mean().reset_index()
    
    # Reshape for stacked bar chart
    time_breakdown_melted = pd.melt(
        time_breakdown, 
        id_vars=['optimization_name'], 
        value_vars=timing_columns, 
        var_name='operation', 
        value_name='time'
    )
    
    # Create prettier operation names
    operation_names = {
        'centroid_time': 'Centroid Computation',
        'nearest_neighbor_time': 'Nearest Neighbor Search',
        'covariance_time': 'Covariance Computation',
        'svd_time': 'SVD Decomposition',
        'transform_time': 'Transform Application'
    }
    time_breakdown_melted['operation'] = time_breakdown_melted['operation'].map(operation_names)
    
    # Plot stacked bar chart
    pivot_time = time_breakdown_melted.pivot(index='optimization_name', columns='operation', values='time')
    pivot_time.plot(kind='bar', stacked=True, ax=axs[0, 0], colormap='viridis')
    
    axs[0, 0].set_xlabel('Optimization Type')
    axs[0, 0].set_ylabel('Time (ms)')
    axs[0, 0].set_title('Execution Time Breakdown by Optimization')
    axs[0, 0].legend(title='Operation')
    axs[0, 0].grid(True, axis='y')
    
    # Plot 2: Pie chart of time breakdown for best optimization
    # Find the optimization with the lowest total time
    best_opt = df.groupby('optimization_name')['total_time'].mean().idxmin()
    best_opt_df = df[df['optimization_name'] == best_opt]
    
    # Average the timing breakdowns for the best optimization
    best_timing = best_opt_df[timing_columns].mean()
    
    # Plot pie chart
    axs[0, 1].pie(best_timing, labels=[operation_names[col] for col in timing_columns], 
                 autopct='%1.1f%%', shadow=True, startangle=90)
    axs[0, 1].set_title(f'Time Breakdown for {best_opt}')
    
    # Plot 3: Nearest neighbor time comparison
    if 'nearest_neighbor_time' in df.columns and df['nearest_neighbor_time'].sum() > 0:
        # Group by point count and optimization
        nn_time = df.groupby(['point_count', 'optimization_name'])['nearest_neighbor_time'].mean().reset_index()
        
        # Pivot for easier plotting
        pivot_nn = nn_time.pivot(index='point_count', columns='optimization_name', values='nearest_neighbor_time')
        
        # Plot each optimization
        colors = ['blue', 'green', 'red', 'purple']
        for i, opt in enumerate(pivot_nn.columns):
            axs[1, 0].plot(pivot_nn.index, pivot_nn[opt], 'o-', 
                        color=colors[i % len(colors)], 
                        linewidth=2, 
                        label=opt)
        
        axs[1, 0].set_xlabel('Point Count')
        axs[1, 0].set_ylabel('Nearest Neighbor Time (ms)')
        axs[1, 0].set_title('Nearest Neighbor Search Time vs. Point Count')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Use log scale if range is large
        if max(pivot_nn.index) / min(pivot_nn.index) > 10:
            axs[1, 0].set_xscale('log')
        
        if pivot_nn.max().max() / pivot_nn.min().min() > 10:
            axs[1, 0].set_yscale('log')
    else:
        axs[1, 0].text(0.5, 0.5, 'Nearest Neighbor Time Data Not Available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axs[1, 0].transAxes, fontsize=14)
    
    # Plot 4: Horizontal bar chart comparing timings for Grid vs. Baseline
    if 'Grid Accel' in df['optimization_name'].values and 'Baseline' in df['optimization_name'].values:
        # Filter for grid acceleration and baseline
        grid_baseline_df = df[df['optimization_name'].isin(['Grid Accel', 'Baseline'])]
        
        # Group by optimization and average over all tests
        comparison = grid_baseline_df.groupby('optimization_name')[timing_columns].mean().reset_index()
        
        # Reshape for grouped bar chart
        comparison_melted = pd.melt(
            comparison, 
            id_vars=['optimization_name'], 
            value_vars=timing_columns, 
            var_name='operation', 
            value_name='time'
        )
        
        # Create prettier operation names
        comparison_melted['operation'] = comparison_melted['operation'].map(operation_names)
        
        # Pivot for easier plotting
        pivot_comparison = comparison_melted.pivot(index='operation', columns='optimization_name', values='time')
        
        # Plot horizontal bar chart
        pivot_comparison.plot(kind='barh', ax=axs[1, 1])
        
        axs[1, 1].set_xlabel('Time (ms)')
        axs[1, 1].set_ylabel('Operation')
        axs[1, 1].set_title('Timing Comparison: Grid Acceleration vs. Baseline')
        axs[1, 1].grid(True, axis='x')
    else:
        axs[1, 1].text(0.5, 0.5, 'Grid Acceleration vs. Baseline Comparison Not Available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axs[1, 1].transAxes, fontsize=14)
    
    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig('detailed_timing_breakdown_plots.png', dpi=300)
    print("Saved detailed timing breakdown plots to detailed_timing_breakdown_plots.png")
    plt.close()

if __name__ == "__main__":
    print("Generating benchmark visualization plots...")
    
    # Plot grid acceleration benchmarks
    if os.path.exists("grid_acceleration_benchmark.csv"):
        plot_grid_acceleration_benchmarks("grid_acceleration_benchmark.csv")
    
    # Plot optimization combinations
    if os.path.exists("optimization_combinations_benchmark.csv"):
        plot_optimization_combinations("optimization_combinations_benchmark.csv")
        plot_detailed_timings("optimization_combinations_benchmark.csv")
    
    # Plot any traditional benchmark results
    if os.path.exists("benchmark_results.csv"):
        print("Traditional benchmark results found. Use plot_benchmarks.py to visualize these.")
    
    print("Visualization complete!")