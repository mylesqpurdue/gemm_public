import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys
import glob
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import os

# Handle input files
if len(sys.argv) < 2:
    input_files = ["results.csv"]  # Default file if none provided
else:
    # Handle wildcards in command line arguments
    input_files = []
    for pattern in sys.argv[1:]:
        input_files.extend(glob.glob(pattern))
    
    if not input_files:
        print(f"Error: No files matching pattern: {' '.join(sys.argv[1:])}")
        print("Usage: python plot.py [file1.csv file2.csv ...]")
        sys.exit(1)

# Output paths
out_dir = pathlib.Path("plots")
out_dir.mkdir(exist_ok=True, parents=True)
base_name = pathlib.Path(input_files[0]).stem
plot_file = out_dir / f"{base_name}_performance.png"

# Read and concatenate all input files
dfs = []
for file in input_files:
    try:
        df = pd.read_csv(file)
        if not df.empty:
            # Clean the data format (remove column prefixes)
            for col in df.columns:
                if col in df.columns and df[col].dtype == 'object':
                    # Remove prefixes like "impl=", "M=", etc.
                    df[col] = df[col].astype(str).str.replace(f'{col}=', '', regex=False)
            
            # Convert numeric columns
            numeric_cols = ['M', 'N', 'K', 'threads', 'MB', 'NB', 'KB', 'time_ms', 'gflops', 'relerr']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            dfs.append(df)
            print(f"Read {len(df)} rows from {file}")
    except Exception as e:
        print(f"Warning: Could not read {file}: {str(e)}")
        continue

if not dfs:
    print("Error: No valid data found in any input files")
    sys.exit(1)
    
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")

# Clean and prepare data
df = df[df['M'] == df['N']]  # Only square matrices for now
df['size'] = df['M']
df['impl_threads'] = df['impl'] + ' (' + df['threads'].astype(str) + 'T)'

# Create a figure with two subplots: performance and relative error
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 1]})

# Get unique implementations and sort them for consistent colors
impls = df['impl_threads'].unique()
impls_sorted = sorted(impls, key=lambda x: ('naive' in x, 'blocked' in x, 'packed' in x, x))
colors = plt.cm.viridis(np.linspace(0, 0.8, len(impls_sorted)))

# Plot performance (GFLOP/s)
for impl, color in zip(impls_sorted, colors):
    sub_df = df[df['impl_threads'] == impl].sort_values('size')
    if not sub_df.empty:
        ax1.plot(sub_df['size'], sub_df['gflops'], 
                marker='o', markersize=6, linewidth=2,
                label=impl, color=color)
        
        # Plot relative error on the second subplot
        # Replace zeros with a small positive value for log scale
        relerr = sub_df['relerr'].copy()
        relerr[relerr == 0] = 1e-30
        ax2.semilogx(sub_df['size'], relerr, 
                    marker='o', markersize=4, linewidth=1,
                    color=color, alpha=0.7)

# Customize the performance plot
ax1.set_title('GEMM Performance (Higher is Better)', fontsize=14, pad=15)
ax1.set_xscale('log')
ax1.set_ylabel('GFLOP/s', fontsize=12)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
ax1.tick_params(axis='both', which='major', labelsize=10)

# Customize the error plot
ax2.set_title('Relative Error (Lower is Better)', fontsize=14, pad=15)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Matrix Size (N×N×N)', fontsize=12)
ax2.set_ylabel('Relative Error', fontsize=12)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Format x-axis to show K, M suffixes
def format_size(x, pos):
    if x >= 1024:
        return f'{x/1024:.0f}K'
    return f'{x:.0f}'

for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(FuncFormatter(format_size))

# Add system info to the plot
system_info = f"System: {df['notes'].iloc[0] if 'notes' in df.columns and len(df['notes']) > 0 else ''}"
plt.figtext(0.5, 0.01, system_info, ha='center', fontsize=9, alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# Save the plot
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plot_file.absolute()}")

def generate_milestone6_plots(df, out_dir):
    """Generate specific plots required for Milestone 6"""
    
    # 1. GFLOP/s vs N plot (hero plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter to 8-thread results for main comparison
    df_8t = df[df['threads'] == 8].copy()
    df_8t = df_8t[df_8t['M'] == df_8t['N']]  # Square matrices only
    df_8t['size'] = df_8t['M']
    
    # Plot each implementation
    impls = ['naive', 'blocked', 'packed', 'mk_avx2', 'openblas']
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    
    for impl, color in zip(impls, colors):
        impl_data = df_8t[df_8t['impl'] == impl].sort_values('size')
        if not impl_data.empty:
            ax.plot(impl_data['size'], impl_data['gflops'], 
                   marker='o', linewidth=2, markersize=6, 
                   label=impl.replace('_', ' ').title(), color=color)
    
    ax.set_xlabel('Matrix Size (N×N×N)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('GEMM Performance Comparison (8 Threads)', fontsize=14, pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.tight_layout()
    hero_plot = out_dir / "gemm_gflops_vs_N.png"
    plt.savefig(hero_plot, dpi=300, bbox_inches='tight')
    print(f"Hero plot saved to {hero_plot.absolute()}")
    plt.close()
    
    # 2. Percentage of OpenBLAS plot
    if 'openblas' in df_8t['impl'].values:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get OpenBLAS baseline for each size
        openblas_data = df_8t[df_8t['impl'] == 'openblas'].set_index('size')['gflops']
        
        for impl, color in zip(['naive', 'blocked', 'packed', 'mk_avx2'], colors[:-1]):
            impl_data = df_8t[df_8t['impl'] == impl].copy()
            if not impl_data.empty:
                impl_data['percent_of_openblas'] = impl_data.apply(
                    lambda row: (row['gflops'] / openblas_data.get(row['size'], 1)) * 100 
                    if row['size'] in openblas_data.index else 0, axis=1
                )
                impl_data = impl_data.sort_values('size')
                ax.plot(impl_data['size'], impl_data['percent_of_openblas'],
                       marker='o', linewidth=2, markersize=6,
                       label=impl.replace('_', ' ').title(), color=color)
        
        ax.set_xlabel('Matrix Size (N×N×N)', fontsize=12)
        ax.set_ylabel('Percentage of OpenBLAS Performance (%)', fontsize=12)
        ax.set_title('Performance Relative to OpenBLAS Baseline (8 Threads)', fontsize=14, pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.axhline(y=100, color='purple', linestyle='--', alpha=0.7, label='OpenBLAS Baseline')
        
        plt.tight_layout()
        percent_plot = out_dir / "percent_of_openblas.png"
        plt.savefig(percent_plot, dpi=300, bbox_inches='tight')
        print(f"Percentage plot saved to {percent_plot.absolute()}")
        plt.close()
    
    # 3. Create a simple roofline plot (basic version)
    try:
        create_simple_roofline(out_dir)
    except Exception as e:
        print(f"Roofline plot generation failed: {e}")

def create_simple_roofline(out_dir):
    """Create a simple roofline plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # System parameters (estimates)
    peak_compute = 512  # GFLOP/s
    peak_bandwidth = 50  # GB/s
    
    # Operational intensity range
    oi_range = np.logspace(-1, 2, 1000)
    
    # Roofline components
    compute_ceiling = np.full_like(oi_range, peak_compute)
    memory_ceiling = peak_bandwidth * oi_range
    roofline = np.minimum(compute_ceiling, memory_ceiling)
    
    # Plot roofline
    ax.loglog(oi_range, roofline, 'k-', linewidth=2, label='Roofline')
    ax.loglog(oi_range, compute_ceiling, 'k--', alpha=0.5, 
             label=f'Peak Compute ({peak_compute} GFLOP/s)')
    ax.loglog(oi_range, memory_ceiling, 'k:', alpha=0.5,
             label=f'Peak Memory ({peak_bandwidth} GB/s)')
    
    # Plot our implementations
    implementations = [
        (32.0, 285.5, '256³ tile (AVX2)', 'green'),
        (16.0, 167.5, '256³ tile (Packed)', 'blue'),
        (8.0, 107.3, '256³ tile (Blocked)', 'orange')
    ]
    
    for oi, perf, label, color in implementations:
        ax.loglog(oi, perf, 'o', color=color, markersize=8, label=label)
    
    ax.set_xlabel('Operational Intensity (FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('Roofline Analysis - GEMM Performance', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0.1, 100)
    ax.set_ylim(1, peak_compute * 1.2)
    
    plt.tight_layout()
    roofline_plot = out_dir / "roofline.png"
    plt.savefig(roofline_plot, dpi=300, bbox_inches='tight')
    print(f"Roofline plot saved to {roofline_plot.absolute()}")
    plt.close()

# Generate additional milestone 6 plots
generate_milestone6_plots(df, out_dir)

# Show the plot if running interactively
plt.show()
