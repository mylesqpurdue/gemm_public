#!/usr/bin/env python3
"""
Quick verification script to check if plots contain data
"""

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys

def verify_csv_data(csv_file):
    """Verify CSV data can be read and processed"""
    print(f"Verifying CSV data: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Read {len(df)} rows")
        
        # Clean the data format (remove column prefixes)
        for col in df.columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(f'{col}=', '', regex=False)
        
        # Convert numeric columns
        numeric_cols = ['M', 'N', 'K', 'threads', 'MB', 'NB', 'KB', 'time_ms', 'gflops', 'relerr']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Data cleaned successfully")
        print(f"✓ Implementations found: {df['impl'].unique()}")
        print(f"✓ Matrix sizes: {sorted(df['N'].unique())}")
        print(f"✓ GFLOP/s range: {df['gflops'].min():.2f} - {df['gflops'].max():.2f}")
        
        # Show top performers
        top_results = df.nlargest(3, 'gflops')[['impl', 'N', 'threads', 'gflops']]
        print("\n🏆 Top 3 Results:")
        for _, row in top_results.iterrows():
            print(f"   {row['impl']} N={row['N']} ({row['threads']}T): {row['gflops']:.2f} GFLOP/s")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def create_quick_plot(df, output_file):
    """Create a quick verification plot"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter to 8-thread results
        df_8t = df[df['threads'] == 8].copy()
        df_8t = df_8t[df_8t['M'] == df_8t['N']]  # Square matrices
        
        # Plot each implementation
        impls = df_8t['impl'].unique()
        colors = plt.cm.tab10(range(len(impls)))
        
        for impl, color in zip(impls, colors):
            impl_data = df_8t[df_8t['impl'] == impl].sort_values('N')
            if not impl_data.empty:
                ax.plot(impl_data['N'], impl_data['gflops'], 
                       marker='o', linewidth=2, markersize=6, 
                       label=impl, color=color)
        
        ax.set_xlabel('Matrix Size (N×N×N)')
        ax.set_ylabel('Performance (GFLOP/s)')
        ax.set_title('GEMM Performance Verification Plot')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Verification plot saved: {output_file}")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        return False

def main():
    # Find the most recent CSV file
    csv_files = list(pathlib.Path("data/runs").glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in data/runs/")
        return False
    
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Using latest CSV: {latest_csv}")
    
    # Verify data
    df = verify_csv_data(latest_csv)
    if df is None:
        return False
    
    # Create verification plot
    output_file = "results/plots/verification_plot.png"
    pathlib.Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    success = create_quick_plot(df, output_file)
    
    if success:
        print("\n✅ Verification complete - plots should contain data!")
    else:
        print("\n❌ Verification failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)