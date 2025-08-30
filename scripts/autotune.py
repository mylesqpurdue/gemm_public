#!/usr/bin/env python3
"""
Auto-tune GEMM tile sizes for different thread counts.
Finds optimal MB, NB, KB parameters for 1, 2, 4, 8 threads.
"""

import subprocess
import json
import os
import sys
from itertools import product
import statistics

def run_benchmark(impl, N, MB, NB, KB, threads, reps=3):
    """Run benchmark and return median GFLOP/s"""
    try:
        # Set environment variables for thread count
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        env['OMP_PLACES'] = 'cores'
        env['OMP_PROC_BIND'] = 'close'
        
        cmd = [
            './gemm_bench.exe',
            '--impl', impl,
            '--N', str(N),
            '--MB', str(MB),
            '--NB', str(NB), 
            '--KB', str(KB),
            '--reps', str(reps)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"Error running benchmark: {result.stderr}")
            return 0.0
            
        # Parse GFLOP/s from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'gflops=' in line:
                # Extract gflops value
                parts = line.split('gflops=')
                if len(parts) > 1:
                    gflops_str = parts[1].split(',')[0]
                    return float(gflops_str)
        
        return 0.0
        
    except Exception as e:
        print(f"Exception running benchmark: {e}")
        return 0.0

def autotune_thread_count(threads, impl='mk_avx2'):
    """Auto-tune tile sizes for a specific thread count"""
    print(f"\n=== Auto-tuning for {threads} threads ===")
    
    # Parameter ranges to search
    MB_values = [128, 192, 256, 320]
    NB_values = [128, 192, 256, 320] 
    KB_values = [96, 128, 160, 192, 256]
    
    # Test sizes
    test_sizes = [2048, 4096]
    
    best_config = None
    best_score = 0.0
    
    total_configs = len(MB_values) * len(NB_values) * len(KB_values)
    config_count = 0
    
    for MB, NB, KB in product(MB_values, NB_values, KB_values):
        config_count += 1
        print(f"Testing config {config_count}/{total_configs}: MB={MB}, NB={NB}, KB={KB}")
        
        # Test on multiple sizes and take average
        scores = []
        for N in test_sizes:
            gflops = run_benchmark(impl, N, MB, NB, KB, threads, reps=2)
            if gflops > 0:
                scores.append(gflops)
                print(f"  N={N}: {gflops:.2f} GFLOP/s")
        
        if scores:
            avg_score = statistics.mean(scores)
            print(f"  Average: {avg_score:.2f} GFLOP/s")
            
            if avg_score > best_score:
                best_score = avg_score
                best_config = {'MB': MB, 'NB': NB, 'KB': KB}
                print(f"  *** NEW BEST: {avg_score:.2f} GFLOP/s ***")
        else:
            print("  Failed to get valid results")
    
    return best_config, best_score

def main():
    """Main auto-tuning function"""
    print("GEMM Auto-Tuner - Finding optimal tile sizes")
    print("=" * 50)
    
    # Check if benchmark executable exists
    if not os.path.exists('./gemm_bench.exe'):
        print("Error: gemm_bench.exe not found. Please build first.")
        sys.exit(1)
    
    # Thread counts to tune
    thread_counts = [1, 2, 4, 8]
    
    results = {}
    
    for threads in thread_counts:
        try:
            config, score = autotune_thread_count(threads)
            if config:
                results[f't{threads}'] = config
                print(f"\nBest config for {threads} threads: {config} ({score:.2f} GFLOP/s)")
            else:
                print(f"\nFailed to find good config for {threads} threads")
        except KeyboardInterrupt:
            print("\nAuto-tuning interrupted by user")
            break
        except Exception as e:
            print(f"\nError tuning {threads} threads: {e}")
    
    # Save results
    if results:
        os.makedirs('data', exist_ok=True)
        with open('data/best_tiles.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Auto-tuning Complete ===")
        print(f"Results saved to data/best_tiles.json")
        print("\nBest configurations:")
        for threads, config in results.items():
            print(f"  {threads}: MB={config['MB']}, NB={config['NB']}, KB={config['KB']}")
    else:
        print("\nNo valid configurations found")

if __name__ == '__main__':
    main()