#!/usr/bin/env python3
"""
Comprehensive GEMM benchmark runner.
Tests all implementations across multiple sizes and generates comparison data.
"""

import subprocess
import os
import sys
import csv
import time
from datetime import datetime

def run_single_benchmark(impl, N, MB=256, NB=256, KB=256, threads=8, reps=3):
    """Run a single benchmark configuration"""
    try:
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
        
        print(f"Running: {impl} N={N} threads={threads}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return None
            
        # Parse results from output
        lines = result.stdout.split('\n')
        for line in lines:
            if line.startswith('impl='):
                # Parse the result line
                parts = line.split(',')
                data = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        data[key] = value
                
                # Convert numeric fields
                numeric_fields = ['M', 'N', 'K', 'threads', 'MB', 'NB', 'KB', 'time_ms', 'gflops', 'relerr']
                for field in numeric_fields:
                    if field in data:
                        try:
                            data[field] = float(data[field])
                        except ValueError:
                            pass
                
                return data
        
        return None
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout for {impl} N={N}")
        return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None

def main():
    """Run comprehensive benchmarks"""
    print("Comprehensive GEMM Benchmark Suite")
    print("=" * 40)
    
    # Check if executable exists
    if not os.path.exists('./gemm_bench.exe'):
        print("Error: gemm_bench.exe not found. Please build first.")
        sys.exit(1)
    
    # Test configurations
    implementations = ['naive', 'blocked', 'packed', 'mk_avx2']
    sizes = [256, 512, 1024, 1536, 2048, 3072, 4096]
    thread_counts = [1, 8]  # Test single-threaded and multi-threaded
    
    # Results storage
    results = []
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_benchmark_{timestamp}.csv"
    
    total_tests = len(implementations) * len(sizes) * len(thread_counts)
    test_count = 0
    
    print(f"Running {total_tests} benchmark configurations...")
    print(f"Results will be saved to: {output_file}")
    
    for impl in implementations:
        for N in sizes:
            for threads in thread_counts:
                test_count += 1
                print(f"\n[{test_count}/{total_tests}] Testing {impl} N={N} threads={threads}")
                
                # Skip naive for large sizes with multiple threads (too slow)
                if impl == 'naive' and N >= 2048 and threads > 1:
                    print("  Skipping (too slow)")
                    continue
                
                # Skip naive for very large sizes even single-threaded
                if impl == 'naive' and N >= 4096:
                    print("  Skipping (too slow)")
                    continue
                
                # Run benchmark
                result = run_single_benchmark(impl, N, threads=threads, reps=2)
                
                if result:
                    results.append(result)
                    print(f"  Result: {result['gflops']:.2f} GFLOP/s, {result['time_ms']:.2f} ms")
                else:
                    print("  Failed")
    
    # Save results to CSV
    if results:
        print(f"\nSaving {len(results)} results to {output_file}")
        
        fieldnames = ['impl', 'M', 'N', 'K', 'threads', 'MB', 'NB', 'KB', 
                     'time_ms', 'gflops', 'relerr', 'notes']
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Ensure all required fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"Results saved to {output_file}")
        
        # Print summary
        print(f"\n=== Performance Summary ===")
        for impl in implementations:
            impl_results = [r for r in results if r['impl'] == impl and r['threads'] == 8]
            if impl_results:
                best = max(impl_results, key=lambda x: x['gflops'])
                print(f"{impl:>10}: {best['gflops']:>8.2f} GFLOP/s (N={int(best['N'])})")
        
        # Generate plots if possible
        try:
            print(f"\nGenerating plots...")
            subprocess.run([sys.executable, 'scripts/plot.py', output_file], check=True)
            print("Plots generated successfully")
        except Exception as e:
            print(f"Plot generation failed: {e}")
    
    else:
        print("No valid results obtained")

if __name__ == '__main__':
    main()