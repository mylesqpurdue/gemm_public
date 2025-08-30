# GEMM Benchmark Build Instructions

## Prerequisites

You need one of the following:
- **CMake + C++ compiler** (recommended)
- **Visual Studio** (Windows)
- **GCC/Clang** (Linux/macOS)

## Method 1: Using CMake (Recommended)

```bash
# Create and enter build directory
mkdir -p build
cd build

# Configure the build (Release mode for performance)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the project
cmake --build . -j

# Run the benchmark
./gemm_bench --N 1024 --reps 5
```

## Method 2: Manual Compilation (if CMake unavailable)

### Windows (Visual Studio Command Prompt)
```cmd
cl /O2 /DNDEBUG /EHsc /I include bench\bench_main.cpp cpu\gemm_naive.cpp /Fe:gemm_bench.exe
```

### Linux/macOS (GCC/Clang)
```bash
g++ -O3 -march=native -DNDEBUG -std=c++17 -I include bench/bench_main.cpp cpu/gemm_naive.cpp -o gemm_bench
```

## Method 3: Using the Build Script (Linux/macOS)

```bash
chmod +x build.sh
./build.sh
```

## Running the Benchmark

After building, you can run various tests:

```bash
# Basic run with default 1024x1024 matrices
./gemm_bench

# Custom dimensions
./gemm_bench --M 512 --N 512 --K 512 --reps 3

# With CSV output
./gemm_bench --N 1024 --reps 5 --csv results.csv

# Test edge cases
./gemm_bench --M 123 --N 77 --K 191 --reps 1

# Different seed for different random matrices
./gemm_bench --seed 456 --reps 3
```

## Expected Output

```
GEMM Benchmark - Milestone 1 Baseline
Config: M=1024, N=1024, K=1024, reps=5, impl=naive

Running 5 timed iterations...
Rep 1: 2847.32 ms, relerr=0.0e+00
Rep 2: 2834.56 ms, relerr=0.0e+00
Rep 3: 2841.78 ms, relerr=0.0e+00
Rep 4: 2839.12 ms, relerr=0.0e+00
Rep 5: 2836.45 ms, relerr=0.0e+00

Best result:
impl=naive,M=1024,N=1024,K=1024,threads=1,MB=0,NB=0,KB=0,time_ms=2834.560,gflops=0.76,relerr=0.0e+00,notes=baseline
```

## Troubleshooting

### Build Issues
- **CMake not found**: Install CMake from https://cmake.org/
- **Compiler not found**: Install Visual Studio (Windows) or build-essential (Linux)
- **OpenMP warnings**: OpenMP is optional, the benchmark will work without it

### Runtime Issues
- **Slow performance**: Make sure you're using Release build (`-DCMAKE_BUILD_TYPE=Release`)
- **Memory errors**: Check available RAM for large matrix sizes
- **Correctness failures**: This indicates a serious bug - please report it

## Performance Notes

The naïve implementation is intentionally simple and will have modest performance:
- **Expected range**: 0.1 - 10 GFLOP/s depending on hardware
- **Scaling**: Performance should increase with matrix size up to memory limits
- **Consistency**: Multiple runs should give similar results (±10%)

This baseline provides a correctness oracle and timing framework for developing optimized GEMM kernels.