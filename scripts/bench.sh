#!/bin/bash
# One-button GEMM benchmark script for Linux/macOS
# Runs comprehensive benchmarks and generates all plots

set -e

SKIP_BUILD=false
QUICK_TEST=false
OUTPUT_DIR="data/runs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--skip-build] [--quick-test] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "🚀 GEMM Benchmark Suite - One-Button Reproduction"
echo "================================================="

# Set optimal OpenMP environment
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Environment: OMP_NUM_THREADS=$OMP_NUM_THREADS, OMP_PLACES=$OMP_PLACES, OMP_PROC_BIND=$OMP_PROC_BIND"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "results/plots"

# Build if not skipped
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "🔨 Building GEMM benchmark..."
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd ..
    
    echo "✅ Build successful!"
fi

# Check if executable exists
EXECUTABLE="./build/gemm_bench"
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ $EXECUTABLE not found. Run without --skip-build or build manually."
    exit 1
fi

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="$OUTPUT_DIR/gemm_benchmark_$TIMESTAMP.csv"

echo ""
echo "📊 Running comprehensive benchmarks..."
echo "Results will be saved to: $CSV_FILE"

# Define test configurations
if [ "$QUICK_TEST" = true ]; then
    SIZES=(256 512 1024 2048)
    REPS=2
    echo "🏃 Quick test mode: sizes ${SIZES[*]}, $REPS reps each"
else
    SIZES=(256 512 1024 1536 2048 3072 4096)
    REPS=3
    echo "🔬 Full benchmark: sizes ${SIZES[*]}, $REPS reps each"
fi

# Implementation configurations
declare -a IMPLEMENTATIONS=(
    "naive:1:Single-threaded baseline"
    "naive:8:Multi-threaded baseline"
    "blocked:8:Cache blocking + OpenMP"
    "packed:8:Panel packing + OpenMP"
    "mk_avx2:8:AVX2 micro-kernels + OpenMP"
    "openblas:8:OpenBLAS baseline"
)

TOTAL_TESTS=$((${#IMPLEMENTATIONS[@]} * ${#SIZES[@]}))
TEST_COUNT=0

# Initialize CSV file
echo "impl,M,N,K,threads,MB,NB,KB,time_ms,gflops,relerr,notes" > "$CSV_FILE"

for impl_config in "${IMPLEMENTATIONS[@]}"; do
    IFS=':' read -r impl_name threads note <<< "$impl_config"
    
    for N in "${SIZES[@]}"; do
        ((TEST_COUNT++))
        echo ""
        echo "[$TEST_COUNT/$TOTAL_TESTS] Testing $impl_name N=$N threads=$threads"
        
        # Skip slow tests
        if [[ "$impl_name" == "naive" && $N -ge 2048 && $threads -gt 1 ]]; then
            echo "  Skipping (too slow)"
            continue
        fi
        if [[ "$impl_name" == "naive" && $N -ge 4096 ]]; then
            echo "  Skipping (too slow)"
            continue
        fi
        
        # Set thread count for this test
        export OMP_NUM_THREADS=$threads
        
        # Run benchmark
        if $EXECUTABLE --impl "$impl_name" --N "$N" --reps "$REPS" --csv "$CSV_FILE" 2>/dev/null; then
            # Extract GFLOP/s from the last line added to CSV
            GFLOPS=$(tail -n 1 "$CSV_FILE" | cut -d',' -f10)
            echo "  ✅ $GFLOPS GFLOP/s"
        else
            echo "  ❌ Failed or not available"
        fi
    done
done

# Reset thread count
export OMP_NUM_THREADS=8

echo ""
echo "📈 Generating plots..."

# Generate performance plots
if python3 scripts/plot.py "$CSV_FILE" 2>/dev/null || python scripts/plot.py "$CSV_FILE" 2>/dev/null; then
    echo "✅ Performance plots generated!"
else
    echo "⚠️  Performance plot generation failed"
fi

# Generate roofline analysis
if python3 scripts/roofline.py 2>/dev/null || python scripts/roofline.py 2>/dev/null; then
    echo "✅ Roofline analysis generated!"
else
    echo "⚠️  Roofline analysis failed"
fi

echo ""
echo "🎉 Benchmark complete!"
echo "📊 Results saved to: $CSV_FILE"
echo "📈 Plots available in: results/plots/"

# Show quick performance summary
if [ -f "$CSV_FILE" ]; then
    echo ""
    echo "📋 Performance Summary:"
    echo "Implementation    Peak GFLOP/s    Best Size"
    echo "============================================"
    
    # Extract best performance for each 8-thread implementation
    awk -F',' '
    NR>1 && $5==8 {
        if ($10 > best[$1]) {
            best[$1] = $10
            size[$1] = $3
        }
    }
    END {
        for (impl in best) {
            printf "%-15s %10.2f      N=%s\n", impl, best[impl], size[impl]
        }
    }' "$CSV_FILE" | sort -k2 -nr
fi

echo ""
echo "🔗 Next steps:"
echo "  • View plots in results/plots/"
echo "  • Check detailed results in $CSV_FILE"
echo "  • Run with --quick-test for faster testing"
echo "  • Run with --skip-build to skip compilation"