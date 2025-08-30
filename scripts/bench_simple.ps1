# Simple one-button GEMM benchmark script for Windows
param([switch]$QuickTest)

Write-Host "🚀 GEMM Benchmark Suite" -ForegroundColor Green

# Set environment
$env:OMP_NUM_THREADS = "8"
$env:OMP_PLACES = "cores"
$env:OMP_PROC_BIND = "close"

# Create directories
New-Item -ItemType Directory -Force -Path "data/runs" | Out-Null
New-Item -ItemType Directory -Force -Path "results/plots" | Out-Null

# Build
Write-Host "🔨 Building..." -ForegroundColor Yellow
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cl /O2 /fp:fast /arch:AVX2 /openmp:experimental /DNDEBUG /EHsc /I include bench\bench_main.cpp cpu\gemm_naive.cpp cpu\gemm_blocked.cpp cpu\gemm_packed.cpp cpu\gemm_mk_avx2.cpp cpu\microkernels\mk_avx2.cpp cpu\gemm_dispatcher.cpp baselines\openblas.cpp /Fe:gemm_bench.exe'

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

# Test configurations
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvFile = "data/runs/benchmark_$timestamp.csv"

if ($QuickTest) {
    $sizes = @(256, 512, 1024)
    $reps = 2
} else {
    $sizes = @(256, 512, 1024, 2048, 4096)
    $reps = 3
}

Write-Host "📊 Running benchmarks..." -ForegroundColor Yellow

# Initialize CSV
"impl,M,N,K,threads,MB,NB,KB,time_ms,gflops,relerr,notes" | Out-File -FilePath $csvFile -Encoding UTF8

# Run tests
$implementations = @("blocked", "packed", "mk_avx2")

foreach ($impl in $implementations) {
    foreach ($N in $sizes) {
        Write-Host "Testing $impl N=$N" -ForegroundColor Cyan
        
        try {
            & .\gemm_bench.exe --impl $impl --N $N --reps $reps --csv $csvFile | Out-Null
            Write-Host "  ✅ Completed" -ForegroundColor Green
        }
        catch {
            Write-Host "  ❌ Failed" -ForegroundColor Red
        }
    }
}

# Generate plots
Write-Host "📈 Generating plots..." -ForegroundColor Yellow
try {
    python scripts/plot.py $csvFile
    Write-Host "✅ Plots generated!" -ForegroundColor Green
}
catch {
    Write-Host "⚠️ Plot generation failed" -ForegroundColor Yellow
}

Write-Host "Complete! Check results/plots/" -ForegroundColor Green