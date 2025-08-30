#!/usr/bin/env pwsh
# Final comprehensive benchmark for Milestone 6

Write-Host "GEMM Final Benchmark - Milestone 6" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

# Set optimal environment
$env:OMP_NUM_THREADS = "8"
$env:OMP_PLACES = "cores"
$env:OMP_PROC_BIND = "close"

# Create directories
New-Item -ItemType Directory -Force -Path "data/runs" | Out-Null
New-Item -ItemType Directory -Force -Path "results/plots" | Out-Null

# Generate timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvFile = "data/runs/final_benchmark_$timestamp.csv"

Write-Host "Results will be saved to: $csvFile"

# Initialize CSV
"impl,M,N,K,threads,MB,NB,KB,time_ms,gflops,relerr,notes" | Out-File -FilePath $csvFile -Encoding UTF8

# Test configurations
$tests = @(
    @{impl="naive"; N=256; threads=1},
    @{impl="naive"; N=512; threads=1},
    @{impl="naive"; N=1024; threads=1},
    @{impl="naive"; N=256; threads=8},
    @{impl="naive"; N=512; threads=8},
    @{impl="blocked"; N=256; threads=8},
    @{impl="blocked"; N=512; threads=8},
    @{impl="blocked"; N=1024; threads=8},
    @{impl="blocked"; N=2048; threads=8},
    @{impl="blocked"; N=4096; threads=8},
    @{impl="packed"; N=256; threads=8},
    @{impl="packed"; N=512; threads=8},
    @{impl="packed"; N=1024; threads=8},
    @{impl="packed"; N=2048; threads=8},
    @{impl="packed"; N=4096; threads=8},
    @{impl="mk_avx2"; N=256; threads=8},
    @{impl="mk_avx2"; N=512; threads=8},
    @{impl="mk_avx2"; N=1024; threads=8},
    @{impl="mk_avx2"; N=2048; threads=8},
    @{impl="mk_avx2"; N=4096; threads=8}
)

$testCount = 0
foreach ($test in $tests) {
    $testCount++
    Write-Host "[$testCount/$($tests.Count)] Testing $($test.impl) N=$($test.N) threads=$($test.threads)" -ForegroundColor Cyan
    
    $env:OMP_NUM_THREADS = $test.threads.ToString()
    
    try {
        $result = & .\gemm_bench.exe --impl $test.impl --N $test.N --reps 2 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            # Parse and append result
            $resultLine = $result | Where-Object { $_ -match "^impl=" }
            if ($resultLine) {
                $resultLine | Out-File -FilePath $csvFile -Append -Encoding UTF8
                
                # Extract GFLOP/s for display
                if ($resultLine -match "gflops=([0-9.]+)") {
                    $gflops = [math]::Round([double]$matches[1], 2)
                    Write-Host "  ✅ $gflops GFLOP/s" -ForegroundColor Green
                }
            }
        } else {
            Write-Host "  ❌ Failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ❌ Exception: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Reset environment
$env:OMP_NUM_THREADS = "8"

Write-Host "`nGenerating plots..." -ForegroundColor Yellow

try {
    python scripts/plot.py $csvFile
    
    # Copy key plots to results directory
    if (Test-Path "plots/gemm_gflops_vs_N.png") {
        Copy-Item "plots/gemm_gflops_vs_N.png" "results/plots/" -Force
    }
    if (Test-Path "plots/roofline.png") {
        Copy-Item "plots/roofline.png" "results/plots/" -Force
    }
    
    Write-Host "Plots generated!" -ForegroundColor Green
} catch {
    Write-Host "Plot generation failed" -ForegroundColor Yellow
}

Write-Host "`nFinal benchmark complete!" -ForegroundColor Green
Write-Host "Results: $csvFile" -ForegroundColor Cyan
Write-Host "Plots: results/plots/" -ForegroundColor Cyan