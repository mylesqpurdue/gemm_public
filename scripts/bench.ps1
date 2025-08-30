#!/usr/bin/env pwsh
# One-button GEMM benchmark script for Windows
# Runs comprehensive benchmarks and generates all plots

param(
    [switch]$SkipBuild,
    [switch]$QuickTest,
    [string]$OutputDir = "data/runs"
)

Write-Host "GEMM Benchmark Suite - One-Button Reproduction" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

# Set optimal OpenMP environment
$env:OMP_NUM_THREADS = "8"
$env:OMP_PLACES = "cores" 
$env:OMP_PROC_BIND = "close"

Write-Host "Environment: OMP_NUM_THREADS=$env:OMP_NUM_THREADS, OMP_PLACES=$env:OMP_PLACES, OMP_PROC_BIND=$env:OMP_PROC_BIND"

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path "results/plots" | Out-Null

# Build if not skipped
if (-not $SkipBuild) {
    Write-Host "`nBuilding GEMM benchmark..." -ForegroundColor Yellow
    
    cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cl /O2 /fp:fast /arch:AVX2 /openmp:experimental /DNDEBUG /EHsc /I include bench\bench_main.cpp cpu\gemm_naive.cpp cpu\gemm_blocked.cpp cpu\gemm_packed.cpp cpu\gemm_mk_avx2.cpp cpu\microkernels\mk_avx2.cpp cpu\gemm_dispatcher.cpp baselines\openblas.cpp /Fe:gemm_bench.exe'
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Build successful!" -ForegroundColor Green
}

# Check if executable exists
if (-not (Test-Path "gemm_bench.exe")) {
    Write-Host "gemm_bench.exe not found. Run without -SkipBuild or build manually." -ForegroundColor Red
    exit 1
}

# Generate timestamp for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvFile = "$OutputDir/gemm_benchmark_$timestamp.csv"

Write-Host "`nRunning comprehensive benchmarks..." -ForegroundColor Yellow
Write-Host "Results will be saved to: $csvFile"

# Define test configurations
if ($QuickTest) {
    $sizes = @(256, 512, 1024, 2048)
    $reps = 2
    Write-Host "Quick test mode: sizes $($sizes -join ', '), $reps reps each"
} else {
    $sizes = @(256, 512, 1024, 1536, 2048, 3072, 4096)
    $reps = 3
    Write-Host "Full benchmark: sizes $($sizes -join ', '), $reps reps each"
}

$implementations = @(
    @{name="naive"; threads=1; note="Single-threaded baseline"},
    @{name="naive"; threads=8; note="Multi-threaded baseline"},
    @{name="blocked"; threads=8; note="Cache blocking + OpenMP"},
    @{name="packed"; threads=8; note="Panel packing + OpenMP"},
    @{name="mk_avx2"; threads=8; note="AVX2 micro-kernels + OpenMP"}
)

$totalTests = $implementations.Count * $sizes.Count
$testCount = 0

# Initialize CSV file
"impl,M,N,K,threads,MB,NB,KB,time_ms,gflops,relerr,notes" | Out-File -FilePath $csvFile -Encoding UTF8

foreach ($impl in $implementations) {
    foreach ($N in $sizes) {
        $testCount++
        Write-Host "`n[$testCount/$totalTests] Testing $($impl.name) N=$N threads=$($impl.threads)" -ForegroundColor Cyan
        
        # Skip slow tests
        if ($impl.name -eq "naive" -and $N -ge 2048 -and $impl.threads -gt 1) {
            Write-Host "  Skipping (too slow)" -ForegroundColor Yellow
            continue
        }
        if ($impl.name -eq "naive" -and $N -ge 4096) {
            Write-Host "  Skipping (too slow)" -ForegroundColor Yellow  
            continue
        }
        
        # Set thread count for this test
        $env:OMP_NUM_THREADS = $impl.threads.ToString()
        
        try {
            # Run benchmark
            $result = & .\gemm_bench.exe --impl $impl.name --N $N --reps $reps 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                # Parse result line and append to CSV
                $resultLine = $result | Where-Object { $_ -match "^impl=" }
                if ($resultLine) {
                    $resultLine | Out-File -FilePath $csvFile -Append -Encoding UTF8
                    
                    # Extract GFLOP/s for display
                    if ($resultLine -match "gflops=([0-9.]+)") {
                        $gflops = [math]::Round([double]$matches[1], 2)
                        Write-Host "  Result: $gflops GFLOP/s" -ForegroundColor Green
                    }
                } else {
                    Write-Host "  Warning: Could not parse result" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  Failed" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "  Exception: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Reset thread count
$env:OMP_NUM_THREADS = "8"

Write-Host "`nGenerating plots..." -ForegroundColor Yellow

try {
    # Generate performance plots
    python scripts/plot.py $csvFile
    
    # Copy key plots to results directory
    if (Test-Path "plots/gemm_gflops_vs_N.png") {
        Copy-Item "plots/gemm_gflops_vs_N.png" "results/plots/" -Force
    }
    if (Test-Path "plots/roofline.png") {
        Copy-Item "plots/roofline.png" "results/plots/" -Force
    }
    
    Write-Host "Plots generated successfully!" -ForegroundColor Green
    Write-Host "Check results/plots/ for output files" -ForegroundColor Cyan
}
catch {
    Write-Host "Plot generation failed: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "You can generate plots manually with: python scripts/plot.py $csvFile" -ForegroundColor Yellow
}

# Summary
Write-Host "`nBenchmark complete!" -ForegroundColor Green
Write-Host "Results saved to: $csvFile" -ForegroundColor Cyan
Write-Host "Plots available in: results/plots/" -ForegroundColor Cyan

# Show quick performance summary
if (Test-Path $csvFile) {
    Write-Host "`nPerformance Summary:" -ForegroundColor Yellow
    
    try {
        $data = Import-Csv $csvFile
        
        # Clean the data format (remove column prefixes)
        foreach ($row in $data) {
            foreach ($prop in $row.PSObject.Properties) {
                if ($prop.Value -match "^$($prop.Name)=(.*)") {
                    $prop.Value = $matches[1]
                }
            }
        }
        
        $summary = $data | Where-Object { $_.threads -eq "8" } | 
                         Group-Object impl | 
                         ForEach-Object { 
                             $best = $_.Group | Sort-Object { [double]$_.gflops } -Descending | Select-Object -First 1
                             [PSCustomObject]@{
                                 Implementation = $_.Name
                                 "Peak GFLOP/s" = [math]::Round([double]$best.gflops, 2)
                                 "Best Size" = "N=$($best.N)"
                             }
                         } | Sort-Object "Peak GFLOP/s" -Descending
        
        $summary | Format-Table -AutoSize
    }
    catch {
        Write-Host "Could not generate summary table" -ForegroundColor Yellow
    }
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  - View plots in results/plots/" 
Write-Host "  - Check detailed results in $csvFile"
Write-Host "  - Run with -QuickTest for faster testing"
Write-Host "  - Run with -SkipBuild to skip compilation"