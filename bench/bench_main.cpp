#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <stdexcept>

// For aligned memory allocation
#ifdef _WIN32
    #include <malloc.h>
#else
    #include <stdlib.h>
    #include <unistd.h>
#endif

// For OpenMP thread detection
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "gemm/gemm.hpp"

// Command line argument parser
struct Config {
    int M = 1024, N = 1024, K = 1024;
    int reps = 5;
    int threads = 1;
    int seed = 42;
    std::string csv_path;
    std::string impl = "naive";
    int MB = 256, NB = 256, KB = 256;
};

Config parse_args(int argc, char** argv) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--M" && i + 1 < argc) {
            config.M = std::stoi(argv[++i]);
        } else if (arg == "--N" && i + 1 < argc) {
            config.N = std::stoi(argv[++i]);
        } else if (arg == "--K" && i + 1 < argc) {
            config.K = std::stoi(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            config.reps = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            config.csv_path = argv[++i];
        } else if (arg == "--impl" && i + 1 < argc) {
            config.impl = argv[++i];
        } else if (arg == "--MB" && i + 1 < argc) {
            config.MB = std::stoi(argv[++i]);
        } else if (arg == "--NB" && i + 1 < argc) {
            config.NB = std::stoi(argv[++i]);
        } else if (arg == "--KB" && i + 1 < argc) {
            config.KB = std::stoi(argv[++i]);
        }
    }
    
    // Set M = N if only N was specified
    if (config.M == 1024 && config.N != 1024) {
        config.M = config.N;
    }
    // Set K = N if only N was specified
    if (config.K == 1024 && config.N != 1024) {
        config.K = config.N;
    }
    
    return config;
}

// Get actual OpenMP thread count
int get_thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Cross-platform aligned memory allocation
float* alloc_aligned(size_t elements) {
    size_t bytes = elements * sizeof(float);
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(bytes, 64);
    if (ptr == nullptr) {
        throw std::runtime_error("Failed to allocate aligned memory");
    }
#else
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        throw std::runtime_error("Failed to allocate aligned memory");
    }
#endif
    
    return static_cast<float*>(ptr);
}

// Cross-platform aligned memory deallocation
void free_aligned(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Fill matrix with deterministic uniform random values in [-1, 1]
void fill_matrix(float* matrix, int rows, int cols, int ld, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * ld + j] = dist(gen);
        }
    }
}

// Zero out matrix
void zero_matrix(float* matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * ld + j] = 0.0f;
        }
    }
}

// Copy matrix
void copy_matrix(const float* src, float* dst, int rows, int cols, int ld_src, int ld_dst) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i * ld_dst + j] = src[i * ld_src + j];
        }
    }
}

// Compute Frobenius norm
double frobenius_norm(const float* matrix, int rows, int cols, int ld) {
    double norm = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = matrix[i * ld + j];
            norm += static_cast<double>(val) * static_cast<double>(val);
        }
    }
    return std::sqrt(norm);
}

// Compute relative error between two matrices
double relative_error(const float* C, const float* C_ref, int rows, int cols, int ld) {
    // Compute ||C - C_ref||_F
    double diff_norm = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = C[i * ld + j] - C_ref[i * ld + j];
            diff_norm += static_cast<double>(diff) * static_cast<double>(diff);
        }
    }
    diff_norm = std::sqrt(diff_norm);
    
    // Compute ||C_ref||_F
    double ref_norm = frobenius_norm(C_ref, rows, cols, ld);
    
    // Return relative error
    return diff_norm / (ref_norm + 1e-30);
}

// Timing helper
class Timer {
public:
    void start() {
        start_time = std::chrono::steady_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1e6; // Convert to milliseconds
    }
    
private:
    std::chrono::steady_clock::time_point start_time;
};

// Write CSV header if file doesn't exist
void write_csv_header(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        std::ofstream csv(filename);
        csv << "impl,M,N,K,threads,MB,NB,KB,time_ms,gflops,relerr,notes\n";
    }
}

// Write CSV row
void write_csv_row(const std::string& filename, const Config& config, 
                   double time_ms, double gflops, double relerr) {
    std::ofstream csv(filename, std::ios::app);
    std::string notes = (config.impl == "blocked") ? "blocked+openmp" : 
                       (config.impl == "packed") ? "packed+openmp" : "baseline";
    csv << config.impl << "," << config.M << "," << config.N << "," << config.K << ","
        << config.threads << "," << config.MB << "," << config.NB << "," << config.KB << "," 
        << std::fixed << std::setprecision(3) << time_ms << ","
        << std::setprecision(2) << gflops << "," << std::scientific << std::setprecision(1) 
        << relerr << "," << notes << "\n";
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);
    
    // Update thread count to actual OpenMP threads
    config.threads = get_thread_count();
    
    std::cout << "GEMM Benchmark - Milestone 2 Blocked + OpenMP\n";
    std::cout << "OpenMP max threads: " << config.threads << "\n";
    std::cout << "Config: M=" << config.M << ", N=" << config.N << ", K=" << config.K 
              << ", reps=" << config.reps << ", impl=" << config.impl << ", threads=" << config.threads;
    if (config.impl == "blocked") {
        std::cout << ", MB=" << config.MB << ", NB=" << config.NB << ", KB=" << config.KB;
        // Calculate working set per thread
        double working_set_mb = (config.MB * config.KB + config.KB * config.NB + config.MB * config.NB) * 4.0 / (1024.0 * 1024.0);
        std::cout << ", working_set=" << std::fixed << std::setprecision(1) << working_set_mb << "MB";
    }
    std::cout << "\n\n";
    
    // Set up leading dimensions (row-major)
    int lda = config.K;  // A is M×K
    int ldb = config.N;  // B is K×N  
    int ldc = config.N;  // C is M×N
    
    // Allocate aligned matrices
    float* A = alloc_aligned(config.M * config.K);
    float* B = alloc_aligned(config.K * config.N);
    float* C = alloc_aligned(config.M * config.N);
    float* C_ref = alloc_aligned(config.M * config.N);
    
    // Initialize matrices with deterministic random values
    std::mt19937 gen(config.seed);
    fill_matrix(A, config.M, config.K, lda, gen);
    fill_matrix(B, config.K, config.N, ldb, gen);
    
    // Create block configuration
    gemm::Block block_sizes{config.MB, config.NB, config.KB};
    
    // Compute reference result for correctness checking (always use naive for reference)
    zero_matrix(C_ref, config.M, config.N, ldc);
    gemm::gemm_naive(config.M, config.N, config.K, A, B, C_ref, lda, ldb, ldc);
    
    // Warmup run (not timed)
    zero_matrix(C, config.M, config.N, ldc);
    gemm::run_gemm(config.impl, config.M, config.N, config.K, A, B, C, lda, ldb, ldc, block_sizes);
    
    std::cout << "Running " << config.reps << " timed iterations...\n";
    
    // Timed runs
    std::vector<double> times;
    Timer timer;
    
    for (int rep = 0; rep < config.reps; rep++) {
        // Reset C to zero for each run
        zero_matrix(C, config.M, config.N, ldc);
        
        // Time the GEMM operation
        timer.start();
        gemm::run_gemm(config.impl, config.M, config.N, config.K, A, B, C, lda, ldb, ldc, block_sizes);
        double time_ms = timer.stop_ms();
        times.push_back(time_ms);
        
        // Verify correctness
        double relerr = relative_error(C, C_ref, config.M, config.N, ldc);
        if (relerr > 1e-6) {
            std::cerr << "ERROR: Relative error " << relerr << " exceeds threshold 1e-6\n";
            return 1;
        }
        
        std::cout << "Rep " << rep + 1 << ": " << std::fixed << std::setprecision(2) 
                  << time_ms << " ms, relerr=" << std::scientific << relerr << "\n";
    }
    
    // Use best (minimum) time for final results
    double best_time_ms = *std::min_element(times.begin(), times.end());
    double best_time_s = best_time_ms / 1000.0;
    
    // Calculate GFLOP/s: 2*M*N*K floating point operations
    double gflops = (2.0 * config.M * config.N * config.K) / (best_time_s * 1e9);
    double final_relerr = relative_error(C, C_ref, config.M, config.N, ldc);
    
    // Print summary
    std::cout << "\nBest result:\n";
    std::string notes = (config.impl == "blocked") ? "blocked+openmp" : 
                       (config.impl == "packed") ? "packed+openmp" :
                       (config.impl == "mk_avx2") ? "mk_avx2+openmp" :
                       (config.impl == "openblas") ? "openblas" : "baseline";
    std::cout << "impl=" << config.impl << ",M=" << config.M << ",N=" << config.N 
              << ",K=" << config.K << ",threads=" << config.threads 
              << ",MB=" << config.MB << ",NB=" << config.NB << ",KB=" << config.KB 
              << ",time_ms=" << std::fixed << std::setprecision(3) << best_time_ms
              << ",gflops=" << std::setprecision(2) << gflops 
              << ",relerr=" << std::scientific << std::setprecision(1) << final_relerr 
              << ",notes=" << notes << "\n";
    
    // Write results to CSV if output file is specified
    if (!config.csv_path.empty()) {
        std::ofstream csv(config.csv_path, std::ios::app);
        if (!csv.is_open()) {
            std::cerr << "Error: Could not open output file " << config.csv_path << std::endl;
        } else {
            // Write header if file is empty
            if (csv.tellp() == 0) {
                csv << "impl,M,N,K,threads,MB,NB,KB,time_ms,gflops,relerr,notes\n";
            }
            
            // Write data row
            std::string notes = (config.impl == "blocked") ? "blocked+openmp" : 
                               (config.impl == "packed") ? "packed+openmp" :
                               (config.impl == "mk_avx2") ? "mk_avx2+openmp" :
                               (config.impl == "openblas") ? "openblas" : "baseline";
            csv << config.impl << "," << config.M << "," << config.N << "," << config.K << ","
                << config.threads << "," << config.MB << "," << config.NB << "," << config.KB << "," 
                << std::fixed << std::setprecision(3) << best_time_ms << ","
                << std::setprecision(2) << gflops << "," << std::scientific << std::setprecision(1) 
                << final_relerr << "," << notes;
            
            std::cout << "Results written to " << config.csv_path << std::endl;
        }
    }
    
    // Cleanup
    free_aligned(A);
    free_aligned(B);
    free_aligned(C);
    free_aligned(C_ref);
    
    return 0;
}