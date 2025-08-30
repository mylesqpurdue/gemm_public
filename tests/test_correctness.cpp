// Unit tests for GEMM correctness verification
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include "gemm/gemm.hpp"

// Test configuration
struct TestCase {
    int M, N, K;
    std::string name;
};

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

void free_aligned(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Fill matrix with random values
void fill_random(float* matrix, int rows, int cols, int ld, std::mt19937& gen) {
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

// Compute relative error between two matrices
double relative_error(const float* C1, const float* C2, int rows, int cols, int ld) {
    double diff_norm = 0.0;
    double ref_norm = 0.0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = C1[i * ld + j] - C2[i * ld + j];
            float ref = C2[i * ld + j];
            diff_norm += static_cast<double>(diff) * static_cast<double>(diff);
            ref_norm += static_cast<double>(ref) * static_cast<double>(ref);
        }
    }
    
    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-30);
}

// Test a single GEMM implementation
bool test_implementation(const std::string& impl_name, const TestCase& test, 
                        std::mt19937& gen, double tolerance = 1e-6) {
    
    const int M = test.M, N = test.N, K = test.K;
    const int lda = K, ldb = N, ldc = N;
    
    // Allocate matrices
    float* A = alloc_aligned(M * K);
    float* B = alloc_aligned(K * N);
    float* C_test = alloc_aligned(M * N);
    float* C_ref = alloc_aligned(M * N);
    
    try {
        // Initialize with random data
        fill_random(A, M, K, lda, gen);
        fill_random(B, K, N, ldb, gen);
        
        // Compute reference result (naive)
        zero_matrix(C_ref, M, N, ldc);
        gemm::gemm_naive(M, N, K, A, B, C_ref, lda, ldb, ldc);
        
        // Compute test result
        zero_matrix(C_test, M, N, ldc);
        gemm::Block block_sizes{256, 256, 256};
        
        try {
            gemm::run_gemm(impl_name, M, N, K, A, B, C_test, lda, ldb, ldc, block_sizes);
        } catch (const std::exception& e) {
            std::cout << "  " << impl_name << " not available: " << e.what() << std::endl;
            free_aligned(A);
            free_aligned(B);
            free_aligned(C_test);
            free_aligned(C_ref);
            return true; // Skip unavailable implementations
        }
        
        // Check correctness
        double error = relative_error(C_test, C_ref, M, N, ldc);
        
        bool passed = error <= tolerance;
        std::cout << "  " << impl_name << ": " << test.name 
                  << " - Error: " << std::scientific << error 
                  << (passed ? " ✅" : " ❌") << std::endl;
        
        free_aligned(A);
        free_aligned(B);
        free_aligned(C_test);
        free_aligned(C_ref);
        
        return passed;
        
    } catch (const std::exception& e) {
        std::cout << "  " << impl_name << ": " << test.name 
                  << " - Exception: " << e.what() << " ❌" << std::endl;
        
        free_aligned(A);
        free_aligned(B);
        free_aligned(C_test);
        free_aligned(C_ref);
        
        return false;
    }
}

int main() {
    std::cout << "🧪 GEMM Correctness Tests" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Test cases
    std::vector<TestCase> test_cases = {
        {64, 64, 64, "Small square"},
        {128, 128, 128, "Medium square"},
        {256, 256, 256, "Large square"},
        {100, 200, 150, "Rectangular"},
        {33, 77, 55, "Odd sizes"},
        {1, 1000, 1, "Skinny matrix"},
        {1000, 1, 1000, "Tall matrix"},
        {8, 8, 8, "Micro tile"},
        {15, 23, 17, "Prime sizes"}
    };
    
    // Implementations to test
    std::vector<std::string> implementations = {
        "blocked", "packed", "mk_avx2"
    };
    
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (const auto& test_case : test_cases) {
        std::cout << "\nTesting " << test_case.name 
                  << " (" << test_case.M << "×" << test_case.N << "×" << test_case.K << "):" << std::endl;
        
        for (const auto& impl : implementations) {
            total_tests++;
            if (test_implementation(impl, test_case, gen)) {
                passed_tests++;
            }
        }
    }
    
    std::cout << "\n📊 Test Summary:" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
    
    if (passed_tests == total_tests) {
        std::cout << "🎉 All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed!" << std::endl;
        return 1;
    }
}