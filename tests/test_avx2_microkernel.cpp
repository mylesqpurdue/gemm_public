#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include "gemm/microkernel.hpp"

// Function to print a matrix
void print_matrix(const char* name, const float* A, int m, int n, int lda) {
    std::cout << name << " (" << m << "x" << n << "):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << A[i * lda + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Function to initialize a matrix with sequential values
void init_matrix(float* A, int m, int n, int lda, float start = 1.0f) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * lda + j] = start + i * n + j;
        }
    }
}

// Function to initialize an identity matrix
void init_identity(float* A, int n, int lda) {
    std::memset(A, 0, n * lda * sizeof(float));
    for (int i = 0; i < n; ++i) {
        A[i * lda + i] = 1.0f;
    }
}

// Function to check if two matrices are equal within a tolerance
bool check_result(const float* C, const float* C_ref, int m, int n, int ldc, float tol = 1e-5f) {
    bool pass = true;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
            if (diff > tol) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " 
                          << C[i * ldc + j] << " != " << C_ref[i * ldc + j] 
                          << " (diff: " << diff << ")" << std::endl;
                pass = false;
            }
        }
    }
    return pass;
}

int main() {
    const int M = 8, N = 8, K = 8;
    float A[M * K], B[K * N], C[M * N], C_ref[M * N];
    
    // Test case 1: Identity matrix multiplication
    std::cout << "Test 1: Identity matrix multiplication\n";
    init_identity(A, M, K);
    init_identity(B, K, N);
    std::memset(C, 0, M * N * sizeof(float));
    std::memset(C_ref, 0, M * N * sizeof(float));
    
    // Reference implementation (naive)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C_ref[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    
    // AVX2 microkernel
    gemm::mk8x8_avx2(K, A, B, C, N);
    
    // Check results
    bool pass1 = check_result(C, C_ref, M, N, N);
    std::cout << "Test 1 " << (pass1 ? "PASSED" : "FAILED") << "\n\n";
    
    // Test case 2: Sequential values
    std::cout << "Test 2: Sequential values\n";
    init_matrix(A, M, K, K, 1.0f);
    init_matrix(B, K, N, N, 1.0f);
    std::memset(C, 0, M * N * sizeof(float));
    std::memset(C_ref, 0, M * N * sizeof(float));
    
    // Reference implementation (naive)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C_ref[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    
    // AVX2 microkernel
    gemm::mk8x8_avx2(K, A, B, C, N);
    
    // Check results
    bool pass2 = check_result(C, C_ref, M, N, N);
    std::cout << "Test 2 " << (pass2 ? "PASSED" : "FAILED") << "\n";
    
    return (pass1 && pass2) ? 0 : 1;
}
