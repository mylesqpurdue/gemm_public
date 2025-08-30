// Packed GEMM - BLIS-style panel packing for optimal memory access
#include "gemm/gemm.hpp"
#include <algorithm>
#include <immintrin.h>
#include <cstring>
#include <stdexcept>

// For aligned memory allocation
#ifdef _WIN32
    #include <malloc.h>
#else
    #include <stdlib.h>
#endif

// Cross-platform aligned memory allocation for panels
static float* alloc_aligned_panel(size_t elements) {
    size_t bytes = elements * sizeof(float);
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(bytes, 64);
    if (ptr == nullptr) {
        throw std::runtime_error("Failed to allocate aligned panel memory");
    }
#else
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        throw std::runtime_error("Failed to allocate aligned panel memory");
    }
#endif
    
    return static_cast<float*>(ptr);
}

static void free_aligned_panel(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Pack A panel: (MB x KB) -> contiguous row-major
static void pack_A_panel(int MB, int KB, const float* A, int lda, float* A_packed) {
    for (int i = 0; i < MB; ++i) {
        for (int k = 0; k < KB; ++k) {
            A_packed[i * KB + k] = A[i * lda + k];
        }
    }
}

// Pack B panel: (KB x NB) -> contiguous row-major  
static void pack_B_panel(int KB, int NB, const float* B, int ldb, float* B_packed) {
    for (int k = 0; k < KB; ++k) {
        for (int j = 0; j < NB; ++j) {
            B_packed[k * NB + j] = B[k * ldb + j];
        }
    }
}

// Micro-kernel: multiply packed A panel by packed B panel into C tile
static void micro_kernel_packed(int MB, int NB, int KB, 
                               const float* A_packed, const float* B_packed, 
                               float* C, int ldc) {
    for (int i = 0; i < MB; ++i) {
        for (int k = 0; k < KB; ++k) {
            const float a = A_packed[i * KB + k];
            const float* b_row = &B_packed[k * NB];
            float* c_row = &C[i * ldc];
            
            // AVX2 vectorized inner loop (8 floats at a time)
            int j = 0;
            const __m256 a8 = _mm256_set1_ps(a);
            for (; j <= NB - 8; j += 8) {
                __m256 b8 = _mm256_loadu_ps(&b_row[j]);
                __m256 c8 = _mm256_loadu_ps(&c_row[j]);
                c8 = _mm256_fmadd_ps(a8, b8, c8);
                _mm256_storeu_ps(&c_row[j], c8);
            }
            
            // Handle remaining elements (tail)
            for (; j < NB; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

void gemm::gemm_packed(int M, int N, int K,
                      const float* __restrict A, const float* __restrict B, float* __restrict C,
                      int lda, int ldb, int ldc, const Block& block_sizes)
{
    const int MB = block_sizes.MB;
    const int NB = block_sizes.NB;
    const int KB = block_sizes.KB;
    
    const int nI = (M + MB - 1) / MB;
    const int nJ = (N + NB - 1) / NB;
    const int nTiles = nI * nJ;
    
    #pragma omp parallel
    {
        // Each thread gets its own packed panel buffers
        float* A_packed = alloc_aligned_panel(MB * KB);
        float* B_packed = alloc_aligned_panel(KB * NB);
        
        #pragma omp for schedule(static)
        for (int t = 0; t < nTiles; ++t) {
            const int ti = t / nJ;
            const int tj = t % nJ;
            const int ii = ti * MB, iMax = std::min(ii + MB, M);
            const int jj = tj * NB, jMax = std::min(jj + NB, N);
            
            const int actual_MB = iMax - ii;
            const int actual_NB = jMax - jj;
            
            for (int kk = 0; kk < K; kk += KB) {
                const int kMax = std::min(kk + KB, K);
                const int actual_KB = kMax - kk;
                
                // Pack A panel: (actual_MB x actual_KB)
                pack_A_panel(actual_MB, actual_KB, &A[ii * lda + kk], lda, A_packed);
                
                // Pack B panel: (actual_KB x actual_NB)
                pack_B_panel(actual_KB, actual_NB, &B[kk * ldb + jj], ldb, B_packed);
                
                // Prefetch next panels (if available) - MSVC doesn't have __builtin_prefetch
                // Skip prefetching for now, focus on correctness
                
                // Compute using packed panels
                micro_kernel_packed(actual_MB, actual_NB, actual_KB, 
                                  A_packed, B_packed, &C[ii * ldc + jj], ldc);
            }
        }
        
        free_aligned_panel(A_packed);
        free_aligned_panel(B_packed);
    }
}