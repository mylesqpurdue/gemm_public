// GEMM with AVX2 8x8 micro-kernel - highest performance implementation
#include "gemm/gemm.hpp"
#include "gemm/microkernel.hpp"
#include <algorithm>
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

// Pack A panel: (MB x KB) -> contiguous row-major for micro-kernel
static void pack_A_panel_mk(int MB, int KB, const float* A, int lda, float* A_packed) {
    for (int i = 0; i < MB; ++i) {
        for (int k = 0; k < KB; ++k) {
            A_packed[i * KB + k] = A[i * lda + k];
        }
    }
}

// Pack B panel: (KB x NB) -> contiguous row-major for micro-kernel
static void pack_B_panel_mk(int KB, int NB, const float* B, int ldb, float* B_packed) {
    for (int k = 0; k < KB; ++k) {
        for (int j = 0; j < NB; ++j) {
            B_packed[k * NB + j] = B[k * ldb + j];
        }
    }
}

void gemm::gemm_mk_avx2(int M, int N, int K,
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
                pack_A_panel_mk(actual_MB, actual_KB, &A[ii * lda + kk], lda, A_packed);
                
                // Pack B panel: (actual_KB x actual_NB)
                pack_B_panel_mk(actual_KB, actual_NB, &B[kk * ldb + jj], ldb, B_packed);
                
                // Process in 8x8 micro-kernel tiles
                for (int i0 = 0; i0 < actual_MB; i0 += 8) {
                    for (int j0 = 0; j0 < actual_NB; j0 += 8) {
                        const int mr = std::min(8, actual_MB - i0);
                        const int nr = std::min(8, actual_NB - j0);
                        
                        if (mr == 8 && nr == 8) {
                            // Full 8x8 tile - use optimized AVX2 micro-kernel
                            const float* Ablk = A_packed + i0 * actual_KB;  // 8xKB contiguous
                            const float* Bblk = B_packed + j0;              // KBx8 contiguous (stride=actual_NB)
                            float* Cblk = C + (ii + i0) * ldc + (jj + j0);
                            
                            // Use optimized micro-kernel that handles strided B access
                            gemm::mk8x8_avx2_strided(actual_KB, Ablk, Bblk, actual_NB, Cblk, ldc);
                        } else {
                            // Edge case - use reference scalar micro-kernel
                            const float* Ablk = A_packed + i0 * actual_KB;
                            const float* Bblk = B_packed + j0;
                            float* Cblk = C + (ii + i0) * ldc + (jj + j0);
                            
                            gemm::mk_ref_strided(mr, nr, actual_KB, Ablk, Bblk, actual_NB, Cblk, ldc);
                        }
                    }
                }
            }
        }
        
        free_aligned_panel(A_packed);
        free_aligned_panel(B_packed);
    }
}