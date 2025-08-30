// Blocked GEMM - robust OpenMP tiling with AVX2 vectorization
#include "gemm/gemm.hpp"
#include <algorithm>
#include <immintrin.h>

void gemm::gemm_blocked(int M, int N, int K,
                       const float* __restrict A, const float* __restrict B, float* __restrict C,
                       int lda, int ldb, int ldc, const Block& block_sizes)
{
    const int MB = block_sizes.MB;
    const int NB = block_sizes.NB;
    const int KB = block_sizes.KB;
    
    const int nI = (M + MB - 1) / MB;
    const int nJ = (N + NB - 1) / NB;
    const int nTiles = nI * nJ;
    
    #pragma omp parallel for schedule(static)
    // One C-tile per thread - guarantees parallelism over tiles
    for (int t = 0; t < nTiles; ++t) {
        const int ti = t / nJ;
        const int tj = t % nJ;
        const int ii = ti * MB, iMax = std::min(ii + MB, M);
        const int jj = tj * NB, jMax = std::min(jj + NB, N);
        
        for (int kk = 0; kk < K; kk += KB) {
            const int kMax = std::min(kk + KB, K);
            for (int i = ii; i < iMax; ++i) {
                for (int k = kk; k < kMax; ++k) {
                    const float a = A[i * lda + k];
                    
                    // AVX2 vectorized inner loop (8 floats at a time)
                    int j = jj;
                    const __m256 a8 = _mm256_set1_ps(a);
                    for (; j <= jMax - 8; j += 8) {
                        __m256 b8 = _mm256_loadu_ps(&B[k * ldb + j]);
                        __m256 c8 = _mm256_loadu_ps(&C[i * ldc + j]);
                        c8 = _mm256_fmadd_ps(a8, b8, c8);
                        _mm256_storeu_ps(&C[i * ldc + j], c8);
                    }
                    
                    // Handle remaining elements (tail)
                    for (; j < jMax; ++j) {
                        C[i * ldc + j] += a * B[k * ldb + j];
                    }
                }
            }
        }
    }
}