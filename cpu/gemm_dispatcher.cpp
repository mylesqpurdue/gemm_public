#include "gemm/gemm.hpp"
#include <stdexcept>

void gemm::run_gemm(const std::string& impl, int M, int N, int K,
                   const float* A, const float* B, float* C,
                   int lda, int ldb, int ldc, const Block& block_sizes)
{
    if (impl == "naive") {
        gemm_naive(M, N, K, A, B, C, lda, ldb, ldc);
    } else if (impl == "blocked") {
        gemm_blocked(M, N, K, A, B, C, lda, ldb, ldc, block_sizes);
    } else if (impl == "packed") {
        gemm_packed(M, N, K, A, B, C, lda, ldb, ldc, block_sizes);
    } else if (impl == "mk_avx2") {
        gemm_mk_avx2(M, N, K, A, B, C, lda, ldb, ldc, block_sizes);
    } else if (impl == "openblas") {
        gemm_openblas(M, N, K, A, B, C, lda, ldb, ldc, block_sizes);
    } else {
        throw std::runtime_error("Unknown implementation: " + impl);
    }
}