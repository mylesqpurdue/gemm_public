// OpenBLAS baseline wrapper for GEMM comparison
#include "gemm/gemm.hpp"
#include <stdexcept>

// For now, we'll create a stub that throws an error
// This can be replaced with actual OpenBLAS integration when available
namespace gemm {

void gemm_openblas(int M, int N, int K,
                   const float* __restrict A, const float* __restrict B, float* __restrict C,
                   int lda, int ldb, int ldc, const Block& block_sizes) {
    // Stub implementation - would call cblas_sgemm in real version
    throw std::runtime_error("OpenBLAS not available - install via vcpkg install openblas:x64-windows");
    
    // Real implementation would be:
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             M, N, K, 1.0f, A, lda, B, ldb, 1.0f, C, ldc);
}

} // namespace gemm