#pragma once
#include <cstddef>

namespace gemm {
    // 8x8 AVX2 micro-kernel: C[0..7,0..7] += A_p(8xKC) * B_p(KCx8)
    void mk8x8_avx2(int KC,
        const float* __restrict A_p,  // 8 x KC, row-major, 64B aligned
        const float* __restrict B_p,  // KC x 8, row-major, 64B aligned
        float* __restrict C,          // points to C[i0, j0]
        int ldc);                     // row-major leading dimension of C
        
    // 8x8 AVX2 micro-kernel with strided B access
    void mk8x8_avx2_strided(int KC,
        const float* __restrict A_p,  // 8 x KC, row-major
        const float* __restrict B_p,  // KC x NB, row-major, access 8 columns starting at offset
        int ldb,                      // leading dimension of B_p
        float* __restrict C,          // points to C[i0, j0]
        int ldc);                     // row-major leading dimension of C
        
    // Reference scalar micro-kernel for edge cases
    void mk_ref(int MR, int NR, int KC,
        const float* __restrict A_p,
        const float* __restrict B_p,
        float* __restrict C,
        int ldc);
        
    // Reference scalar micro-kernel with strided B access
    void mk_ref_strided(int MR, int NR, int KC,
        const float* __restrict A_p,
        const float* __restrict B_p,
        int ldb,
        float* __restrict C,
        int ldc);
}