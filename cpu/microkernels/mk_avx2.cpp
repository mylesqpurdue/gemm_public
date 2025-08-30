#include <immintrin.h>
#include "gemm/microkernel.hpp"

namespace gemm {

// 8x8 AVX2 FMA micro-kernel with K unrolling
// Assumes A_p and B_p come from packed buffers (64B aligned)
void mk8x8_avx2(int KC,
    const float* __restrict A,  // 8xKC
    const float* __restrict B,  // KCx8
    float* __restrict C, int ldc) {
    
    // Load initial C values (8 rows × 8 columns)
    __m256 c0 = _mm256_loadu_ps(C + 0*ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1*ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2*ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3*ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4*ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5*ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6*ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7*ldc);
    
    // Main K loop with unrolling by 4
    int k = 0;
    for (; k <= KC - 4; k += 4) {
        // Unroll 1: k+0
        {
            __m256 b = _mm256_load_ps(B + (k+0)*8);  // 32B aligned if B base is 64B
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+0)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+0)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+0)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+0)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+0)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+0)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+0)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+0)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Unroll 2: k+1
        {
            __m256 b = _mm256_load_ps(B + (k+1)*8);
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+1)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+1)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+1)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+1)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+1)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+1)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+1)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+1)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Unroll 3: k+2
        {
            __m256 b = _mm256_load_ps(B + (k+2)*8);
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+2)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+2)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+2)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+2)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+2)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+2)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+2)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+2)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Unroll 4: k+3
        {
            __m256 b = _mm256_load_ps(B + (k+3)*8);
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+3)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+3)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+3)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+3)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+3)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+3)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+3)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+3)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Prefetch upcoming data
        _mm_prefetch((const char*)(B + (k+8)*8), _MM_HINT_T0);
        _mm_prefetch((const char*)(A + 0*KC + (k+32)), _MM_HINT_T0);
    }
    
    // Handle remaining K iterations (KC % 4)
    for (; k < KC; ++k) {
        __m256 b = _mm256_load_ps(B + k*8);
        
        __m256 a0 = _mm256_broadcast_ss(&A[0*KC + k]);
        __m256 a1 = _mm256_broadcast_ss(&A[1*KC + k]);
        __m256 a2 = _mm256_broadcast_ss(&A[2*KC + k]);
        __m256 a3 = _mm256_broadcast_ss(&A[3*KC + k]);
        __m256 a4 = _mm256_broadcast_ss(&A[4*KC + k]);
        __m256 a5 = _mm256_broadcast_ss(&A[5*KC + k]);
        __m256 a6 = _mm256_broadcast_ss(&A[6*KC + k]);
        __m256 a7 = _mm256_broadcast_ss(&A[7*KC + k]);
        
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
        c4 = _mm256_fmadd_ps(a4, b, c4);
        c5 = _mm256_fmadd_ps(a5, b, c5);
        c6 = _mm256_fmadd_ps(a6, b, c6);
        c7 = _mm256_fmadd_ps(a7, b, c7);
    }
    
    // Write back results (minimizes memory traffic)
    _mm256_storeu_ps(C + 0*ldc, c0);
    _mm256_storeu_ps(C + 1*ldc, c1);
    _mm256_storeu_ps(C + 2*ldc, c2);
    _mm256_storeu_ps(C + 3*ldc, c3);
    _mm256_storeu_ps(C + 4*ldc, c4);
    _mm256_storeu_ps(C + 5*ldc, c5);
    _mm256_storeu_ps(C + 6*ldc, c6);
    _mm256_storeu_ps(C + 7*ldc, c7);
}

// 8x8 AVX2 micro-kernel with strided B access - optimized for packed panels
void mk8x8_avx2_strided(int KC,
    const float* __restrict A,  // 8xKC
    const float* __restrict B,  // KCxNB, access 8 columns starting at offset
    int ldb,                    // leading dimension of B
    float* __restrict C, int ldc) {
    
    // Load initial C values (8 rows × 8 columns)
    __m256 c0 = _mm256_loadu_ps(C + 0*ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1*ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2*ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3*ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4*ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5*ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6*ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7*ldc);
    
    // Main K loop with unrolling by 4
    int k = 0;
    for (; k <= KC - 4; k += 4) {
        // Unroll 1: k+0
        {
            __m256 b = _mm256_loadu_ps(B + (k+0)*ldb);  // Load 8 floats from B row
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+0)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+0)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+0)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+0)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+0)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+0)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+0)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+0)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Unroll 2: k+1
        {
            __m256 b = _mm256_loadu_ps(B + (k+1)*ldb);
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+1)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+1)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+1)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+1)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+1)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+1)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+1)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+1)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Unroll 3: k+2
        {
            __m256 b = _mm256_loadu_ps(B + (k+2)*ldb);
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+2)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+2)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+2)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+2)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+2)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+2)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+2)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+2)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Unroll 4: k+3
        {
            __m256 b = _mm256_loadu_ps(B + (k+3)*ldb);
            
            __m256 a0 = _mm256_broadcast_ss(&A[0*KC + (k+3)]);
            __m256 a1 = _mm256_broadcast_ss(&A[1*KC + (k+3)]);
            __m256 a2 = _mm256_broadcast_ss(&A[2*KC + (k+3)]);
            __m256 a3 = _mm256_broadcast_ss(&A[3*KC + (k+3)]);
            __m256 a4 = _mm256_broadcast_ss(&A[4*KC + (k+3)]);
            __m256 a5 = _mm256_broadcast_ss(&A[5*KC + (k+3)]);
            __m256 a6 = _mm256_broadcast_ss(&A[6*KC + (k+3)]);
            __m256 a7 = _mm256_broadcast_ss(&A[7*KC + (k+3)]);
            
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Prefetch upcoming data
        _mm_prefetch((const char*)(B + (k+8)*ldb), _MM_HINT_T0);
        _mm_prefetch((const char*)(A + 0*KC + (k+32)), _MM_HINT_T0);
    }
    
    // Handle remaining K iterations (KC % 4)
    for (; k < KC; ++k) {
        __m256 b = _mm256_loadu_ps(B + k*ldb);
        
        __m256 a0 = _mm256_broadcast_ss(&A[0*KC + k]);
        __m256 a1 = _mm256_broadcast_ss(&A[1*KC + k]);
        __m256 a2 = _mm256_broadcast_ss(&A[2*KC + k]);
        __m256 a3 = _mm256_broadcast_ss(&A[3*KC + k]);
        __m256 a4 = _mm256_broadcast_ss(&A[4*KC + k]);
        __m256 a5 = _mm256_broadcast_ss(&A[5*KC + k]);
        __m256 a6 = _mm256_broadcast_ss(&A[6*KC + k]);
        __m256 a7 = _mm256_broadcast_ss(&A[7*KC + k]);
        
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
        c4 = _mm256_fmadd_ps(a4, b, c4);
        c5 = _mm256_fmadd_ps(a5, b, c5);
        c6 = _mm256_fmadd_ps(a6, b, c6);
        c7 = _mm256_fmadd_ps(a7, b, c7);
    }
    
    // Write back results (minimizes memory traffic)
    _mm256_storeu_ps(C + 0*ldc, c0);
    _mm256_storeu_ps(C + 1*ldc, c1);
    _mm256_storeu_ps(C + 2*ldc, c2);
    _mm256_storeu_ps(C + 3*ldc, c3);
    _mm256_storeu_ps(C + 4*ldc, c4);
    _mm256_storeu_ps(C + 5*ldc, c5);
    _mm256_storeu_ps(C + 6*ldc, c6);
    _mm256_storeu_ps(C + 7*ldc, c7);
}

// Reference scalar micro-kernel for edge cases
void mk_ref(int MR, int NR, int KC,
    const float* __restrict A_p,
    const float* __restrict B_p,
    float* __restrict C,
    int ldc) {
    
    for (int i = 0; i < MR; ++i) {
        for (int k = 0; k < KC; ++k) {
            const float a = A_p[i * KC + k];
            const float* b_row = &B_p[k * NR];
            float* c_row = &C[i * ldc];
            
            for (int j = 0; j < NR; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

// Reference scalar micro-kernel with strided B access
void mk_ref_strided(int MR, int NR, int KC,
    const float* __restrict A_p,
    const float* __restrict B_p,
    int ldb,
    float* __restrict C,
    int ldc) {
    
    for (int i = 0; i < MR; ++i) {
        for (int k = 0; k < KC; ++k) {
            const float a = A_p[i * KC + k];
            const float* b_row = &B_p[k * ldb];
            float* c_row = &C[i * ldc];
            
            for (int j = 0; j < NR; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

} // namespace gemm