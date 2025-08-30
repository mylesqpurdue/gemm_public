#include "gemm/gemm.hpp"

void gemm::gemm_naive(int M, int N, int K,
  const float* __restrict A, const float* __restrict B, float* __restrict C,
  int lda, int ldb, int ldc)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      float a = A[i * lda + k];
      #pragma omp simd
      for (int j = 0; j < N; j++) {
        C[i * ldc + j] += a * B[k * ldb + j];
      }
    }
  }
}