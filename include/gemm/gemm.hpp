#pragma once
#include <string>

namespace gemm {
  struct Block { 
    int MB = 256, NB = 256, KB = 256; 
  };
  
  void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc);
  
  void gemm_blocked(int M, int N, int K, const float* A, const float* B, float* C, 
                   int lda, int ldb, int ldc, const Block& block_sizes);
  
  void gemm_packed(int M, int N, int K, const float* A, const float* B, float* C, 
                  int lda, int ldb, int ldc, const Block& block_sizes);
  
  void gemm_mk_avx2(int M, int N, int K, const float* A, const float* B, float* C, 
                   int lda, int ldb, int ldc, const Block& block_sizes);
  
  void gemm_openblas(int M, int N, int K, const float* A, const float* B, float* C, 
                    int lda, int ldb, int ldc, const Block& block_sizes);
  
  void run_gemm(const std::string& impl, int M, int N, int K,
               const float* A, const float* B, float* C,
               int lda, int ldb, int ldc, const Block& block_sizes);
}