#include "TMVA/SOFIE/ROperator_GEMM_CUDA.hxx"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ROperator_GEMM_CUDA::ROperator_GEMM_CUDA(std::string A, std::string B, std::string C, std::string Y,
                                         float alpha, float beta, bool transA, bool transB)
   : fA(std::move(A)), fB(std::move(B)), fC(std::move(C)), fY(std::move(Y)),
     fAlpha(alpha), fBeta(beta), fTransA(transA), fTransB(transB) {}

void ROperator_GEMM_CUDA::Execute(RExecutionContext& context) {
   auto A = context.GetTensor<float>(fA);
   auto B = context.GetTensor<float>(fB);
   auto C = context.GetTensor<float>(fC);
   auto Y = context.GetTensor<float>(fY);

   const int M = A->GetShape()[0];
   const int K = A->GetShape()[1];
   const int N = B->GetShape()[1];

   float *d_A, *d_B, *d_C, *d_Y;
   cudaMalloc(&d_A, sizeof(float) * M * K);
   cudaMalloc(&d_B, sizeof(float) * K * N);
   cudaMalloc(&d_C, sizeof(float) * M * N);
   cudaMalloc(&d_Y, sizeof(float) * M * N);

   cudaMemcpy(d_A, A->GetData(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B->GetData(), sizeof(float) * K * N, cudaMemcpyHostToDevice);
   cudaMemcpy(d_C, C->GetData(), sizeof(float) * M * N, cudaMemcpyHostToDevice);

   cublasHandle_t handle;
   cublasCreate(&handle);

   cublasOperation_t opA = fTransA ? CUBLAS_OP_T : CUBLAS_OP_N;
   cublasOperation_t opB = fTransB ? CUBLAS_OP_T : CUBLAS_OP_N;

   int lda = fTransA ? M : K;
   int ldb = fTransB ? K : N;
   int ldc = N;

   cublasSgemm(handle, opB, opA,
               N, M, K,
               &fAlpha,
               d_B, ldb,
               d_A, lda,
               &fBeta,
               d_Y, ldc);

   cudaMemcpy(Y->GetData(), d_Y, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   cudaFree(d_Y);
   cublasDestroy(handle);
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
