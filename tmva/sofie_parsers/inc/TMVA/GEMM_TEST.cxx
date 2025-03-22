#include "TMVA/SOFIE/ROperator_GEMM_CUDA.hxx"
#include "TMVA/SOFIE/RExecutionContext.hxx"
#include "TMVA/SOFIE/RTensor.hxx"
#include <iostream>

using namespace TMVA::Experimental::SOFIE;

int main() {
   const int M = 2, K = 3, N = 2;

   std::vector<float> A_data = {1, 2, 3, 4, 5, 6};   // (2x3)
   std::vector<float> B_data = {7, 8, 9, 10, 11, 12}; // (3x2)
   std::vector<float> C_data = {0, 0, 0, 0};         // (2x2)

   RExecutionContext context;

   auto A = std::make_shared<RTensor<float>>(std::vector<size_t>{M, K}, A_data);
   auto B = std::make_shared<RTensor<float>>(std::vector<size_t>{K, N}, B_data);
   auto C = std::make_shared<RTensor<float>>(std::vector<size_t>{M, N}, C_data);
   auto Y = std::make_shared<RTensor<float>>(std::vector<size_t>{M, N});

   context.RegisterTensor("A", A);
   context.RegisterTensor("B", B);
   context.RegisterTensor("C", C);
   context.RegisterTensor("Y", Y);

   ROperator_GEMM_CUDA gemm("A", "B", "C", "Y", 1.0f, 1.0f, false, false);
   gemm.Execute(context);

   auto result = context.GetTensor<float>("Y")->GetData();
   std::cout << "Result of GEMM (CUDA):\n";
   for (int i = 0; i < M * N; ++i) {
      std::cout << result[i] << " ";
      if ((i + 1) % N == 0) std::cout << "\n";
   }

   return 0;
}

// The output of this code is:
// 58 64
// 139 154