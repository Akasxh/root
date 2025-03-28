#ifndef TMVA_SOFIE_ROPERATOR_GEMM_CUDA
#define TMVA_SOFIE_ROPERATOR_GEMM_CUDA

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <vector>

// CUDA kernel for GEMM operation when A is MxK, B is KxN, and C is MxN
template <typename T>
__global__ void gemmKernel(const T *A, const T *B, T *C, int M, int N, int K) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   if (row < M && col < N) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
         sum += A[row * K + k] * B[k * N + col];
      }
      C[row * N + col] = sum;
   }
}

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Gemm final : public ROperator {

private:
   // Names for input and output tensors
   std::string fNA; // Tensor A
   std::string fNB; // Tensor B
   std::string fNC; // Tensor C (output)

   // Shapes for tensors (expected to be 2D)
   std::vector<Dim> fShapeA;
   std::vector<Dim> fShapeB;
   std::vector<Dim> fShapeC;

   // Dimensions
   int fM, fN, fK;

public:
   ROperator_Gemm() : fM(0), fN(0), fK(0) {}

   // Constructor with tensor names for A, B and C
   ROperator_Gemm(std::string nameA, std::string nameB, std::string nameC)
      : fNA(UTILITY::Clean_name(nameA)), fNB(UTILITY::Clean_name(nameB)), fNC(UTILITY::Clean_name(nameC)),
        fM(0), fN(0), fK(0)
   {
      fInputTensorNames = { fNA, fNB };
      fOutputTensorNames = { fNC };
   }

   // The output type is assumed to be the same as the type of A (and B)
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return { input[0] };
   }

   // Shape inference:
   // A: [M, K] and B: [K, N] => C: [M, N]
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() < 2 || input[0].size() != 2 || input[1].size() != 2) {
         throw std::runtime_error("GEMM requires two 2D input matrices for tensors A and B");
      }
      fM = input[0][0];
      fK = input[0][1];
      if (input[1][0] != static_cast<size_t>(fK)) {
         throw std::runtime_error("GEMM dimension mismatch: Number of columns of A must equal number of rows of B");
      }
      fN = input[1][1];
      // Set output shape: [M, N]
      fShapeC = { static_cast<Dim>(fM), static_cast<Dim>(fN) };
      return { input[0], input[1], {static_cast<size_t>(fM), static_cast<size_t>(fN)} };
   }

   // Initialization: Check input tensors, retrieve shapes, and register output tensor.
   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNA)) {
         throw std::runtime_error("TMVA SOFIE GEMM Op: Input Tensor A (" + fNA + ") not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw std::runtime_error("TMVA SOFIE GEMM Op: Input Tensor B (" + fNB + ") not found in model");
      }
      fShapeA = model.GetDynamicTensorShape(fNA);
      fShapeB = model.GetDynamicTensorShape(fNB);
      if (fShapeA.size() != 2 || fShapeB.size() != 2) {
         throw std::runtime_error("TMVA SOFIE GEMM Op: Both input tensors must be 2D");
      }
      fM = fShapeA[0];
      fK = fShapeA[1];
      if (fShapeB[0] != static_cast<Dim>(fK)) {
         throw std::runtime_error("TMVA SOFIE GEMM Op: Dimension mismatch between A and B");
      }
      fN = fShapeB[1];
      fShapeC = { static_cast<Dim>(fM), static_cast<Dim>(fN) };
      model.AddIntermediateTensor(fNC, model.GetTensorType(fNA), fShapeC);

      if (model.Verbose()) {
         std::cout << "GEMM (CUDA): " << fNA << " (" << fM << "x" << fK << ") * " << fNB << " (" << fK << "x" << fN
                   << ") = " << fNC << " (" << fM << "x" << fN << ")" << std::endl;
      }
   }

   // Generate CUDA code to launch the GEMM kernel
   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      std::stringstream out;
      // Set up a 2D grid and block configuration (16x16 threads per block)
      out << "\n//------ GEMM (CUDA)\n";
      out << "   dim3 threadsPerBlock(16, 16);\n";
      out << "   dim3 blocksPerGrid(( " << fN << " + 15) / 16, ( " << fM << " + 15) / 16);\n";
      out << "   gemmKernel<<<blocksPerGrid, threadsPerBlock>>>(tensor_" << fNA << ", tensor_" << fNB << ", tensor_" << fNC
          << ", " << fM << ", " << fN << ", " << fK << ");\n";
      out << "   cudaDeviceSynchronize();\n";
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_GEMM_CUDA
