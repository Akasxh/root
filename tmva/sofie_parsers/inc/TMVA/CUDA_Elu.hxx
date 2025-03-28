#ifndef TMVA_SOFIE_ROPERATOR_ELU_CUDA
#define TMVA_SOFIE_ROPERATOR_ELU_CUDA

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <cmath>

// CUDA kernel for ELU operation
// ELU: f(x) = x if x >= 0, else alpha * (exp(x) - 1)
template <typename T>
__global__ void eluKernel(const T *input, T *output, size_t length, T alpha) {
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id < length) {
      T inVal = input[id];
      output[id] = (inVal >= static_cast<T>(0)) ? inVal : alpha * (exp(inVal) - static_cast<T>(1));
   }
}

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Elu final : public ROperator {

private:
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;
   T fAlpha; // Parameter for ELU, default is typically 1.0

public:
   ROperator_Elu() : fAlpha(static_cast<T>(1.0)) {}
   ROperator_Elu(std::string nameX, std::string nameY, T alpha = static_cast<T>(1.0))
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fAlpha(alpha) {
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE Elu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDynamicTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

      if (model.Verbose()) {
         std::cout << "Elu (CUDA): " << fNX << " -> " << fNY << " " 
                   << ConvertDynamicShapeToString(fShape)
                   << " with alpha " << fAlpha << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Elu (CUDA) called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);

      out << "\n//------ ELU (CUDA)\n";
      out << "   int threadsPerBlock = 256;\n";
      out << "   int blocksPerGrid = (" << length << " + threadsPerBlock - 1) / threadsPerBlock;\n";
      out << "   eluKernel<<<blocksPerGrid, threadsPerBlock>>>(tensor_" << fNX << ", tensor_" << fNY << ", " << length << ", " << fAlpha << ");\n";
      out << "   cudaDeviceSynchronize();\n";

      return out.str();
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_ELU_CUDA
