#ifndef TMVA_SOFIE_ROPERATOR_LEAKY_RELU_CUDA
#define TMVA_SOFIE_ROPERATOR_LEAKY_RELU_CUDA

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

// CUDA kernel for LeakyReLU operation
// LeakyReLU: f(x) = x if x > 0, otherwise alpha * x
template <typename T>
__global__ void leakyReluKernel(const T *input, T *output, size_t length, T alpha) {
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id < length) {
      output[id] = (input[id] > static_cast<T>(0)) ? input[id] : alpha * input[id];
   }
}

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_LeakyRelu final : public ROperator {

private:
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;
   T fAlpha; // Slope for negative inputs

public:
   // Default constructor with default alpha = 0.01
   ROperator_LeakyRelu() : fAlpha(static_cast<T>(0.01)) {}
   
   // Constructor with names and optional alpha parameter
   ROperator_LeakyRelu(std::string nameX, std::string nameY, T alpha = static_cast<T>(0.01))
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
      if (model.CheckIfTensorAlreadyExist(fNX) == false) { // Input tensor must exist
         throw std::runtime_error("TMVA SOFIE LeakyRelu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDynamicTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

      if (model.Verbose()) {
         std::cout << "LeakyRelu (CUDA): " << fNX << " -> " << fNY << " " 
                   << ConvertDynamicShapeToString(fShape)
                   << " with alpha " << fAlpha << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator LeakyRelu (CUDA) called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);

      out << "\n//------ LeakyReLU (CUDA)\n";
      out << "   int threadsPerBlock = 256;\n";
      out << "   int blocksPerGrid = (" << length << " + threadsPerBlock - 1) / threadsPerBlock;\n";
      out << "   leakyReluKernel<<<blocksPerGrid, threadsPerBlock>>>(tensor_" << fNX << ", tensor_" << fNY << ", " << length << ", " << fAlpha << ");\n";
      out << "   cudaDeviceSynchronize();\n";

      return out.str();
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_LEAKY_RELU_CUDA
