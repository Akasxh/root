#ifndef TMVA_SOFIE_ROPERATOR_TANH_CUDA
#define TMVA_SOFIE_ROPERATOR_TANH_CUDA

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <cmath>

// CUDA kernel for Tanh operation
template <typename T>
__global__ void tanhKernel(const T *input, T *output, size_t length) {
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id < length) {
      output[id] = tanh(input[id]);
   }
}

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Tanh final : public ROperator {

private:
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Tanh() {}
   ROperator_Tanh(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)) {
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
         throw std::runtime_error("TMVA SOFIE Tanh Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDynamicTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

      if (model.Verbose()) {
         std::cout << "Tanh (CUDA): " << fNX << " -> " << fNY << " " 
                   << ConvertDynamicShapeToString(fShape) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Tanh (CUDA) called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);

      out << "\n//------ Tanh (CUDA)\n";
      out << "   int threadsPerBlock = 256;\n";
      out << "   int blocksPerGrid = (" << length << " + threadsPerBlock - 1) / threadsPerBlock;\n";
      out << "   tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(tensor_" << fNX << ", tensor_" << fNY << ", " << length << ");\n";
      out << "   cudaDeviceSynchronize();\n";

      return out.str();
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_TANH_CUDA
