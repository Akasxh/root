#ifndef TMVA_SOFIE_ROPERATOR_CUDA_RELU
#define TMVA_SOFIE_ROPERATOR_CUDA_RELU

#include "TMVA/SOFIE_ROperator.hxx"
#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class ROperator_CUDA_Relu : public ROperator {
public:
   ROperator_CUDA_Relu() = default;

   ROperator_CUDA_Relu(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs)
      : ROperator(inputs, outputs) {}

   std::string GetName() const override { return "CUDA_Relu"; }

   std::unique_ptr<ROperator> Clone() const override {
      return std::make_unique<ROperator_CUDA_Relu>(*this);
   }

   void Generate(std::ostream &out, std::string indent = "") const override {
      const std::string &input = fInputs[0];
      const std::string &output = fOutputs[0];

      out << indent << "// CUDA kernel for ReLU\n";
      out << indent << "__global__ void relu_kernel(float* input, float* output, int size) {\n";
      out << indent << "   int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
      out << indent << "   if (i < size) {\n";
      out << indent << "      output[i] = fmaxf(0.0f, input[i]);\n";
      out << indent << "   }\n";
      out << indent << "}\n\n";

      out << indent << "// Launch the ReLU kernel\n";
      out << indent << "{\n";
      out << indent << "   int relu_size = " << input << "_size;\n";
      out << indent << "   int blockSize = 256;\n";
      out << indent << "   int gridSize = (relu_size + blockSize - 1) / blockSize;\n";
      out << indent << "   relu_kernel<<<gridSize, blockSize>>>(" << input << ", " << output << ", relu_size);\n";
      out << indent << "   cudaDeviceSynchronize();\n";
      out << indent << "}\n";
   }

   ~ROperator_CUDA_Relu() override = default;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_CUDA_RELU
