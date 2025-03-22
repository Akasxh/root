#ifndef TMVA_SOFIE_ROPERATOR_CUDA_RELU
#define TMVA_SOFIE_ROPERATOR_CUDA_RELU

#include "TMVA/SOFIE_ROperator.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class ROperator_CUDA_Relu : public ROperator {
public:
   ROperator_CUDA_Relu() = default;

   ROperator_CUDA_Relu(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs);

   // Generate the source code to run this operator on GPU
   void Generate(std::ostream &out, std::string indent = "") const override;

   // Returns a unique string name for this operator
   std::string GetName() const override { return "CUDA_Relu"; }

   // Clone function for deep copying
   std::unique_ptr<ROperator> Clone() const override {
      return std::make_unique<ROperator_CUDA_Relu>(*this);
   }

   ~ROperator_CUDA_Relu() override = default;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_CUDA_RELU
