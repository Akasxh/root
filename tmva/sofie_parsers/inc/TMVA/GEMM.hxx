#ifndef TMVA_SOFIE_ROPERATOR_GEMM_CUDA
#define TMVA_SOFIE_ROPERATOR_GEMM_CUDA

#include "TMVA/SOFIE_ROperator.hxx"
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class ROperator_GEMM_CUDA : public ROperator {
private:
   std::string fA, fB, fC, fY;
   float fAlpha = 1.0f;
   float fBeta = 1.0f;
   bool fTransA = false;
   bool fTransB = false;

public:
   ROperator_GEMM_CUDA(std::string A, std::string B, std::string C, std::string Y,
                       float alpha, float beta, bool transA, bool transB);

   void Execute(RExecutionContext& context) override;
   std::string GetName() const override { return "GEMM_CUDA"; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
