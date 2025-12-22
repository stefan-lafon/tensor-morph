#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

namespace {
struct FoldBatchNormPass 
    : public PassWrapper<FoldBatchNormPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldBatchNormPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Traverse the IR to find the Conv -> BatchNorm chain
    module.walk([&](Operation *op) {
      auto opName = op->getName().getStringRef();

      // Look for the NHWC Convolution variant
      if (opName == "linalg.conv_2d_nhwc_fhwc") {
        llvm::outs() << "TensorMorph: Found a Convolution (NHWC): " << *op << "\n";
      }

      // Look for the BatchNorm
      if (opName == "test.batch_norm") {
        llvm::outs() << "TensorMorph: Found a BatchNorm to fold: " << *op << "\n";
      }
    });
  }

  llvm::StringRef getArgument() const final { return "fold-batchnorm"; }
  llvm::StringRef getDescription() const final { return "Fuses BatchNorm into Conv2D weights"; }
};
} // namespace

namespace mlir {
void registerFoldBatchNormPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<FoldBatchNormPass>();
    });
}
} // namespace