#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

namespace {
// This struct defines the actual 'surgery' logic of the pass.
struct FoldBatchNormPass 
    : public PassWrapper<FoldBatchNormPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldBatchNormPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The 'walk' function visits every operation in the entire module.
    module.walk([&](Operation *op) {
      auto opName = op->getName().getStringRef();

      // 1. We look for our target 'BatchNorm'
      if (opName == "test.batch_norm") {
        llvm::outs() << "\n[TensorMorph] Found a potential BatchNorm target.\n";

        // 2. Get the value being fed into the BatchNorm (Operand 0)
        Value bnInput = op->getOperand(0);

        // 3. Follow the Use-Def chain back to the 'Defining Op'
        Operation *definingOp = bnInput.getDefiningOp();

        // 4. Verify that the producer is specifically our target Convolution
        if (definingOp && 
            definingOp->getName().getStringRef() == "linalg.conv_2d_nhwc_fhwc") {
          
          llvm::outs() << "  -> SUCCESS: Connection verified.\n";
          llvm::outs() << "  -> Producer: " << definingOp->getName() << "\n";
          
          // 5. Reach into the Convolution to find its weight tensor (Operand 1)
          // For NHWC Convolution, inputs are [InputData, Weights]
          Value convWeights = definingOp->getOperand(1);
          
          // 6. Access the shape metadata to prove we can see the dimensions
          auto weightType = convWeights.getType().cast<ShapedType>();
          auto shape = weightType.getShape();
          
          llvm::outs() << "  -> Conv Weights identified. Shape: " 
                       << shape[0] << "x" << shape[1] << "x" 
                       << shape[2] << "x" << shape[3] << "\n";
          
          // The pass is now ready for the actual math transformation.
        } else {
          llvm::outs() << "  -> SKIPPED: BatchNorm input is not a direct Convolution output.\n";
        }
      }
    });
  }

  // These provide the metadata for the 'tensormorph-opt' tool to list and run the pass.
  llvm::StringRef getArgument() const final { return "fold-batchnorm"; }
  llvm::StringRef getDescription() const final { return "Fuses BatchNorm into Conv2D weights"; }
};
} // namespace

namespace mlir {
// This registration function makes the pass available to the registry we built in main.cpp.
void registerFoldBatchNormPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<FoldBatchNormPass>();
    });
}
} // namespace