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

    module.walk([&](Operation *op) {
      auto opName = op->getName().getStringRef();

      // We focus on our BatchNorm consumer
      if (opName == "test.batch_norm") {
        
        // 1. Safety Check: A valid BN for folding must have 5 operands
        // [input, scale, shift, mean, variance]
        if (op->getNumOperands() != 5) {
          llvm::outs() << "[TensorMorph] Skipping: BatchNorm does not have 5 operands.\n";
          return;
        }

        // 2. Trace back to the producer
        Value bnInput = op->getOperand(0);
        Operation *definingOp = bnInput.getDefiningOp();

        if (definingOp && 
            definingOp->getName().getStringRef() == "linalg.conv_2d_nhwc_fhwc") {
          
          llvm::outs() << "\n[TensorMorph] Pattern Found: Conv2D -> BatchNorm\n";

          // 3. Extract the 4 BN parameters
          Value scale = op->getOperand(1);
          Value shift = op->getOperand(2);
          Value mean  = op->getOperand(3);
          Value var   = op->getOperand(4);

          // 4. Extract the Convolution weights
          Value convWeights = definingOp->getOperand(1);

          // 5. Verification: Ensure the BN params match the Conv output channels
          auto weightType = convWeights.getType().cast<ShapedType>();
          int64_t outChannels = weightType.getShape()[0]; // F dimension in FHWCc

          auto scaleType = scale.getType().cast<ShapedType>();
          int64_t scaleSize = scaleType.getShape()[0];

          if (outChannels == scaleSize) {
            llvm::outs() << "  -> Parameter Extraction Successful.\n";
            llvm::outs() << "  -> Channels: " << outChannels << "\n";
            llvm::outs() << "  -> Status: Ready for Mathematical Folding.\n";
          } else {
            llvm::outs() << "  -> Error: Channel mismatch! Conv has " << outChannels 
                         << " but BN has " << scaleSize << ".\n";
          }
        }
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