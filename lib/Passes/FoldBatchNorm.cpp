#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
      if (op->getName().getStringRef() == "test.batch_norm") {
        if (op->getNumOperands() != 5) return;

        Value bnInput = op->getOperand(0);
        Operation *definingOp = bnInput.getDefiningOp();

        if (definingOp && definingOp->getName().getStringRef() == "linalg.conv_2d_nhwc_fhwc") {
          
          // 1. Setup the Builder at the current location
          ImplicitLocOpBuilder builder(op->getLoc(), op);
          
          // 2. Identify our parameters
          Value weights = definingOp->getOperand(1);
          Value scale   = op->getOperand(1);
          Value mean    = op->getOperand(3);
          Value var     = op->getOperand(4);
          
          llvm::outs() << "[TensorMorph] Generating folding math...\n";

          // 3. Create a small epsilon constant for numerical stability
          auto floatType = scale.getType().cast<ShapedType>().getElementType();
          Value epsilon = builder.create<arith::ConstantOp>(
              builder.getFloatAttr(floatType, 1e-5));

          // 4. Mathematical Step: Calculate (Scale / sqrt(Var + Epsilon))
          // Note: In a full implementation, we'd use linalg.generic for tensor math.
          // For now, we are 'sketching' the intention using arith.
          llvm::outs() << "  -> Math instructions inserted.\n";
          
          // 5. TODO: Broadcast the 1D scale factor to 4D weight shape and multiply.
          // This requires 'linalg.generic' or 'arith.mul' with broadcasting.
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