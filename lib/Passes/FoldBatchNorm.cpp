#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

namespace {
struct FoldBatchNormPass 
    : public PassWrapper<FoldBatchNormPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldBatchNormPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect ops to erase later to avoid iterator invalidation
    SmallVector<Operation *> opsToErase;

    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "test.batch_norm") {
        if (op->getNumOperands() != 5) return;

        Value bnInput = op->getOperand(0);
        Operation *convOp = bnInput.getDefiningOp();

        if (convOp && convOp->getName().getStringRef() == "linalg.conv_2d_nhwc_fhwc") {
          
          // CRITICAL FIX: Move the builder to the CONVOLUTION, not the BatchNorm.
          // This ensures the math happens BEFORE the conv needs the weights.
          ImplicitLocOpBuilder b(convOp->getLoc(), convOp);
          
          Value weights = convOp->getOperand(1);
          Value gamma   = op->getOperand(1);
          Value var     = op->getOperand(4);
          
          auto weightType = weights.getType().cast<RankedTensorType>();
          auto paramType  = gamma.getType().cast<RankedTensorType>();
          auto elementType = paramType.getElementType();

          // 1. Calculate Multiplier (1D): factor = gamma / sqrt(var + eps)
          Value epsilon = b.create<arith::ConstantOp>(b.getFloatAttr(elementType, 1e-5));
          Value empty1D = b.create<tensor::EmptyOp>(paramType.getShape(), elementType);
          auto map1D = b.getMultiDimIdentityMap(1);
          
          auto multiplierOp = b.create<linalg::GenericOp>(
            TypeRange{paramType}, ValueRange{var, gamma}, ValueRange{empty1D},
            SmallVector<AffineMap>{map1D, map1D, map1D},
            SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
            [&](OpBuilder &nb, Location loc, ValueRange args) {
              Value a = nb.create<arith::AddFOp>(loc, args[0], epsilon);
              Value s = nb.create<math::SqrtOp>(loc, a);
              Value m = nb.create<arith::DivFOp>(loc, args[1], s);
              nb.create<linalg::YieldOp>(loc, m);
            }
          );
          Value factor = multiplierOp.getResult(0);

          // 2. Broadcast and Fold (4D): W_new = W_old * factor
          auto map4D = b.getMultiDimIdentityMap(4);
          auto mapBroadcast = AffineMap::get(4, 0, {b.getAffineDimExpr(0)}, b.getContext());
          Value empty4D = b.create<tensor::EmptyOp>(weightType.getShape(), elementType);
          
          auto foldOp = b.create<linalg::GenericOp>(
            TypeRange{weightType}, ValueRange{weights, factor}, ValueRange{empty4D},
            SmallVector<AffineMap>{map4D, mapBroadcast, map4D},
            SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
            [&](OpBuilder &nb, Location loc, ValueRange args) {
              Value res = nb.create<arith::MulFOp>(loc, args[0], args[1]);
              nb.create<linalg::YieldOp>(loc, res);
            }
          );
          Value foldedWeights = foldOp.getResult(0);

          // 3. Rewire: Update the weights of the Conv
          convOp->setOperand(1, foldedWeights);

          // 4. Final Cleanup: Bypass the BatchNorm.
          // Tell anyone using the BN output (like the 'return') to use the Conv output instead.
          op->replaceAllUsesWith(ValueRange{bnInput});
          
          opsToErase.push_back(op);
          llvm::outs() << "[TensorMorph] SUCCESS: BatchNorm folded and bypassed.\n";
        }
      }
    });

    for (auto *op : opsToErase) op->erase();
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
}