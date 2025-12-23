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
    SmallVector<Operation *> opsToErase;

    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "test.batch_norm") {
        if (op->getNumOperands() != 5) return;

        Value bnInput = op->getOperand(0);
        Operation *convOp = bnInput.getDefiningOp();

        if (convOp && convOp->getName().getStringRef() == "linalg.conv_2d_nhwc_fhwc") {
          
          ImplicitLocOpBuilder b(convOp->getLoc(), convOp);
          
          Value weights = convOp->getOperand(1);
          Value gamma   = op->getOperand(1);
          Value beta    = op->getOperand(2);
          Value mean    = op->getOperand(3);
          Value var     = op->getOperand(4);
          
          auto weightType = weights.getType().cast<RankedTensorType>();
          auto paramType  = gamma.getType().cast<RankedTensorType>();
          auto outType    = bnInput.getType().cast<RankedTensorType>();
          auto elementType = paramType.getElementType();

          // 1. Calculate Multiplier: factor = gamma / sqrt(var + eps)
          Value epsilon = b.create<arith::ConstantOp>(b.getFloatAttr(elementType, 1e-5));
          Value empty1D = b.create<tensor::EmptyOp>(paramType.getShape(), elementType);
          auto map1D = b.getMultiDimIdentityMap(1);
          
          auto multOp = b.create<linalg::GenericOp>(
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
          Value factor = multOp.getResult(0);

          // 2. Calculate New Bias: bias = beta - (factor * mean)
          auto biasOp = b.create<linalg::GenericOp>(
            TypeRange{paramType}, ValueRange{beta, factor, mean}, ValueRange{empty1D},
            SmallVector<AffineMap>{map1D, map1D, map1D, map1D},
            SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
            [&](OpBuilder &nb, Location loc, ValueRange args) {
              Value prod = nb.create<arith::MulFOp>(loc, args[1], args[2]);
              Value res  = nb.create<arith::SubFOp>(loc, args[0], prod);
              nb.create<linalg::YieldOp>(loc, res);
            }
          );
          Value newBias = biasOp.getResult(0);

          // 3. Fold Weights: W_new = W_old * factor
          auto map4D = b.getMultiDimIdentityMap(4);
          auto mapBroadcast = AffineMap::get(4, 0, {b.getAffineDimExpr(0)}, b.getContext());
          Value empty4D = b.create<tensor::EmptyOp>(weightType.getShape(), elementType);
          
          auto weightFoldOp = b.create<linalg::GenericOp>(
            TypeRange{weightType}, ValueRange{weights, factor}, ValueRange{empty4D},
            SmallVector<AffineMap>{map4D, mapBroadcast, map4D},
            SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
            [&](OpBuilder &nb, Location loc, ValueRange args) {
              Value res = nb.create<arith::MulFOp>(loc, args[0], args[1]);
              nb.create<linalg::YieldOp>(loc, res);
            }
          );
          convOp->setOperand(1, weightFoldOp.getResult(0));

          // 4. Apply Bias after Conv: output = conv_result + newBias
          // We move the builder to AFTER the conv now
          b.setInsertionPointAfter(convOp);
          
          auto mapData = b.getMultiDimIdentityMap(4);
          auto mapBias = AffineMap::get(4, 0, {b.getAffineDimExpr(3)}, b.getContext()); // Map to Channel dim
          Value emptyOut = b.create<tensor::EmptyOp>(outType.getShape(), elementType);

          auto finalAddOp = b.create<linalg::GenericOp>(
            TypeRange{outType}, ValueRange{convOp->getResult(0), newBias}, ValueRange{emptyOut},
            SmallVector<AffineMap>{mapData, mapBias, mapData},
            SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
            [&](OpBuilder &nb, Location loc, ValueRange args) {
              Value res = nb.create<arith::AddFOp>(loc, args[0], args[1]);
              nb.create<linalg::YieldOp>(loc, res);
            }
          );

          // 5. Final Rewire
          op->replaceAllUsesWith(finalAddOp.getResults());
          opsToErase.push_back(op);
          
          llvm::outs() << "[TensorMorph] SUCCESS: Full Weight + Bias folding complete.\n";
        }
      }
    });

    for (auto *op : opsToErase) op->erase();
  }

  llvm::StringRef getArgument() const final { return "fold-batchnorm"; }
  llvm::StringRef getDescription() const final { return "Fuses BatchNorm into Conv2D weights and adds bias"; }
};
} // namespace

namespace mlir {
void registerFoldBatchNormPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<FoldBatchNormPass>();
    });
}
}