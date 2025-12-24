/*
 * This file implements a set of greedy optimization patterns for the TOSA dialect.
 * The primary goal here is to identify chains of linear operations (like those 
 * produced by BatchNorm or BiasAdd layers) and "fold" them directly into the 
 * preceding Convolution's weights and bias.
 *
 * By doing this at the MLIR level, we reduce the number of operations the 
 * backend has to lower, which simplifies buffer allocation and improves 
 * cache locality.
 */

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

namespace {

/**
 * This pattern anchors on a tosa::Conv2DOp and walks forward through the 
 * instruction stream. It attempts to fuse any subsequent constant-based 
 * Add, Sub, or Mul operations into the convolution's parameters.
 */
struct FoldLinearMathIntoConv : public OpRewritePattern<tosa::Conv2DOp> {
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, 
                                PatternRewriter &rewriter) const override {
    
    // We need constant weights and bias to perform the compile-time math
    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    
    if (!weightConst || !biasConst) return failure();

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();

    // Copy original values into local buffers for mutation
    SmallVector<float> currentWeights(weightAttr.getValues<float>());
    SmallVector<float> currentBias(biasAttr.getValues<float>());
    
    int numOutChannels = biasAttr.getNumElements();
    int weightsPerChannel = currentWeights.size() / numOutChannels;

    Operation *lastOp = convOp;
    bool changed = false;

    // Greedy forward walk through the IR
    while (lastOp->getResult(0).hasOneUse()) {
      Operation *nextOp = *lastOp->getResult(0).getUsers().begin();

      // Handle Scaling (Mul)
      if (auto mulOp = llvm::dyn_cast<tosa::MulOp>(nextOp)) {
        auto scaleConst = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!scaleConst) break;
        
        auto sValues = scaleConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) {
          float s = sValues[oc];
          currentBias[oc] *= s;
          for (int i = 0; i < weightsPerChannel; ++i) {
            currentWeights[oc * weightsPerChannel + i] *= s;
          }
        }
        lastOp = nextOp;
        changed = true;
      } 
      // Handle Shifting (Add)
      else if (auto addOp = llvm::dyn_cast<tosa::AddOp>(nextOp)) {
        auto shiftConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!shiftConst) break;

        auto shValues = shiftConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) {
          currentBias[oc] += shValues[oc];
        }
        lastOp = nextOp;
        changed = true;
      } 
      // Handle Shifting (Sub)
      else if (auto subOp = llvm::dyn_cast<tosa::SubOp>(nextOp)) {
        auto subConst = subOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!subConst) break;

        auto sValues = subConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) {
          currentBias[oc] -= sValues[oc];
        }
        lastOp = nextOp;
        changed = true;
      }
      else {
        break; 
      }
    }

    if (!changed) return failure();

    // Create the updated attributes and constants
    auto newWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(currentWeights));
    auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(currentBias));
    
    auto newWConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), weightAttr.getType(), newWAttr);
    auto newBConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), biasAttr.getType(), newBAttr);

    // Update the convolution in place
    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.setOperand(1, newWConst);
      convOp.setOperand(2, newBConst);
    });

    // Replace the end of the chain with the output of our new Conv
    rewriter.replaceOp(lastOp, convOp.getResult());
    
    return success();
  }
};

/**
 * TosaOptimizationsPass: This pass runs all patterns defined in this file.
 * Currently it focuses on linear algebra folding, but it's built to 
 * accommodate more patterns (like activation fusion) in the future.
 */
struct TosaOptimizationsPass : 
    public PassWrapper<TosaOptimizationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaOptimizationsPass)
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    patterns.add<FoldLinearMathIntoConv>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  
  llvm::StringRef getArgument() const final { return "tosa-opt"; }
};

} // namespace

namespace mlir {
void registerTosaOptimizationsPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<TosaOptimizationsPass>();
    });
}
}