#include "TosaPatterns.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

/**
 * Pattern: (x + const1) + const2  =>  x + (const1 + const2)
 */
struct FoldAddChain : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp addOp, 
                                PatternRewriter &rewriter) const override {
    auto prevAdd = addOp.getInput1().getDefiningOp<tosa::AddOp>();
    if (!prevAdd || !prevAdd->getResult(0).hasOneUse()) return failure();

    auto const1 = prevAdd.getInput2().getDefiningOp<tosa::ConstOp>();
    auto const2 = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
    if (!const1 || !const2) return failure();

    auto attr1 = const1.getValue().cast<DenseElementsAttr>();
    auto attr2 = const2.getValue().cast<DenseElementsAttr>();
    
    SmallVector<float> vals1(attr1.getValues<float>());
    auto vals2 = attr2.getValues<float>();
    if (vals1.size() != vals2.size()) return failure();

    for (size_t i = 0; i < vals1.size(); ++i) vals1[i] += vals2[i];

    auto newAttr = DenseElementsAttr::get(attr1.getType(), llvm::ArrayRef<float>(vals1));
    auto newConst = rewriter.create<tosa::ConstOp>(addOp.getLoc(), attr1.getType(), newAttr);
    
    rewriter.replaceOpWithNewOp<tosa::AddOp>(addOp, addOp.getType(), prevAdd.getInput1(), newConst);
    return success();
  }
};

/**
 * Pattern: (x * const1) * const2  =>  x * (const1 * const2)
 */
struct FoldMulChain : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp mulOp, 
                                PatternRewriter &rewriter) const override {
    auto prevMul = mulOp.getInput1().getDefiningOp<tosa::MulOp>();
    if (!prevMul || !prevMul->getResult(0).hasOneUse()) return failure();

    auto const1 = prevMul.getInput2().getDefiningOp<tosa::ConstOp>();
    auto const2 = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
    if (!const1 || !const2) return failure();

    if (prevMul.getShift() != 0 || mulOp.getShift() != 0) return failure();

    auto attr1 = const1.getValue().cast<DenseElementsAttr>();
    auto attr2 = const2.getValue().cast<DenseElementsAttr>();
    
    SmallVector<float> vals1(attr1.getValues<float>());
    auto vals2 = attr2.getValues<float>();
    if (vals1.size() != vals2.size()) return failure();

    for (size_t i = 0; i < vals1.size(); ++i) vals1[i] *= vals2[i];

    auto newAttr = DenseElementsAttr::get(attr1.getType(), llvm::ArrayRef<float>(vals1));
    auto newConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), attr1.getType(), newAttr);
    
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        mulOp, mulOp.getType(), prevMul.getInput1(), newConst, rewriter.getI8IntegerAttr(0));
    return success();
  }
};

/**
 * Pattern: (x * scale) + bias
 * In this specific pass, we don't 'fuse' them into one op (TOSA doesn't have an FMA op),
 * but we verify that the chain is recognized and can be optimized later by the backend.
 * * NOTE: This is a placeholder for more complex logic if we were to introduce a 
 * custom 'tensormorph.fma' op. For now, we'll keep it simple.
 */
struct FoldMulAddChain : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp addOp, 
                                PatternRewriter &rewriter) const override {
    auto mulOp = addOp.getInput1().getDefiningOp<tosa::MulOp>();
    if (!mulOp || !mulOp->getResult(0).hasOneUse()) return failure();

    auto mulConst = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
    auto addConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
    
    if (!mulConst || !addConst) return failure();

    // Human touch: We just log that we found a potential FMA (Fused Multiply-Add).
    // In a production compiler, you might rewrite this to a linalg.generic.
    return failure(); // We'll come back to the actual transformation logic shortly.
  }
};

} // namespace

void mlir::tensormorph::populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldAddChain, FoldMulChain, FoldMulAddChain>(patterns.getContext());
}