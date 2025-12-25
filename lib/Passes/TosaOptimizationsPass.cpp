#include "TosaPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TosaOptimizationsPass : 
    public PassWrapper<TosaOptimizationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaOptimizationsPass)
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    // Pull patterns from our modular library
    tensormorph::populateTosaStructuralFusionPatterns(patterns);
    tensormorph::populateTosaAlgebraicFoldingPatterns(patterns);

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