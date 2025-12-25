#include "TosaPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TosaOptimizationsPass : 
    public PassWrapper<TosaOptimizationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaOptimizationsPass)
  
  TosaOptimizationsPass() = default;
  
  // Custom copy constructor for MLIR's cloning mechanism
  TosaOptimizationsPass(const TosaOptimizationsPass &other) : PassWrapper(other) {
    this->fuseFanout = other.fuseFanout;
    this->fuseActivations = other.fuseActivations;
    this->foldAlgebraic = other.foldAlgebraic;
  }

  // Command-line options
  Option<bool> fuseFanout{*this, "fuse-fanout", 
    llvm::cl::desc("Allow cloning operations to enable fusion across multiple users."), 
    llvm::cl::init(true)};

  Option<bool> fuseActivations{*this, "fuse-activations", 
    llvm::cl::desc("Enable/Disable fusion of Clamp/ReLU into Conv."), 
    llvm::cl::init(true)};

  Option<bool> foldAlgebraic{*this, "fold-algebraic", 
    llvm::cl::desc("Enable/Disable pure algebraic identities (Add+Add, etc)."), 
    llvm::cl::init(true)};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    tensormorph::populateTosaStructuralFusionPatterns(patterns, fuseFanout, fuseActivations);
    
    if (foldAlgebraic) {
      tensormorph::populateTosaAlgebraicFoldingPatterns(patterns);
    }

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  
  // Required override when using Options with PassWrapper
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<TosaOptimizationsPass>(*this);
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