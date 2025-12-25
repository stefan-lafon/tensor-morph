#include "TosaPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TosaOptimizationsPass : 
    public PassWrapper<TosaOptimizationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaOptimizationsPass)
  
  TosaOptimizationsPass() = default;
  
  TosaOptimizationsPass(const TosaOptimizationsPass &other) : PassWrapper(other) {
    this->fuseFanout = other.fuseFanout;
    this->fuseActivations = other.fuseActivations;
    this->fuseTranspose = other.fuseTranspose;
    this->fusePadding = other.fusePadding;
    this->fuseLinear = other.fuseLinear;
    this->foldAlgebraic = other.foldAlgebraic;
  }

  Option<bool> fuseFanout{*this, "fuse-fanout", 
    llvm::cl::desc("Allow cloning operations to enable fusion across multiple users."), 
    llvm::cl::init(true)};

  Option<bool> fuseActivations{*this, "fuse-activations", 
    llvm::cl::desc("Enable/Disable fusion of Clamp/ReLU into Conv."), 
    llvm::cl::init(true)};

  Option<bool> fuseTranspose{*this, "fuse-transpose", 
    llvm::cl::desc("Enable/Disable folding of Transpose into Conv weights."), 
    llvm::cl::init(true)};

  Option<bool> fusePadding{*this, "fuse-padding", 
    llvm::cl::desc("Enable/Disable elimination of explicit Pad ops into Conv."), 
    llvm::cl::init(true)};

  Option<bool> fuseLinear{*this, "fuse-linear", 
    llvm::cl::desc("Enable/Disable folding of Add/Sub/Mul into Conv weights/bias."), 
    llvm::cl::init(true)};

  Option<bool> foldAlgebraic{*this, "fold-algebraic", 
    llvm::cl::desc("Enable/Disable pure algebraic identities (Add+Add, x+0, etc)."), 
    llvm::cl::init(true)};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    // Pass the new fuseLinear boolean to the structural registry
    tensormorph::populateTosaStructuralFusionPatterns(
        patterns, fuseFanout, fuseActivations, fuseTranspose, fusePadding, fuseLinear);
    
    if (foldAlgebraic) {
      tensormorph::populateTosaAlgebraicFoldingPatterns(patterns);
    }

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  
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