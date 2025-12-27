#include "TosaPatterns.h"
#include "experimental/Advisor.h"
#include "experimental/codegen/MemoryAdvisor.h"
#include "experimental/codegen/ComputeAdvisor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

using namespace mlir;
using namespace mlir::tensormorph;

namespace {

/**
 * Main pass for TensorMorph's TOSA optimizations.
 * This class orchestrates structural fusions (AI-guided) and structural/algebraic folding (deterministic).
 */
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
    this->advisorMode = other.advisorMode;
    this->minProfit = other.minProfit;
    this->debugAi = other.debugAi;
  }

  // --- AI Advisor Settings ---

  enum AdvisorMode { None, Memory, Compute };
  
  Option<AdvisorMode> advisorMode{*this, "ai-advisor",
    llvm::cl::desc("Select the AI advisor profile for targeted hardware optimization."),
    llvm::cl::init(None),
    llvm::cl::values(
      clEnumValN(None, "none", "Use standard greedy optimization (no AI)."),
      clEnumValN(Memory, "memory", "Use Memory-Bound hardware advisor."),
      clEnumValN(Compute, "compute", "Use Compute-Bound hardware advisor.")
    )};

  Option<float> minProfit{*this, "min-profit",
    llvm::cl::desc("Minimum predicted profit ratio required to apply an AI-guided fusion."),
    llvm::cl::init(1.2f)};

  Option<bool> debugAi{*this, "debug-ai", 
    llvm::cl::desc("Enable detailed diagnostic logging for AI veto decisions."), 
    llvm::cl::init(false)};

  // --- Pass Capability Toggles ---

  Option<bool> fuseFanout{*this, "fuse-fanout", 
    llvm::cl::desc("Enable cloning of anchors to facilitate fusion across multiple users."), 
    llvm::cl::init(true)};

  Option<bool> fuseActivations{*this, "fuse-activations", 
    llvm::cl::desc("Enable fusion of Clamp/ReLU operations into Convolution anchors."), 
    llvm::cl::init(true)};

  Option<bool> fuseTranspose{*this, "fuse-transpose", 
    llvm::cl::desc("Enable compile-time folding of Transpose operations into weight constants."), 
    llvm::cl::init(true)};

  Option<bool> fusePadding{*this, "fuse-padding", 
    llvm::cl::desc("Enable elimination of explicit Pad operations via Convolution attributes."), 
    llvm::cl::init(true)};

  Option<bool> fuseLinear{*this, "fuse-linear", 
    llvm::cl::desc("Enable folding of Add/Sub/Mul math into weights and bias (AI-guided)."), 
    llvm::cl::init(true)};

  Option<bool> foldAlgebraic{*this, "fold-algebraic", 
    llvm::cl::desc("Enable deterministic algebraic identities (x + 0, x * 1)."), 
    llvm::cl::init(true)};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    // Select the concrete advisor implementation based on CLI flags.
    std::unique_ptr<Advisor> activeAdvisor;

    if (advisorMode == Memory) {
      activeAdvisor.reset(new MemoryAdvisor());
    } else if (advisorMode == Compute) {
      activeAdvisor.reset(new ComputeAdvisor());
    }

    // Populate guided structural fusion patterns.
    tensormorph::populateTosaStructuralFusionPatterns(
        patterns, 
        activeAdvisor.get(),
        minProfit,
        fuseFanout, 
        fuseActivations, 
        fuseTranspose, 
        fusePadding, 
        fuseLinear,
        debugAi);
    
    // Populate anchor-less deterministic patterns.
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
/**
 * Global registration for the TosaOptimizations pass.
 */
void registerTosaOptimizationsPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<TosaOptimizationsPass>();
    });
}
}