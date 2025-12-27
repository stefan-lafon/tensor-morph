#ifndef TENSORMORPH_TOSAPATTERNS_H
#define TENSORMORPH_TOSAPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensormorph {

/**
 * Populates a RewritePatternSet with structural optimization patterns.
 * * @param patterns The pattern set to populate.
 * @param advisorMode The active advisor policy (None, Memory, or Compute).
 * @param minProfit The threshold for AI-guided decisions.
 * @param fuseFanout Allows cloning for nodes with multiple users.
 * @param fuseActivations Fuses tosa.clamp into convolution.
 * @param fuseTranspose Folds tosa.transpose into weights.
 * @param fusePadding Absorbs explicit tosa.pad into convolution.
 * @param fuseLinear Folds Add/Sub/Mul into weights/bias.
 */
void populateTosaStructuralFusionPatterns(
    RewritePatternSet &patterns, 
    int advisorMode,
    float minProfit,
    bool fuseFanout, 
    bool fuseActivations, 
    bool fuseTranspose,
    bool fusePadding,
    bool fuseLinear);

/**
 * Populates a RewritePatternSet with pure algebraic folding patterns.
 */
void populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns);

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_TOSAPATTERNS_H