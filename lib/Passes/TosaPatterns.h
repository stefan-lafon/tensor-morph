#ifndef TENSORMORPH_TOSAPATTERNS_H
#define TENSORMORPH_TOSAPATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "experimental/Advisor.h"

namespace mlir {
namespace tensormorph {

/**
 * Populates a RewritePatternSet with structural optimization patterns.
 * * @param patterns The pattern set to populate.
 * @param advisor The AI Advisor instance to consult for fusion decisions.
 * @param minProfit The minimum predicted profit ratio required to apply a fusion.
 * @param fuseFanout Allow cloning operations to enable fusion across multiple users.
 * @param fuseActivations Enable fusion of Clamp/ReLU operations into Convolution.
 * @param fuseTranspose Enable folding of Transpose operations into Convolution weights.
 * @param fusePadding Enable elimination of explicit Pad operations into Convolution.
 * @param fuseLinear Enable folding of Add/Sub/Mul math into Convolution weights and bias.
 * @param debugAi If true, output diagnostic logs for AI decision making to stderr.
 */
void populateTosaStructuralFusionPatterns(
    RewritePatternSet &patterns, 
    Advisor *advisor,
    float minProfit,
    bool fuseFanout, 
    bool fuseActivations, 
    bool fuseTranspose,
    bool fusePadding,
    bool fuseLinear,
    bool debugAi);

/**
 * Populates a RewritePatternSet with pure algebraic folding patterns.
 * These are deterministic identities (e.g., x + 0, x * 1) that do not require AI guidance.
 */
void populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns);

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_TOSAPATTERNS_H