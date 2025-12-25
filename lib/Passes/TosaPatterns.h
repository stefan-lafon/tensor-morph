#ifndef TENSORMORPH_TOSAPATTERNS_H
#define TENSORMORPH_TOSAPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensormorph {

/**
 * @brief Populates a RewritePatternSet with structural optimization patterns.
 * * Structural patterns target "anchor" operations (like tosa.conv2d) and attempt
 * to absorb surrounding operations into the anchor's attributes or constants.
 *
 * @param patterns The pattern set to populate.
 * @param fuseFanout If true, allows cloning operations to enable fusion for nodes with multiple users.
 * @param fuseActivations If true, enables fusing tosa.clamp (ReLU) into convolution attributes.
 * @param fuseTranspose If true, enables folding tosa.transpose into convolution weights.
 * @param fusePadding If true, enables absorbing explicit tosa.pad operations into convolution padding.
 * @param fuseLinear If true, enables the "Eating Machine" (folding Add/Sub/Mul into weights/bias).
 */
void populateTosaStructuralFusionPatterns(
    RewritePatternSet &patterns, 
    bool fuseFanout, 
    bool fuseActivations, 
    bool fuseTranspose,
    bool fusePadding,
    bool fuseLinear);

/**
 * @brief Populates a RewritePatternSet with pure algebraic and pointwise folding patterns.
 * * Algebraic patterns focus on math-to-math simplifications that do not necessarily 
 * require a convolution anchor.
 * * Examples include:
 * - Arithmetic Identities: x + 0 => x, x * 1 => x
 * - Pointwise Chains: (x + C1) + C2 => x + (C1 + C2)
 *
 * @param patterns The pattern set to populate.
 */
void populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns);

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_TOSAPATTERNS_H