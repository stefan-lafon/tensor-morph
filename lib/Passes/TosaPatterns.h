#ifndef TENSORMORPH_TOSA_PATTERNS_H
#define TENSORMORPH_TOSA_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensormorph {

/**
 * Registers patterns that anchor on tosa.conv2d.
 * @param fuseFanout: If true, allows cloning Conv ops to enable fusion with 
 * multiple downstream users.
 * @param fuseActivations: If true, allows folding Clamp/ReLU into Conv attributes.
 * @param fuseTranspose: If true, allows folding Transpose into a Conv op.
 */
void populateTosaStructuralFusionPatterns(RewritePatternSet &patterns, bool fuseFanout, bool fuseActivations, bool fuseTranspose);
/**
 * Registers patterns for general algebraic simplifications.
 */
void populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns);

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_TOSA_PATTERNS_H