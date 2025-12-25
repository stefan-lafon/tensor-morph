#ifndef TENSORMORPH_TOSA_PATTERNS_H
#define TENSORMORPH_TOSA_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensormorph {

/**
 * Registers patterns that anchor on tosa.conv2d to fold following
 * elementwise operations (Add, Sub, Mul, Clamp) into weights/attributes.
 */
void populateTosaStructuralFusionPatterns(RewritePatternSet &patterns);

/**
 * Registers patterns for general algebraic simplifications that do 
 * not require a convolution anchor (e.g., Add + Add folding).
 */
void populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns);

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_TOSA_PATTERNS_H