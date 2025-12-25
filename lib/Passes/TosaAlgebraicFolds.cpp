#include "TosaPatterns.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

void mlir::tensormorph::populateTosaAlgebraicFoldingPatterns(RewritePatternSet &patterns) {
  // We will add non-anchored pointwise patterns here in the next step.
}