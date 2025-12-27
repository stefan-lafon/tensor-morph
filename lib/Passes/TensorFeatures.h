#ifndef TENSORMORPH_TENSORFEATURES_H
#define TENSORMORPH_TENSORFEATURES_H

#include "mlir/IR/Operation.h"
#include <vector>

namespace mlir {
namespace tensormorph {

/**
 * Data structure representing the 9-feature vector for the AI Advisor.
 * These must align perfectly with the indices in schema.py.
 */
struct TensorFeatures {
  float in_h = 0.0f;     // [0]
  float in_w = 0.0f;     // [1]
  float in_c = 0.0f;     // [2]
  float out_c = 0.0f;    // [3]
  float kernel = 0.0f;   // [4]
  float stride = 0.0f;   // [5]
  float is_dw = 0.0f;    // [6] (1.0 for Depthwise, 0.0 for Conv2D)
  float chain_len = 0.0f; // [7] (Count of Add/Sub/Mul)
  float has_act = 0.0f;   // [8] (1.0 if chain ends in Clamp)

  // Returns the features as a flat vector for model inference.
  std::vector<float> toVector() const {
    return {in_h, in_w, in_c, out_c, kernel, stride, is_dw, chain_len, has_act};
  }
};

/**
 * The Scout: Extracts features from a Convolution anchor by walking the IR.
 * This handles the look-ahead logic to calculate the full chain depth.
 */
TensorFeatures extractFeatures(Operation *op);

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_TENSORFEATURES_H