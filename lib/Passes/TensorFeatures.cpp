#include "TensorFeatures.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tensormorph;

TensorFeatures mlir::tensormorph::extractFeatures(Operation *op) {
  TensorFeatures f;
  
  // 1. Identify the anchor type.
  auto convOp = dyn_cast<tosa::Conv2DOp>(op);
  auto dwOp = dyn_cast<tosa::DepthwiseConv2DOp>(op);
  
  if (!convOp && !dwOp) return f;

  f.is_dw = dwOp ? 1.0f : 0.0f;

  // 2. Extract Input Shapes (NHWC).
  Value input = op->getOperand(0);
  if (auto type = input.getType().dyn_cast<RankedTensorType>()) {
    auto shape = type.getShape();
    if (shape.size() == 4) {
      f.in_h = static_cast<float>(shape[1]);
      f.in_w = static_cast<float>(shape[2]);
      f.in_c = static_cast<float>(shape[3]);
    }
  }

  // 3. Extract Output Channel Count.
  if (auto outType = op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
    auto outShape = outType.getShape();
    if (outShape.size() == 4) {
      f.out_c = static_cast<float>(outShape[3]);
    }
  }

  // 4. Extract Kernel and Stride from attributes.
  if (convOp) {
    auto strideAttr = convOp.getStride();
    f.stride = static_cast<float>(strideAttr[0]);
    
    auto weightType = convOp.getWeight().getType().cast<RankedTensorType>();
    f.kernel = static_cast<float>(weightType.getShape()[1]); 
  } else if (dwOp) {
    auto strideAttr = dwOp.getStride();
    f.stride = static_cast<float>(strideAttr[0]);
    
    auto weightType = dwOp.getWeight().getType().cast<RankedTensorType>();
    f.kernel = static_cast<float>(weightType.getShape()[0]);
  }

  // 5. Calculate chain_len and has_act via look-ahead walk.
  int chainCount = 0;
  Operation *current = op;

  while (current->getResult(0).hasOneUse()) {
    Operation *next = *current->getResult(0).getUsers().begin();
    
    if (isa<tosa::AddOp, tosa::SubOp, tosa::MulOp>(next)) {
      chainCount++;
      current = next;
    } else if (isa<tosa::ClampOp>(next)) {
      f.has_act = 1.0f;
      break; // Clamp is always the terminal point.
    } else {
      break; // Non-fusible operation encountered.
    }
  }

  f.chain_len = static_cast<float>(chainCount);

  return f;
}