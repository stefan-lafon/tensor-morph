#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

// Explicit Dialect Includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
    void registerFoldBatchNormPass();
}

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    
    mlir::DialectRegistry registry;
    
    // Only register what we actually use
    registry.insert<mlir::func::FuncDialect,
                   mlir::linalg::LinalgDialect,
                   mlir::arith::ArithDialect,
                   mlir::tensor::TensorDialect,
                   mlir::memref::MemRefDialect>();

    // Register our custom TensorMorph pass
    mlir::registerFoldBatchNormPass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "TensorMorph Optimizer\n", registry)
    );
}