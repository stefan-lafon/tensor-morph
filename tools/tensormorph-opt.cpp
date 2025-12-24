#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

// Dialect Includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir {
    // Forward declaration for the updated optimization pass registration
    void registerTosaOptimizationsPass();
}

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    
    mlir::DialectRegistry registry;
    
    // Registering the full suite for TFLite -> TOSA -> LLVM flow
    registry.insert<mlir::func::FuncDialect,
                   mlir::linalg::LinalgDialect,
                   mlir::arith::ArithDialect,
                   mlir::tosa::TosaDialect,
                   mlir::scf::SCFDialect,
                   mlir::tensor::TensorDialect,
                   mlir::bufferization::BufferizationDialect,
                   mlir::math::MathDialect,
                   mlir::memref::MemRefDialect>();

    // Register our reorganized TensorMorph logic
    mlir::registerTosaOptimizationsPass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "TensorMorph Compiler\n", registry)
    );
}