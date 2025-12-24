#!/bin/bash
# TensorMorph Project Driver (tm-cli)

# Usage: ./tm-cli <command> [args]

COMMAND=$1
shift 

# Determine the absolute path to the directory where this script lives
# This makes the CLI work anywhere (Colab, Linux, Docker, etc.)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Define internal paths relative to the project root
BUILD_DIR="$PROJECT_ROOT/build"
BINARY="$BUILD_DIR/bin/tensormorph-opt"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

case $COMMAND in
  "build")
    echo "--- Initiating Build in: $BUILD_DIR ---"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit
    
    # We still use the environment's MLIR/LLVM install paths
    cmake .. -G Ninja \
        -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
        -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
        -DCMAKE_BUILD_TYPE=Release
    ninja
    ;;

  "ingest")
    # Dispatch to shared Python logic using the relative scripts directory
    python3 "$SCRIPTS_DIR/ingest.py" "$@"
    ;;

  "optimize")
    if [ ! -f "$BINARY" ]; then
        echo "Error: Optimizer binary not found at $BINARY. Run './tm-cli build' first."
        exit 1
    fi
    "$BINARY" "$@"
    ;;

  "benchmark")
    # Dispatch to shared shell logic using the relative scripts directory
    bash "$SCRIPTS_DIR/benchmark.sh" "$@"
    ;;

  *)
    echo "Usage: tm-cli {build|ingest|optimize|benchmark} [options]"
    exit 1
    ;;
esac