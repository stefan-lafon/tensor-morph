#!/bin/bash
# Shared Benchmarking Logic.

MLIR_FILE=$1
LOWER_SCRIPT="/content/tensormorph_local/lower.sh"
RUNNER="/usr/lib/llvm-18/bin/mlir-cpu-runner"

# Safety check
if [ ! -f "$MLIR_FILE" ]; then
    echo "Error: MLIR file not found: $MLIR_FILE"
    exit 1
fi

# 1. Lower TOSA to LLVM Dialect
# We use a temporary file for the lowered output to keep the workspace clean
EXEC_LLVM="/tmp/exec_$(date +%s).mlir"

echo "--- Lowering to LLVM ---"
$LOWER_SCRIPT "$MLIR_FILE" "$EXEC_LLVM" > /dev/null

if [ $? -ne 0 ]; then
    echo "Error: Lowering failed."
    exit 1
fi

# 2. Execute via MLIR CPU Runner
# This invokes the 'benchmark_entry' function defined in your model
echo "--- Starting Execution ---"
$RUNNER "$EXEC_LLVM" \
    -e benchmark_entry -entry-point-result=void \
    -shared-libs=/usr/lib/llvm-18/lib/libmlir_c_runner_utils.so

# Clean up
rm "$EXEC_LLVM"