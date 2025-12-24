# TensorMorph

TensorMorph is a command-line utility built on MLIR (LLVM) for experimenting with graph-level optimizations. The tool provides a sandbox for implementing operator fusion patterns and graph restructuring using the `PatternRewriter` infrastructure.

Currently, the project supports a "dual-path" workflow: it can ingest and lower official TOSA dialect operations for performance analysis, while maintaining support for experimental prototyping via custom operation matching.

## Project Structure

* `tools/`: Contains `tensormorph-opt`, the main driver that registers TOSA, SCF, and Linalg dialects.
* `lib/Passes/`: Core transformation logic, including the current TOSA-based fusion patterns.
* `TensorMorph-Lab.ipynb`: A Colab notebook used for remote environment setup, build orchestration, and performance benchmarking.
* `tests/`: A regression suite of MLIR files verified via `FileCheck`.

## Supported Transformation Patterns

The optimizer currently implements a greedy forward-pass to consolidate pointwise operations into anchor nodes.

### Structural Fusion
* **Conv2D + Elementwise Folding**: Consumes `tosa.add`, `tosa.sub`, and `tosa.mul` following a convolution. It re-calculates the weight and bias constants channel-wise to eliminate the extra operations.
* **Activation Injection**: Fuses `tosa.clamp` into preceding `tosa.conv2d` operations by materializing `fused_activation`, `clamp_min`, and `clamp_max` as IR attributes for backend kernel generation.

### Algebraic Identities
* **Additive Identity**: Automatically folds $x + 0.0 \rightarrow x$.
* **Multiplicative Identity**: Automatically folds $x \times 1.0 \rightarrow x$.
* **Subtractive Identity**: Automatically folds $x - 0.0 \rightarrow x$.

## Current State

The toolchain is capable of:
1. Identifying and fusing specific TOSA operation sequences (e.g., Conv2D + Add + Clamp).
2. Lowering fused IR through a multi-stage pipeline (TOSA -> Linalg -> LLVM).
3. Executing and timing the resulting machine code on the host CPU.