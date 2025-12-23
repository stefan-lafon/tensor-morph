# TensorMorph

TensorMorph is a command-line utility built on MLIR (LLVM) for experimenting with graph-level optimizations. The tool provides a sandbox for implementing operator fusion patterns and graph restructuring using the `PatternRewriter` infrastructure.

Currently, the project supports a "dual-path" workflow: it can ingest and lower official TOSA dialect operations for performance analysis, while maintaining support for experimental prototyping via custom operation matching.

## Project Structure

* `tools/`: Contains `tensormorph-opt`, the main driver that registers TOSA, SCF, and Linalg dialects.
* `lib/Passes/`: Core transformation logic, including the current TOSA-based fusion patterns.
* `include/`: Header definitions and pass declarations for out-of-tree builds.
* `benchmark/`: MLIR wrapper scripts used to measure execution latency via `mlir-cpu-runner`.

## Current State

The toolchain is capable of:
1. Identifying and fusing specific TOSA operation sequences (e.g., Conv2D + Add).
2. Lowering fused IR through a multi-stage pipeline (TOSA -> Linalg -> LLVM).
3. Executing and timing the resulting machine code on the host CPU.