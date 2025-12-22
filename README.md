# TensorMorph

TensorMorph is a command-line utility built on MLIR (LLVM) designed for experimenting with graph-level optimizations in AI models. The tool focuses on implementing common operator fusion patterns and graph restructuring to improve inference efficiency, particularly for mobile-centric workloads.

The project is structured as an out-of-tree MLIR tool, utilizing the `PatternRewriter` infrastructure to apply transformations to TFLite and TOSA dialects.

## Project Structure

* `tools/`: Main entry point for the `tensormorph-opt` CLI.
* `lib/Passes/`: Implementation of optimization passes (e.g., Conv-BatchNorm folding).
* `test/`: Regression tests using `.mlir` snippets and `FileCheck`.
* `include/`: Header definitions and pass declarations.
