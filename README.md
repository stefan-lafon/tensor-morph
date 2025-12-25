# TensorMorph

TensorMorph is a command-line utility built on MLIR (LLVM) for experimenting with graph-level optimizations. The tool provides a sandbox for implementing operator fusion patterns and graph restructuring using the `PatternRewriter` infrastructure.

Currently, the project focuses on high-performance TOSA-to-TOSA transformations, specifically targeting the reduction of memory bandwidth overhead through greedy operation fusion.

## Project Structure

* `tools/`: Contains `tensormorph-opt`, the main driver registering TOSA, SCF, and Linalg dialects.
* `lib/Passes/`: Core transformation logic, modularized into Structural Fusion and Algebraic Folding libraries.
* `tests/`: A regression suite of MLIR files verified via `FileCheck`.
* `TensorMorph-Lab.ipynb`: The primary development environment for orchestration and benchmarking.

## Supported Transformation Patterns

The optimizer implements a greedy driver to consolidate pointwise operations into anchor nodes or collapse them into simpler mathematical identities.

### Structural Fusion (Anchor: tosa.conv2d)
* **Linear Math Folding**: Greedily consumes `tosa.add`, `tosa.sub`, and `tosa.mul` chains following a convolution. It re-calculates weight and bias constants at compile-time to eliminate runtime overhead.
* **Activation Injection**: Fuses `tosa.clamp` (ReLU/ReLU6) into preceding convolutions by materializing `fused_activation` and clamp range attributes.

### Algebraic & Pointwise Folding
* **Pointwise Chains**: Collapses back-to-back additions or multiplications—e.g., $(x + C_1) + C_2 \rightarrow x + (C_1 + C_2)$.
* **Arithmetic Identities**: Automatically eliminates no-op patterns:
    * $x + 0.0 \rightarrow x$
    * $x \times 1.0 \rightarrow x$
    * $x - 0.0 \rightarrow x$

## Optimization Control Flags

The tool provides granular control over the optimization pipeline via command-line flags passed to `--tosa-opt`:

| Flag | Default | Description |
| :--- | :--- | :--- |
| `fuse-activations` | `true` | Enables/Disables folding of `tosa.clamp` into `tosa.conv2d`. |
| `fold-algebraic` | `true` | Enables/Disables pure algebraic identities and pointwise chains. |
| `fuse-fanout` | `true` | Allows cloning operations to enable fusion across nodes with multiple users. |

## Current State

The toolchain effectively collapses complex sequences (e.g., `Conv2D -> Mul -> Add -> Clamp`) into a single highly-specialized operation. The resulting IR can be lowered through a standard `TOSA -> Linalg -> LLVM` pipeline for execution on host hardware.