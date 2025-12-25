# TensorMorph

TensorMorph is a command-line utility built on MLIR (LLVM) for experimenting with graph-level optimizations. The tool provides a sandbox for implementing operator fusion patterns and graph restructuring using the `PatternRewriter` infrastructure.

## Supported Transformation Patterns

### Structural Fusion (Anchor: tosa.conv2d)
* **Linear Math Folding**: Greedily consumes `tosa.add`, `tosa.sub`, and `tosa.mul` chains following a convolution.
* **Activation Injection**: Fuses `tosa.clamp` (ReLU/ReLU6) into convolution attributes.
* **Transpose Folding**: Eliminates `tosa.transpose` operations by permuting convolution weights at compile-time. This effectively converts runtime data movement into static constant rearrangement.
* **Fan-out Cloning**: Enables fusion even when a convolution has multiple consumers. The optimizer will "clone" the convolution for each branch, allowing local fusions to proceed and improving data locality at the expense of redundant computation.

### Algebraic & Pointwise Folding
* **Pointwise Chains**: Collapses back-to-back additions or multiplications.
* **Arithmetic Identities**: Eliminates $x + 0$, $x \times 1$, and $x - 0$.

## Optimization Control Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `fuse-activations` | `true` | Fuses `tosa.clamp` into `tosa.conv2d`. |
| `fuse-transpose` | `true` | Folds `tosa.transpose` into convolution weights. |
| `fuse-fanout` | `true` | Allows cloning operations to enable fusion across multiple users. |
| `fold-algebraic` | `true` | Enables pure algebraic identities (Add+Add, etc). |

## Build and Usage
```bash
mkdir build && cd build
cmake .. -GNinja
ninja
./tools/tensormorph-opt --tosa-opt input.mlir