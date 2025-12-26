# TensorMorph

TensorMorph is a specialized MLIR-based optimizer designed to "morph" standard TOSA (Tensor Operator Set Architecture) graphs into hardware-efficient execution kernels. It provides a modular sandbox for folding redundant spatial, algebraic, and linear operations into convolution anchors.

![Benchmark Results](./assets/benchmark_snapshot.png)

## Transformation Suite

### Structural Fusion (primary anchors: tosa.conv2d, tosa.depthwise_conv2d)
* **Linear Math Folding**: Greedily consumes `tosa.add`, `tosa.sub`, and `tosa.mul` layers following a convolution, baking their effects into weights and biases.
* **Padding Elimination**: Absorbs explicit `tosa.pad` operations into the internal convolution padding attribute, eliminating redundant memory copies.
* **Transpose Folding**: Bakes input spatial permutations into weight constants at compile-time, converting runtime overhead into zero-cost constant rearrangement.
* **Activation Injection**: Fuses `tosa.clamp` (ReLU/ReLU6) operations directly into convolution attributes.
* **Fan-out Cloning**: Automatically clones anchors when they have multiple consumers to preserve fusion potential and data locality.

### Algebraic & Pointwise Folding (anchor-less)
* **Pointwise Chains**: Collapses sequential operations (e.g., `(x + 5) + 10`) into a single combined operation.
* **Arithmetic Identities**: Detects and removes redundant operations such as $x + 0$ or $x \times 1$.

## Optimization Control Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `fuse-linear` | `true` | Toggles the folding of Add/Sub/Mul into convolutions. |
| `fuse-padding` | `true` | Toggles the absorption of explicit Pad ops. |
| `fuse-transpose` | `true` | Toggles weight-based transpose folding. |
| `fuse-activations` | `true` | Toggles ReLU/Clamp fusion. |
| `fuse-fanout` | `true` | Toggles cloning for multi-user nodes. |
| `fold-algebraic` | `true` | Toggles pure math identities (x+0, etc). |

## Build and Usage

```bash
mkdir build && cd build
cmake .. -GNinja
ninja
./tools/tensormorph-opt --tosa-opt input.mlir