"""
TensorMorph Model Ingestion Utility
-----------------------------------
This script serves as the bridge between high-level machine learning frameworks 
(like TensorFlow/Keras) and the TensorMorph MLIR-based optimization pipeline.

It performs two primary roles:
1. Model Acquisition: Loads existing models or generates standard test architectures 
   (like 'sample_mnist') via a factory pattern.
2. Dialect Conversion: Freezes the model graph and lowers it into the TensorFlow 
   Executor dialect within the MLIR ecosystem.

Usage via CLI:
    ./tm-cli ingest --model <path_or_keyword> --output <filename.mlir>
"""

import os
import argparse
import tensorflow as tf

def get_mnist_model():
    """Builds a small Keras model for pipeline testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model

def main():
    parser = argparse.ArgumentParser(description="TensorMorph Model Ingestor")
    parser.add_argument("--model", type=str, required=True, help="Path to model or 'sample_mnist'")
    parser.add_argument("--output", type=str, default="test.mlir", help="Output MLIR filename")
    args = parser.parse_args()

    # 1. Acquire Model
    if args.model == "sample_mnist":
        print(f"--- Generating internal factory model: {args.model} ---")
        model = get_mnist_model()
    else:
        print(f"--- Loading external model: {args.model} ---")
        model = tf.keras.models.load_model(args.model)

    # 2. Convert to MLIR
    print(f"--- Converting to MLIR -> {args.output} ---")
    
    # Freeze the model into a concrete function to capture the graph
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )
    
    try:
        # We use an empty pass pipeline here to produce raw TF dialect.
        # This avoids scheduling errors during initial ingestion.
        mlir_graph = tf.mlir.experimental.convert_graph_def(
            concrete_func.graph.as_graph_def(),
            pass_pipeline='' 
        )
        
        with open(args.output, "w") as f:
            f.write(mlir_graph)
        print(f"Successfully ingested {args.model} to {args.output}")
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    main()