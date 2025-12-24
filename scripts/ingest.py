# @title 1. Shared Ingestion Logic (scripts/ingest.py)
import tensorflow as tf
import argparse
import os
from tensorflow.python.compiler.mlir import mlir_convert

def lower_to_tosa(model_path, output_path):
    """Lowers a SavedModel or Keras model to TOSA MLIR."""
    print(f"--- Loading Model from: {model_path} ---")
    
    # Load the model using Keras
    model = tf.keras.models.load_model(model_path)
    
    print(f"--- Converting to TOSA IR (Official Bridge) ---")
    # This uses the same logic as the tf-tfl-translate tool
    tosa_ir = mlir_convert(
        model,
        pass_pipeline='tf-standard-pipeline,tf-to-tosa-pipeline',
        show_debug_info=False
    )
    
    # Write the IR to the specified output
    with open(output_path, 'w') as f:
        f.write(tosa_ir)
    print(f"--- Conversion Successful: {output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorMorph Model Ingestor")
    parser.add_argument("--input", required=True, help="Path to input model")
    parser.add_argument("--output", required=True, help="Path to save output .mlir file")
    
    args = parser.parse_args()
    lower_to_tosa(args.input, args.output)