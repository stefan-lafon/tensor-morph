# Feature names used for training and C++ inference.
FEATURES = [
    'in_h',      # Input tensor height.
    'in_w',      # Input tensor width.
    'in_c',      # Input channels (depth).
    'out_c',     # Output channels (filters).
    'kernel',    # Square kernel size (e.g., 3 for 3x3).
    'stride',    # Convolution stride value.
    'is_dw',     # Flag: 1 if depthwise, 0 if standard.
    'chain_len', # Number of subsequent pointwise ops.
    'has_act'    # Flag: 1 if fused activation exists.
]

# The value we are trying to predict.
TARGET = 'profit_ratio'