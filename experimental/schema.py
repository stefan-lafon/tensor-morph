# experimental/schema.py

# This is the single source of truth for the TensorMorph feature vector.
# Adding or removing features here will update all notebooks.
FEATURES = [
    'in_h', 'in_w', 'in_c', 'out_c', 
    'kernel', 'stride', 'is_dw', 'chain_len', 'has_act'
]

# The target variable the AI is trying to predict.
TARGET = 'profit_ratio'