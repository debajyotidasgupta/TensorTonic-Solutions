import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length, dtype=np.float32)[:, None] # (L, 1)
    i = np.arange(d_model, dtype=np.float32)[None, :]      # (1, D)
    ang_rates = 1 / np.power(10000.0, 2 * (i // 2) / d_model)
    ang = pos @ ang_rates                                  # (L, D)
    pe = np.zeros((seq_length, d_model), dtype=np.float32) # (L, D)
    pe[:, 0::2] = np.sin(ang[:, 0::2]) 
    pe[:, 1::2] = np.cos(ang[:, 1::2]) 
    return pe