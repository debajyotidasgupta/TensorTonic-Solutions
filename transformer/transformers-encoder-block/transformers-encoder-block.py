import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    z_score = (x - mu) / np.sqrt(std ** 2 + eps)
    return gamma * z_score + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, T, d_model = Q.shape
    d_head = d_model // num_heads

    Qh = (Q @ W_q).reshape(B, T, num_heads, d_head).transpose([0, 2, 1, 3])
    Kh = (K @ W_k).reshape(B, T, num_heads, d_head).transpose([0, 2, 1, 3])
    Vh = (V @ W_v).reshape(B, T, num_heads, d_head).transpose([0, 2, 1, 3])

    raw = (Qh @ Kh.transpose([0, 1, 3, 2])) / np.sqrt(d_head)
    scores = softmax(raw, axis=-1)
    attn = raw @ Vh
    return attn.transpose([0, 2, 1, 3]).reshape(B, T, d_model)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x1 = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x2 = layer_norm(x1 + x, gamma1, beta1)
    x3 = feed_forward(x2, W1, b1, W2, b2)
    x4 = layer_norm(x3 + x2, gamma2, beta2)
    return x4