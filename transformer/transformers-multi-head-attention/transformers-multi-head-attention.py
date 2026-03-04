import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
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