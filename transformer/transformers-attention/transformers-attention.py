import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    dk = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
    return F.softmax(scores, dim=-1) @ V