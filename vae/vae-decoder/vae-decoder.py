import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    B, latent_dim = z.shape

    W1 = np.random.randn(latent_dim, latent_dim)
    b1 = np.zeros((latent_dim,), dtype=z.dtype)
    h = z @ W1  + b1

    W2 = np.random.randn(latent_dim, output_dim)
    b2 = np.zeros((output_dim,), dtype=z.dtype)
    o = h @ W2  + b2

    return o