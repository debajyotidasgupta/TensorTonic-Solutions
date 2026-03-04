import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """
    B, input_dim = x.shape
    
    W_mu = np.random.randn(input_dim, latent_dim)
    b_mu = np.zeros((latent_dim,), dtype=x.dtype)
    mu = x @ W_mu + b_mu
    
    W_logvar = np.random.randn(input_dim, latent_dim)
    b_logvar = np.zeros((latent_dim,), dtype=x.dtype)
    logvar = x @ W_logvar + b_logvar

    return mu, logvar