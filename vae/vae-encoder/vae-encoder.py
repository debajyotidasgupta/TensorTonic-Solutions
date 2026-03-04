import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """
    B, input_dim = x.shape

    W = np.random.randn(input_dim, latent_dim)
    b = np.zeros((latent_dim,), dtype=x.dtype)
    h = x @ W  + b
    
    W_mu = np.random.randn(latent_dim, latent_dim)
    b_mu = np.zeros((latent_dim,), dtype=x.dtype)
    mu = h @ W_mu + b_mu
    
    W_logvar = np.random.randn(latent_dim, latent_dim)
    b_logvar = np.zeros((latent_dim,), dtype=x.dtype)
    logvar = h @ W_logvar + b_logvar

    return mu, logvar