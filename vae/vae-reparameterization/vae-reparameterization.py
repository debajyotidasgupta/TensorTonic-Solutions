import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    return mu + np.exp(0.5 * log_var) * np.random.randn(*mu.shape)
    