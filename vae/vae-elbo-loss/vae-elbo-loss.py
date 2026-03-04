import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    mse = np.mean((x - x_recon) ** 2)
    kl_div = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
    return {
        "total": mse + kl_div,
        "recon": mse,
        "kl": kl_div,
    }