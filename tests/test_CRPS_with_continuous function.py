import numpy as np
from scipy.special import erf

def crps_gaussian(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """CRPS for Gaussian distribution.

    Args:
        x (np.ndarray): Ground truth data of size (n_samples, n_features)
        mu (np.ndarray): Mean of size (n_samples, n_features)
        std (np.ndarray): Standard deviation of size (n_samples, n_features)

    Returns:
        crps (np.ndarray): CRPS of size (n_samples, n_features)
    """
    sqrtPi = np.sqrt(np.pi)
    z = (x - mu) / std
    phi = np.exp(-z ** 2 / 2) / (np.sqrt(2) * sqrtPi) #standard normal pdf
    crps = std * (z * erf(z / np.sqrt(2)) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005
    return crps



def pdf_gaussian(z, mu, std):
    'Create gaussian PDF'
    sqrtPi = np.sqrt(np.pi)
    phi = np.exp(-z ** 2 / 2) / (np.sqrt(2) * sqrtPi) #standard normal pdf
    return phi


def crps_continuous(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    sqrtPi = np.sqrt(np.pi)
    z = (x - mu) / std
    phi = pdf_gaussian(z, mu, std)
    crps = std * (z * erf(z / np.sqrt(2)) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005
