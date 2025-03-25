import numpy as np


def lp_error(preds: np.ndarray, targets: np.ndarray, p=1):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    return np.sum(errors, axis=-1) ** (1 / p)
