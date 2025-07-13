import numpy as np

def l1_error(preds, targets):
    """
    Compute the L1 error (Mean Absolute Error) between predictions and targets.

    This function calculates the Mean Absolute Error (MAE) across all dimensions
    of the input arrays. The inputs are reshaped to flatten spatial dimensions
    while preserving the batch structure for efficient computation.

    Parameters
    ----------
    preds : numpy.ndarray
        Predicted values with shape (num_samples, (num_eval_rollouts_plus_one*
        label_seq_length), num_channels, *spatial_dims).
    targets : numpy.ndarray
        Ground truth values with the same shape as preds.

    Returns
    -------
    float
        The L1 error (MAE) computed as the mean of absolute differences
        between predictions and targets across all elements.

    Notes
    -----
    The L1 error is computed as:
    L1 = mean(|preds - targets|)

    Both input arrays are reshaped to (num_samples, flattened_features, -1)
    where flattened_features combines the temporal, channel, and sequence
    dimensions for efficient computation.
    """

    diff = preds - targets
    l1_error = np.mean(np.abs(diff))
    return l1_error


def l2_error(preds, targets):
    """
    Compute the L2 error (Root Mean Squared Error) between predictions and targets.

    This function calculates the Root Mean Squared Error (RMSE) across all
    dimensions of the input arrays. The inputs are reshaped to flatten spatial
    dimensions while preserving the batch structure for efficient computation.

    Parameters
    ----------
    preds : numpy.ndarray
        Predicted values with shape (num_samples, (num_eval_rollouts_plus_one*
        label_seq_length), num_channels, *spatial_dims).
    targets : numpy.ndarray
        Ground truth values with the same shape as preds.

    Returns
    -------
    float
        The L2 error (RMSE) computed as the square root of the mean squared
        differences between predictions and targets across all elements.

    Notes
    -----
    The L2 error is computed as:
    L2 = sqrt(mean((preds - targets)^2))

    Both input arrays are reshaped to (num_samples, flattened_features, -1)
    where flattened_features combines the temporal, channel, and sequence
    dimensions for efficient computation.
    """
    diff = preds - targets
    l2_error = np.mean((diff) ** 2) ** 0.5
    return l2_error
