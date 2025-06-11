import numpy as np
from transformers.trainer_utils import EvalPrediction

def l1_error(preds, targets): #MAE loss
    num_samples, num_eval_rollouts_plus_one, label_seq_length, num_channels, *spatial_dims = preds.shape
    preds = preds.reshape(num_samples, num_eval_rollouts_plus_one*label_seq_length*num_channels, -1) #shape: (270,9,160)
    targets = targets.reshape(num_samples, num_eval_rollouts_plus_one*label_seq_length*num_channels, -1) #shape: (270,9,160)
    diff = (preds - targets)
    l1_error = np.mean(np.abs(diff))
    return l1_error
    #270 is the number of accumulated samples == len(eval_dataset)

def l2_error(preds, targets): #RMSE loss
    num_samples, num_eval_rollouts_plus_one, label_seq_length, num_channels, *spatial_dims = preds.shape
    preds = preds.reshape(num_samples, num_eval_rollouts_plus_one*label_seq_length*num_channels, -1) #shape: (270,9,160)
    targets = targets.reshape(num_samples, num_eval_rollouts_plus_one*label_seq_length*num_channels, -1) #shape: (270,9,160)
    diff = (preds - targets)
    l2_error = np.mean(np.abs(diff) ** 2) ** 0.5
    return l2_error