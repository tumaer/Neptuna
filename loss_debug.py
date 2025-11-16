import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.config_utils import prepare_config
from utils.loss_utils import fetch_loss_metric

@hydra.main(version_base="1.3", config_path="./config", config_name="defaults.yaml")
def main(cfg):
    cfg = prepare_config(cfg)
    print(cfg)

    ## Load debug data 
    with open('./temporary/loss_debug_data/2D_AB_prediction.pkl', 'rb') as f:
        predictions = pickle.load(f)
    with open('./temporary/loss_debug_data/2D_AB_label.pkl', 'rb') as f:
        logits = pickle.load(f)

    # Fetch loss function
    loss_fn = fetch_loss_metric(cfg)

    # Compute loss
    model = None  # Not used in L2Loss
    loss = loss_fn(model, predictions, logits)

    print(f"Loss: {loss.item()}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Logits shape: {logits.shape}")
    


if __name__ == "__main__":
    main()