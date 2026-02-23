import hydra
from omegaconf import DictConfig, OmegaConf
from utils.config_utils import prepare_config
from bench.run_ft import run

@hydra.main(version_base="1.3", config_path="./config", config_name="defaults.yaml")
def main(cfg: DictConfig):
    """Hydra entry-point – patches the config then delegates to bench.run.run."""
    
    cfg = prepare_config(cfg)
    
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        print("#" * 79, "\nStarting a benchmarking run with the following config:")
        print(OmegaConf.to_yaml(cfg))
        print("#" * 79)

    run(cfg)

if __name__ == "__main__":
    main()

## NOTE:
### How to get the number of iterations as seen in the progress bar?: number_of_training_iterations = (len(train_index_map)/batch_size) * num_epochs
### Number of eval iterations: number_of_eval_iterations = (len(eval_index_map)/batch_size) there are no epochs during eval
