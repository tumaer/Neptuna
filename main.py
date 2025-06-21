from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from bench.config_utils import prepare_config
from bench.run import run


@hydra.main(version_base="1.3", config_path="./config", config_name="defaults.yaml")
def main(cfg: DictConfig):  # noqa: D401
    """Hydra entry-point – patches the config then delegates to bench.run.run."""
    print("#" * 79, "\nStarting a benchmarking run with the following config:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    cfg = prepare_config(cfg)
    run(cfg)

if __name__ == "__main__":
    main()


## NOTE: 
### How to get the number of iterations as seen in the progress bar?: number_of_training_iterations = (len(train_index_map)/batch_size) * num_epochs
### Number of eval iterations: number_of_eval_iterations = (len(eval_index_map)/batch_size) there are no epochs during eval 

##TODO:
# 4 Inference code
# most of the training arguments to be passed from the config file
#TODO: See how to increase the n_jobs inside integration_utils.py > run_hp_search_optuna() (it is by default 1)
#why is it too slow at the end of the evaluati