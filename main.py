import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.3", config_path="./config", config_name="defaults.yaml")
def main(cfg: DictConfig):
    print("#" * 79, "\nStarting a benchmarking run with the following config:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)


if __name__=="__main__":
    main()