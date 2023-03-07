import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="test.yaml", version_base="1.1")
def main(config: DictConfig):

    from src.testing_pipeline import test

    # Train model
    return test(config)


if __name__ == "__main__":
    main()
