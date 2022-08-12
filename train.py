import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="train.yaml", version_base="1.1")
def main(config: DictConfig):

    from src.training_pipeline import train

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
