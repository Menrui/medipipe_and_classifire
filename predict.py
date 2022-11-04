import argparse
from pathlib import Path
from hydra import initialize_config_dir, compose


def parse_args():
    parser = argparse.ArgumentParser("Classifire")
    parser.add_argument("--data_source", type=str)
    parser.add_argument(
        "--log_dir", type=str, default="logs/experiments/runs/default/2022-11-01_16-54-20"
    )
    return parser.parse_args()


def main():
    from src.predict_pipline import predict

    args = parse_args()
    data_source = Path(args.data_source)
    log_dir = Path(args.log_dir)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(log_dir.joinpath("config").absolute()), job_name="app"
    ):
        cfg = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=["original_work_dir=${hydra.runtime.cwd}"],
        )
        predict(config=cfg, log_dir=log_dir, data_source=data_source)


if __name__ == "__main__":
    main()
