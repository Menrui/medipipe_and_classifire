from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from tqdm import tqdm
from src import utils
from src.datamodules.components.prediction_detector_dataset import PredictionDetectorDataset

from src.utils.logger import get_stream_logger


log = get_stream_logger(__name__)


def predict(config: DictConfig, log_dir: Path, data_source: Path) -> None:
    log.info("Start prediction...")
    if config.get("seed"):
        utils.seed_everything(config.seed)

    ckpt_dir = log_dir.joinpath("checkpoints")
    ckpt_path = ckpt_dir.joinpath("best.pth")

    log.info(f"Instantiating dataset <{config.datamodule._target_}>")
    category = pd.read_csv(log_dir.joinpath("category.txt"), header=None, index_col=None).values[0]
    if config.model.get("num_classes") == -1:
        config.model.num_classes = len(category)

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    checkpoint = torch.load(str(ckpt_path))
    model.load_state_dict(checkpoint["state_dict"]),

    if config.get("gpu"):
        device = "cuda"
        model = model.cuda()
    else:
        device = "cpu"

    dataset = PredictionDetectorDataset(data_source=data_source)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    log.info("Start prediction loop")
    model.eval()
    preds = []
    pred_categorys = []
    filepaths = []

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch[0].to(device)
            filepath = batch[1]

            output = model(images)
            pred = torch.argmax(output, dim=1)
            pred_category = [category[_p.item()] for _p in pred]

            preds.append(pred)
            pred_categorys.extend(pred_category)
            filepaths.extend(filepath)

    df = pd.DataFrame([filepaths, pred_categorys], index=["filepath", "category"]).T
    data_dir = data_source if data_source.is_dir() else data_source.parent
    df.to_csv(data_dir.joinpath("classifire_prediction_result.csv"))
