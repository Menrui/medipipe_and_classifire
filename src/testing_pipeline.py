import os
from pathlib import Path
from typing import Literal, Optional, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from torchmetrics import Accuracy
from tqdm import tqdm

from src import utils
from src.utils.logger import get_logger

log = get_logger(__name__)


def test(config: DictConfig) -> Optional[float]:
    log.info("Start testining!")
    if config.get("seed"):
        utils.seed_everything(config.seed)

    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(
            hydra.utils.get_original_cwd(), config.ckpt_path
        )

    log.info(f"Instantiating dataset <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule)
    if config.model.get("num_classes") == -1:
        config.model.num_classes = datamodule.num_classes

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    if config.get("gpu"):
        device = "cuda"
        model = model.cuda()
    else:
        device = "cpu"

    test_loader = datamodule.test_dataloader()
    ckpt_path = config.ckpt_path

    checkpoint = torch.load(str(ckpt_path))
    test_loader = datamodule.test_dataloader()
    model.load_state_dict(checkpoint["state_dict"]),
    acc, preds, targets = test_loop(
        model=model,
        loader=test_loader,
        ckpt_path=None,
        device=device,
        num_classes=datamodule.num_classes,
    )
    log.info(f"test acc : {acc}")
    class_tags = list(datamodule.data_test.class_to_idx.keys())
    calcurate_cls_score(
        preds=preds,
        targets=targets,
        classes=class_tags,
        savepath=Path("confusion_matrix.png"),
    )

    acc_cal = Accuracy(
        top_k=1, task="multiclass", num_classes=datamodule.num_classes, average=None
    )
    classwise_acc = acc_cal(preds, targets)
    log.info(
        f"class wise acc | {' | '.join([f'{tag}:{classwise_acc[i]:.3f}' for i, tag in enumerate(class_tags)])}"
    )

    # print(datamodule.data_test.data_paths.shape)
    # print(datamodule.data_test.labels.shape)
    # print(targets.shape)
    # print(preds.shape)

    result_df = pd.DataFrame(
        [
            datamodule.data_test.data_paths,
            # datamodule.data_test.labels,
            targets.numpy(),
            preds.numpy(),
        ],
        index=["PATH", "TARGETS", "PREDs"],
    ).T
    result_df.to_csv("result.csv", index=None)

    with open("category.txt", "w") as o:
        print(*list(datamodule.data_test.class_to_idx.keys()), file=o, sep=",")


def test_loop(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    ckpt_path: Optional[Path],
    device: Literal["cuda", "cpu"],
    num_classes: int,
):
    if ckpt_path is not None:
        checkpoint = torch.load(str(ckpt_path))
        model = model.load_state_dict(checkpoint["state_dict"])
    acc1 = 0.0
    # acc3 = 0.0
    top1_acc = Accuracy(top_k=1, task="multiclass", num_classes=num_classes).to(device)
    # top3_acc = Accuracy(top_k=3, task="multiclass", num_classes=num_classes).to(device)

    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(loader) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description("[Test]")
                pbar.set_postfix(
                    acc1=f"{acc1:.2f}",
                    # acc3=f"{acc3:.2f}",
                )

                images = batch[0].to(device)
                target = batch[1].to(device)

                output = model(images)
                pred = torch.argmax(output, dim=1)

                acc1 = top1_acc(output, target)
                # acc3 = top3_acc(output, target)

                preds.append(pred)
                targets.append(target)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    print(f"Top1 Accuracy : {top1_acc.compute()}")
    # print(f"Top3 Accuracy : {top3_acc.compute()}")

    return top1_acc.compute(), preds.cpu().detach(), targets.cpu().detach()


def calcurate_cls_score(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    classes: list[str],
    savepath: Path,
) -> None:
    cm = confusion_matrix(targets, preds)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        savepath.parent.joinpath(savepath.stem + ".csv")
    )
    sns.heatmap(
        cm,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        annot=True,
        fmt="d",
        # cmap=
    )
    plt.savefig(savepath)
