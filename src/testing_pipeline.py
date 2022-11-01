from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from sklearn.metrics import confusion_matrix


def test():
    pass


def test_loop(
    model,
    loader,
    ckpt_path,
    device,
):
    if ckpt_path is not None:
        checkpoint = torch.load(str(ckpt_path))
        model = model.load_state_dict(checkpoint["state_dict"])
    acc1 = 0.0
    acc3 = 0.0
    top1_acc = Accuracy(top_k=1).to(device)
    top3_acc = Accuracy(top_k=3).to(device)

    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(loader) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description("[Validation]")
                pbar.set_postfix(
                    acc1=f"{acc1:.2f}",
                    acc3=f"{acc3:.2f}",
                )

                images = batch[0].to(device)
                target = batch[1].to(device)

                output = model(images)
                pred = torch.argmax(output, dim=1)

                acc1 = top1_acc(output, target)
                acc3 = top3_acc(output, target)

                preds.append(pred)
                targets.append(target)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    print(f"Top1 Accuracy : {top1_acc.compute()}")
    print(f"Top3 Accuracy : {top3_acc.compute()}")

    return top1_acc.compute(), preds.cpu().detach(), targets.cpu().detach()


def calcurate_cls_score(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    classes: list[str],
    savepath: Path,
):
    cm = confusion_matrix(targets, preds)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        savepath.parent.joinpath(savepath.stem + ".csv")
    )
    sns.heatmap(
        cm,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        annot=False,
        # cmap=
    )
    plt.savefig(savepath)
