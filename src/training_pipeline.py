import os
from pathlib import Path
import hydra

from omegaconf import DictConfig
from typing import Optional, OrderedDict, Tuple
from torchmetrics import MeanMetric, Accuracy, MaxMetric
from tqdm import tqdm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn

from src import utils
from src.utils.logger import get_logger
from src.testing_pipeline import calcurate_cls_score, test_loop
from src.utils.history import History

log = get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    log.info("Start training!")
    if config.get("seed"):
        utils.seed_everything(config.seed)

    ckpt_dir = Path("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log.info(f"Instantiating dataset <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule)
    if config.model.get("num_classes") == -1:
        config.model.num_classes = datamodule.num_classes

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    if config.get("scheduler"):
        scheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)

    if config.get("gpu"):
        device = "cuda"
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        device = "cpu"

    train_loader, val_loader = (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
    )
    iteration_history = History(
        keys=("train_loss", "val_loss", "train_acc", "val_acc"),
        output_dir="./",
    )
    epoch_history = History(
        keys=("train_loss", "val_loss", "train_acc", "val_acc"),
        output_dir="./",
    )

    best_acc1 = MaxMetric().to(device)
    log.info("Start learning loop")
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_loop(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            history=iteration_history,
            device=device,
        )

        val_loss, val_acc = val_loop(
            loader=val_loader,
            model=model,
            criterion=criterion,
            history=iteration_history,
            device=device,
        )

        if config.get("scheduler"):
            scheduler.step()

        is_best = val_acc >= best_acc1.compute() if epoch != 0 else False
        best_acc1.update(val_acc)

        epoch_history(
            {
                "train_loss": train_loss.cpu().numpy(),
                "train_acc": train_acc.cpu().numpy(),
                "val_loss": val_loss.cpu().numpy(),
                "val_acc": val_acc.cpu().numpy(),
            }
        )

        # if is_best or epoch % config.get("checkpoint") == 0:
        # save checkpoint
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": config.model.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1.compute(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if config.get("scheduler") else None,
            },
            is_best,
            save_dir=ckpt_dir,
        )

    log.info("Plot learning curve")
    iteration_history.plot_graph(["train_loss", "val_loss"], "lc_loss_iter.png")
    iteration_history.plot_graph(
        ["train_acc", "val_acc"],
        "lc_acc_iter.png",
        ylabel="Accuracy",
        title="Accuracy.",
    )
    epoch_history.plot_graph(["train_loss", "val_loss"], "lc_loss_epoch.png")
    epoch_history.plot_graph(
        ["train_acc", "val_acc"],
        "lc_acc_epoch.png",
        ylabel="Accuracy",
        title="Accuracy.",
    )
    log.info("Complete training!")

    if config.get("test"):
        log.info("Starting testing!")
        ckpt_path = ckpt_dir.joinpath("best.pth")
        # if not config.get("train"):
        #     ckpt_path = None
        checkpoint = torch.load(str(ckpt_path))
        test_loader = datamodule.test_dataloader()
        model.load_state_dict(checkpoint["state_dict"]),
        acc, preds, targets = test_loop(
            model=model,
            loader=test_loader,
            ckpt_path=None,
            device=device,
        )
        log.info(f"test acc : {acc}")
        calcurate_cls_score(
            preds=preds,
            targets=targets,
            classes=list(datamodule.data_test.class_to_idx.keys()),
            savepath="confusion_matrix.png",
        )

    return best_acc1.compute()


def train_loop(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: History,
    device: Literal["cuda", "cpu"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    losses = MeanMetric().to(device)
    acc1 = 0.0
    acc3 = 0.0
    top1_acc = Accuracy(top_k=1).to(device)
    top3_acc = Accuracy(top_k=3).to(device)

    model.train()
    scaler = torch.cuda.amp.GradScaler()

    with tqdm(loader) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"[Train] Epoch {epoch+1}")
            pbar.set_postfix(
                OrderedDict(
                    loss=f"{losses.compute().item() if i != 0 else 0:.3f}",
                    acc1=f"{acc1:.2f}",
                    acc3=f"{acc3:.2f}",
                )
            )

            images = batch[0].to(device)
            target = batch[1].to(device)

            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            # print(output.shape)
            # print(target.shape)

            acc1 = top1_acc(output, target).cpu().item()
            acc3 = top3_acc(output, target).cpu().item()

            losses.update(loss.item())

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            history({"train_loss": loss.cpu().item(), "train_acc": acc1})

    return losses.compute(), top1_acc.compute()


def val_loop(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: nn.Module,
    history: History,
    device: Literal["cuda", "cpu"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    losses = MeanMetric().to(device)
    acc1 = 0.0
    acc3 = 0.0
    top1_acc = Accuracy(top_k=1).to(device)
    top3_acc = Accuracy(top_k=3).to(device)

    model.eval()

    with torch.no_grad():
        with tqdm(loader) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description("[Validation]")
                pbar.set_postfix(
                    loss=f"{losses.compute().item() if i != 0 else 0:.3f}",
                    acc1=f"{acc1:.2f}",
                    acc3=f"{acc3:.2f}",
                )

                images = batch[0].to(device)
                target = batch[1].to(device)

                output = model(images)
                loss = criterion(output, target)

                acc1 = top1_acc(output, target).item()
                acc3 = top3_acc(output, target).item()

                losses.update(loss.item())

                history({"val_loss": loss.item(), "val_acc": acc1})

    return losses.compute(), top1_acc.compute()
