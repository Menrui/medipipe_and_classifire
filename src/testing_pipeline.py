import torch
from torchmetrics import Accuracy
from tqdm import tqdm


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

                acc1 = top1_acc(output, target)
                acc3 = top3_acc(output, target)

    return top1_acc.compute()
