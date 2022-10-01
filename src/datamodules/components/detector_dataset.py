from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class DetectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source: Path = Path("data/kinkazan"),
        stage: Literal["train", "val", "test"] = "train",
        transform: transforms.Compose = None,
    ) -> None:
        super().__init__()
        assert stage in [
            "train",
            "val",
            "test",
        ], f"Invalid Value of stage: {stage}, please choice  ['train', 'val', 'test']"
        if isinstance(data_source, str):
            data_source = Path(data_source)

        if data_source.is_dir():
            # print("directory")
            pass
        elif data_source.is_file and data_source.suffix == ".json":
            # print("json")
            pass
        elif data_source.is_file and data_source.suffix == ".csv":
            # print("csv")
            stage2idx = {k: v for k, v in zip(["train", "val", "test"], [0, 1, 2])}
            df = pd.read_csv(str(data_source))
            stage_label = df["learning_phase"].values
            self.data_paths = df["filepath"].values[stage_label == stage2idx.get(stage)]
            self.label_set = np.unique(
                df["label"].values[stage_label == stage2idx.get(stage)]
            )
            self.label2idx = {
                k: v for k, v in zip(self.label_set, range(len(self.label_set)))
            }
            self.targets = [
                self.label2idx.get(li)
                for li in df["label"].values[stage_label == stage2idx.get(stage)]
            ]
        else:
            raise ValueError(f"Invalid value of data_source: {data_source}")

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    @property
    def num_classes(self):
        return len(self.label_set)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        target = self.targets[index]

        return self.preprocess(data_path), target

    def preprocess(self, item):
        img = np.float32(np.array(Image.open(item)) / 255)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = np.uint8(img * 255)
            img = Image.fromarray(img)
        else:
            img = Image.open(item)

        img = self.transform(img)
        return img
