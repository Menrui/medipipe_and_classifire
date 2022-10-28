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
import torchvision.transforms.functional as TF
from src.datamodules.components.transforms import LongsideResizeSquarePadding


class DetectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source: Path = Path("data/d4cls/d4cls.csv"),
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
            if stage2idx.get(stage) is None:
                raise ValueError(
                    f"Invalid Value of stage: {stage}, please input ['train', 'val', 'test']"
                )
            df = pd.read_csv(str(data_source))
            stage_tags = df["learning_phase"].values
            self.data_paths = df["fullpath"].values[stage_tags == stage2idx.get(stage)]
            self.classes = np.unique(df["category"].values[stage_tags == stage2idx.get(stage)])
            self.class_to_idx = {k: v for k, v in zip(self.classes, range(len(self.classes)))}
            self.idx_to_class = {k: v for k, v in zip(range(len(self.classes)), self.classes)}
            self.targets = [
                self.class_to_idx.get(li)
                for li in df["category"].values[stage_tags == stage2idx.get(stage)]
            ]
        else:
            raise ValueError(f"Invalid value of data_source: {data_source}")

        if transform is None:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize(224),
                    LongsideResizeSquarePadding(
                        size=224,
                        interpolation=TF.InterpolationMode.NEAREST,
                        antialias=False,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    @property
    def num_classes(self):
        return len(self.classes)

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
