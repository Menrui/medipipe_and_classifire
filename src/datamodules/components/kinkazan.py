from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class KINKAZANDataset(torch.utils.data.Dataset):
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

        if data_source.is_dir:
            pass
        elif data_source.is_file and data_source.suffix == ".json":
            pass
        elif data_source.is_file and data_source.suffix == ".csv":
            df = pd.read_csv(str(data_source))
            stage_label = df["phase"].values
            self.data_paths = df["path"].values[stage_label == stage]
            self.targets = df["label"].values[stage_label == stage]

        if transform is None:
            # self.transform =
            pass
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        target = self.targets[index]

        return self._preprocess(data_path), target

    def preprocess(self, item):
        item = np.float32(np.array(Image.open(item)) / 255)
        if item.ndim == 2:
            item = np.expand_dims(item, axis=2)
            item = np.concatenate([item, item, item], axis=2)
            item = np.uint8(item * 255)
            item = Image.fromarray(item)
        else:
            item = Image.open(item)

        item = self.transform(item)
        return item
