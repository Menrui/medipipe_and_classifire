from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MPDataset(Dataset):
    def __init__(
        self,
        data_dir: Path = Path("data"),
        stage: str = "train",
        val_test_split_process: tuple[str, str] = ("human", "human"),
        human_ids: list[list[str]] = [["A", "B", "C", "D"], ["E", "F"], ["G", "H"]],
        train_val_test_split_rate: tuple[float, float, float] = (0.6, 0.2, 0.2),
        transform: transforms.Compose = None,
        data_type: Literal["image", "mesh", "render"] = "render",
    ):
        super().__init__()
        assert stage in [
            "train",
            "val",
            "test",
        ], f"Invalid Value of stage: {stage}, please choice  ['train', 'val', 'test']"
        val_test_split_process = (
            val_test_split_process
            if isinstance(val_test_split_process, tuple)
            else tuple(val_test_split_process)
        )
        if val_test_split_process == ("human", "human"):
            assert (
                len(human_ids) == 3
            ), f"Invalid Value of human_ids: {human_ids}, please 3 lists with IDs"
        elif val_test_split_process in [("human", "length"), ("length", "human")]:
            assert (
                len(human_ids) == 2
            ), f"Invalid Value of human_ids: {human_ids}, please 2 lists with IDs"
        elif val_test_split_process == ("length", "length"):
            pass
        else:
            assert (
                False
            ), f"Invalid Value of val_test_split_process: {val_test_split_process}, \
                please input [(human, human), (human, length), (length, human), (length, length)]"

        self.data_type = data_type
        self.class_to_idx = {"non_contact": 0, "contact": 1}
        data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        if data_type in ["image", "mesh"]:
            self.hel_data_dir = data_dir / "photographed_data"
            if data_type == "image":
                self.hel_data_table = pd.read_csv(
                    self.hel_data_dir.joinpath("processed_data_table.csv")
                )
            elif data_type == "mesh":
                self.hel_data_table = pd.read_csv(
                    self.hel_data_dir.joinpath("facemesh.csv")
                )
        elif data_type in ["render"]:
            self.hel_data_dir = data_dir / "mediapipe_render"
            self.hel_data_table = pd.read_csv(
                self.hel_data_dir.joinpath("processed_data_table.csv")
            )

        if stage == "train":
            data_df = self.hel_data_table[
                self.hel_data_table["HUMAN_ID"].isin(human_ids[0])
            ]
        elif stage == "val":
            data_df = self.hel_data_table[
                self.hel_data_table["HUMAN_ID"].isin(human_ids[1])
            ]
        elif stage == "test":
            data_df = self.hel_data_table[
                self.hel_data_table["HUMAN_ID"].isin(human_ids[2])
            ]
        self.data_paths = data_df["PATH"].values
        self.labels = data_df["CLASS"].values
        if data_type == "mesh":
            self.data = torch.tensor(data_df.iloc[:, 3:].values)

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[index]

        return self._preprocess(index), label

    def _preprocess(self, index: int) -> torch.Tensor:
        if not self.data_type == "mesh":
            data_path = self.data_paths[index]
            item = np.float32(
                np.array(Image.open(self.hel_data_dir.joinpath(data_path))) / 255
            )
            if item.ndim == 2:
                item = np.expand_dims(item, axis=2)
                item = np.concatenate([item, item, item], axis=2)
            #     item = np.uint8(item * 255)
            #     item = Image.fromarray(item)
            # else:
            #     item = Image.fromarray(item)
            item = np.uint8(item * 255)
            item = Image.fromarray(item)
            item = self.transform(item)
            return item
        else:
            return self.data[index].float()


if __name__ == "__main__":
    import os

    print(os.getcwd())
    dataset = MPDataset(
        # data_dir=Path("../../../datasets/"),
        stage="train",
        data_type="mesh",
    )
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=2, pin_memory=True, batch_size=1
    )
    for batch in loader:
        print(batch[0].shape, batch[0])
        print(batch[1])
        break
