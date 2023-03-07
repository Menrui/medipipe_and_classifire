from pathlib import Path
from typing import Literal, Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.datamodules.components.mp_dataset import MPDataset
from src.datamodules.sampler.balanced_sampler import BalancedBatchSampler

from . import BaseDataModule


class MPDataModule(BaseDataModule):
    def __init__(
        self,
        data_source: Path = Path("data/"),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_type: Literal["image", "mesh", "render"] = "render",
    ) -> None:
        super().__init__()

        self.data_source = (
            data_source if isinstance(data_source, Path) else Path(data_source)
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_type = data_type

        self.transforms = transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomRotation(degrees=2),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                # transforms.RandomAffine(degrees=1, translate=(0.1, 0.1)),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            (
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.prepare_data()
        self.setup()

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MPDataset(
                data_dir=self.data_source,
                stage="train",
                transform=self.transforms,
                data_type=self.data_type,
            )
            self.data_val = MPDataset(
                data_dir=self.data_source,
                stage="val",
                transform=self.test_transform,
                data_type=self.data_type,
            )
            self.data_test = MPDataset(
                data_dir=self.data_source,
                stage="test",
                transform=self.test_transform,
                data_type=self.data_type,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # shuffle=True,
            batch_sampler=BalancedBatchSampler(
                labels=self.data_train.labels,
                label_set=sorted(list(set(self.data_train.labels))),
                label_idxs=[
                    np.where(self.data_train.labels == target)[0].tolist()
                    for target in sorted(set(self.data_train.labels))
                ],
                n_classes=self.num_classes,
                n_samples=self.batch_size // self.num_classes,
                episode_per_epoch=len(self.data_train) // self.batch_size,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data_val is not None
        return DataLoader(
            dataset=self.data_val,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # shuffle=False,
            batch_sampler=BalancedBatchSampler(
                labels=self.data_val.labels,
                label_set=sorted(list(set(self.data_val.labels))),
                label_idxs=[
                    np.where(self.data_val.labels == target)[0].tolist()
                    for target in sorted(set(self.data_val.labels))
                ],
                n_classes=self.num_classes,
                n_samples=self.batch_size // self.num_classes,
                episode_per_epoch=len(self.data_val) // self.batch_size,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data_test is not None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
