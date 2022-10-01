from typing import Optional

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from . import BaseDataModule
from src.datamodules.components.detector_dataset import DetectorDataset


class DetectorDataModule(BaseDataModule):
    def __init__(
        self,
        data_source: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.prepare_data()
        self.setup()

    @property
    def num_classes(self) -> int:
        return self.data_train.num_classes

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = DetectorDataset(
                data_source=self.data_source, stage="train", transform=self.transforms
            )
            self.data_val = DetectorDataset(
                data_source=self.data_source, stage="val", transform=self.transforms
            )
            self.data_test = DetectorDataset(
                data_source=self.data_source, stage="test", transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
