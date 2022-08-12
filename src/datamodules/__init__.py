from abc import abstractclassmethod
from typing import Optional
from torch.utils.data.dataloader import DataLoader


class BaseDataModule:
    def __init__(self) -> None:
        pass

    @property
    @abstractclassmethod
    def num_classes(self) -> int:
        raise NotImplementedError()

    def prepare_data(self):
        pass

    @abstractclassmethod
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass
