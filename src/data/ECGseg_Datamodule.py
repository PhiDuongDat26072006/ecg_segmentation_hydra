from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision.datasets import MNIST
from src.data.components.transform import *
from src.data.components.datareader import *
import os

class ECGseg_DataModule(LightningDataModule):
    def __init__(self, use_sampler, data_dir: str = "data/", batch_size: int = 64,
                 pin_memory: bool = True) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.use_sampler = use_sampler
        self.shuffle = None
        self.train_sampler = None
        self.train_dataset = None
        self.test_dataset = None

    @property
    def seg_num_classes(self) -> int:
        return 4

    @property
    def cls_num_classes(self) -> int:
        return 2

    def setup(self, stage: Optional[str] = None) -> None:
        n_ludb_train = 180  # 180/200
        ludb_files = [os.path.abspath(os.path.join(self.data_dir, p))[:-4] for p in os.listdir(self.data_dir) if
                      p.endswith('.hea')]
        ludb_files_train = ludb_files[:n_ludb_train]
        ludb_files_test = ludb_files[n_ludb_train:]

        X_train, y_seg_train, y_cls_train = load_ludb_tensors(ludb_files_train)
        X_test, y_seg_test, y_cls_test = load_ludb_tensors(ludb_files_test)
        if self.use_sampler:
            target = y_cls_train
            weight = torch.tensor([1. / torch.sum(target == t) for t in torch.unique(target)])
            samples_weight = torch.tensor([weight[int(t)] for t in target]).double()
            self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            self.shuffle = None
        else:
            self.train_sampler = None
            self.shuffle = True

        self.train_dataset = CustomTensorDataset(tensors=(X_train, y_seg_train, y_cls_train), transform=Compose([
                                RandomCrop(2000, start=1000, end=4000),
                                BaselineWander(prob=0.2),
                                GaussianNoise(prob=0.2),
                                PowerlineNoise(prob=0.2),
                                ChannelResize(),
                                BaselineShift(prob=0.2),
                            ]))
        self.test_dataset = CustomTensorDataset(tensors=(X_test, y_seg_test, y_cls_test))

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle = self.shuffle,
            sampler = self.train_sampler
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def teardown(self, stage: Optional[str] = None) -> None:

        pass

    def state_dict(self) -> Dict[Any, Any]:

        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:

        pass


if __name__ == "__main__":
    _ = ECGseg_DataModule()
