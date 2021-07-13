from typing import Optional, Type, Callable, Dict

import albumentations
import torch
from torch import nn, Tensor
from torch.utils import data
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch.transforms import ToTensorV2 as AlbToTensor, ToTensorV2

from dataset.GausToCircle import GaussToCircle
from dataset.tracker import TrackerDataset

from parameters.path import Paths


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch



class AbstractLoader:
    pass


class TrajectoryDatasetLoader:

    batch_size = 4

    def __init__(self):
        self.dataset_train = TrackerDataset(traj_len=48)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=TrajectoryDatasetLoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("TrajectoryDataset initialize")
        print(f"train size: {len(self.dataset_train)}")


class GausToCircleLoader:

    batch_size = 64

    def __init__(self):
        self.dataset_train = GaussToCircle()

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=GausToCircleLoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("GaussToCircleDataset initialize")
        print(f"train size: {len(self.dataset_train)}")


class LazyLoader:

    saved = {}

    trajectory_save: Optional[TrajectoryDatasetLoader] = None
    circle_save: Optional[GausToCircleLoader] = None

    @staticmethod
    def register_loader(cls: Type[AbstractLoader]):
        LazyLoader.saved[cls.__name__] = None

    @staticmethod
    def gauss_to_circle() -> GausToCircleLoader:
        if not LazyLoader.circle_save:
            LazyLoader.circle_save = GausToCircleLoader()
        return LazyLoader.circle_save

    @staticmethod
    def trajectory() -> TrajectoryDatasetLoader:
        if not LazyLoader.trajectory_save:
            LazyLoader.trajectory_save = TrajectoryDatasetLoader()
        return LazyLoader.trajectory_save
