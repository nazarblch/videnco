import os
from os.path import isdir

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import re
import albumentations
from albumentations.pytorch.transforms import ToTensor as AlbToTensor

from dataset.toheatmap import ToGaussHeatMap
from parameters.path import Paths


class TrackerDataset(Dataset):
    def __init__(self, traj_len: int):
        self.traj_len = traj_len
        trajectories_path = os.path.join(Paths.default.data(), "trajectory/numpy_data/")
        image_folders = [x for x in os.listdir(trajectories_path) if x.endswith(".npy")]

        self.heatmapper = ToGaussHeatMap(128, 20)
        self.trajectories = []

        for traj in image_folders:
            traj_path = os.path.join(trajectories_path, traj)
            if os.path.isfile(traj_path):
                self.trajectories += [traj_path]

    def __getitem__(self, index):
        traj = torch.from_numpy(np.load(self.trajectories[index]).astype(np.float32))
        shift = np.random.randint(0, traj.shape[0] - self.traj_len)
        traj = traj[None, shift: shift + self.traj_len]
        mask = self.heatmapper.forward(traj)
        return traj.squeeze(), mask.squeeze()[:, None, ...]

    def __len__(self):
        return len(self.trajectories)
