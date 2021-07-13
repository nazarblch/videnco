import os
from os.path import isdir

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class GaussToCircle(Dataset):

    def __init__(self, n: int = 10000):

        self.x: Tensor = torch.randn(n, 2)
        self.y = self.x / (self.x * self.x).sum(-1, keepdim=True).sqrt()

    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
