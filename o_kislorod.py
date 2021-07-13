import random
import time
import os, sys

from matplotlib import pyplot as plt

from typing import List
import torch.utils.data
from torch import Tensor

from conv_rnn import ConvLSTM
from dataset.lazy_loader import LazyLoader
from loss.nce import PatchNCELoss, InfoNCE
from parameters.path import Paths
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from utils.wr import WR
from view import View


def trolebus_func(x: Tensor):
    return ((x + 1) ** 2) / 2

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
).cuda()

loss_func = InfoNCE()
writer = SummaryWriter(f"{Paths.default.board()}/O_Kislorod_{int(time.time())}")
WR.writer = writer

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(10000):
    print(i)
    optimizer.zero_grad()

    input = torch.randn(100, 8)
    neg = torch.randn(1000, 8)
    target = trolebus_func(input)
    input, target, neg = input.to(device), target.to(device), neg.to(device)
# model output (100, 128)
    loss = loss_func(model(input), model(target), model(neg))
    loss.backward()
    optimizer.step()
    print(loss.item())
    writer.add_scalar("loss", loss.item(), i)
