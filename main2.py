import random
import time
import os, sys
from itertools import chain

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans-pytorch/'))
import json
from matplotlib import pyplot as plt
from optim.accumulator import Accumulator
from typing import List
import torch.utils.data
from torch import Tensor
from gan.nn.stylegan.components import EqualLinear
from transforms.augment import ToNumpy, ToTensor, NumpyBatch
from utils.loss_utils import Loss
from conv_encoder import fast_lnn_coord, fast_hm_cnn, enc_hm_to_coord, coord_to_coord, CoordDisc
from conv_rnn import ConvLSTM
from dataset.lazy_loader import LazyLoader
from loss.nce import PatchNCELoss, InfoNCE
from parameters.path import Paths
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from utils.wr import WR
import albumentations
from view import View
import numpy as np
from gan.models.stylegan import StyleGanModel, StyleGANLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
manualSeed = 79
random.seed(manualSeed)
torch.manual_seed(manualSeed)


writer = SummaryWriter(f"{Paths.default.board()}/circle_{int(time.time())}")
WR.writer = writer

generator = nn.Sequential(
    EqualLinear(2, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 2, activation=None)
).cuda()

discriminator = nn.Sequential(
    EqualLinear(2, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 1, activation=None)
).cuda()

gan_model = StyleGanModel(generator, StyleGANLoss(discriminator, r1=1), (0.001, 0.0015))
accumulator = Accumulator(generator, decay=0.98, write_every=100)

x_network = nn.Sequential(
    EqualLinear(2, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 20, activation=None),
).cuda()

y_network = nn.Sequential(
    EqualLinear(2, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 20, activation=None),
).cuda()

xy_opt = torch.optim.Adam(chain(x_network.parameters(), y_network.parameters()), lr=0.001)

def transformed_patch_maker(x: Tensor, window: int):
    B = x.shape[0]
    res = []
    def trancf(xi):
        return xi + torch.randn_like(xi) * 0.1
    for i in range(0, B):
        res.append(torch.cat([trancf(x[i])[None,] for _ in range(window)], dim=0))
    return torch.stack(res, dim=0)


def contrastive_loss(x: Tensor, y: Tensor) -> Tensor:

    B = x.shape[0]
    x_patch = transformed_patch_maker(x, 5).view(-1, 2)
    y_patch = generator(x_patch)

    x_feat = x_network(x_patch + torch.randn_like(x_patch) * 0.02)
    y_feat = y_network(y_patch)

    x_feat_p, y_feat_p = x_feat.view(B, 5, 20), y_feat.view(B, 5, 20)

    return PatchNCELoss(nce_T=0.1)(x_feat_p, y_feat_p) + InfoNCE()(x_feat, y_feat) * 0.5

c2c_model = nn.Sequential(
    EqualLinear(2, 20, activation=None),
    nn.LeakyReLU(0.2, inplace=True),
    EqualLinear(20, 2, activation=None),
).cuda()
c2c_optim = torch.optim.Adam(c2c_model.parameters(), lr=0.001)

for i in range(50000):
    print(i)

    coefs = json.load(open(os.path.join(sys.path[0], "parameters/loss.json")))

    x, y = next(LazyLoader.gauss_to_circle().loader_train_inf)
    x, y = x.cuda(), y.cuda()

    f = generator(x)
    gan_model.discriminator_train([y], [f.detach()])
    c_loss = Loss(contrastive_loss(x, f))
    (gan_model.generator_loss([y], [f]) + c_loss * coefs["c_loss"])\
        .minimize_step(gan_model.optimizer.opt_min, xy_opt)

    accumulator.step(i)

    pred = generator(x)
    pred_loss = Loss(nn.MSELoss()(pred, y))
    print(pred_loss.item())

    if i % 1000 == 0:
        # print(x.shape, y.shape, f.shape)
        x = x.cpu().numpy()
        f = f.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        print((np.sum((y - f)**2, -1)**(1/2)).mean())
        plt.scatter(f[:, 0], f[:, 1])
        plt.scatter(y[:, 0], y[:, 1])
        # plt.scatter(f[:, 1], f[:, 2])
        plt.show()






