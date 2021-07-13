import random
import time
import os, sys
from itertools import chain

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans-pytorch/'))

from matplotlib import pyplot as plt

from typing import List
import torch.utils.data
from torch import Tensor
from optim.accumulator import Accumulator
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
from gan.models.stylegan import StyleGanModel, StyleGANLoss

def patch_maker(x: Tensor, stride: int, window: int):
    T = x.shape[1]
    res = []
    for i in range(0, T - window, stride):
        res.append(x[:, i: i + window, ...])
    return torch.stack(res, dim=1)  # B x NP x P x DATA


def transformed_patch_maker(x: Tensor, transform: albumentations.Compose, stride: int, window: int):
    B = x.shape[0]
    T = x.shape[1]
    res = []
    def trancf(x):
        return transform(image=x)["image"]
    for i in range(0, T-window, stride):
        res.append(torch.cat([trancf(x[:, i:i+1, ...]) for _ in range(window)], dim=1))
    return torch.stack(res, dim=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
manualSeed = 79
random.seed(manualSeed)
torch.manual_seed(manualSeed)

P = 4
model = enc_hm_to_coord(48)
model = model.to(device)
enc_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

hm_network = fast_hm_cnn(P, 128)
coords_network = fast_lnn_coord(P, 128)

A = chain(hm_network.parameters(), coords_network.parameters())
optim_coord_hm = torch.optim.Adam(A, lr=2e-5)

# c2c_model = coord_to_coord(48)
# c2c_optim = torch.optim.Adam(c2c_model.parameters(), lr=1e-4)

writer = SummaryWriter(f"{Paths.default.board()}/lstm_{int(time.time())}")
WR.writer = writer


def contrastive_loss(hm: Tensor, enc: nn.Module) -> Tensor:

    hm = hm[:, :, 0]
    # hm_patched = torch.cat(
    #     [patch_maker(hm, P - 2, P), transformed_patch_maker(hm, transform, P - 2, P)],
    #     dim=1
    # )
    hm_patched = patch_maker(transform(image=hm)["image"], P-2, P)
    coords_patched = patch_maker(enc(hm), P-2, P)

    B = hm_patched.shape[0]
    NP = hm_patched.shape[1]

    coords_patched = coords_patched.view(B * NP, *coords_patched.shape[2:])
    hm_patched = hm_patched.view(B * NP, *hm_patched.shape[2:])
    # coords_patched = enc(hm_patched)
    coords_feat = coords_network(coords_patched)
    hm_feat = hm_network(hm_patched)

    coords_feat_p = coords_feat.view(B, NP, *coords_feat.shape[1:])
    hm_feat_p = hm_feat.view(B, NP, *hm_feat.shape[1:])

    return PatchNCELoss(nce_T=0.1)(hm_feat_p, coords_feat_p) + InfoNCE()(hm_feat, coords_feat) * 0.5


transform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ShiftScaleRotate(p=1.0, rotate_limit=5, scale_limit=0, shift_limit=0.02),
        ])),
        ToTensor(device),
])


discriminator = CoordDisc(48)
# discriminator.load_state_dict(torch.load(f'{Paths.default.models()}/videnc_{str(20000).zfill(6)}.pt', map_location="cpu")["disc"])
discriminator = discriminator.cuda()
c2c_model_tuda = coord_to_coord(48)
gan_model = StyleGanModel(c2c_model_tuda, StyleGANLoss(discriminator), (0.001, 0.0015))

accumulator = Accumulator(model, decay=0.99, write_every=100)


for i in range(50000):
    print(i)

    coords, heatmap = next(LazyLoader.trajectory().loader_train_inf)
    coords, heatmap = coords.to(device), heatmap.to(device)

    fake_coords = c2c_model_tuda(model(heatmap.squeeze()))
    pred = fake_coords.detach()
    gan_model.discriminator_train([coords], [fake_coords.detach()])

    gan_model.generator_loss([coords], [fake_coords]).minimize_step(gan_model.optimizer.opt_min, enc_optimizer)

    encoded = model(heatmap.squeeze())
    fake_coords = c2c_model_tuda(encoded)
    Loss(nn.MSELoss()(encoded, fake_coords.detach()) + nn.MSELoss()(fake_coords, encoded.detach()))\
        .minimize_step(gan_model.optimizer.opt_min, enc_optimizer)

    c_loss = Loss(contrastive_loss(heatmap, model))
    print(c_loss.item())
    writer.add_scalar("loss", c_loss.item(), i)
    c_loss.__mul__(0.02).minimize_step(optim_coord_hm, enc_optimizer)

    # pred = c2c_model(model(heatmap.squeeze()).detach())
    pred_loss = Loss(nn.MSELoss()(encoded, coords))
    writer.add_scalar("verka", pred_loss.item(), i)

    accumulator.step(i)

    if i % 2000 == 0 and i > 0:
        torch.save(
            {
                "enc": model.state_dict(),
                # "disc": discriminator.state_dict(),
            },
            f'{Paths.default.models()}/videnc2_{str(i).zfill(6)}.pt'
        )




