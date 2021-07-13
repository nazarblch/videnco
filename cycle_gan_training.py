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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
manualSeed = 79
random.seed(manualSeed)
torch.manual_seed(manualSeed)

model = enc_hm_to_coord(48)
# model.load_state_dict(torch.load(f'{Paths.default.models()}/videnc_{str(4000).zfill(6)}.pt', map_location="cpu")["enc"])
model = model.to(device)
enc_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(f"{Paths.default.board()}/gan_{int(time.time())}")
WR.writer = writer


c2c_model_tuda = coord_to_coord(48)
c2c_model_obratno = coord_to_coord(48)

discriminator_tuda = CoordDisc(48)
discriminator_tuda = discriminator_tuda.cuda()
gan_model_tuda = StyleGanModel(c2c_model_tuda, StyleGANLoss(discriminator_tuda), (0.001, 0.0015))


discriminator_obratno = CoordDisc(48)
discriminator_obratno = discriminator_obratno.cuda()
gan_model_obratno = StyleGanModel(c2c_model_obratno, StyleGANLoss(discriminator_obratno), (0.001, 0.0015))

accumulator = Accumulator(model, decay=0.98, write_every=100)
# accumulator2 = Accumulator(c2c_model_obratno, decay=0.98, write_every=100)


def patch_maker(x: Tensor, stride: int, window: int):
    T = x.shape[1]
    res = []
    for i in range(0, T - window, stride):
        res.append(x[:, i: i + window, ...])
    return torch.stack(res, dim=1)  # B x NP x P x DATA


def contrastive_loss(hm: Tensor, enc: nn.Module) -> Tensor:

    hm = hm[:, :, 0]
    P = 4
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

hm_network = fast_hm_cnn(4, 128)
coords_network = fast_lnn_coord(4, 128)

A = chain(hm_network.parameters(), coords_network.parameters())
optim_coord_hm = torch.optim.Adam(A, lr=2e-5)


for i in range(500000):
    print(i)

    coords, heatmap = next(LazyLoader.trajectory().loader_train_inf)
    coords, heatmap = coords.to(device), heatmap.to(device)

    enc_coords = model(heatmap.squeeze())

    coords_tuda = c2c_model_tuda(enc_coords)
    gan_model_tuda.discriminator_train([coords], [coords_tuda.detach()])
    gan_model_tuda.generator_loss([coords], [coords_tuda]).minimize_step(gan_model_tuda.optimizer.opt_min, enc_optimizer)

    enc_coords = enc_coords.detach()

    coords_obratno = c2c_model_obratno(coords)
    gan_model_obratno.discriminator_train([enc_coords], [coords_obratno.detach()])
    gan_model_obratno.generator_loss([enc_coords], [coords_obratno]).minimize_step(gan_model_obratno.optimizer.opt_min)

    tuda_obratno = c2c_model_obratno(c2c_model_tuda(enc_coords))
    tuda_obratno_loss = Loss(nn.MSELoss()(tuda_obratno, enc_coords)).__mul__(1)\
        .minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    obratno_tuda = c2c_model_tuda(c2c_model_obratno(coords))
    obratno_tuda_loss = Loss(nn.MSELoss()(obratno_tuda, coords)).__mul__(1)\
        .minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    writer.add_scalar("tuda_obratno_loss", tuda_obratno_loss, i)
    writer.add_scalar("obratno_tuda_loss", obratno_tuda_loss, i)

    enc_coords = model(heatmap.squeeze())
    coords_tuda = c2c_model_tuda(enc_coords)
    pred_ll = Loss(nn.MSELoss()(enc_coords, coords_tuda))

    c_loss = Loss(contrastive_loss(heatmap, model))
    writer.add_scalar("loss", c_loss.item(), i)
    (c_loss * 0.1 + pred_ll * 0.1).minimize_step(optim_coord_hm, enc_optimizer)

    pred_loss = nn.MSELoss()(coords_tuda, coords)
    writer.add_scalar("verka", pred_loss.item(), i)

    accumulator.step(i)

    if i % 2000 == 0 and i > 0:
        torch.save(
            {
                "d1": discriminator_tuda.state_dict(),
                "d2": discriminator_obratno.state_dict(),
                "g1": c2c_model_tuda.state_dict(),
                "g2": c2c_model_obratno.state_dict()
            },
            f'{Paths.default.models()}/videnc_gan_{str(i).zfill(6)}.pt'
        )




