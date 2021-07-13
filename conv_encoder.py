from torch import nn, Tensor

from gan.discriminator import Discriminator
from gan.nn.stylegan.components import EqualLinear, ConvLayer
from view import View


def enc_hm_to_coord(n_points: int = 20):

    return nn.Sequential(
        ConvLayer(n_points, 20, 3, downsample=True),
        ConvLayer(20, 40, 3, downsample=True),
        ConvLayer(40, 80, 3, downsample=True),
        ConvLayer(80, 160, 3, downsample=True),
        ConvLayer(160, 100, 3, downsample=True),
        View(-1),
        EqualLinear(4 * 4 * 100, n_points * 2, activation='fused_lrelu'),
        EqualLinear(n_points * 2, n_points * 2, activation=None),
        nn.Sigmoid(),
        View(n_points, 2)
    ).cuda()


def fast_hm_cnn(patch_size: int, out_dim: int):
    return nn.Sequential(
        nn.Conv2d(patch_size, 20, 4, 2, 1),
        nn.BatchNorm2d(20),
        nn.ReLU(inplace=True),
        nn.Conv2d(20, 40, 4, 2, 1),
        nn.BatchNorm2d(40),
        nn.ReLU(inplace=True),
        nn.Conv2d(40, 80, 4, 2, 1),
        nn.BatchNorm2d(80),
        nn.ReLU(inplace=True),
        nn.Conv2d(80, 160, 4, 2, 1),
        nn.BatchNorm2d(160),
        nn.ReLU(inplace=True),
        nn.Conv2d(160, 100, 4, 2, 1),
        nn.BatchNorm2d(100),
        nn.ReLU(inplace=True),
        View(-1),
        # nn.Dropout(p=0.2),
        nn.Linear(4 * 4 * 100, out_dim),
        nn.ReLU(inplace=True),
        nn.Linear(out_dim, out_dim)
    ).cuda()


def fast_lnn_coord(patch_size: int, out_dim: int):
    return nn.Sequential(
        View(-1),
        nn.Linear(patch_size * 2, 100),
        nn.ReLU(inplace=True),
        nn.Linear(100, 100),
        # nn.Dropout(p=0.2),
        nn.ReLU(inplace=True),
        nn.Linear(100, out_dim)
    ).cuda()


def coord_to_coord(n_points: int = 20):
    return nn.Sequential(
        View(-1),
        EqualLinear(n_points * 2, 100, activation='fused_lrelu'),
        EqualLinear(100, 100, activation='fused_lrelu'),
        EqualLinear(100, 100, activation='fused_lrelu'),
        EqualLinear(100, n_points * 2, activation=None),
        nn.Sigmoid(),
        View(n_points, 2)
    ).cuda()


class CoordDisc(Discriminator):

    def __init__(self, n_points: int):
        super().__init__()

        self.net = nn.Sequential(
            View(-1),
            EqualLinear(n_points * 2, 256, activation='fused_lrelu'),
            EqualLinear(256, 256, activation='fused_lrelu'),
            EqualLinear(256, 256, activation='fused_lrelu'),
            EqualLinear(256, 1, activation=None),
        ).cuda()

    def forward(self, *x: Tensor) -> Tensor:
        return self.net(x[0])

