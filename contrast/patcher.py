from abc import ABC
import torch
from torch import nn, Tensor


def merge_with_batch(x: Tensor, dim: int) -> Tensor:
    perm = [dim] + list(range(dim)) + list(range(dim+1, x.ndim))
    combined_shape = [-1] + list(x.shape[1:dim]) + list(x.shape[dim + 1:])
    return x.permute(*perm).view(*combined_shape)


class Patcher1D(ABC):

    def make(self, x: Tensor) -> Tensor: pass


class SlidePatcher1D(Patcher1D, nn.Module):

    def __init__(self, stride: int, window: int, dim: int):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.window = window

    def make(self, x: Tensor) -> Tensor:
        perm = list(range(0, self.dim + 1)) + [x.ndim] + list(range(self.dim + 1, x.ndim))
        return x.unfold(self.dim, self.window, self.stride).permute(*perm).contiguous()

