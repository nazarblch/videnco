from typing import Dict
from albumentations import *
import albumentations
from albumentations import DualTransform, BasicTransform
from albumentations.pytorch import ToTensor
import torch
import numpy as np


class ToNumpy(DualTransform):
    def __init__(self):
        super(ToNumpy, self).__init__(1)

    def apply(self, img: torch.Tensor, **params):
        return np.transpose(img.detach().cpu().numpy(), [0, 2, 3, 1])

    def apply_to_mask(self, mask: torch.Tensor, **params):
        return np.transpose(mask.detach().cpu().numpy(), [0, 2, 3, 1])

    def apply_to_keypoint(self, keypoint, **params):
        x, y, a, s = keypoint
        return [x.detach().cpu().numpy(), y.detach().cpu().numpy(), a, s]


class ToTensor(DualTransform):
    def __init__(self, device):
        super(ToTensor, self).__init__(1)
        self.device = device

    def apply(self, img: np.array, **params):
        return torch.tensor(np.transpose(img, [0, 3, 1, 2]), device=self.device)

    def apply_to_mask(self, img: np.array, **params):
        return torch.tensor(np.transpose(img, [0, 3, 1, 2]), device=self.device)


class NumpyBatch(BasicTransform):

    def __init__(self, transform: BasicTransform):
        super(NumpyBatch, self).__init__(1)
        self.transform = transform
        # self.par = Parallel(n_jobs=20)

    def __call__(self, force_apply=False, **kwargs):

        keys = ["image"]
        if "mask" in kwargs:
            keys.append("mask")

        def compute(transform, tdata: Dict[str, np.ndarray]):

            data_i = transform(**tdata)
            return data_i

        # processed_list = Parallel(n_jobs=2)(delayed(compute)(
        #     self.transform, {k: kwargs[k][i] for k in keys}) for i in range(kwargs["image"].shape[0])
        #                           )

        processed_list = [compute(
            self.transform, {k: kwargs[k][i] for k in keys}) for i in range(kwargs["image"].shape[0])
                                  ]

        batch = {key: [] for key in keys}

        for data in processed_list:
            for key in keys:
                batch[key].append(data[key][np.newaxis, ...])

        return {key: np.concatenate(batch[key], axis=0) for key in keys}