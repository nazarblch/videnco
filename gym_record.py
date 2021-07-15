import torch
from torch import nn, Tensor

import gym
env = gym.make('CartPole-v0')
from matplotlib import pyplot as plt
from albumentations import Resize
import numpy as np
path = "/raid/data/cartpole_records"

for ep in range(1000):

    ll = 0
    images = []
    coords = []

    while ll < 33:

        images = []
        coords = []
        env.reset()

        for i in range(100):
            data = Resize(128, 128)(image=env.render(mode="rgb_array"))['image']
            data = np.transpose(data, (2, 0, 1))
            observation, reward, done, info = env.step(env.action_space.sample())
            images.append(data[np.newaxis,])
            coords.append(observation[np.newaxis,])
            if done:
                break

        ll = len(images)

    images = np.concatenate(images)
    coords = np.concatenate(coords)
    print(ep, images.shape, coords.shape)
    np.save(f"{path}/img/{ep}.npy", images)
    np.save(f"{path}/coord/{ep}.npy", coords)


env.close()
