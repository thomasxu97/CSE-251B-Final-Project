from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np

import matplotlib.pyplot as plt


## TetrisA-vo environment
## Action space: Discrete(6)
## SIMPLE_MOVEMENT = [
#     ['NOOP'],
#     ['A'],
#     ['B'],
#     ['right'],
#     ['left'],
#     ['down'],
# ]
## State shape: (240, 256, 3)

class Environment():
    def __init__(self):
        self.env = gym_tetris.make('TetrisA-v2')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env.reset()

    # state preprocess
    # (240, 256, 3) -> (84, 84, 1)
    # pixel value range [0, 255] -> [-0.5, 0.5]
    def step(self, action, render=False):
        if action > 6:
            action = 0
        state, reward, done, info = self.env.step(action)
        if action == 5:
            reward += 0.001
        if render:
            self.env.render()
        if done:
            self.env.reset()
        state = state[44:212:2, 80:248:2, :]
        state = (0.2989 * state[:,:,0] + 0.5870 * state[:,:,1] + 0.1140 * state[:,:,2] - 127)/255
        return state, reward, done

    def close(self):
        self.env.close()




