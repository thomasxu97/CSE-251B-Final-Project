from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np

import matplotlib.pyplot as plt

ACTION_REPEAT_NUM = 4


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
    def state_preprocess(self, state):
        state = state[44:212:2, 80:248:2, :]
        state = (0.2989 * state[:,:,0] + 0.5870 * state[:,:,1] + 0.1140 * state[:,:,2] - 127)/255
        state = state.reshape(1, 1, 84, 84)
        return state

    def step(self, action, render=False):
        if action > 6:
            action = 0
        reward = 0
        for i in range(ACTION_REPEAT_NUM):
            state, r, done, info = self.env.step(action)
            reward += r
            if done:
                break
        if render:
            self.env.render()
        if done:
            self.env.reset()       
        return self.state_preprocess(state), reward, done

    def reset(self):
        state = self.env.reset()
        return self.state_preprocess(state)

    def close(self):
        self.env.close()


