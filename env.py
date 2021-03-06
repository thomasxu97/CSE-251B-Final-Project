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
        self.env = gym_tetris.make('TetrisA-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env.reset()
        self.number_of_lines = 0
        self.board_height = 0
        self.score = 0

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
        for i in range(ACTION_REPEAT_NUM):
            state, r, done, info = self.env.step(action)
            if done:
                break
        board_height = info["board_height"]
        number_of_lines = info["number_of_lines"]
        reward = (number_of_lines - self.number_of_lines) * 100 - (board_height - self.board_height)
        self.number_of_lines = number_of_lines
        self.board_height = board_height
        self.score = info["score"]
        if action == 5:
            reward += 0.01
        if render:
            self.env.render()
        if done:
            self.env.reset()
            reward -= 400
        return self.state_preprocess(state), reward, done

    def reset(self):
        state = self.env.reset()
        self.number_of_lines = 0
        self.board_height = 0
        self.score = 0
        return self.state_preprocess(state)

    def close(self):
        self.env.close()

# env = Environment()
# env.reset()
# for i in range(1000):
#     s, r, d = env.step(np.random.randint(6), render=True)
#     print(d)
#     if d:
#         break

# print(env.score)

