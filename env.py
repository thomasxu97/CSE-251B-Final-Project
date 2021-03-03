from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np


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
        return state, reward, done

    def close(self):
        self.env.close()

env = Environment()

for i in range(5000):
    _, r, d = env.step(np.random.randint(6), render = True)
    print(r)
env.close()




