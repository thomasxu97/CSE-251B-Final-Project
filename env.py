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

    # state preprocess
    # (240, 256, 3) -> (84, 84, 1)
    def state_preprocess(self, state):
        state = state[44:212:2, 80:190:2, :]
        state = 0.2989 * state[:,:,0] + 0.5870 * state[:,:,1] + 0.1140 * state[:,:,2]
        state = state.astype('uint8')
        return state

    def step(self, action, render=False):
        actions = [action] + [0 for i in range(ACTION_REPEAT_NUM-1)]
        reward = 0
        for i in range(ACTION_REPEAT_NUM):
            state, r, done, info = self.env.step(actions[i])
            reward += r
            if done:
                break
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
        return self.state_preprocess(state)

    def close(self):
        self.env.close()

env = Environment()
env.reset()
for i in range(1000):
    s, r, d = env.step(np.random.randint(6), render=True)
    if d:
        break

np.save("env.npy", s)
plt.imshow(s.reshape(84, 55))
plt.savefig("sample_Tetris.png")

