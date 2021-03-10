import gym
import numpy as np

import matplotlib.pyplot as plt

ACTION_REPEAT_NUM = 4


## Pong-v0 environment


class Environment():
    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.env.reset()

    # state preprocess
    # (240, 256, 3) -> (84, 84, 4)
    # pixel value range [0, 255] -> [-0.5, 0.5]
    def state_preprocess(self, state):
        state = state[30:198:2,::2,:]/255
        state = np.hstack([state, np.ones((84, 4, 3))])
        state = (0.2989 * state[:,:,0] + 0.5870 * state[:,:,1] + 0.1140 * state[:,:,2] - 127)/255
        state = state.reshape(1, 1, 84, 84)
        return state

    def step(self, action, render=False):
        state, reward, done, info = self.env.step(action)
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

