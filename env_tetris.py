import numpy as np

import matplotlib.pyplot as plt
from modified_tetris import Tetris


## Modified Tetris Environment
## Concert 20*10 board environment to 84*84 screen input
## Action space: Discrete(40)

class Environment():
    def __init__(self):
        self.env = Tetris()
        self.env.reset()
        self.screen = np.load("sample_game_frame.npy")
        self.screen = self.screen.astype('uint8')
        self.box_1 = np.array([[251,157,157,157],[157,157,157,157],[157,157,157,157],[157,157,157,157]])
        self.box_2 = np.array([[251,79,79,79],[79,251,251,79],[79,251,251,79],[79,79,79,79]])
        self.box_0 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.topoffset = 2
        self.leftoffset = 8

    # state preprocess
    def state_preprocess(self, state):
        s = self.screen.copy()
        for i in range(10):
            for j in range(20):
                if state[i][j] == 1:
                    s[self.topoffset+4*i:self.topoffset+4*i+4,self.leftoffset+4*j:self.leftoffset+4*j+4] = box_1
                elif state[i][j] == 2:
                    s[self.topoffset+4*i:self.topoffset+4*i+4,self.leftoffset+4*j:self.leftoffset+4*j+4] = box_2
                else:
                    s[self.topoffset+4*i:self.topoffset+4*i+4,self.leftoffset+4*j:self.leftoffset+4*j+4] = box_0
        s = s.astype('uint8')
        return s

    def step(self, action, render=False):
        reward, done = self.env.play(int(action/4), action%4)
        state = self.env._get_complete_board()
        return self.state_preprocess(state), reward, done

    def reset(self):
        state = self.env.reset()
        return self.state_preprocess(state)

# env = Environment()
# env.reset()
# for i in range(1000):
#     s, r, d = env.step(np.random.randint(40), render=True)
#     if d:
#         break

# np.save("env.npy", s)

