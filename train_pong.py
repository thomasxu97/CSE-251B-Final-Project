from basicDQN import DQN
from collections import deque
from os import path
from env_pong import Environment
import numpy as np
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys


NUM_ITERATION = 100
LEARNING_RATE = 0.00025
MAX_MEMORY = 50000
EXPLORE_FREQUENCY = 5000
TRAIN_FREQUENCY = 5000
EVALUATION_EPISODES = 10
BATCH_SIZE = 32
GAMMA = 0.99
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
INITIAL_EXPLORATION_FRAME = 25000
FINAL_EXPLORATION_FRAME = 250000
ACTION_SAPCE = 6

def progressBar(i, max, text):
    bar_size = 30
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()

class Agent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = DQN(ACTION_SAPCE)
        self.training_net = DQN(ACTION_SAPCE)
        self.policy_net.apply(self.init_weights)
        self.training_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.double().to(self.device)
        self.training_net = self.training_net.double().to(self.device)
        self.memory = deque(maxlen = MAX_MEMORY)
        self.optimizer = optim.Adam(self.training_net.parameters(), lr=LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()

    def init_weights(self, m):
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            torch.nn.init.zeros_(m.bias.data)

    def add_to_memory(self, a, r, d, s):
        self.memory.append((a, r, d, s))

    def evaluate_q(self, state):
        state = (state - 127)/255
        inputs = torch.from_numpy(state.copy()).to(self.device)
        q_value = self.policy_net(inputs).detach().cpu().numpy()
        return q_value

    def optimal_action(self, state):
        state = (state - 127)/255
        inputs = torch.from_numpy(state.copy()).to(self.device)
        q_value = self.policy_net(inputs).detach().cpu().numpy()
        return np.argmax(q_value)

    def optimize_step(self, state_batch, targetQ):
        self.training_net.zero_grad()
        state_batch = (state_batch - 127)/255
        inputs = torch.from_numpy(state_batch.copy()).to(self.device)
        outputs = self.training_net(inputs)
        targetQ = torch.from_numpy(targetQ.copy()).to(self.device)
        loss = self.criterion(outputs, targetQ)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


class TrainSolver:
    def __init__(self, load = True):
        self.agent = Agent()
        self.env = Environment()
        self.env.reset()
        self.iteration = 0
        self.frame = 0
        self.savepath = "./ckpt/model.pt"
        if path.exists(self.savepath) and load:
            self.loadCheckpoint(self.savepath)

    # with some probability p: pick random action
    # other wise: pick optimal_action given
    # p is a function of self.frame
    def exploration_policy(self, optimal_action):
        if self.frame < INITIAL_EXPLORATION_FRAME:
            return np.random.randint(ACTION_SAPCE)
        else:
            p = INITIAL_EXPLORATION - (INITIAL_EXPLORATION - FINAL_EXPLORATION)*(self.frame - INITIAL_EXPLORATION_FRAME)/(FINAL_EXPLORATION_FRAME - INITIAL_EXPLORATION_FRAME)
            if p < FINAL_EXPLORATION:
                p = FINAL_EXPLORATION
            if np.random.uniform() < p:
                return np.random.randint(ACTION_SAPCE)
            else:
                return optimal_action

    def exploration(self):
        i = 0
        state = self.env.reset()
        for j in range(3):
            self.agent.add_to_memory(None, None, None, None)
        self.agent.add_to_memory(None, None, None, state.copy())
        prev_four_state = np.zeros((1, 4, 84, 84), dtype='uint8')
        prev_four_state[0,3,:,:] = state
        while i < EXPLORE_FREQUENCY:
            action = self.agent.optimal_action(prev_four_state)
            action = self.exploration_policy(action)
            state, reward, done = self.env.step(action)
            self.agent.add_to_memory(action, reward, done, state.copy())
            prev_four_state[0,0:3,:,:] = prev_four_state[0,1:4,:,:]
            prev_four_state[0,3,:,:] = state
            if done:
                state = self.env.reset()
                prev_four_state = np.zeros((1, 4, 84, 84), dtype='uint8')
                prev_four_state[0,3,:,:] = state
                for j in range(3):
                    self.agent.add_to_memory(None, None, None, None)
                self.agent.add_to_memory(None, None, None, state.copy())
            i += 1
            self.frame += 1

    def training(self):
        i = 0
        sum_loss = 0
        state_batch = np.zeros((BATCH_SIZE, 4, 84, 84))
        next_state_batch = np.zeros((BATCH_SIZE, 4, 84, 84))
        while i < TRAIN_FREQUENCY:
            j = 0
            index_l = []
            while j < BATCH_SIZE:
                idx = np.random.randint(4, len(self.agent.memory))
                if self.agent.memory[idx][3] is None or self.agent.memory[idx-1][3] is None:
                    continue
                else:
                    index_l.append(idx)
                    for k in range(4):
                        if self.agent.memory[idx-4+k] != None:
                            state_batch[j,0,:,:] = self.agent.memory[idx-4+k][3]
                    next_state_batch[j,0:3,:,:] = state_batch[j,1:4,:,:]
                    next_state_batch[j,3,:,:] = self.agent.memory[idx][3]
                j += 1
            assert len(index_l) == BATCH_SIZE
            V = np.amax(self.agent.evaluate_q(next_state_batch), axis=1)
            Q = self.agent.evaluate_q(state_batch)
            for j in range(BATCH_SIZE):
                action, reward, done, state = self.agent.memory[index_l[j]]
                if done:
                    Q[j][action] = reward
                else:
                    Q[j][action] = reward + GAMMA * V[j]
            loss = self.agent.optimize_step(state_batch, Q)
            sum_loss += loss
            progressBar(i + 1, TRAIN_FREQUENCY, "Iteration " + str(self.iteration) + ": Train Progress")
            i += 1
        self.policy_net.load_state_dict(self.training_net.state_dict())
        print(" - Loss: " + str(sum_loss/TRAIN_FREQUENCY))
        print("Sampled Q Value: ")
        print(str(Q))

    def evaluation(self):
        total_score = 0
        for i in range(EVALUATION_EPISODES):
            done = False
            state = self.env.reset()
            prev_four_state = np.zeros((1, 4, 84, 84), dtype='uint8')
            prev_four_state[:,3,:,:] = state
            while not done:
                action = self.agent.optimal_action(prev_four_state)
                state, reward, done = self.env.step(action)
                self.state[:,0:3,:,:] = self.state[:,1:4,:,:]
                self.state[:,3,:,:] = state
                total_score += reward
            progressBar(i + 1, EVALUATION_EPISODES, "Iteration " + str(self.iteration) + ": Evaluation Progress")
        print(" - Average Score: " + str(total_score/EVALUATION_EPISODES))

    def evaluation_with_render(self):
        done = False
        state = self.env.reset()
        prev_four_state = np.zeros((1, 4, 84, 84), dtype='uint8')
        prev_four_state[:,3,:,:] = state
        while not done:
            action = self.agent.optimal_action(self.state)
            print(action)
            print(self.agent.evaluate_q(self.state))
            state, reward, done = self.env.step(action, render = True)
            self.state[:,0:3,:,:] = self.state[:,1:4,:,:]
            self.state[:,3,:,:] = state

    def checkpoint(self, savepath):
        torch.save({
            'epoch': self.iteration,
            'frame': self.frame,
            'model_state_dict': self.agent.policy_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'memory': self.agent.memory
        }, savepath)

    def loadCheckpoint(self, savepath):
        checkpoint = torch.load(savepath)
        self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.agent.training_net.load_state_dict(checkpoint['model_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['epoch']
        self.frame = checkpoint['frame']
        self.agent.memory = checkpoint['memory']


    def trainSolver(self):
        while self.iteration < NUM_ITERATION:
            self.exploration()
            self.training()
            self.evaluation()
            self.iteration += 1
            self.checkpoint(self.savepath, self.memorysavepath)


trainSolver = TrainSolver()
trainSolver.trainSolver()

