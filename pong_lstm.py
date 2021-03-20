from LSTMdqn import *
from env_pong import Environment
from collections import deque
import numpy as np
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys




NUM_ITERATION = 100
LEARNING_RATE = 0.00025
MAX_MEMORY = 1000000
EXPLORE_FREQUENCY = 10000
TRAIN_FREQUENCY = 10000
EVALUATION_EPISODES = 10
BATCH_SIZE = 32
GAMMA = 0.99
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
INITIAL_EXPLORATION_FRAME = 50000
FINAL_EXPLORATION_FRAME = 1000000
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
        self.LSTMDQN = LSTMDQN(ACTION_SAPCE)
        self.LSTMDQN.apply(self.init_weights)
        self.LSTMDQN = self.LSTMDQN.double()
        
        self.LSTMDQN_hidden_state, self.LSTMDQN_cell_state = self.LSTMDQN.init_states()
        
        
        self.memory = deque(maxlen = MAX_MEMORY)
        self.optimizer = optim.Adam(self.LSTMDQN.parameters(), lr=LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()
        self.use_gpu = torch.cuda.is_available()
        
        
        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.LSTMDQN.to(self.device)

    def init_weights(self, m):
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            torch.nn.init.zeros_(m.bias.data)

    def add_to_memory(self, s1, a, r, s2, done):
        self.memory.append((s1, a, r, s2, done))

    def evaluate_q(self, state):
        inputs = torch.from_numpy(state.copy()).to(self.device)
        #lstm update train 
        action, self.LSTMDQN_hidden_state, self.LSTMDQN_cell_state = self.LSTMDQN.forward(inputs, self.LSTMDQN_hidden_state, self.LSTMDQN_cell_state)
        q_value = action.detach().cpu().numpy()
        return q_value

    def optimal_action(self, state):
        inputs = torch.from_numpy(state.copy()).to(self.device)
        #lstm update optimize 
        action, self.LSTMDQN_hidden_state, self.LSTMDQN_cell_state = self.LSTMDQN.forward(inputs, self.LSTMDQN_hidden_state, self.LSTMDQN_cell_state)
        q_value = action.detach().cpu().numpy()

        return np.argmax(q_value)

    #lstm update
    def optimize_step(self, state_batch):
        self.dqn.zero_grad()
        inputs = torch.from_numpy(state_batch.copy()).to(self.device)
        #lstm update
        outputs, self.LSTMDQN_hidden_state, self.LSTMDQN_cell_state = self.LSTMDQN.forward(inputs, LSTMDQN_hidden_state, LSTMDQN_cell_state)
        
        targetQ = torch.from_numpy(targetQ.copy()).to(self.device)
        loss = self.criterion(outputs, targetQ)
        loss.backward()
        self.optimizer.step()


class TrainSolver:
    def __init__(self):
        self.agent = Agent()
        self.env = Environment()
        self.iteration = 0
        self.frame = 0
        self.state = np.zeros((1, 4, 84, 84))

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
        #lstm update
        self.agent.LSTMDQN_hidden_state, self.agent.LSTMDQN_cell_state = self.agent.LSTMDQN.reset_states(self.agent.LSTMDQN_hidden_state, self.agent.LSTMDQN_cell_state)
        
        self.state = np.zeros((1, 4, 84, 84))
        self.state[:,3,:,:] = state
        while i < EXPLORE_FREQUENCY:
            #lstm update
            action = self.agent.optimal_action(self.state)
            
            action = self.exploration_policy(action)
            state, reward, done = self.env.step(action)
            prev_state = self.state.copy()
            self.state[:,0:3,:,:] = self.state[:,1:4,:,:]
            self.state[:,3,:,:] = state
            self.agent.add_to_memory(prev_state, action, reward, self.state, done)
            if done:
                state = self.env.reset()
                self.state = np.zeros((1, 4, 84, 84))
                self.state[:,3,:,:] = state
            i += 1
            self.frame += 1

    def training(self):
        i = 0
        while i < TRAIN_FREQUENCY:
            minibatch = random.sample(self.agent.memory, BATCH_SIZE)
            state_batch = np.zeros((BATCH_SIZE, 4, 84, 84))
            next_state_batch = np.zeros((BATCH_SIZE, 4, 84, 84))
            for j in range(BATCH_SIZE):
                state, action, reward, next_state, done = minibatch[j]
                state_batch[j,:,:,:] = state
                next_state_batch[j,:,:,:] = next_state
            V = np.amax(self.agent.evaluate_q(next_state_batch), axis=1)
            Q = self.agent.evaluate_q(state_batch)
            for j in range(BATCH_SIZE):
                state, action, reward, next_state, done = minibatch[j]
                if done:
                    Q[j][action] = reward
                else:
                    Q[j][action] = reward + GAMMA * V[j]
                    
            #lstm update
            self.agent.optimize_step(state_batch, Q)
            
            progressBar(i + 1, TRAIN_FREQUENCY, "Iteration " + str(self.iteration) + ": Train Progress")
            i += 1

    def evaluation(self):
        total_score = 0
        for i in range(EVALUATION_EPISODES):
            done = False
            state = self.env.reset()
            self.state = np.zeros((1, 4, 84, 84)) 
            self.state[:,3,:,:] = state
            while not done:
                #lstm update
                action = self.agent.optimal_action(self.state)
                
                state, reward, done = self.env.step(action)
                self.state[:,0:3,:,:] = self.state[:,1:4,:,:]
                self.state[:,3,:,:] = state
                total_score += reward
            progressBar(i + 1, EVALUATION_EPISODES, "Iteration " + str(self.iteration) + ": Evaluation Progress")
        print(" - Average Score: " + str(total_score/EVALUATION_EPISODES))


    def trainSolver(self):
        while self.iteration < NUM_ITERATION:
            self.exploration()
            self.training()
            self.evaluation()
            self.iteration += 1


trainSolver = TrainSolver()
trainSolver.trainSolver()
