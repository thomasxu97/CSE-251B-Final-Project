from basicDQN import DQN
from env import Environment
from collections import deque
import numpy as np
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys


DEVICE = "cpu"
NUM_ITERATION = 40
LEARNING_RATE = 0.00025
MAX_MEMORY = 1000000
EXPLORE_FREQUENCY = 100
TRAIN_FREQUENCY = 100
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
        self.dqn = DQN(ACTION_SAPCE)
        self.dqn.apply(self.init_weights)
        self.dqn = self.dqn.double()
        self.memory = deque(maxlen = MAX_MEMORY)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()
        # use_gpu = torch.cuda.is_available()
        # if use_gpu:
        #     self.dqn = self.dqn.cuda()

    def init_weights(self, m):
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            torch.nn.init.zeros_(m.bias.data)

    def add_to_memory(self, s1, a, r, s2, done):
        self.memory.append((s1, a, r, s2, done))

    def evaluate_q(self, state):
        inputs = torch.from_numpy(state.copy())
        q_value = self.dqn(inputs).detach().numpy()
        return q_value

    def optimal_action(self, state):
        inputs = torch.from_numpy(state.copy())
        q_value = self.dqn(inputs).detach().numpy()
        return np.argmax(q_value)

    def optimize_step(self, state_batch, targetQ):
        self.dqn.zero_grad()
        inputs = torch.from_numpy(state_batch.copy())
        outputs = self.dqn(inputs)
        targetQ = torch.from_numpy(targetQ.copy())
        loss = self.criterion(outputs, targetQ)
        loss.backward()
        self.optimizer.step()


class TrainSolver:
    def __init__(self):
        self.agent = Agent()
        self.env = Environment()
        self.iteration = 0
        self.frame = 0

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
        while i < EXPLORE_FREQUENCY:
            action = self.agent.optimal_action(state)
            action = self.exploration_policy(action)

            next_state, reward, done = self.env.step(action)
            self.agent.add_to_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.env.reset()

            i += 1
            self.frame += 1

    def training(self):
        i = 0
        while i < TRAIN_FREQUENCY:
            minibatch = random.sample(self.agent.memory, BATCH_SIZE)
            state_batch = np.zeros((BATCH_SIZE, 1, 84, 84))
            next_state_batch = np.zeros((BATCH_SIZE, 1, 84, 84))
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
            self.agent.optimize_step(state_batch, Q)
            progressBar(i + 1, TRAIN_FREQUENCY, "Train Progress")
            i += 1


trainSolver = TrainSolver()
trainSolver.exploration()
trainSolver.training()



# class TestSolver:
#     def __init__(self, max_len, average_of_last_runs, model = None):
#         self.max_len = max_len
#         self.score_table = deque(maxlen=self.max_len)
#         self.model = model
#         self.average_of_last_runs = average_of_last_runs

#     def store_score(self, episode, step):
#         self.score_table.append([episode, step])

#     def plot_evaluation(self, title = "Training"):
#         print(self.model.summary()) if self.model is not None else print("Model not defined!")
#         avg_score = mean(self.score_table[1])
#         x = []
#         y = []
#         for i in range(len(self.score_table)):
#             x.append(self.score_table[i][0])
#             y.append(self.score_table[i][1])

#         average_range = self.average_of_last_runs if self.average_of_last_runs is not None else len(x)
#         plt.plot(x, y, label="score per run")
#         plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
#                  label="last " + str(average_range) + " runs average")
#         title = "CartPole-v1 " + str(title)
#         plt.title(title)
#         plt.xlabel("Runs")
#         plt.ylabel("Score")
#         plt.show()



