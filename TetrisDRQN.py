from Tetris import *
from TetrisDRQNAgent import *
from datetime import datetime
from statistics import mean, median
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

env = Tetris()
episodes = 1500
max_steps = None
epsilon_stop_episode = 1500
mem_size = 20000
discount = 0.95
batch_size = 512
epochs = 1
render_every = 50
log_every = 50
replay_start_size = 2000
train_every = 1
n_neurons = [32, 32]
render_delay = None
activations = ['relu', 'relu', 'linear']

agent = DRQNAgent(env.get_state_size(),
                  n_neurons=n_neurons, activations=activations,
                  epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                  discount=discount, replay_start_size=replay_start_size)



DRQNscores = []
DRQNmeanscores = []
DRQNsd = []

for episode in tqdm(range(episodes)):
    current_state = env.reset()
    done = False
    steps = 0

    if render_every and episode % render_every == 0:
        render = True
    else:
        render = False

    # Game
    while not done and (not max_steps or steps < max_steps):
        next_states = env.get_next_states()
        best_state = agent.best_state(next_states.values())
        
        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        reward, done = env.play(best_action[0], best_action[1], render=render,
                                render_delay=render_delay)
        
        agent.add_to_memory(current_state, next_states[best_action], reward, done)
        current_state = next_states[best_action]
        steps += 1

    DRQNscores.append(env.get_game_score())
    
    
    # print(scores)
    # episodes = list(range(episode))
    # print(episodes)
    # plt.plot(episodes, scores, label = 'score') 
    # # naming the x axis 
    # plt.xlabel('episode') 
    # # naming the y axis 
    # plt.ylabel('score') 
  
    # # giving a title to my graph 
    # plt.title('Scare Graph') 
      
    # # function to show the plot 
    # plt.show()

    # Train
    if episode % train_every == 0:
        agent.train(batch_size=batch_size, epochs=epochs)

    # Logs
    if log_every and episode and episode % log_every == 0:
        avg_score = mean(DRQNscores[-log_every:])
        sd = statistics.stdev(DRQNscores[-log_every:])
        
        DRQNmeanscores.append(avg_score)
        DRQNsd.append(sd)
 
        # episodes = list(range(episode))

        # # plotting the points  
        # plt.plot(episodes, meanloss, label = 'mean loss') 
        # plt.plot(episodes, maxloss, label = 'max loss') 
        # plt.plot(episodes, minloss, label = 'min loss') 
        # # naming the x axis 
        # plt.xlabel('episode') 
        # # naming the y axis 
        # plt.ylabel('loss') 
      
        # # giving a title to my graph 
        # plt.title('Loss Graph') 
          
        # # function to show the plot 
        # plt.show()


import json
with open('DRQNScore.txt', 'w') as f:
    f.write(json.dumps(DRQNscores))

with open('DRQNmeanscores.txt', 'w') as f:
    f.write(json.dumps(DRQNmeanscores))

with open('DRQNsd.txt', 'w') as f:
    f.write(json.dumps(DRQNsd))
