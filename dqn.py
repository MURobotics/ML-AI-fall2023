import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import GameAI

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = 8 #TODO make nicer function for this
# Get the number of state observations
state = ["x", "y", "orientation"]
n_observations = len(state)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
#Deep Q-Network
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, layers=[128,128]):
        super(DQN, self).__init__()
        self.layer_num = len(layers)
        self.layers = []

        in_amount = n_observations

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(in_amount, layers[i]))
            in_amount = layers[i]

        self.layers.append(in_amount, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)
    
    # Used to save the model's parameters
    # Used after training
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean square error (Qnew - Q)^2
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        # Converting data to tensors (can be multi-demensional; one-dimensional unless we are training long-memory)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Converting Tensors to right format (state could be tuples or one val)
        if len(state.shape) == 1:
            # Convert to shape (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # Comma necessary to make done a tuple rather than just parenthetical expression
            done = (done, )
        
        # 1: predicted Q values with current state (pred is 9-length output array for action)
        pred = self.model(state)

        # 2: Q_new = r + y * Max(next_predicted Q value) -> only do this if not done
        # NOTE: using target variable, pred clone
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            # If we are done, we don't factor in next_state for Bellman Equation
            if not done[i]:
                # torch.max(self.model(next_state[i])) returns max of the action returned by the model for the next_state
                # That value is applied to Bellman Equation with discount rate gamma and the current state's reward
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            
            # torch.argmax(action).item() retunrns index of action currently taken. We set that to the Q_new
            target[i][torch.argmax(action).item()] = Q_new

        # Empties gradient (pytorch specific)
        self.optimizer.zero_grad()
        # Loss calculated (target = Q_new while pred = Q)
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0 # Number of Games Played
        self.epsilon = 0.9 # Randomness
        self.gamma = 0.9 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() when maxlen is reached
        self.num_state_variables = 3
        self.num_actions = 8
        self.model = DQN(3, 8, layers=[128,128])
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    # Use game to get state!
    def get_state(self, car):

        state = car.get_x_y_orient()

        return np.array(state, dtype=int)

    # Store Memory of previous moves
    def remember(self, state_old, action, reward, state_new, game_over):
        self.memory.append((state_old, action, reward, state_new, game_over))

    # Training current state / information
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    # Training batch of past
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        # Tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    # Obtain move from model
    def get_action(self, state):
        # Update randomness based on number of games
        self.epsilon = max(0.97 * self.epsilon, 0.05)
        action = np.zeores(self.num_actions)
        if random.random() < self.epsilon:
            move = random.randint(0, 5)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediciton = self.model(state0)
            move = torch.argmax(prediciton).item()
            action[move] = 1
            # print(action)

        return action

# Driver Function
def train():
    agent = Agent()
    game = GameAI()

    while True:

        # get old/current state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old)

        # perform move (recieve reward and if game is over)
        # and get new state
        reward, game_over = game.play_move(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        # remember current move
        agent.remember(state_old, action, reward, state_new, game_over)

        # if new game, save/plot data!
        if game_over:
            agent.n_games += 1
            agent.train_long_memory()

train()