import torch
import random
import numpy as np
from game import DriveGameAI
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0 # Number of Games Played
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() when maxlen is reached
        self.model = Linear_QNet(8, 256, 9)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: MODEL AND TRAINER
    
    # Use game to get state!
    def get_state(self, game):

        state = [
            round((game.car.vel), 1) * 10,
            round((game.car.angle % 360)/ 36, 2),
            round((game.car.x), 1),
            round((game.car.y), 1),
            game.walldistancearray[0] < 30,
            game.walldistancearray[1] < 30,
            game.walldistancearray[2] < 30,
            game.walldistancearray[3] < 30
        ]
        # Velocity
        # Angle
        # X
        # Y
        # Left Distance
        # Right Distance
        # Up Distance
        # Down Distance
        # Distance to Reward Gate

        # NOTE: np.array is faster due to homogeneous nature
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

    # Obtain move from model!!!
    def get_action(self, state):
        # Update randomness based on number of games
        self.epsilon = 80 - self.n_games*2
        action = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 8)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediciton = self.model(state0)
            move = torch.argmax(prediciton).item()
            action[move] = 1
            # print(prediciton)

        return action

# Driver Function
def train():
    agent = Agent()
    game = DriveGameAI()

    while True:

        # get old/current state
        state_old = agent.get_state(game)

        print(state_old)

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

