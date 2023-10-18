import torch
import random
import numpy as np
from game import DriveGameAI
from collections import deque
from model import Linear_QNet, QTrainer

# Distances based on car angle
# Trey Technique: Inside/Outside Track detection for optimal direction detection
# Delete Some Inputs????
# 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0 # Number of Games Played
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() when maxlen is reached
        self.model = Linear_QNet(11, 2567, 9)#Changed number of inputs to 11 to accomodate for new rays and correct direction input.
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: MODEL AND TRAINER
    
    # Use game to get state!
    def get_state(self, game):

        state = [
            int(game.car.vel * 10),
            int((game.car.angle % 360)/ 36),
            int(game.car.x / 10),
            int(game.car.y / 10),
            int(game.walldistancearray[0] / 10),
            int(game.walldistancearray[1] / 10),
            int(game.walldistancearray[2] / 10),
            int(game.walldistancearray[3] / 10),
            int(game.walldistancearray[4] / 10),
            int(game.walldistancearray[5] / 10),
            int(game.car.correctDirection())# 1 == Correct direction and 0 == Wrong direction you can change this value it doesn't matter.
        ]
        # Velocity
        # Angle
        # X
        # Y
        # Left Distance
        # Right Distance
        # Forward Left Distance - This is new
        # Forward Distance
        # Forward Right Distance - and this new
        # Down Distance
        # Distance to Reward Gate
        # Right direction? - Can maybe remove this and make a negative reward out of this.
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
    game = DriveGameAI()

    while True:

        # get old/current state
        state_old = agent.get_state(game)

        # print(state_old)

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

