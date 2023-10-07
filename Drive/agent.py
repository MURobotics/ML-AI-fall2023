# import torch
import random
from game import DriveGameAI
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0 # Number of Games Played
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() when maxlen is reached
        # self.model
        # self.trainer 
        # TODO: MODEL AND TRAINER
    
    # Use game to get state!
    def get_state(self, game):

        state = [
            game.car.vel,
            game.car.angle,
            game.car.x,
            game.car.y
            # George help us!
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
        return state

    # Store Memory of previous moves (state, action, reward, next_state, done/game over)
    def remember(self):
        pass
    
    # Training current state / information
    def train_short_memory(self):
        pass
    
    # Training batch of past
    def train_short_memory(self):
        pass

    # Obtain move from model!!!
    def get_action(self):
        # Update randomness based on number of games
        self.epsilon = 101 # - self.n_games
        action = [0, 0]
        if random.randint(0, 100) < self.epsilon:
            action[0] = random.randint(0, 2)
            action[1] = random.randint(0, 2)
            pass
        else:
            # Get move from model
            pass

        return action

# Driver Function
def train():
    agent = Agent()
    game = DriveGameAI()

    while True:

        # get old/current state

        # get move
        action = agent.get_action()

        # perform move (recieve reward and if game is over)
        # and get new state

        game.play_move(action)

        # train short memory

        # remember current move

        # if new game, save/plot data!

train()

