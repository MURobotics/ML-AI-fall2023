#tutorials used to help write this program:
# https://www.youtube.com/watch?v=wc-FxNENg9U
# https://www.youtube.com/watch?v=L8ypSXwyBds
# https://www.youtube.com/watch?v=L3ktUWfAMPg

import torch
import random
import numpy as np
from game import DriveGameAI
from collections import deque
from model import Linear_QNet, QTrainer
import math



MAX_MEMORY = 100000
BATCH_SIZE = 64
LR = .001

class Agent:
    def __init__(self):
        self.n_games = 0 # Number of Games Played
        self.epsilon = 0.9 # Randomness
        self.epsilonend = .1
        self.epsilondecay = 5e-4
        self.gamma = 0.99 # Discount Rate
        self.policynet = Linear_QNet(7, 50, 3)
        self.trainer = QTrainer(self.policynet, lr=LR)
        #self.policynet.load_state_dict(torch.load("model/model.pth")) uncomment this  to load a saved model
        #self.trainer.optimizer.load_state_dict(torch.load("model/optimizer.pth")) uncomment this to load a saved optimizer

        #memory setup
        self.mem_size = MAX_MEMORY
        self.state_memory = np.zeros((self.mem_size, 7), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 7), dtype = np.float32)
        self.action_memory = np.zeros((self.mem_size), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.mem_cntr = 0
        
    
    # Use game to get state!
    def get_state(self, game):

        state = [
            (game.car.vel ),
            #((game.car.angle % 360)/ 36),
            #(game.car.x / 10),
            #(game.car.y / 10),
            (game.walldistancearray[0]),
            (game.walldistancearray[1]),
            (game.walldistancearray[2]),
            (game.walldistancearray[3]),
            (game.walldistancearray[4]),
            (game.walldistancearray[5]),
            #(game.car.correctDirection())# 1 == Correct direction and 0 == Wrong direction you can change this value it doesn't matter.
        ]
        return state

    # Store Memory of previous moves
    def remember(self, state_old, action, reward, state_new, game_over):
        index = self.mem_cntr % MAX_MEMORY
        self.state_memory[index] = state_old
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = game_over
        self.mem_cntr +=1

        
    # Obtain move from model!!!
    def get_action(self, state):
        action = [0, 0, 0]
        if random.random() <= self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
            #print(f"random move selected  {move}      {self.epsilon}")
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            actions = self.policynet.forward(state0)
            move = torch.argmax(actions).item()
            action[move] = 1
            #print(f"selected move {move}              {self.epsilon}")

        return move
    

    #model learns from past states.
    def train_step(self, state_memory, new_state_memory, reward_memory, terminal_memory,action_memory, max_memory, batch_size):
        if self.mem_cntr < batch_size:
            return
        self.trainer.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, max_memory)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        batch_index = np.arange(batch_size, dtype=np.int32)
        state_batch = torch.tensor(state_memory[batch])
        new_state_batch = torch.tensor(new_state_memory[batch])
        reward_batch = torch.tensor(reward_memory[batch])
        terminal_batch = torch.tensor(terminal_memory[batch])
        action_batch = action_memory[batch]

        
        q_eval = self.policynet.forward(state_batch)[batch_index,action_batch]
        q_next = self.policynet.forward(new_state_batch) 
        

        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next,dim=1)[0]

        
        loss = self.trainer.criterion((q_target),(q_eval))
        loss.backward()
        self.trainer.optimizer.step()
        

      


# Driver Function
def train():
    agent = Agent()
    game = DriveGameAI()
    best_episode_score=0
    episode_score=0
    while True:
       
        # get old/current state
        state_old = agent.get_state(game)
        # get move
        action = agent.get_action(state_old)

        # perform move (recieve reward and if game is over)
        # and get new state
        reward, game_over = game.play_move(action)
        state_new = agent.get_state(game)

       

        # remember current move
        agent.remember(state_old, action, reward, state_new, game_over)

        agent.train_step(agent.state_memory, agent.new_state_memory,agent.reward_memory,agent.terminal_memory,agent.action_memory, MAX_MEMORY, BATCH_SIZE)
        # if new game, save/plot data!
        episode_score += reward
        if game_over:
            agent.n_games += 1
            #uncomment this block to save agent and optimizer.
            # if(episode_score > best_episode_score):
            #     best_episode_score = episode_score
            #     agent.policynet.save()
            #     torch.save(agent.trainer.optimizer.state_dict(),"./model/optimizer.pth")
            #     print(f"agent and optimizer saved")
            #     print(episode_score)
            episode_score=0
        if (agent.epsilon > agent.epsilonend):
            agent.epsilon = agent.epsilon - agent.epsilondecay
        else:
            agent.epsilon = agent.epsilonend
        

train()

