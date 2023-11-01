import torch
from game import DriveGameAI
from model import Linear_QNet
import os.path

def get_state(game):#Stole from agent.py
    state = [
        int(game.car.vel),
        game.walldistancearray[0]/50,
        game.walldistancearray[1]/50,
        game.walldistancearray[2]/50,
        game.walldistancearray[3]/50,
        game.walldistancearray[4]/50,
        game.walldistancearray[5]/50,
    ]       
    return state


def get_action(state, model):
    action = [0, 0, 0, 0, 0, 0, 0, 0, 0]#Removed randomness element from get_action, this is all straight AI.
    state0 = torch.tensor(state, dtype=torch.float)
    prediciton = model(state0)
    move = torch.argmax(prediciton).item()
    action[move] = 1

    return action

def viewModel():#Loads the best model you have saved and displays it.
    game = DriveGameAI()
    model = Linear_QNet(7, 9, 9)
    if not os.path.isfile("model/bestModel.pt"):
        print("You don't have a bestModel.pt silly!\nRun agent.py before trying to view a model.")
        return
    model.load_state_dict(torch.load("model/bestModel.pt"))#Can change the path if you have models saved somewhere else or want to look at different models.
    model.eval()
    while True:#Model plays the game.
        state = get_state(game)
        action = get_action(state,model)
        game.play_move(action)

viewModel()