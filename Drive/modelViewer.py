import torch
from game import DriveGameAI
from model import Linear_QNet


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
    # Update randomness based on number of games
    action = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)
    prediciton = model(state0)
    move = torch.argmax(prediciton).item()
    action[move] = 1

    return action

def viewModel():
    game = DriveGameAI()
    model = Linear_QNet(7, 9, 9)
    model.load_state_dict(torch.load("model/bestModel.pt"))
    model.eval()
    while True:
        state = get_state(game)
        action = get_action(state,model)
        game.play_move(action)

viewModel()