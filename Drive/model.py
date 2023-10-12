import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Feed forward Neural Net as info goes in one direction (input -> output through network) 
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 1 layer neural network (only 1 hidden layer)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    # Called when running self.model(state), returning the length-9 output array 
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    # Used to "save" model once trained (for storage on computer)
    # NOTE: this has nothing to do with training the model
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
        # Converting data to tensors (Arrays used to represent data in ML, can be multi-demensional but these are one-dimensional
        # unless we are training long-memory)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # print(state.shape, state.size(), state.dim())

        # Converting Tensors to right format (state could be tuples or one val)
        # NOTE: state.shape / state.size() returns the dimensions of the state tensor
        #       if state.shape has one dimension (short-memory training), we convert it 
        #       to two dimensions using unsqueeze (thus we can do both types of training
        #       in this one function)
        # EX: tensor([1, 4, 5]) -> tensor([[1, 4, 5]]) =
        #     tensor([
        #           [1, 4, 5]
        #           ])
        if len(state.shape) == 1:
            # Convert to shape (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # Comma necessary to make done a tuple rather than just parenthetical expression
            done = (done, )
        
        # print(state.shape, state.size(), state.dim())
        
        # print(state.dim())
        
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
                # EX: [-2.0265,  1.9955, -6.3561,  6.1161,  5.1550, 11.7539,  6.9249,  8.3696, 4.8816] -> 11.7359
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