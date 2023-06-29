import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.fc2 = nn.Linear(action_dim + hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, obs, action):
        x = self.relu(self.fc1(obs))
        x = torch.cat((x, action), dim=1) # dim=0 is the batch dimension

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        
    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        return x