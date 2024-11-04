import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import torch.nn.functional as F

# placeholder hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.0001
EPS_DECAY = 3000
TAU = 0.005
LR = 5e-5

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# This is to save the agent's experiences and sample from them later when training the NN 
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Network architecture
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
