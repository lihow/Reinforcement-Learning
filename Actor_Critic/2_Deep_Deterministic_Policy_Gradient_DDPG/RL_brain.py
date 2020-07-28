import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Memory(object):
  def __init__(self, capacity, dims):
    self.capacity = capacity
    self.data = np.zeros((capacity, dims))
    self.pointer = 0

  def store_transition(self, s, a, r, s_):
    transition = np.hstack((s, a, [r], s_))
    index = self.pointer % self.capacity  # replace the old memory with new memory
    self.data[index, :] = transition
    self.pointer += 1

  def sample(self, n):
    assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
    indices = np.random.choice(self.capacity, size=n)
    return self.data[indices, :]

class ActorNet(nn.Module):
  def __init__(self, n_state, n_actions, action_board, n_middle=30):
    super(ActorNet, self).__init__()
    self.action_board = torch.Tensor(action_board)
    self.fc1 = nn.Linear(n_state, n_middle)
    self.fc1.weight.data.normal_(0, 0.1)
    self.fc2 = nn.Linear(n_middle, n_actions)
    self.fc2.weight.data.normal_(0, 0.1)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x, inplace=True)
    x = self.fc2(x)
    x = F.tanh(x)
    scaled_x = x * self.action_board
    return scaled_x

class Actor(object):
  def __init__(self, state_dim, action_dim, action_bound, learning_rate, replacement):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.action_bound = action_bound
    self.lr = learning_rate 
    self.replacement = replacement
    self.t_replace_counter = 0

    self._build_net()
    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    
  def _build_net(self, s, scope, trainable):
    self.eval_net = ActorNet(self.state_dim, self.action_dim, self.action_bound)
    self.target_net = ActorNet(self.state_dim, self.action_dim, self.action_bound)

  def learn(self, s):
    
    
