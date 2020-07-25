import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

np.random.seed(2)
torch.manual_seed(2) 

class ActorNet(nn.Module):
  def __init__(self, n_states, n_actions, n_middle=30):
    super(ActorNet, self).__init__()
    self.l1 = nn.Linear(n_states, n_middle)
    self.l1.weight.data.normal_(0, 0.1)
    self.mu = nn.Linear(n_middle, 1)
    self.mu.weight.data.mormal_(0. 0.1)
    self.sigma = nn.Linear(n_middle, 1)
    self.sigma.weight.data.normal_(0, 0.1)

  def forward(self, x):
    x = self.l1(x)
    x = F.relu(x)
    x_mu = self.mu(x)
    x_mu = F.tanh(x_mu)
    x_sigma = self.sigma(x)
    '''
     a = torch.Tensor([[[[1], [2], [4]]]])  torch.Size([1, 1, 3, 1])
     b = torch.squeeze(a)                   torch.Size([3])
    '''
    x_mu, x_sigma = torch.squeeze(x_mu*2), torch.squeeze(x_sigma+0.1)

    

    self.fc2 = nn.Linear(n_middle, n_actions)
    self.fc2.weight.data.normal_(0, 0.1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x, inplace=True)
    x = self.fc2(x)
    output = F.tanh(x, dim=1)
    # output = F.softmax(x, dim=1)
    return output

class CriticNet(nn.Module):
  def __init__(self, n_states, n_middle=30):
    super(CriticNet, self).__init__()
    self.fc1 = nn.Linear(n_states, n_middle)
    self.fc1.weight.data.normal_(0, 0.1)
    self.fc2 = nn.Linear(n_middle, 1)
    self.fc2.weight.data.normal_(0, 0.1)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x, inplace=True)
    output = self.fc2(x)
    return output

class Actor(object):
  def __init__(self, n_features, n_actions, lr=0.001):
    self.n_features = n_features
    self.n_actions = n_actions
    self.net = ActorNet(n_features, n_actions)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

  def learn(self, s, a, td):
    output = self.net(torch.Tensor(s[np.newaxis, :]))
    one_hot = F.one_hot(torch.arange(0, self.n_actions))[a]
    # one_hot = torch.zeros(1, self.n_actions).scatter_(1, torch.LongTensor([a]).view(-1, 1), 1)
    neg = torch.sum(-torch.log(output) * one_hot, 1)
    loss = (neg * torch.Tensor(td)).mean()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def choose_action(self, observation):
    prob_weights = self.net(torch.Tensor(observation[np.newaxis, :]))
    action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.view(-1).detach().numpy())
    return action

class Critic(object):
  def __init__(self, n_features, lr=0.01):
    self.n_features = n_features
    self.net = CriticNet(n_features)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

  def learn(self, s, r, s_):
    s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
    v = self.net(s)
    v_ = self.net(s_)

    td_error = torch.Tensor([r]) + GAMMA * v_ - v
    loss = torch.square(td_error)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return td_error.data.numpy()
  
# Superparameters
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F, lr=LR_C)

for i_episode in range(MAX_EPISODE):
  s = env.reset()
  t = 0
  track_r = []
  while True:
    if RENDER: env.render()

    a = actor.choose_action(s)
    s_, r, done, info = env.step(a)

    if done: r = -20
    track_r.append(r)

    td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
    actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

    s = s_
    t += 1

    if done or t >= MAX_EP_STEPS:
      ep_rs_sum = sum(track_r)
      if 'running_reward' not in globals():
        running_reward = ep_rs_sum
      else:
        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
      if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
      print("episode:", i_episode, "  reward:", int(running_reward))
      break

