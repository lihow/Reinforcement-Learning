import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, n_states, n_middles, n_actions):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_states, n_middles)
    self.fc1.weight.data.normal_(0, 0.1)
    self.out = nn.Linear(n_middles, n_actions)
    self.out.weight.data.normal_(0, 0.1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    actions_value = self.out(x)
    return F.softmax(actions_value, dim=1)

class PolicyGradient:
  def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr = learning_rate
    self.gamma = reward_decay

    self.ep_obs, self.ep_as, self.ep_rs = [], [], []
    
    self._build_net()
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

  def _build_net(self):
    self.net = Net(self.n_features, 10, self.n_actions)

  def choose_action(self, observation):
    prob_weights = self.net(torch.Tensor(observation[np.newaxis, :]))
    action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.view(-1).detach().numpy())
    return action

  def store_transition(self, s, a, r):
    self.ep_obs.append(s)
    self.ep_as.append(a)
    self.ep_rs.append(r)

  def learn(self):
    # discount and normalize episode reward
    discounted_ep_rs_norm = self._discount_and_norm_reward()
    # train on episode
    output = self.net(torch.Tensor(self.ep_obs))
    one_hot = torch.zeros(len(self.ep_as), self.n_actions).scatter_(1, torch.LongTensor(self.ep_as).view(-1, 1), 1)
    neg = torch.sum(-torch.log(output) * one_hot, 1)
    loss = (neg * torch.Tensor(discounted_ep_rs_norm)).mean()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.ep_as = []
    self.ep_obs = []
    self.ep_rs = []
    return discounted_ep_rs_norm

  def _discount_and_norm_reward(self):
    # discount episode rewards
    discounted_ep_rs = np.zeros_like(self.ep_rs)
    runing_add = 0
    for t in reversed(range(0, len(self.ep_rs))):
      runing_add = runing_add * self.gamma + self.ep_rs[t]
      discounted_ep_rs[t] = runing_add

    # normalize episode reward
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return discounted_ep_rs
