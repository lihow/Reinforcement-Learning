#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from NN_module import CNN_Net, DuelingNet
from Buffer_module import Buffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQN():
  batch_size = 128
  lr = 1e-4
  epsilon = 0.15
  memory_capacity = int(1e4)
  gamma = 0.99
  replace_target_iter  = 200
  clip_norm_max = 1

  def __init__(self, num_state, num_action, dueling=False, prioritized=True):
    super(DQN, self).__init__()
    self.num_state = num_state
    self.num_action = num_action
    self.state_len = int(np.sqrt(self.num_state))
    self.dueling = dueling
    self.prioritized = prioritized
    self._build_net()
    self.learn_step_counter = 0
    self.buffer = Buffer(self.num_state, 'priority', self.memory_capacity)
    self.initial_epsilon = self.epsilon
    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    self.loss_func = nn.MSELoss()

  def _build_net(self):
    if self.dueling:
      self.eval_net = DuelingNet(self.state_len, self.num_action).to(device)
      self.target_net = DuelingNet(self.state_len, self.num_action).to(device)
    else:
      self.eval_net = CNN_Net(self.state_len, self.num_action).to(device)
      self.target_net = CNN_Net(self.state_len, self.num_action).to(device)

  def select_action(self, state, random=False, deterministic=False):
    """
    torch.unsqueeze() 给指定位置加上维数为一的维度
    a = torch.randn(2, 3) => torch.Size([2, 3])
    b = torch.unsqueeze(a, 0) => torch.Size([1, 2, 3])
    c = torch.unsqueeze(a, 1) => torch.Size([2, 1, 3])
    """
    state = torch.unsqueeze(torch.FloatTensor(state), 0) 
    if not random and np.random.random() > self.epsilon or deterministic:
      action_value = self.eval_net.forward(state.to(device))
      action = torch.max(action_value.reshape(-1,4), 1)[1].data.cpu().numpy()[0]
    else:
      action = np.random.randint(0, self.num_action)
    return action

  def store_transition(self, state, action, reward, next_state):
    state = state.reshape(-1)
    next_state = next_state.reshape(-1)
    transition = np.hstack((state, [action, reward], next_state))
    self.buffer.store(transition)

  def learn(self):
    if self.learn_step_counter % self.replace_target_iter  ==0:
      self.target_net.load_state_dict(self.eval_net.state_dict())

    if self.prioritized:
      batch_memory, (tree_idx, ISWeights) = self.buffer.sample(self.batch_size)
    else:
      batch_memory, _ = self.buffer.sample(self.batch_size)

    batch_state = torch.FloatTensor(batch_memory[:, :self.num_state]).to(device)
    batch_action = torch.LongTensor(batch_memory[:, self.num_state: self.num_state+1].astype(int)).to(device)
    batch_reward = torch.FloatTensor(batch_memory[:, self.num_state+1: self.num_state+2]).to(device)
    batch_next_state = torch.FloatTensor(batch_memory[:,-self.num_state:]).to(device)

    q_eval = self.eval_net(batch_state).gather(1, batch_action)
    q_next = self.target_net(batch_next_state).detach()

    max_action_indexes = self.eval_net(batch_next_state).detach().argmax(1)
    select_q_next = q_next.gather(1, max_action_indexes.unsqueeze(1))

    q_target = batch_reward + self.gamma * select_q_next

    loss = self.loss_func(q_eval, q_target)

    if self.prioritized:
      loss = loss * torch.FloatTensor(ISWeights).to(device)
      td_errors  = loss.cpu().detach().numpy()
      loss = torch.mean(loss)
      self.buffer.update(tree_idx, td_errors)

    self.optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪，防止梯度消失或爆炸
    nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.clip_norm_max)
    self.optimizer.step()

    self.learn_step_counter+=1

  def save(self, path=None, name='dqn_net.pkl'):
    path = self.save_path if not path else path
    # utils.check_path_exist(path)
    torch.save(self.eval_net.state_dict(), path + name)

  def load(self, path=None, name='dqn_net.pkl'):
    path = self.save_path if not path else path
    self.eval_net.load_state_dict(torch.load(path + name))

  def epsilon_decay(self, episode, total_episode):
    self.epsilon = self.initial_epsilon * (1 - episode / total_episode)