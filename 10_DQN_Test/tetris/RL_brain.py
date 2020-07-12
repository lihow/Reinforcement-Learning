#encoding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, sample
from collections import deque
np.random.seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Net(nn.Module):
  def __init__(self, n_states, n_middle, n_actions):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_states, n_middle)
    self.fc1.weight.data.normal_(0, 0.1)
    self.fc2 = nn.Linear(n_middle, n_middle)
    self.fc2.weight.data.normal_(0, 0.1)
    self.out = nn.Linear(n_middle, n_actions)
    self.out.weight.data.normal_(0, 0.1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    actions_value = self.out(x)
    return actions_value

class DuelingNet(nn.Module):
  def __init__(self, n_states, n_middle, n_actions):
    super(DuelingNet, self).__init__()
    self.fc1 = nn.Linear(n_states, n_middle)
    self.fc1.weight.data.normal_(0, 0.1)
    self.fc2 = nn.Linear(n_middle, n_middle)
    self.fc2.weight.data.normal_(0, 0.1)
    self.adv = nn.Linear(n_middle, n_actions)
    self.adv.weight.data.normal_(0, 0.1)
    self.val = nn.Linear(n_middle, 1)
    self.val.weight.data.normal_(0, 0.1)
    
  
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    adv = self.adv(x)
    # val = self.val(x).expand(adv.size())
    # actions_value = val + adv - adv.mean().expand(adv.size())
    val = self.val(x)
    actions_value = val + adv -adv.mean()
    return actions_value


# DQNPrioritizedReplay based DDQN
class DQNPrioritizedReplay:
  def __init__(self, 
                n_actions, 
                n_features, 
                learning_rate=0.01, 
                reward_decay=0.99,
                e_greedy=0.9,
                replace_target_iter=300,
                memory_size=500,
                batch_size=32,
                e_greedy_increment=None,
                prioritized = True,
                dueling = False,
                test = False,
                ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon_min = 1 - e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epsilon_increment = e_greedy_increment
    self.epsilon = 1 if e_greedy_increment is not None else self.epsilon_min

    self.prioritized = prioritized
    self.dueling = dueling

    # total learning step
    self.learn_step_counter = 0
    if self.prioritized:
      self.memory = Memory(capacity=self.memory_size)
    else:
      self.memory = deque(maxlen=self.memory_size)
    # consist of [target_net, evaluate_net]
    self.test = test
    if self.test is False:
      self._build_net()
      self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    self.loss_func = nn.MSELoss()
    self.cost_his = []

  def _build_net(self):
    if self.dueling:
      self.eval_net = DuelingNet(self.n_features, 64, self.n_actions).to(device)
      self.target_net = DuelingNet(self.n_features, 64, self.n_actions).to(device)
    else:      
      self.eval_net = Net(self.n_features, 50, self.n_actions).to(device)
      self.target_net = Net(self.n_features, 50, self.n_actions).to(device)
    
  def load_model(self, model_path):
    if torch.cuda.is_available():
      self.eval_net = torch.load(model_path)
    else:
      self.eval_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    self.eval_net.eval()

  def store_transition(self,state, reward, next_state, done):
    if self.prioritized:
      transition = np.hstack((state, reward, next_state, done))
      self.memory.store(transition)
    else:
      self.memory.append([state, reward, next_state, done])
  
  def choose_action(self, next_actions, next_states, epsilon=None, is_random=True):
    if epsilon is not None:
      self.epsilon = epsilon
    else:
      # decrease epsilon
      self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min

    next_states = torch.stack(tuple(torch.FloatTensor(state) for state in next_states))

    predictions = self.eval_net(next_states.to(device))[:, 0]

    if is_random and np.random.uniform() < self.epsilon:
      index = randint(0, next_states.shape[0] - 1)
    else:
      index = torch.argmax(predictions).item()
    
    action = next_actions[index]
    next_state = next_states[index, :]

    return action, next_state

  def learn(self):
    # target parameter update
    if self.learn_step_counter % self.replace_target_iter == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict())

    if self.prioritized:
      # batch_memory = (state, reward, next_state, done)
      # batch过大或削弱惩罚？ 
      tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
      state_batch = torch.FloatTensor(batch_memory[:, :self.n_features]).to(device)
      reward_batch = torch.FloatTensor(batch_memory[:, self.n_features:self.n_features+1]).to(device)
      next_state_batch = torch.FloatTensor(batch_memory[:, self.n_features+1:2*self.n_features+1]).to(device)
      done_batch = torch.FloatTensor(batch_memory[:, -1:]).to(device)
    else:
      batch_memory = sample(self.memory, min(len(self.memory), self.batch_size))
      state_batch, reward_batch, next_state_batch, done_batch = zip(*batch_memory)
      state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
      reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
      next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)


    q_values = self.eval_net(state_batch)
    next_prediction_batch  = self.target_net(next_state_batch).detach().to(device)
    y_batch = torch.cat(
        tuple(reward if done else reward + self.gamma * prediction for reward, done, prediction in
              zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

    loss = self.loss_func(q_values, y_batch)

    if self.prioritized:
      loss = loss * torch.FloatTensor(ISWeights).to(device)
      td_errors  = loss.cpu().detach().numpy()
      loss = torch.mean(loss)
      self.memory.batch_update(tree_idx, td_errors)
    
    # print(loss)
    self.cost_his.append(loss.cpu().detach().numpy())

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()  

    self.learn_step_counter += 1
  
  def save_model(self, path):
    torch.save(self.eval_net, path)

  def plot_cost(self):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()