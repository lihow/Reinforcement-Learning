import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
  def __init__(self, n_states, n_middle, n_actions):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_states, n_middle)
    self.fc1.weight.data.normal_(0, 0.1)
    self.out = nn.Linear(n_middle, n_actions)
    self.out.weight.data.normal_(0, 0.1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    actions_value = self.out(x)
    return actions_value



# Deep Q Network off-policy
class DeepQNetwork:
  def __init__(self, 
                n_actions, 
                n_features, 
                learning_rate=0.01, 
                reward_decay=0.9,
                e_greedy=0.9,
                replace_target_iter=300,
                memory_size=500,
                batch_size=32,
                e_greedy_increment=None,
                output_graph=False,
                ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon_max = e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epsilon_increment = e_greedy_increment
    self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

    # total learning step
    self.learn_step_counter = 0
    # initialize zero memory [s, a, r, s_]
    self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
    # consist of [target_net, evaluate_net]
    self._build_net()
    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    self.loss_func = nn.MSELoss()
    self.cost_his = []

  def _build_net(self):
    self.eval_net = Net(self.n_features, 50, self.n_actions)
    self.target_net = Net(self.n_features, 50, self.n_actions)

  def store_transition(self, s, a, r, s_):
    if not hasattr(self, 'memory_counter'):
      self.memory_counter = 0
    transition = np.hstack((s, [a, r], s_))
    # 记录下所有经历过的步, 这些步可以进行反复的学习, 所以是一种 off-policy 方法,
    # replace the old memory with new memory
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition
    self.memory_counter += 1
  
  def choose_action(self, x):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    # input only one sample
    if np.random.uniform() < self.epsilon:   # greedy
        actions_value = self.eval_net.forward(x)
        # 通常需要使用max()函数对softmax函数的输出值进行操作，求出预测值索引
        action = torch.max(actions_value, 1)[1].data.numpy()[0]
    else:   # random
        action = np.random.randint(0, self.n_actions)
    return action  

  def learn(self):
    # target parameter update
    if self.learn_step_counter % self.replace_target_iter == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # sample batch transitions
    if self.memory_counter > self.memory_size:
      sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
      sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    
    batch_memory  = self.memory[sample_index, :]
    # batch_memory = (s, [a, r], s_)
    b_s = torch.FloatTensor(batch_memory [:, :self.n_features])
    b_a = torch.LongTensor(batch_memory [:, self.n_features:self.n_features+1].astype(int))
    b_r = torch.FloatTensor(batch_memory [:, self.n_features+1:self.n_features+2])
    b_s_ = torch.FloatTensor(batch_memory [:, -self.n_features:])

    # q_eval w.r.t the action in experience
    q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
    q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
    q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
    loss = self.loss_func(q_eval, q_target)
    # print(loss)
    self.cost_his.append(loss.data.numpy())

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()  

    # increase epsilon
    self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
    self.learn_step_counter += 1
  
  def plot_cost(self):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
  
