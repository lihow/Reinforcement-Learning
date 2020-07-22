#encoding=utf-8
from RL_brain import DQN
from Game2048Env import Game2048Env
import torch
import numpy as np 
import time
import os 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

train_episodes = 20000
# train_episodes = 20
test_episodes = 50
ifrender = False
eval_interval = 25
epsilon_decay_interval = 100

def log2_shaping(s, divide=16):
    s = np.log2(1 + s) / divide
    return s

def train(RL):
  scores = []
  episodes = train_episodes
  env = Game2048Env()

  for i in range(episodes):
    state, reward, done, info = env.reset()
    # 归一化
    state = log2_shaping(state)

    # start = time.time()
    # loss = None
    while True:
      if RL.buffer.memory_counter <= RL.memory_capacity:
        action = RL.select_action(state, random=True)
      else:
        action = RL.select_action(state)

      next_state, reward, done, info = env.step(action)
      next_state = log2_shaping(next_state)
      reward = log2_shaping(reward, divide=1)

      RL.store_transition(state, action, reward, next_state)
      state = next_state
      
      if ifrender:
        env.render_img()

      if RL.buffer.memory_counter > RL.memory_capacity:
        RL.learn()
      
      if done:
        print("Epoch: {}/{}, highest: {}".format(i, episodes, info['highest']))
        scores.append(info['highest'])
        if i % epsilon_decay_interval == 0:
          RL.epsilon_decay(i, episodes)
        break
  return scores


if __name__ == "__main__":
  RL = DQN(num_state=16, num_action=4, dueling=True)
  scores = train(RL)

  plt.figure(figsize=(18, 6), dpi=200)
  plt.figure(1)
  plt.plot(np.array(scores), c='r')
  plt.ylabel('highest score')
  plt.xlabel('training steps')
  plt.savefig('result.jpg')