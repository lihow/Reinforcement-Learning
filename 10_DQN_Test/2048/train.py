#encoding=utf-8
from RL_brain import DQN
from Game2048Env import Game2048Env
import torch
import numpy as np 
import time
import os 

train_episodes = 20000
test_episodes = 50
ifrender = False
eval_interval = 25
epsilon_decay_interval = 100

def log2_shaping(s, divide=16):
    s = np.log2(1 + s) / divide
    return s

def train():
  episodes = train_episodes
  agent = DQN(num_state=16, num_action=4)
  env = Game2048Env()

  eval_max_score = 0
  for i in range(episodes):
    state, reward, done, info = env.reset()
    # 归一化
    state = log2_shaping(state)

    start = time.time()
    loss = None
    while True:
      if agent.buffer.memory_counter <= agent.memory_capacity:
        action = agent.select_action(state, random=True)
      else:
        action = agent.select_action(state)

      next_state, reward, done, info = env.step(action)
      next_state = log2_shaping(next_state)
      reward = log2_shaping(reward, divide=1)

      agent.store_transition(state, action, reward, next_state)
      state = next_state
      
      if ifrender:
        env.render_img()

      if agent.buffer.memory_counter % agent.train_interval == 0 and agent.buffer.memory_counter > agent.memory_capacity:
        loss = agent.update()
      
      if done:
        print("Epoch: {}/{}, highest: {}".format(i, episodes, info['highest']))
        if i % epsilon_decay_interval == 0:
          agent.epsilon_decay(i, episodes)
        break

  end = time.time()
  print('episode time:{} s\n'.format(end - start))


if __name__ == "__main__":
  train()