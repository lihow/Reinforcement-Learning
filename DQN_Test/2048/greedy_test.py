from Game2048Env import Game2048Env
import torch
import numpy as np 
import time
import os 
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

EPISODES = 10
ACTIONS = ["UP", "Right", "Down", "Left"]

def print_ground(ground):
    for e_i in range(ground.shape[0]):
      for e_j in range(ground.shape[1]):
        print(ground[e_i][e_j], end=" ")
      print("")

def greedy_policy(ifrender=True):
  scores = []
  env = Game2048Env()
  for i in range(EPISODES):
    state, reward, done, info = env.reset()
    while True:
      action = 0
      max_reward = 0
      for act in range(4):     
        _, _, done, info = env.step(act)
        reward = np.sum(env.Matrix, axis=(0, 1))
        # reward = info['highest']
        if reward > max_reward:
          max_reward = reward
          action = act
        env.Matrix = state

      next_state, reward, done, info = env.step(action)
      state = next_state

      if ifrender:
        env.render_img()
      if done:
        scores.append(info['highest'])
        print("Epoch: {}/{}, highest: {}".format(i, EPISODES, info['highest']))
        break
  return scores

if __name__ == "__main__":
  scores = greedy_policy()
  plt.figure(figsize=(18, 6), dpi=200)
  plt.figure(1)
  plt.plot(np.array(scores), c='r')
  plt.ylabel('highest score')
  plt.xlabel('training steps')
  plt.savefig('greedy_result.jpg')
  plt.show()
  