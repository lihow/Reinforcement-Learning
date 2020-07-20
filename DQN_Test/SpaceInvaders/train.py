"""
ref: 
https://towardsdatascience.com/optimized-deep-q-learning-for-automated-atari-space-invaders-an-implementation-in-tensorflow-2-0-80352c744fdc
http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/ **
https://blog.csdn.net/u012465304/article/details/81328116
https://github.com/chenghongkuan/AIGame/blob/master/run_SpaceInvaders.py
"""

import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

env = gym.make("SpaceInvaders-v0")

n_putputs = env.action_space.n 
print(n_putputs)
print(env.get_action_meanings())

def preprocess(observation):
  observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
  observation = observation[26:110,:]
  ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
  return np.reshape(observation,(84,84,1))

def test_preprocess():
  env.reset()
  action0 = 0  # do nothing
  observation0, reward0, terminal, info = env.step(action0)
  print("Before processing: " + str(np.array(observation0).shape))
  plt.imshow(np.array(observation0))
  plt.show()
  observation0 = preprocess(observation0)
  print("After processing: " + str(np.array(observation0).shape))
  plt.imshow(np.array(np.squeeze(observation0)))
  plt.show()

