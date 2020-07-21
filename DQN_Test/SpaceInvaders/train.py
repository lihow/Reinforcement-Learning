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
from RL_brain import DQNPrioritizedReplay

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

EPISODE = 20
MEMORY_SIZE = 100

def train(RL):
  total_steps = 0
  observation = env.reset()
  for i_episode in range(EPISODE):
    observation = env.reset()
    observation = preprocess(observation)
    total_reward = 0
    while True:
      env.render()
      action = RL.choose_action(observation)
      observation_, reward, done, info = env.step(action)

      reward = reward / 200 # reward归一化
      total_reward += reward
      observation_ = preprocess(observation_)

      RL.store_transition(observation, action, reward, observation_)

      if total_steps > MEMORY_SIZE:
        RL.learn()
      
      if done:
        print('i_episode: '+str(i_episode)+' finished')
        break

      observation = observation_
      total_steps += 1
  env.close()


if __name__ == "__main__":
  RL_prio_dueling = DQNPrioritizedReplay(n_actions=env.action_space.n,
                  n_features=2,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=True)  
  train(RL_prio_dueling)


