import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import torch

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11



def train(RL):
  total_steps = 0
  observation = env.reset()
  ep_r = 0
  while True:
      env.render()

      action = RL.choose_action(observation)
      f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)

      observation_, reward, done, info = env.step(np.array([f_action]))

      reward /= 10 

      RL.store_transition(observation, action, reward, observation_)

      ep_r += reward
      if total_steps > MEMORY_SIZE:
          RL.learn()

      if total_steps - MEMORY_SIZE > 20000:
          print('episode: ', i_episode,
                'ep_r: ', round(ep_r, 2),
                ' epsilon: ', round(RL.epsilon, 2))
          break

      observation = observation_
      total_steps += 1  
  return RL.q


if __name__ == "__main__":
  DDQN = DoubleDQN(n_actions=ACTION_SPACE,
                  n_features=3,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001, double_q=True)
  q_double = train(DDQN)
  DQN = DoubleDQN(n_actions=ACTION_SPACE,
                  n_features=3,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001, double_q=False)
  q_nature = train(DQN)
  plt.plot(np.array(q_natural), c='r', label='natural')
  plt.plot(np.array(q_double), c='b', label='double')
  plt.legend(loc='best')
  plt.ylabel('Q eval')
  plt.xlabel('training steps')
  plt.grid()
  plt.show()