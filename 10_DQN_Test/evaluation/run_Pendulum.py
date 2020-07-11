import gym
from RL_brain import DQNPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 5


def train(RL):
    acc_r = [0]
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps-MEMORY_SIZE > 9000: 
        # env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10      # normalize to a range of (-1, 0)
        acc_r.append(reward + acc_r[-1])  # accumulated reward

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps-MEMORY_SIZE > 15000:
            break

        observation = observation_
        total_steps += 1

    env.close()
    return RL.cost_his, acc_r

if __name__ == "__main__":
  RL_prio = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=3,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=False)
  
  RL_dueling = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=3,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=False, dueling=True)

  RL_prio_dueling = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=3,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=True)

  cost_prio, raward_prio = train(RL_prio)
  print("RL_prio Finished!")
  cost_dueling, raward_dueling = train(RL_dueling)
  print("RL_dueling Finished!")
  cost_prio_dueling, raward_prio_dueling = train(RL_prio_dueling)
  print("RL_prio_dueling Finished!")

  plt.figure(1)
  plt.plot(np.array(cost_prio), c='r', label='prioritized DQN')
  plt.plot(np.array(cost_dueling), c='b', label='dueling DQN')
  plt.plot(np.array(cost_prio_dueling), c='g', label='prioritized dueling DQN')
  plt.legend(loc='best')
  plt.ylabel('cost')
  plt.xlabel('training steps')
  plt.grid()

  plt.figure(2)
  plt.plot(np.array(raward_prio), c='r', label='prioritized DQN')
  plt.plot(np.array(raward_dueling), c='b', label='dueling DQN')
  plt.plot(np.array(raward_prio_dueling), c='g', label='prioritized dueling DQN')
  plt.legend(loc='best')
  plt.ylabel('accumulated reward')
  plt.xlabel('training steps')
  plt.grid()

  plt.show()