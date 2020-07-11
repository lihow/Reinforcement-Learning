import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import numpy as np
# import collections

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000
# SAVE_INTERVAL = 5
EPISODES = 20


def train(RL):
  total_steps = 0
  observation = env.reset()
  steps = []
  episodes = []
  # mean_test = collections.deque(maxlen=10)
  # best_reward = 0
  for i_episode in range(EPISODES):
    observation = env.reset()
    # Train
    while True:
      # env.render()
      action = RL.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      if done:
        reward = 10

      RL.store_transition(observation, action, reward, observation_)

      if total_steps > MEMORY_SIZE:
          RL.learn()

      if done:
        print('episode', i_episode, 'finished')
        steps.append(total_steps)
        episodes.append(i_episode)
        break

      observation = observation_
      total_steps += 1  

    # # Save best model
    # if i_episode % SAVE_INTERVAL == 0 or (i_episode == EPISODES - 1):
    #   observation = env.reset()
    #   test_reward = 0
    #   try_action_count = 0
    #   while True:
    #     env.render()
    #     action = RL.choose_action(observation, is_random=True)
    #     observation_, reward, done, info = env.step(action)
    #     test_reward += reward
    #     observation = observation_
    #     # 完成任务，或者 陷入局部最优
    #     if done or try_action_count > 1000:
    #       env.close()
    #       break
    #     try_action_count += 1
    #   print('episode: {} , test_reward: {}'.format(i_episode, round(test_reward, 3)))
    #   mean_test.append(test_reward)
    #   if try_action_count < 1000:
    #     if np.mean(mean_test) > best_reward:
    #       best_reward = np.mean(mean_test)
    #       RL.save_models(i_episode, i_episode)
  env.close()
  return RL.cost_his, np.vstack((episodes, steps))

def test(RL):
    observation = env.reset()
    test_reward = 0
    try_action_count = 0
    while True:
      env.render()
      action = RL.choose_action(observation, is_random=False)
      observation_, reward, done, info = env.step(action)
      test_reward += reward
      observation = observation_
      try_action_count += 1
      # 完成任务，或者 陷入局部最优失败
      if done:
        env.close()
        print("success!")
        break
      if try_action_count > 1000:
        print("failed!")
        break

if __name__ == "__main__":
  RL_prio = DQNPrioritizedReplay(n_actions=3,
                  n_features=2,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=False)
  cost_prio, his_prio  = train(RL_prio)

  RL_dueling = DQNPrioritizedReplay(n_actions=3,
                  n_features=2,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=False, dueling=True)
  cost_dueling, his_dueling  = train(RL_dueling)

  RL_prio_dueling = DQNPrioritizedReplay(n_actions=3,
                  n_features=2,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=True)
  cost_prio_dueling, his_prio_dueling  = train(RL_prio_dueling)

  plt.figure(1)
  plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='prioritized DQN')
  plt.plot(his_dueling[0, :], his_dueling[1, :] - his_dueling[1, 0], c='b', label='dueling DQN')
  plt.plot(his_prio_dueling[0, :], his_prio_dueling[1, :] - his_prio_dueling[1, 0], c='g', label='prioritized dueling DQN')
  plt.legend(loc='best')
  plt.ylabel('total training time')
  plt.xlabel('episode')
  plt.grid()

  plt.figure(2)
  plt.plot(np.array(cost_prio), c='r', label='prioritized DQN')
  plt.plot(np.array(cost_dueling), c='b', label='dueling DQN')
  plt.plot(np.array(cost_prio_dueling), c='g', label='prioritized dueling DQN')
  plt.legend(loc='best')
  plt.ylabel('cost')
  plt.xlabel('training steps')
  plt.grid()

  plt.show()

