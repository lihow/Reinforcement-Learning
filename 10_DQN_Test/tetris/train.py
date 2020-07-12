#encoding=utf-8
from RL_brain import DQNPrioritizedReplay
from random import random, randint, sample
from tetris import Tetris
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

TETRIS_WIDTH = 10
TETRIS_HEIGHT = 20
TETRIS_BLOCK_SIZE = 30

N_FEATURES = 4
ACTION_SPACE = 1

NUM_EPOCH = 3000
NUM_DECAY_EPOCHS = 2000
INITIAL_EPSILON = 1
FINAL_EPSION = 1e-3
MEMORY_SIZE = 32
#当为普通DQN时设置和batch_size相关效果最好?
SAVE_INTERVAL = 500

def train(model, saved_path="model"):
  if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
  else:
    torch.manual_seed(123)
  cleared_lines = []
  env = Tetris(width=TETRIS_WIDTH, height=TETRIS_HEIGHT, block_size=TETRIS_BLOCK_SIZE)
  state = torch.FloatTensor(env.reset())

  epoch = 0
  while epoch < NUM_EPOCH:
    # epsilon[1, 0]随着训练次数逐渐减少，使随机选择的动作减少，减少学习错误的次数
    epsilon = FINAL_EPSION + (max(NUM_DECAY_EPOCHS - epoch, 0) * (
            INITIAL_EPSILON - FINAL_EPSION) / NUM_DECAY_EPOCHS)

    next_steps = env.get_next_states()
    next_actions, next_states = zip(*next_steps.items())

    action, next_state = model.choose_action(next_actions, next_states, epsilon)
    reward, done = env.step(action, render=False)

    # model.store_transition(state, action, reward, next_state)
    model.store_transition(state, reward, next_state, done)

    if done:
      final_score = env.score
      final_tetrominoes = env.tetrominoes
      final_cleared_lines = env.cleared_lines
      state = torch.FloatTensor(env.reset())
    else:
      state = next_state
      continue

    if epoch > MEMORY_SIZE:
        model.learn()

    epoch += 1
    print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
        epoch,
        NUM_EPOCH,
        action,
        final_score,
        final_tetrominoes,
        final_cleared_lines))
    if epoch > 0 and epoch % SAVE_INTERVAL == 0:
      model.save_model("{}tetris_{}.pkl".format(saved_path, epoch))

    cleared_lines.append(final_cleared_lines)

  model.save_model("{}tetris.pkl".format(saved_path))
  return cleared_lines

def plot_train(DQN, prio_DQN, prio_duel_DQN):
  plt.figure(figsize=(18, 54), dpi=200)
  plt.figure(1)
  ax1 = plt.subplot(3, 1, 1)
  plt.plot(np.array(cl_DQN), c='r')
  plt.ylabel('cleared lines')
  plt.xlabel('training steps')

  ax2 = plt.subplot(3, 1, 2)
  plt.plot(np.array(cl_prio_DQN), c='b')
  plt.ylabel('cleared lines')
  plt.xlabel('training steps')

  ax3 = plt.subplot(3, 1, 3)
  plt.plot(np.array(cl_prio_duel_DQN), c='g')
  plt.ylabel('cleared lines')
  plt.xlabel('training steps')


  ax1.set_title("DQN")
  ax2.set_title("prio_DQN")
  ax3.set_title("prio_duel_DQN")

  plt.tight_layout()
  plt.savefig('result.jpg')
  # plt.show()

if __name__ == "__main__":
  DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=5, memory_size=MEMORY_SIZE, batch_size=32,
                  e_greedy_increment=0.00005, prioritized=False, dueling=False)

  prio_DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=5, memory_size=128, batch_size=32,
                  e_greedy_increment=0.00005, prioritized=True, dueling=False)

  prio_duel_DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=5, memory_size=128, batch_size=32,
                  e_greedy_increment=0.00005, prioritized=True, dueling=True)


  cl_DQN = train(DQN, "model/dqn_") 
  cl_prio_DQN = train(prio_DQN, "model/prio_dqn_")  
  cl_prio_duel_DQN = train(prio_duel_DQN, "model/duel_prio_dqn_")   

  plot_train(cl_DQN, cl_prio_DQN, cl_prio_duel_DQN)