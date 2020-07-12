#encoding=utf-8
from RL_brain import DQNPrioritizedReplay
from random import random, randint, sample
from tetris import Tetris
import torch
import matplotlib.pyplot as plt

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

def train(model):
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

    cleared_lines.append(final_cleared_lines)
  return cleared_lines


if __name__ == "__main__":
  DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=5, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=False, dueling=False)
  prio_DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=5, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=False)
  prio_duel_DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=5, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=True, dueling=True)

  cl_DQN = train(DQN) 
  cl_prio_DQN = train(prio_DQN)  
  cl_prio_duel_DQN = train(prio_duel_DQN)   

  plt.plot(np.array(cl_DQN), c='r', label='DQN')
  plt.plot(np.array(cl_prio_DQN), c='b', label='prio_DQN')
  plt.plot(np.array(cl_prio_duel_DQN), c='g', label='prio_duel_DQN')
  plt.legend(loc='best')
  plt.ylabel('cleared lines')
  plt.xlabel('training steps')
  plt.grid()  
  plt.savefig('result.jpg')
  # plt.show()