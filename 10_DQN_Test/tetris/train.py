from RL_brain import DQNPrioritizedReplay
from random import random, randint, sample
from tetris import Tetris
import torch

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
  env = Tetris(width=TETRIS_WIDTH, height=TETRIS_HEIGHT, block_size=TETRIS_BLOCK_SIZE)
  state = torch.FloatTensor(env.reset())
  epoch = 0

  while epoch < NUM_EPOCH:
    next_steps = env.get_next_states()
    epsilon = FINAL_EPSION + (max(NUM_DECAY_EPOCHS - epoch, 0) * (
            INITIAL_EPSILON - FINAL_EPSION) / NUM_DECAY_EPOCHS)

    next_actions, next_states = zip(*next_steps.items())
    action, next_state = model.choose_action(next_actions, next_states)
    reward, done = env.step(action, render=True)

    # model.store_transition(state, action, reward, next_state)
    model.store_transition(state, reward, next_state, done)

    if done:
      final_score = env.score
      final_tetrominoes = env.tetrominoes
      final_cleared_lines = env.cleared_lines
      state = torch.FloatTensor(env.reset())
    else:
      state = next_state

    if epoch > MEMORY_SIZE:
        model.learn()

    epoch += 1

if __name__ == "__main__":
  RL_prio = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=5, memory_size=MEMORY_SIZE ,
                  e_greedy_increment=0.00005, prioritized=False, dueling=False)
  train(RL_prio)  