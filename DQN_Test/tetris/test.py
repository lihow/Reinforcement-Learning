#encoding=utf-8
import torch
import cv2
from tetris import Tetris
from RL_brain import DQNPrioritizedReplay

TETRIS_WIDTH = 10
TETRIS_HEIGHT = 20
TETRIS_BLOCK_SIZE = 30

N_FEATURES = 4
ACTION_SPACE = 1

def test(model, model_path):
  if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
  else:
    torch.manual_seed(123)

  model.load_model(model_path)

  env = Tetris(width=TETRIS_WIDTH, height=TETRIS_HEIGHT, block_size=TETRIS_BLOCK_SIZE)
  env.reset()

  if torch.cuda.is_available():
    model.cuda()
  
  while True:
    next_steps = env.get_next_states()
    next_actions, next_states = zip(*next_steps.items())
    action, _ = model.choose_action(next_actions, next_states, is_random=False)
    _, done = env.step(action, render=True)

    if done:
      print("Cleared: {}".format(env.cleared_lines))
      break

if __name__ == "__main__":
  prio_duel_DQN = DQNPrioritizedReplay(n_actions=ACTION_SPACE,
                  n_features=N_FEATURES,
                  test=True)
  model_path = "model/dqn_tetris.pkl"
  test(prio_duel_DQN, model_path)


