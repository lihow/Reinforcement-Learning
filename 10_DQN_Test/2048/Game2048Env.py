#encoding=utf-8
import itertools
import numpy as np
import sys
from six import StringIO
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import os

class IllegalMove(Exception):
    pass

class Game2048Env:
  max_steps = 10000

  def __init__(self):
    self.size = 4
    self.w = self.size
    self.h = self.size
    self.squares = self.size * self.size
    self.score = 0
    self.set_illegal_move_reward(0.)
    self.set_max_tile(65536)

    self.max_illegal = 10
    self.num_illegal = 0

    _, self.digit_block_dict = self.read_digit_block("digit")

  def _get_info(self, info=None):
    if not info:
      info = {}
    else:
      assert type(info) == dict, 'info should be of type dict'
    
    info['highest'] = self.highest()
    info['score'] = self.score
    info['steps'] = self.steps
    return info


  def set_illegal_move_reward(self, reward):
    self.illegal_move_reward = reward
    self.reward_range = (self.illegal_move_reward, float(2**self.squares))

  def set_max_tile(self, max_tile):
    assert max_tile is None or isinstance(max_tile, int)
    self.max_tile = max_tile

  def highest(self):
    return np.max(self.Matrix)

  def reset(self):
    self.Matrix = np.zeros((self.h, self.w), np.int)
    self.score = 0
    self.steps = 0
    self.num_illegal = 0
    self.add_tile()
    self.add_tile()
    return self.Matrix, 0, False, self._get_info()

  def add_tile(self):
    possible_tiles = np.array([2, 4])
    tile_probabilities = np.array([0.9, 0.1])
    val = np.random.choice(possible_tiles, p =tile_probabilities)
    empties = self.empties()
    assert empties.shape[0]
    empty_idx = np.random.choice(empties.shape[0])
    empty = empties[empty_idx]
    self.set(empty[0], empty[1], val)

  def set(self, x, y, val):
    self.Matrix[x, y] = val
  
  def get(self, x, y):
    return self.Matrix[x, y]

  def empties(self):
    """Return a 2d numpy array with the location of empty squares."""
    """
    a = np.array([[2, 3, 0],
                  [0, 1, 1],
                  [5, 4, 0]])
    b = np.argwhere(a) => b = [[0, 2], [1, 0], [2, 2]]
    """
    return np.argwhere(self.Matrix == 0)

  def step(self, action):
    self.steps += 1
    score = 0
    done = None
    info = {
      'illegal_move': False,
    }
    try:
      score = float(self.move(action))
      self.score += score
      assert score <= 2**(self.w*self.h)
      self.add_tile()
      done = self.isend()
      reward = float(score)
    except IllegalMove as e:
      info['illegal_move'] = True
      if self.steps > self.max_steps:
        done = True
      else:
        done = False
      reward = self.illegal_move_reward
      self.num_illegal += 1
      if self.num_illegal >= self.max_illegal:
        done = True

    info = self._get_info(info)
    return self.Matrix, reward, done, info

  def isend(self):
    if self.max_tile is not None and self.highest() == self.max_tile:
      return True
    if self.steps >= self.max_steps:
      return True
    for direction in range(4):
      try:
        self.move(direction, trial=True)
        return False
      except IllegalMove:
        pass
    return True

  def move(self, direction, trial=False):
    """
    direction == 0 up
    direction == 1 Right
    direction == 2 Down
    direction == 3 Left
    """
    changed = False
    move_score = 0
    dir_div_two = int(direction / 2) 
    dir_mod_two = int(direction % 2) # 0: up or down
    shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

    rx = list(range(self.w))
    ry = list(range(self.h))

    if dir_mod_two == 0:
      # Up or down, split into columns
      for y in range(self.h):
        old = [self.get(x, y) for x in rx]
        (new, ms) = self.shift(old, shift_direction)
        move_score += ms
        if old != new:
          changed = True
          if not trial:
            for x in rx:
              self.set(x, y, new[x])
    else:
      # Left or right, split into rows
      for x in range(self.w):
        old = [self.get(x, y) for y in ry]
        (new, ms) = self.shift(old, shift_direction)
        move_score += ms
        if old != new:
          changed = True
          if not trial:
            for y in ry:
              self.set(x, y, new[y])
    if changed != True:
      raise IllegalMove

    return move_score

  def shift(self, row, direction):
    length = len(row)
    assert length == self.size
    assert direction == 0 or direction == 1

    shifted_row = [i for i in row if i != 0]
    # 因为combine为从右向左合并，所以需要改变方向  
    if direction:
      """
      a = [1, 2, 3, 4]
      a.reverse() => a = [4, 3, 2, 1]
      """
      shifted_row.reverse()
    (combined_row, move_score) = self.combine(shifted_row)
    if direction:
      combined_row.reverse()

    assert len(combined_row) == self.size
    return (combined_row, move_score)

  def combine(self, shifted_row):
    """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
    move_score = 0
    combined_row = [0] * self.size
    skip = False
    output_index = 0
    for p in self.pairwise(shifted_row):
      if skip:
        skip = False
        continue
      combined_row[output_index] = p[0]
      if p[0] == p[1]:
        combined_row[output_index] += p[1]
        move_score += p[0] + p[1]
        skip = True
      output_index += 1
    if shifted_row and not skip:
      combined_row[output_index] = shifted_row[-1]
    return (combined_row, move_score)

  def pairwise(self, iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

  def get_board(self):
    return self.Matrix

  def set_board(self, new_board):
    self.Matrix = new_board

  def render(self, mode='human'):
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    s = 'Score: {}\n'.format(self.score)
    s += 'Highest: {}\n'.format(self.highest())
    npa = np.array(self.Matrix)
    grid = npa.reshape((self.size, self.size))
    s += "{}\n\n".format(grid)
    outfile.write(s)
    return outfile

  def render_img(self):
    ground = np.array([[(125,135,146) for i in range(450)] for j in range(450)], dtype=np.uint8)
    for i in range(4):
      for j in range(4):
        x_start = 10 + 110 * i
        y_start = 10 + 110 * j
        x_end = x_start + 100
        y_end = y_start + 100
        block = self.digit_block_dict[str(self.Matrix[i][j])]
        # cv2.imshow("block", block)
        # cv2.waitKey()
        ground[x_start:x_end, y_start:y_end] = block
    cv2.imshow("2048", ground)
    cv2.waitKey(2)


  def read_digit_block(self, path):
    digit_block_dict = {}
    dirs = os.listdir(path)
    for file in dirs:
      if file[-3:] == "png":
        digit = file[:-4]
        img = cv2.imread(os.path.join(path, file), 1)
        img = cv2.resize(img, (100, 100))
        digit_block_dict[digit] = img
    if len(digit_block_dict) is not 17:
      return False, digit_block_dict
    return True, digit_block_dict

if __name__ == "__main__":
  pass


