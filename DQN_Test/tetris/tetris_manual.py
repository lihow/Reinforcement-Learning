import numpy as np
from PIL import Image
import cv2
import random
import time
import copy

class Tetris:
  piece_colors = [
      (0, 0, 0),
      (255, 255, 0),
      (147, 88, 254),
      (54, 175, 144),
      (255, 0, 0),
      (102, 217, 238),
      (254, 151, 32),
      (0, 0, 255)
  ]

  pieces = [
    [[1, 1],
      [1, 1]],

    [[0, 2, 0],
      [2, 2, 2]],

    [[0, 3, 3],
      [3, 3, 0]],

    [[4, 4, 0],
      [0, 4, 4]],

    [[5, 5, 5, 5]],

    [[0, 0, 6],
      [6, 6, 6]],

    [[7, 0, 0],
      [7, 7, 7]]
  ]

  def __init__(self, height=20, width=10, block_size=30):
    self.height = height
    self.width = width
    self.block_size = block_size
    self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3), 
                              dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
    self.text_color = (200, 20, 220)
    self.reset()
  
  def reset(self):
    self.board = [[0] * self.width for _ in range(self.height)]
    self.score = 0
    self.tetrominoes = 0
    self.cleared_lines = 0
    self.bag = list(range(len(self.pieces)))
    random.shuffle(self.bag)
    self.ind = self.bag.pop()
    self.piece = [row[:] for row in self.pieces[self.ind]]
    self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
    self.gameover = False

  def new_piece(self):
    if not len(self.bag):
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
    self.ind = self.bag.pop()
    self.piece = [row[:] for row in self.pieces[self.ind]]
    self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                        "y": 0
                        }
    if self.check_collision(self.piece, self.current_pos):
        self.gameover = True

  def rotate(self, piece):
    num_rows_orig = num_cols_new = len(piece)
    num_rows_new = len(piece[0])
    rotated_array = []
    for i in range(num_rows_new):
      new_row = [0] * num_cols_new
      for j in range(num_cols_new):
        new_row[j] = piece[(num_rows_orig - 1) - j][i]
      rotated_array.append(new_row)
    return rotated_array

  def check_collision(self, piece, pos):
    future_y = pos["y"] + 1
    for y in range(len(piece)):
      for x in range(len(piece[y])):
        if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
          return True
    return False

  def check_move_collision(self, piece, pos):
    valid_xs = self.width - len(piece[0])
    if pos["x"] > valid_xs  or pos["x"] < 0:
      return True
    if pos["y"] + len(piece) > self.height - 1:
      return True
    for y in range(len(piece)):
      for x in range(len(piece[y])):
        if self.board[pos["y"] + y][pos["x"]  + x] and piece[y][x]:
          return True
    return False

  def get_current_board_state(self):
    board = [x[:] for x in self.board]
    for y in range(len(self.piece)):
      for x in range(len(self.piece[y])):
        board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
    return board

  def check_cleared_rows(self, board):
    to_delete = []
    for i, row in enumerate(board[::-1]):
        if 0 not in row:
            to_delete.append(len(board) - 1 - i)
    if len(to_delete) > 0:
        board = self.remove_row(board, to_delete)
    return len(to_delete), board

  def remove_row(self, board, indices):
    for i in indices[::-1]:
        del board[i]
        board = [[0 for _ in range(self.width)]] + board
    return board

  def store(self, piece, pos):
    board = [x[:] for x in self.board]
    for y in range(len(piece)):
      for x in range(len(piece[y])):
        if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
          board[y + pos["y"]][x + pos["x"]] = piece[y][x]
    return board

  def truncate(self, piece, pos):
    gameover = False
    last_collision_row = -1
    for y in range(len(piece)):
      for x in range(len(piece[y])):
        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
            if y > last_collision_row:
                last_collision_row = y

    if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
      while last_collision_row >= 0 and len(piece) > 1:
        gameover = True
        last_collision_row = -1
        del piece[0]
        for y in range(len(piece)):
          for x in range(len(piece[y])):
            if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                last_collision_row = y
    return gameover

  def render(self):
    if not self.gameover:
      img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
    else:
      img = [self.piece_colors[p] for row in self.board for p in row]
    img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
    img = img[..., ::-1]
    img = Image.fromarray(img, "RGB")

    img = img.resize((self.width * self.block_size, self.height * self.block_size), resample=Image.NEAREST)
    img = np.array(img)

    img[[i * self.block_size for i in range(self.height)], :, :] = 0
    img[:, [i * self.block_size for i in range(self.width)], :] = 0

    img = np.concatenate((img, self.extra_board), axis=1)

    cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    cv2.putText(img, str(self.score),
                (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

    cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    cv2.putText(img, str(self.tetrominoes),
                (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

    cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    cv2.putText(img, str(self.cleared_lines),
                (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)    

    old_pos = copy.deepcopy(self.current_pos)
    old_piece = copy.deepcopy(self.piece)
    cv2.imshow("Tetris Game", img)
    key = cv2.waitKey(500)
    if ord("a") == key:
      self.current_pos["x"] -= 1
    if ord("d") == key:
      self.current_pos["x"] += 1
    if ord("s") == key:
      self.current_pos["y"] += 2
    if ord("w") == key:
      self.piece = self.rotate(self.piece) 
    if self.check_move_collision(self.piece, self.current_pos):
      self.piece = copy.deepcopy(old_piece)
      self.current_pos = copy.deepcopy(old_pos)

  def play(self):
    while not self.gameover:
      self.new_piece()

      while not self.check_collision(self.piece, self.current_pos):
        self.current_pos["y"] += 1
        self.render()


      overflow = self.truncate(self.piece, self.current_pos)
      if overflow:
        self.gameover = True

      self.board = self.store(self.piece, self.current_pos)

      lines_cleared, self.board = self.check_cleared_rows(self.board)
      score = 1 + (lines_cleared ** 2) * self.width
      self.score += score
      self.tetrominoes += lines_cleared
      self.cleared_lines += lines_cleared


if __name__ == "__main__":
  env = Tetris()
  env.play()