#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from NN_module import CNN_Net

class DQN():
  batch_size = 128
  lr = 1e-4
  epsilon = 0.15
  memory_capacity =  int(1e4)
  gamma = 0.99
  q_network_iteration = 200
  oft_update_theta = 0.1