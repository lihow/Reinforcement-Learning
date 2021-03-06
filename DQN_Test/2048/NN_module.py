#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

"""
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
  padding=0, dilation=1, groups=1, bias=True)
in_channels: Number of channels in the input image
out_channels: Number of channels produced by convolution
kernel_size: Size of the convolving. Default: 1
padding: Zero-padding added to both sides of input. Default: 0
dilation: Spacing between kernel elements. Default: 1
H_out = Floor((H_in + 2 x padding - dilation x (kernel_size - 1) - 1) / stride) + 1)
if kernel_size = 3, stride = 1, padding = 1, dilation = 1:
H_out = Floor((H_in + 2 - (3 - 1) - 1) / 1) + 1) = Floor(H_in) => H_out = H_in
"""

class CNN_Net(nn.Module):
  def __init__(self, input_len, output_num, conv_size=(32, 64), fc_size=(1024, 128), out_softmax=False):
    super(CNN_Net, self).__init__()
    self.input_len = input_len
    self.output_num = output_num
    self.out_softmax = out_softmax

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, conv_size[0], kernel_size=3, stride=1, padding=1),
      # 如果指定inplace=True，则对于上层网络传递下来的tensor直接进行修改，可以少存储变量，节省运算内存
      nn.ReLU(inplace=True)
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(conv_size[0], conv_size[1], kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True)
    )

    self.fc1 = nn.Linear(conv_size[1]*self.input_len*self.input_len, fc_size[0])
    self.fc2 = nn.Linear(fc_size[0], fc_size[1])
    self.head = nn.Linear(fc_size[1], self.output_num)

  def forward(self, x):
    """
    a = torch.rand(4*28*28)
    a.shape => torch.Size([3136])
    b = a.reshape(-1, 1, 28, 28)
    b.shape => torch.Size(4, 1, 28, 28)
    """
    x = x.reshape(-1,1,self.input_len, self.input_len)
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    output = self.head(x)
    if self.out_softmax:
      output = F.softmax(output, dim=1)
    return output

class DuelingNet(nn.Module):
  def __init__(self, input_len, output_num):
    super(DuelingNet, self).__init__()
    self.input_len = input_len
    # input [4, 4, 1]
    self.cnn_layer = nn.Sequential(
      # H_out = Floor((H_in + 2 x padding - dilation x (kernel_size - 1) - 1) / stride) + 1)
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False), # [4, 4, 32]
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False), # [2, 2, 64]
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False), # [2, 2, 128]
      nn.BatchNorm2d(128)
    )
    self.fc_layer = nn.Sequential(
      nn.Linear(2*2*128, 1024),
      nn.Linear(1024, 512)
    )
    self.val_layer = nn.Linear(512, 1)
    self.act_layer = nn.Linear(512, output_num)
  
  def forward(self, x):
    x = x.reshape(-1,1,self.input_len, self.input_len)
    x = self.cnn_layer(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layer(x)
    val = self.val_layer(x)
    act = self.act_layer(x)
    action_val = val + act - val.mean()
    return action_val
