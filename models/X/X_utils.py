import math
import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, config, input_resolution=None, dim=None, time_res=False):
        super().__init__()
        kernel_size = 3
        self.input_resolution = input_resolution
        self.time_res = time_res
        pad = (kernel_size - 1) // 2

        self.T_out = math.ceil(config.sequence_info[1]/config.patch_time)
        self.T_in = math.ceil(config.sequence_info[0]/config.patch_time)
        

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) 
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)

        if self.time_res:

            self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) 
            self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
            # self.conv3 = nn.Conv1d(self.T_in, self.T_out, kernel_size=kernel_size, stride=1, padding=pad)


        # self.bn1 = nn.BatchNorm2d(dim) # in my case not allowed!
        # self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, **kwargs):

        batch_size, sequence_length, hidden_size = x.shape

        if self.time_res:
            input = x 
            x = x.permute(0, 2, 1) 
            x = self.conv1(x) 
            x = nn.functional.leaky_relu(x)
            x = self.conv2(x)
            x = x.permute(0, 2, 1) 
            x = x + input
            # x = self.conv3(x)


        else:
        
            input = x 
            x = x.reshape(batch_size, self.input_resolution[0], self.input_resolution[1], hidden_size) 
            x = x.permute(0, 3, 1, 2)
            x = self.conv1(x) 
            # x = self.bn1(x)
            x = nn.functional.leaky_relu(x)
            x = self.conv2(x)
            # x = self.bn2(x)
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(batch_size, sequence_length, hidden_size) 
            x = x + input  


        return x