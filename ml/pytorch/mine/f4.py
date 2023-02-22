#ONEDNN VERBOSE is working
# Line11 开始有输出
import torch
import torch.nn as nn
import os

print(os.getpid())

m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)


m = torch.nn.Linear(4096, 4096)
input = torch.randn(128, 4096)
output = m(input)
print(output.size())
