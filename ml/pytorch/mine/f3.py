from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
#    tensor1 = torch.randn(1,16,1024,256,dtype=torch.float32)
#    tensor2 = torch.randn(1,16,1024,256,dtype=torch.float32)
#    tensor3 = torch.matmul(tensor1, tensor2.transpose(-1, -2))
#    print(tensor3)

with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    m = torch.nn.Linear(4096, 4096)
    input = torch.randn(128, 4096)
    output = m(input)
    print(output.size())