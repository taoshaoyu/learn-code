import torch
import intel_extension_for_pytorch as ipex
import torchvision
from torch import nn
import sys
import os

print(torch.__version__)
print(torchvision.__version__)

print(os.getpid())

inputs=torch.randn(5, 3, 224, 224)
m=torchvision.models.resnet50()
im=ipex.optimize(m.eval())

im(inputs)
im(inputs)
im(inputs)
im(inputs)
im(inputs)
