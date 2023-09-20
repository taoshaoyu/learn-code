import torch

import torchvision
from torch import nn
import sys
import os

print(torch.__version__)
print(torchvision.__version__)

m=torchvision.models.resnet50().eval()
print(os.getpid())

#import intel_extension_for_pytorch as ipex


inputs=torch.randn(5, 3, 224, 224)
#m=torch.jit.trace(m,inputs)
#m=torch.jit.freeze(m)
m(inputs)
m(inputs)
