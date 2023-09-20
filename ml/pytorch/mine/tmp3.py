
import torch
import intel_extension_for_pytorch as ipex
import os

a=torch.tensor(1)

def foo(a):
    b = torch.conv2d(a, torch.randn(1, 1, 1, 1)) # not fusible
    x = torch.mul(b, b)                          # fusible
    y = torch.sin(x)                             # fusible
    z = torch.mul(y, y)                          # fusible
    return z

#torch._C._jit_override_can_fuse_on_cpu(True)
print(os.getpid())

a = torch.randn(1, 1, 128, 128)

scripted = torch.jit.script(foo)

# do several runs:
for _ in range(10):
    scripted(a)