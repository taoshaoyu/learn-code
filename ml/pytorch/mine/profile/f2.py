# https://pytorch.org/tutorials/beginner/profiler.html
# 学习 pt profiler 的使用

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
prof.export_chrome_trace("f2.json")