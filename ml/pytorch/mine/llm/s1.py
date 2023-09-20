from torchinfo import summary
import torch
from torch import nn
import intel_extension_for_pytorch as ipex

rnn = nn.RNN(10, 20, 2)
summary(rnn,depth=7)

rnn.eval()
rnn = ipex.optimize(rnn)
summary(rnn,depth=7)

input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(output)
print(hn)