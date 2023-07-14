import torch

a=torch.arange(15).reshape(3,5).to(dtype=torch.float16).requires_grad_()
b=torch.arange(15).reshape(3,5).to(dtype=torch.float16)
# 1
c=a*a+b*b*b
c.backward(torch.ones(3,5))
print(a.grad)
print(b.grad)
# 2
a.grad.zero_()
c=a*a+b*b*b
c.backward(torch.ones(3,5))
print(a.grad)
print(b.grad)
# 3
a.grad.zero_()
with torch.no_grad():
    c=a*a+b*b*b
 #   c.backward(torch.ones(3,5))
    print(a.grad)
    print(b.grad)
    print(c.requires_grad)
print(c.requires_grad)