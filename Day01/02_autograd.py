import torch

x = torch.tensor(
    3.0,
    requires_grad=True
)

y = x ** 2

y.backward()

print(y)

print(x.grad)