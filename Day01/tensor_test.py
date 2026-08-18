import torch

x = torch.tensor([1,2,3,4])

x=x.to("cuda")

print(x)

print(x.device)