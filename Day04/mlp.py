import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784,128)

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(128,10)

    def forward(self,x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x

model = MLP()

x = torch.randn(32,784)

output = model(x)

print(output.shape)
