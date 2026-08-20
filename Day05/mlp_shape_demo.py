import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)

        )

    def forward(self,x):
        print("Original shape:",x.shape)

        x =self.flatten(x)
        print("Flattend shape:",x.shape)

        logits = self.layers(x)
        print("Logits shape:",logits.shape)

        return logits

model = MLP()

images = torch.randn(64,1,28,28)

logits = model(images)

predictions = torch.argmax(logits,dim=1)

print("Predictions shape:", predictions.shape)
