import torch
import torch.nn as nn


dropout = nn.Dropout(p=0.5)
x = torch.ones(10)

print("Input:")
print(x)

# TODO 1：切换到训练模式
dropout.train()

print("\nTraining mode:")
for _ in range(3):
    print(dropout(x))

# TODO 2：切换到评估模式
dropout.eval()

print("\nEvaluation mode:")
with torch.no_grad():
    for _ in range(3):
        print(dropout(x))