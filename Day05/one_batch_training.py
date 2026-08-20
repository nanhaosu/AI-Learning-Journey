import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


model = MLP().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

images, labels = next(iter(loader))

# TODO 1：把images和labels移动到device
images = images.to(device)
labels = labels.to(device)

# TODO 2：清空上一轮梯度
optimizer.zero_grad()

# TODO 3：前向传播，得到logits
logits = model(images)

# TODO 4：计算损失
loss =loss_function(logits,labels)

print("Loss before update:", loss.item())

# TODO 5：反向传播
loss.backward()

# TODO 6：更新参数
optimizer.step()

# 使用更新后的模型，再计算同一个batch的损失
with torch.no_grad():
    new_logits = model(images)
    new_loss = loss_function(new_logits, labels)

print("Loss after update:", new_loss.item())