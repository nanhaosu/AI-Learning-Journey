import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = datasets.CIFAR10(
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

# 把你在cnn_shape_demo.py中写的CNN类放在这里
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # 两次池化后：32×32 → 16×16 → 8×8
        # TODO 1：补全Linear的输入特征数量
        self.classifier = nn.Linear(
            2048,
            10
        )

    def forward(self, x):
        print("Input:       ", x.shape)

        x = self.conv1(x)
        print("After conv1: ", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("After pool1: ", x.shape)

        x = self.conv2(x)
        print("After conv2: ", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("After pool2: ", x.shape)

        x = self.flatten(x)
        print("After flatten:", x.shape)

        logits = self.classifier(x)
        print("Logits:      ", logits.shape)

        return logits

model = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

images, labels = next(iter(loader))
images = images.to(device)
labels = labels.to(device)

# TODO 1：清除旧梯度
optimizer.zero_grad()

# TODO 2：前向传播
logits = model(images)

# TODO 3：计算loss
loss = loss_function(logits,labels)

print("Images:", images.shape)
print("Logits:", logits.shape)
print("Labels:", labels.shape)
print("Loss before update:", loss.item())

# TODO 4：反向传播
loss.backward()

# TODO 5：更新参数
optimizer.step()

# 使用更新后的模型重新计算同一个batch
with torch.no_grad():
    new_logits = model(images)
    new_loss = loss_function(new_logits, labels)

print("Loss after update:", new_loss.item())